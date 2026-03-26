#!/usr/bin/env python3
# Copyright 2026 Innodisk Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def dequant(arr: np.ndarray, scale: float, zp: int) -> np.ndarray:
    if scale and scale != 0:
        return (arr.astype(np.float32) - float(zp)) * float(scale)
    return arr.astype(np.float32)


def find_boxes_scores(output_details):
    box_candidates = []
    cls_candidates = []

    for od in output_details:
        shp = tuple(int(v) for v in od["shape"])
        if len(shp) != 3 or shp[0] != 1:
            continue
        if shp[1] in (4, 64):
            box_candidates.append((od, 2))
        elif shp[2] in (4, 64):
            box_candidates.append((od, 1))
        else:
            cls_candidates.append(od)

    if len(box_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one raw box output [1,4/64,N], "
            f"got {[tuple(od['shape']) for od in output_details]}"
        )
    out_box, anchor_axis = box_candidates[0]
    anchor_count = int(out_box["shape"][anchor_axis])
    matching_cls = [
        od
        for od in cls_candidates
        if int(od["shape"][1]) == anchor_count or int(od["shape"][2]) == anchor_count
    ]
    if not matching_cls:
        raise RuntimeError(
            "Expected class output [1,C,N] or [1,N,C] with the same anchor dimension as box output, "
            f"got {[tuple(od['shape']) for od in output_details]}"
        )
    out_cls = sorted(
        matching_cls,
        key=lambda od: (
            int(od["shape"][2])
            if int(od["shape"][1]) == anchor_count
            else int(od["shape"][1])
        ),
        reverse=True,
    )[0]

    return out_box, out_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-qnn", action="store_true")
    ap.add_argument("--qnn-lib", default="/usr/lib/libQnnTFLiteDelegate.so")
    ap.add_argument("--backend", default="htp")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.model, "rb") as f:
        model_content = f.read()

    delegates = []
    if not args.no_qnn:
        delegate = tf.lite.experimental.load_delegate(
            args.qnn_lib,
            options={"backend_type": args.backend},
        )
        delegates = [delegate]

    interpreter = tf.lite.Interpreter(
        model_content=model_content,
        experimental_delegates=delegates,
    )
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    out_box, out_cls = find_boxes_scores(output_details)

    input_files = sorted(input_dir.glob("*.npy"))
    if not input_dir.exists():
        raise RuntimeError(
            f"Input directory does not exist: {input_dir}. "
            "Host-side shared input generation or adb push likely failed."
        )
    if not input_files:
        raise RuntimeError(
            f"Input directory contains no .npy files: {input_dir}. "
            "Host-side shared input generation selected zero images or adb push did not upload any inputs."
        )

    for in_file in input_files:
        x = np.load(in_file)
        interpreter.set_tensor(input_detail["index"], x)
        interpreter.invoke()

        box_raw = interpreter.get_tensor(out_box["index"])
        cls_raw = interpreter.get_tensor(out_cls["index"])

        b_sc, b_zp = out_box.get("quantization", (0.0, 0))
        c_sc, c_zp = out_cls.get("quantization", (0.0, 0))

        boxes = dequant(box_raw, b_sc, b_zp)
        scores = dequant(cls_raw, c_sc, c_zp)

        stem = in_file.stem
        np.save(output_dir / f"{stem}_boxes.npy", boxes)
        np.save(output_dir / f"{stem}_scores.npy", scores)

        print(f"done: {in_file.name}")


if __name__ == "__main__":
    main()
