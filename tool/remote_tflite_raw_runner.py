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
import time

import numpy as np

# Runtime change (TF -> LiteRT): ai-edge-litert replaces tensorflow as the LiteRT
# interpreter. It exposes Interpreter/load_delegate with the same signatures as
# tf.lite.Interpreter / tf.lite.experimental.load_delegate.
from ai_edge_litert.interpreter import Interpreter, load_delegate

# tool/postprocess is pushed next to this script (see push_postprocess_package).
try:
    from tool.postprocess.outputs import save_avg_invoke_ms, save_raw_outputs
except ModuleNotFoundError:
    from postprocess.outputs import save_avg_invoke_ms, save_raw_outputs


def dequant(arr: np.ndarray, scale: float, zp: int) -> np.ndarray:
    if scale and scale != 0:
        return (arr.astype(np.float32) - float(zp)) * float(scale)
    return arr.astype(np.float32)


def _output_order_key(name, position):
    """Graph output order; converted models name their outputs output_0, output_1."""
    if name and name.startswith("output_") and name[len("output_") :].isdigit():
        return int(name[len("output_") :]), position
    return position, position


def find_boxes_scores(output_details, anchor_count=8400):
    """Pick the (box, class) outputs.

    A class tensor with 4 or 64 classes has the same shape as a box tensor, so
    shape alone cannot tell them apart; ties are broken by output order, since
    every iQ-Foundry export returns (boxes, scores).
    """
    channels = {}
    for idx, od in enumerate(output_details):
        shp = tuple(int(v) for v in od["shape"])
        if len(shp) != 3 or shp[0] != 1:
            continue
        if shp[2] == anchor_count:
            channels[idx] = shp[1]
        elif shp[1] == anchor_count:
            channels[idx] = shp[2]

    order = {
        idx: _output_order_key(output_details[idx].get("name"), idx) for idx in channels
    }
    box_candidates = [idx for idx, ch in channels.items() if ch in (4, 64)]
    if not box_candidates:
        raise RuntimeError(
            f"Expected a raw box output [1,4/64,{anchor_count}], "
            f"got {[tuple(od['shape']) for od in output_details]}"
        )
    box_idx = min(box_candidates, key=order.__getitem__)
    cls_candidates = [idx for idx in channels if idx != box_idx]
    if not cls_candidates:
        raise RuntimeError(
            "Expected class output [1,C,N] or [1,N,C] with the same anchor "
            "dimension as box output, "
            f"got {[tuple(od['shape']) for od in output_details]}"
        )
    cls_idx = min(cls_candidates, key=order.__getitem__)
    return output_details[box_idx], output_details[cls_idx]


def find_extra_outputs(output_details, used_indices, count):
    """The outputs after boxes and scores (pose: kpts; segment: mask_coef, proto), in graph order."""
    rest = [
        (idx, od) for idx, od in enumerate(output_details) if od["index"] not in used_indices
    ]
    if len(rest) != count:
        raise RuntimeError(
            f"Expected {count} outputs besides boxes and scores, got "
            f"{[tuple(int(v) for v in od['shape']) for od in output_details]}"
        )
    rest.sort(key=lambda item: _output_order_key(item[1].get("name"), item[0]))
    return [od for _, od in rest]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-qnn", action="store_true")
    ap.add_argument("--qnn-lib", default="/usr/lib/libQnnTFLiteDelegate.so")
    ap.add_argument("--backend", default="htp")
    ap.add_argument("--task", choices=["detect", "pose", "segment", "classify"], default="detect")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.model, "rb") as f:
        model_content = f.read()

    delegates = []
    if not args.no_qnn:
        # Runtime change (TF -> LiteRT): was tf.lite.experimental.load_delegate.
        delegate = load_delegate(
            args.qnn_lib,
            options={"backend_type": args.backend},
        )
        delegates = [delegate]

    # Runtime change (TF -> LiteRT): was tf.lite.Interpreter.
    interpreter = Interpreter(
        model_content=model_content,
        experimental_delegates=delegates,
    )
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    if args.task == "classify":
        # A classifier has one output, the raw class scores.
        if len(output_details) != 1:
            raise RuntimeError(
                "Expected one output (logits) for --task classify, got "
                f"{[tuple(int(v) for v in od['shape']) for od in output_details]}"
            )
    else:
        out_box, out_cls = find_boxes_scores(output_details)
        extra_roles = {"detect": (), "pose": ("kpts",), "segment": ("mask_coef", "proto")}[
            args.task
        ]
        extra_outs = dict(
            zip(
                extra_roles,
                find_extra_outputs(
                    output_details, {out_box["index"], out_cls["index"]}, len(extra_roles)
                ),
            )
        )

    input_files = sorted(input_dir.glob("*.npy"))
    if not input_dir.exists():
        raise RuntimeError(
            f"Input directory does not exist: {input_dir}. "
            "Host-side shared input generation or adb push likely failed."
        )
    if not input_files:
        raise RuntimeError(
            f"Input directory contains no .npy files: {input_dir}. "
            "Host-side shared input generation selected zero images or adb "
            "push did not upload any inputs."
        )

    invoke_seconds = []
    for in_file in input_files:
        x = np.load(in_file)
        expected_dtype = np.dtype(input_detail["dtype"])
        expected_shape = tuple(int(v) for v in input_detail["shape"])
        if tuple(int(v) for v in x.shape) != expected_shape:
            raise RuntimeError(
                f"Input tensor shape mismatch for {in_file.name}: "
                f"expected {expected_shape}, got {tuple(int(v) for v in x.shape)}"
            )
        if x.dtype != expected_dtype:
            if np.issubdtype(x.dtype, np.floating) and np.issubdtype(
                expected_dtype, np.floating
            ):
                x = x.astype(expected_dtype, copy=False)
            elif np.issubdtype(x.dtype, np.integer) and np.issubdtype(
                expected_dtype, np.integer
            ):
                x = x.astype(expected_dtype, copy=False)
            else:
                raise RuntimeError(
                    f"Input tensor dtype mismatch for {in_file.name}: "
                    f"expected {expected_dtype}, got {x.dtype}"
                )
        interpreter.set_tensor(input_detail["index"], x)
        t0 = time.perf_counter()
        interpreter.invoke()
        invoke_seconds.append(time.perf_counter() - t0)

        stem = in_file.stem
        if args.task == "classify":
            od = output_details[0]
            sc, zp = od.get("quantization", (0.0, 0))
            outputs = {"logits": dequant(interpreter.get_tensor(od["index"]), sc, zp)}
            save_raw_outputs(output_dir, stem, outputs)
            print(f"done: {in_file.name}")
            continue

        box_raw = interpreter.get_tensor(out_box["index"])
        cls_raw = interpreter.get_tensor(out_cls["index"])

        b_sc, b_zp = out_box.get("quantization", (0.0, 0))
        c_sc, c_zp = out_cls.get("quantization", (0.0, 0))

        boxes = dequant(box_raw, b_sc, b_zp)
        scores = dequant(cls_raw, c_sc, c_zp)

        outputs = {"boxes": boxes, "scores": scores}
        for role, od in extra_outs.items():
            sc, zp = od.get("quantization", (0.0, 0))
            outputs[role] = dequant(interpreter.get_tensor(od["index"]), sc, zp)
        save_raw_outputs(output_dir, stem, outputs)

        print(f"done: {in_file.name}")
    save_avg_invoke_ms(output_dir, invoke_seconds)


if __name__ == "__main__":
    main()
