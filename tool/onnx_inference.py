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
"""Shared ONNX Runtime inference helpers for test mode and mAP evaluation."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

cv2 = None
np = None
onnx = None
ort = None
yaml = None

IMG_W = 640
IMG_H = 640
REG_MAX = 16
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
IQ9_ARCH_ALIASES = {"aarch64", "arm64"}
DEFAULT_ORT_QNN_WHEEL = str(
    Path(__file__).resolve().parent.parent
    / "wheels"
    / "onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl"
)
EXPECTED_BOX_MODES = {
    "yolov10": "dfl64",
    "yolov11": "dfl64",
    "yolov26": "ltrb4",
}
DEFAULT_TFLITE_QNN_LIB = "/usr/lib/libQnnTFLiteDelegate.so"
DEFAULT_ORT_QNN_BACKEND_PATH = "libQnnHtp.so"


@dataclass(frozen=True)
class QuantParams:
    scale: float
    zero_point: int


@dataclass(frozen=True)
class ORTModelMeta:
    model_path: str
    model_type: str
    precision: str
    input_name: str
    input_layout: str
    input_dtype: object
    box_output_name: str
    class_output_name: str
    box_mode: str
    class_count: int
    box_quant: QuantParams | None = None
    class_quant: QuantParams | None = None


@dataclass
class ResolvedONNXArtifact:
    requested_path: str
    model_path: str
    sidecars: tuple[Path, ...]
    temp_dir_obj: tempfile.TemporaryDirectory | None = None

    def cleanup(self) -> None:
        if self.temp_dir_obj is not None:
            self.temp_dir_obj.cleanup()
            self.temp_dir_obj = None


def _ensure_runtime_deps() -> None:
    global cv2, np, yaml
    if cv2 is None:
        import cv2 as _cv2

        cv2 = _cv2
    if np is None:
        import numpy as _np

        np = _np
    if yaml is None:
        try:
            import yaml as _yaml
        except ModuleNotFoundError:
            yaml = False
        else:
            yaml = _yaml


def _ensure_ort() -> None:
    global ort
    if ort is None:
        try:
            import onnxruntime as _ort
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "onnxruntime is required for ONNX Runtime execution paths."
            ) from exc
        required_prefix = os.environ.get("IQF_REQUIRE_ORT_PREFIX")
        if required_prefix:
            origin = getattr(_ort, "__file__", None)
            resolved_origin = str(Path(origin).resolve()) if origin else None
            if resolved_origin is None:
                raise RuntimeError(
                    "onnxruntime was imported, but its module origin could not be "
                    "resolved while enforcing the remote venv requirement."
                )
            origin_path = Path(resolved_origin)
            required_prefix_path = Path(required_prefix).resolve()
            try:
                origin_path.relative_to(required_prefix_path)
            except ValueError as exc:
                raise RuntimeError(
                    "Remote ONNX Runtime must be imported from "
                    f"{required_prefix_path}, but resolved to {resolved_origin}."
                ) from exc
        ort = _ort


def _ensure_onnx() -> None:
    global onnx
    if onnx is None:
        try:
            import onnx as _onnx
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The Python package 'onnx' is required to inspect W8A16 ONNX models."
            ) from exc
        onnx = _onnx


def _adb_prefix(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def _run_cmd(cmd: Sequence[str]) -> None:
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(list(cmd), check=True)


def _adb_shell(serial: str | None, command: str) -> None:
    _run_cmd(_adb_prefix(serial) + ["shell", command])


def _adb_push(serial: str | None, src: str, dst: str) -> None:
    _run_cmd(_adb_prefix(serial) + ["push", src, dst])


def _adb_pull(serial: str | None, src: str, dst: str) -> None:
    _run_cmd(_adb_prefix(serial) + ["pull", src, dst])


def _resolve_remote_ort_prefix(remote_python: str) -> str:
    return str(Path(remote_python).parent.parent)


def _format_remote_onnx_command(cmd: Sequence[str], remote_python: str) -> str:
    return (
        "set -e; "
        f"export IQF_REQUIRE_ORT_PREFIX={shlex.quote(_resolve_remote_ort_prefix(remote_python))}; "
        + " ".join(shlex.quote(c) for c in cmd)
    )


def _strip_trailing_slash(path_value: str) -> str:
    stripped = path_value.rstrip("/")
    return stripped or "/"


def _is_iq9_native_runtime() -> bool:
    return platform.machine().lower() in IQ9_ARCH_ALIASES


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sigmoid_clip(x):
    _ensure_runtime_deps()
    x = x.astype(np.float32, copy=False)
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def letterbox_bgr(
    image_bgr,
    new_shape: tuple[int, int] = (IMG_H, IMG_W),
    color: tuple[int, int, int] = (114, 114, 114),
):
    _ensure_runtime_deps()
    h0, w0 = image_bgr.shape[:2]
    new_h, new_w = new_shape

    ratio = min(new_h / h0, new_w / w0)
    resized_w = int(round(w0 * ratio))
    resized_h = int(round(h0 * ratio))

    dw = (new_w - resized_w) / 2.0
    dh = (new_h - resized_h) / 2.0
    resized = cv2.resize(
        image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, (float(ratio), float(dw), float(dh))


def load_class_names(yaml_path: str) -> list[str]:
    _ensure_runtime_deps()
    if yaml is False:
        names: list[str] = []
        in_names = False

        with open(yaml_path, encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("names:"):
                    in_names = True
                    continue
                if not in_names:
                    continue
                if raw_line.startswith("  - "):
                    names.append(raw_line.split("-", 1)[1].strip())
                    continue
                if ":" in stripped and not raw_line.startswith(" "):
                    break

        if not names:
            raise RuntimeError(f"Could not parse class names from {yaml_path}")
        return names

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]
    if isinstance(names, dict):
        ordered = []
        for key, value in names.items():
            try:
                idx = int(key)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "YAML 'names' dict keys must be integer-like class ids."
                ) from exc
            ordered.append((idx, value))
        ordered.sort(key=lambda pair: pair[0])
        indices = [idx for idx, _ in ordered]
        expected = list(range(len(indices)))
        if indices != expected:
            raise RuntimeError(
                "YAML 'names' dict class ids must be contiguous and start at 0. "
                f"Expected {expected}, got {indices}."
            )
        return [str(value) for _, value in ordered]
    return [str(x) for x in names]


def build_class_colors(num_classes: int) -> list[tuple[int, int, int]]:
    _ensure_runtime_deps()
    n = max(1, num_classes)
    rng = np.random.default_rng()
    hues = rng.random(n) * 179.0
    sats = rng.uniform(170.0, 255.0, n)
    vals = rng.uniform(190.0, 255.0, n)
    hsv = np.stack([hues, sats, vals], axis=1).astype(np.uint8).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return [(int(c[0]), int(c[1]), int(c[2])) for c in bgr]


def class_color(
    class_colors: list[tuple[int, int, int]], class_id: int
) -> tuple[int, int, int]:
    if not class_colors:
        return (0, 255, 0)
    return class_colors[int(class_id) % len(class_colors)]


def collect_image_files(img_dir: str) -> list[Path]:
    image_dir = Path(img_dir)
    if not image_dir.is_dir():
        raise RuntimeError(f"Image directory not found: {img_dir}")
    files = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
    )
    if not files:
        raise RuntimeError(f"No images found in {img_dir}")
    return files


def build_anchor_centers_8400():
    _ensure_runtime_deps()
    centers = []
    strides = []
    for grid, stride in ((80, 8.0), (40, 16.0), (20, 32.0)):
        ys, xs = np.meshgrid(
            np.arange(grid, dtype=np.float32),
            np.arange(grid, dtype=np.float32),
            indexing="ij",
        )
        ax = (xs + 0.5) * stride
        ay = (ys + 0.5) * stride
        centers.append(np.stack([ax, ay], axis=-1).reshape(-1, 2))
        strides.append(np.full((grid * grid,), stride, dtype=np.float32))
    return np.concatenate(centers, axis=0), np.concatenate(strides, axis=0)


def dfl_decode_to_ltrb_pixels(box64x_n, strides):
    _ensure_runtime_deps()
    num_anchors = box64x_n.shape[1]
    reg = box64x_n.reshape(4, REG_MAX, num_anchors).astype(np.float32)
    reg = reg - reg.max(axis=1, keepdims=True)
    exp = np.exp(reg)
    prob = exp / exp.sum(axis=1, keepdims=True)
    proj = np.arange(REG_MAX, dtype=np.float32).reshape(1, REG_MAX, 1)
    dist_bins = (prob * proj).sum(axis=1)
    return dist_bins * strides.reshape(1, num_anchors)


def dfl_to_xyxy_pixels(box64x_n, centers_xy, strides):
    _ensure_runtime_deps()
    ltrb = dfl_decode_to_ltrb_pixels(box64x_n, strides)
    left, top, right, bottom = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
    ax, ay = centers_xy[:, 0], centers_xy[:, 1]
    xyxy = np.stack([ax - left, ay - top, ax + right, ay + bottom], axis=1)
    xyxy = xyxy.astype(np.float32)
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0.0, IMG_W)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0.0, IMG_H)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0.0, IMG_W)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0.0, IMG_H)
    return xyxy


def ltrb_grid_to_xyxy_pixels(box4x_n, centers_xy, strides):
    _ensure_runtime_deps()
    box = np.maximum(box4x_n.astype(np.float32).T, 0.0)
    left, top, right, bottom = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    ax, ay = centers_xy[:, 0], centers_xy[:, 1]
    xyxy = np.stack(
        [
            ax - left * strides,
            ay - top * strides,
            ax + right * strides,
            ay + bottom * strides,
        ],
        axis=1,
    ).astype(np.float32)
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0.0, IMG_W)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0.0, IMG_H)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0.0, IMG_W)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0.0, IMG_H)
    return xyxy


def classwise_nms_xyxy(
    xyxy,
    scores,
    class_ids,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> list[int]:
    _ensure_runtime_deps()
    kept: list[int] = []

    for cid in np.unique(class_ids):
        idx = np.where(class_ids == cid)[0]
        if idx.size == 0:
            continue
        idx = idx[scores[idx] >= conf_thres]
        if idx.size == 0:
            continue

        tlwh = []
        cls_scores = scores[idx].tolist()
        for x1, y1, x2, y2 in xyxy[idx].astype(np.float32):
            tlwh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

        keep_local = cv2.dnn.NMSBoxes(tlwh, cls_scores, conf_thres, iou_thres)
        if len(keep_local) == 0:
            continue

        if isinstance(keep_local, np.ndarray):
            keep_local = keep_local.flatten().astype(int).tolist()
        else:
            keep_local = [int(i) for i in keep_local]
        kept.extend(idx[keep_local].tolist())

    kept = sorted(kept, key=lambda i: float(scores[i]), reverse=True)
    if max_det > 0 and len(kept) > max_det:
        kept = kept[:max_det]
    return kept


def sort_and_cap(scores, max_det: int) -> list[int]:
    _ensure_runtime_deps()
    keep = np.argsort(scores)[::-1].astype(int).tolist()
    if max_det > 0 and len(keep) > max_det:
        keep = keep[:max_det]
    return keep


def _extract_box_layout(shape: Sequence[int]) -> tuple[int, int] | None:
    shp = tuple(int(x) for x in shape)
    if len(shp) != 3 or shp[0] != 1:
        return None
    if shp[1] in (4, 64):
        return int(shp[1]), int(shp[2])
    if shp[2] in (4, 64):
        return int(shp[2]), int(shp[1])
    return None


def _class_dim_for_anchor_count(shape: Sequence[int], anchor_count: int) -> int | None:
    shp = tuple(int(x) for x in shape)
    if len(shp) != 3 or shp[0] != 1:
        return None
    if int(shp[2]) == anchor_count:
        return int(shp[1])
    if int(shp[1]) == anchor_count:
        return int(shp[2])
    return None


def _normalize_output_layout(
    tensor,
    *,
    tensor_name: str,
    channel_sizes: tuple[int, ...] | None = None,
    anchor_count: int | None = None,
):
    _ensure_runtime_deps()
    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
        raise RuntimeError(
            f"Unexpected {tensor_name} tensor rank/shape: "
            f"{tuple(int(x) for x in tensor.shape)}"
        )
    if channel_sizes is not None:
        if int(tensor.shape[1]) in channel_sizes:
            return tensor
        if int(tensor.shape[2]) in channel_sizes:
            return np.transpose(tensor, (0, 2, 1))
        raise RuntimeError(
            f"Unexpected {tensor_name} tensor shape: "
            f"{tuple(int(x) for x in tensor.shape)}"
        )
    if anchor_count is not None:
        if int(tensor.shape[2]) == anchor_count:
            return tensor
        if int(tensor.shape[1]) == anchor_count:
            return np.transpose(tensor, (0, 2, 1))
        raise RuntimeError(
            f"Unexpected {tensor_name} tensor shape: "
            f"{tuple(int(x) for x in tensor.shape)}"
        )
    raise ValueError("Either channel_sizes or anchor_count must be provided.")


def ort_type_to_numpy_dtype(ort_type: str):
    _ensure_runtime_deps()
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(uint16)": np.uint16,
        "tensor(int16)": np.int16,
        "tensor(uint32)": np.uint32,
        "tensor(int32)": np.int32,
        "tensor(uint64)": np.uint64,
        "tensor(int64)": np.int64,
        "tensor(bool)": np.bool_,
    }
    if ort_type not in mapping:
        raise RuntimeError(f"Unsupported ORT tensor type: {ort_type}")
    return np.dtype(mapping[ort_type])


def _resolve_input_layout(shape: Sequence[int]) -> str:
    dims = [int(v) for v in shape]
    if dims == [1, 3, IMG_H, IMG_W]:
        return "nchw"
    if dims == [1, IMG_H, IMG_W, 3]:
        return "nhwc"
    raise RuntimeError(
        "Expected fixed ONNX input shape [1,3,640,640] or [1,640,640,3], "
        f"got {dims}"
    )


def _load_w8a16_output_quant_params(model_path: str) -> dict[str, QuantParams]:
    _ensure_runtime_deps()
    _ensure_onnx()
    from onnx import numpy_helper

    model = onnx.load(model_path, load_external_data=True)
    initializers = {
        init.name: numpy_helper.to_array(init) for init in model.graph.initializer
    }
    producers = {}
    for node in model.graph.node:
        for output_name in node.output:
            producers[output_name] = node

    output_quants: dict[str, QuantParams] = {}
    for output_meta in model.graph.output:
        output_name = output_meta.name
        producer = producers.get(output_name)
        if producer is None or producer.op_type != "QuantizeLinear" or len(producer.input) < 3:
            raise RuntimeError(
                f"Graph output '{output_name}' is not produced by a terminal QuantizeLinear node."
            )

        scale_name = producer.input[1]
        zero_point_name = producer.input[2]
        if scale_name not in initializers or zero_point_name not in initializers:
            raise RuntimeError(
                f"Missing quantization initializers for graph output '{output_name}'."
            )

        scale = float(np.asarray(initializers[scale_name]).reshape(-1)[0])
        zero_point = int(np.asarray(initializers[zero_point_name]).reshape(-1)[0])
        output_quants[output_name] = QuantParams(scale=scale, zero_point=zero_point)
    return output_quants


def _validate_model_type_compatibility(meta: ORTModelMeta, model_type: str) -> None:
    expected_mode = EXPECTED_BOX_MODES.get(model_type)
    if expected_mode is None:
        raise RuntimeError(f"Unsupported model_type: {model_type}")
    if meta.box_mode != expected_mode:
        raise RuntimeError(
            f"Output shape mismatch: --type {model_type} expects {expected_mode}, "
            f"but model exposes {meta.box_mode}. Check that --type and --model match."
        )


def _validate_class_name_count(meta: ORTModelMeta, yaml_path: str) -> None:
    class_names = load_class_names(yaml_path)
    if len(class_names) != meta.class_count:
        raise RuntimeError(
            "Class count mismatch: "
            f"YAML defines {len(class_names)} classes, "
            f"but model outputs {meta.class_count}. "
            "Check that --yaml matches --model."
        )


def _onnx_bundle_uses_external_data(model_path: Path) -> bool:
    _ensure_onnx()
    model = onnx.load(str(model_path), load_external_data=False)
    external_location = getattr(onnx.TensorProto, "EXTERNAL", 1)
    for initializer in model.graph.initializer:
        if int(getattr(initializer, "data_location", 0)) == int(external_location):
            return True
        if getattr(initializer, "external_data", None):
            return True
    return False


def collect_model_sidecars(model_path: str) -> list[Path]:
    model = Path(model_path).expanduser().resolve()
    sidecars: list[Path] = []
    candidate = model.with_name("model.data")
    if candidate.is_file():
        sidecars.append(candidate)
    return sidecars


def resolve_onnx_model_artifact(model_path: str) -> ResolvedONNXArtifact:
    requested = Path(model_path).expanduser().resolve()
    if not requested.is_file():
        raise RuntimeError(f"ONNX model not found: {requested}")

    if requested.suffix.lower() != ".zip":
        return ResolvedONNXArtifact(
            requested_path=str(requested),
            model_path=str(requested),
            sidecars=tuple(collect_model_sidecars(str(requested))),
        )

    if not zipfile.is_zipfile(requested):
        raise RuntimeError(
            f"Expected a valid ONNX bundle zip at {requested}, but the file is not a zip archive."
        )

    tmpdir_obj = tempfile.TemporaryDirectory(prefix="iqf_onnx_bundle_")
    extract_root = Path(tmpdir_obj.name)
    try:
        with zipfile.ZipFile(requested) as archive:
            archive.extractall(extract_root)

        model_members = sorted(
            path for path in extract_root.rglob("model.onnx") if path.is_file()
        )
        if not model_members:
            raise RuntimeError(
                f"ONNX bundle {requested} does not contain model.onnx."
            )
        if len(model_members) != 1:
            raise RuntimeError(
                f"Expected exactly one model.onnx inside ONNX bundle {requested}, "
                f"found {len(model_members)}."
            )

        resolved_model = model_members[0].resolve()
        resolved_sidecars: list[Path] = []
        data_member = resolved_model.with_name("model.data")
        if data_member.is_file():
            resolved_sidecars.append(data_member)
        elif _onnx_bundle_uses_external_data(resolved_model):
            raise RuntimeError(
                f"ONNX bundle {requested} contains model.onnx that references external "
                "tensor data, but model.data is missing."
            )

        print(
            f"[warn] ONNX bundle input detected: {requested}. "
            "Extracting the bundle and executing the contained model.onnx entrypoint."
        )
        print(f"[info] resolved onnx model: {resolved_model}")
        return ResolvedONNXArtifact(
            requested_path=str(requested),
            model_path=str(resolved_model),
            sidecars=tuple(resolved_sidecars),
            temp_dir_obj=tmpdir_obj,
        )
    except Exception:
        tmpdir_obj.cleanup()
        raise


def load_onnx_model_metadata(
    model_path: str,
    model_type: str,
    precision: str,
    box_quant: QuantParams | None = None,
    class_quant: QuantParams | None = None,
) -> ORTModelMeta:
    _ensure_runtime_deps()
    _ensure_ort()
    model = str(Path(model_path).expanduser().resolve())
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise RuntimeError(f"Expected exactly one ONNX model input, got {len(inputs)}")
    if len(outputs) != 2:
        raise RuntimeError(
            f"Expected exactly two raw ONNX outputs, got {len(outputs)}"
        )

    input_meta = inputs[0]
    input_layout = _resolve_input_layout(input_meta.shape)
    input_dtype = ort_type_to_numpy_dtype(input_meta.type)

    box_meta = None
    class_meta = None
    box_mode = None
    class_count = None
    for output_meta in outputs:
        shape = tuple(int(v) for v in output_meta.shape)
        box_layout = _extract_box_layout(shape)
        if box_layout is not None and box_layout[1] == 8400:
            box_meta = output_meta
            box_mode = "dfl64" if box_layout[0] == 64 else "ltrb4"
            continue

        cls_dim = _class_dim_for_anchor_count(shape, anchor_count=8400)
        if cls_dim is not None:
            class_meta = output_meta
            class_count = int(cls_dim)

    if box_meta is None or class_meta is None or box_mode is None or class_count is None:
        raise RuntimeError(
            "Could not identify raw ONNX box/class outputs. "
            f"Outputs were: {[(o.name, tuple(int(v) for v in o.shape), o.type) for o in outputs]}"
        )

    resolved_box_quant = box_quant
    resolved_class_quant = class_quant
    if precision == "fp32":
        if input_dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
            raise RuntimeError(
                f"Expected FP32/FP16 ONNX input tensor for precision={precision}, got {input_meta.type}"
            )
    elif precision == "w8a16":
        if input_dtype != np.dtype(np.uint16):
            raise RuntimeError(
                f"Expected uint16 ONNX input tensor for precision={precision}, got {input_meta.type}"
            )
        # ONNX ADB runners pass output quant metadata explicitly so the target does not
        # need to import the heavyweight `onnx` package just to inspect graph outputs.
        if resolved_box_quant is None or resolved_class_quant is None:
            output_quants = _load_w8a16_output_quant_params(model)
            if box_meta.name not in output_quants or class_meta.name not in output_quants:
                raise RuntimeError("Missing W8A16 output quantization parameters.")
            resolved_box_quant = output_quants[box_meta.name]
            resolved_class_quant = output_quants[class_meta.name]
    else:
        raise RuntimeError(f"Unsupported ONNX precision: {precision}")

    meta = ORTModelMeta(
        model_path=model,
        model_type=model_type,
        precision=precision,
        input_name=input_meta.name,
        input_layout=input_layout,
        input_dtype=input_dtype,
        box_output_name=box_meta.name,
        class_output_name=class_meta.name,
        box_mode=box_mode,
        class_count=class_count,
        box_quant=resolved_box_quant,
        class_quant=resolved_class_quant,
    )
    _validate_model_type_compatibility(meta, model_type)
    return meta


def _dequant_w8a16_output(tensor, quant: QuantParams):
    _ensure_runtime_deps()
    return (tensor.astype(np.float32) - float(quant.zero_point)) * float(quant.scale)


def _warn_unused_onnx_backend_flag(backend: str) -> None:
    if backend and backend != "htp":
        print(
            f"[warn] --backend {backend} is not consumed by the ONNX Runtime QNN path and will be ignored."
        )


def resolve_onnx_qnn_backend_path(qnn_lib: str) -> str:
    if qnn_lib == DEFAULT_TFLITE_QNN_LIB:
        print(
            "[warn] The LiteRT default --qnn-lib path does not apply to the ONNX Runtime QNN EP. "
            f"Using {DEFAULT_ORT_QNN_BACKEND_PATH} instead."
        )
        return DEFAULT_ORT_QNN_BACKEND_PATH
    return qnn_lib


def create_ort_session(model_path: str, qnn_lib: str, no_qnn: bool):
    _ensure_ort()
    if no_qnn:
        return ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    qnn_lib = resolve_onnx_qnn_backend_path(qnn_lib)
    available = ort.get_available_providers()
    if "QNNExecutionProvider" not in available:
        print(
            "[warn] QNNExecutionProvider is not available. "
            "--qnn-lib was not used and execution will continue on CPU."
        )
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    providers = [
        (
            "QNNExecutionProvider",
            {
                "backend_path": qnn_lib,
            },
        ),
        "CPUExecutionProvider",
    ]
    try:
        return ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        print(
            "[warn] Failed to create an ONNX Runtime QNN session with "
            f"--qnn-lib {qnn_lib}: {exc}. Falling back to CPUExecutionProvider."
        )
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def build_input_tensor_for_meta(image_bgr, meta: ORTModelMeta):
    _ensure_runtime_deps()
    padded_bgr, (ratio, pad_w, pad_h) = letterbox_bgr(image_bgr, (IMG_H, IMG_W))
    rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)

    if meta.precision == "w8a16":
        tensor = rgb.astype(np.uint16) * 257
    else:
        tensor = rgb.astype(np.float32) / 255.0

    if meta.input_layout == "nchw":
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    else:
        tensor = tensor[None, ...]
    return tensor.astype(meta.input_dtype, copy=False), ratio, pad_w, pad_h


def _resolve_effective_flow(requested_flow: str, default_flow: str) -> str:
    if requested_flow == "auto":
        return default_flow
    return requested_flow


def _should_run_nms(flow: str, iou_thres: float, o2o_nms: bool) -> bool:
    if flow in ("o2m", "default"):
        return iou_thres > 0
    if flow == "o2o":
        return bool(o2o_nms and iou_thres > 0)
    raise ValueError(f"Unsupported flow: {flow}")


def postprocess_output(
    box_tensor,
    class_tensor,
    *,
    meta: ORTModelMeta,
    conf_thres: float,
    iou_thres: float,
    topk: int,
    max_det: int,
    flow: str,
    o2o_nms: bool,
):
    _ensure_runtime_deps()
    box_raw = _normalize_output_layout(
        box_tensor,
        tensor_name="box",
        channel_sizes=(4, 64),
    )[0]
    class_raw = _normalize_output_layout(
        class_tensor,
        tensor_name="class",
        anchor_count=int(box_raw.shape[1]),
    )[0]

    if meta.precision == "w8a16":
        if meta.box_quant is None or meta.class_quant is None:
            raise RuntimeError("Missing W8A16 quantization metadata.")
        box_head = _dequant_w8a16_output(box_raw, meta.box_quant)
        class_head = _dequant_w8a16_output(class_raw, meta.class_quant)
    else:
        box_head = box_raw.astype(np.float32)
        class_head = class_raw.astype(np.float32)

    class_prob = sigmoid_clip(class_head)
    class_ids = np.argmax(class_prob, axis=0).astype(np.int32)
    scores = np.max(class_prob, axis=0).astype(np.float32)

    keep = scores >= float(conf_thres)
    if not np.any(keep):
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            0,
        )

    keep_idx = np.nonzero(keep)[0]
    scores = scores[keep_idx]
    class_ids = class_ids[keep_idx]
    pre_nms_count = int(scores.shape[0])

    if topk > 0 and scores.shape[0] > topk:
        order = np.argsort(scores)[-topk:][::-1]
        keep_idx = keep_idx[order]
        scores = scores[order]
        class_ids = class_ids[order]

    centers_all, strides_all = build_anchor_centers_8400()
    if meta.box_mode == "dfl64":
        xyxy = dfl_to_xyxy_pixels(
            box_head[:, keep_idx],
            centers_all[keep_idx],
            strides_all[keep_idx],
        )
    else:
        xyxy = ltrb_grid_to_xyxy_pixels(
            box_head[:, keep_idx],
            centers_all[keep_idx],
            strides_all[keep_idx],
        )

    if _should_run_nms(flow, iou_thres, o2o_nms):
        keep2 = classwise_nms_xyxy(
            xyxy=xyxy,
            scores=scores,
            class_ids=class_ids,
            conf_thres=float(conf_thres),
            iou_thres=float(iou_thres),
            max_det=int(max_det),
        )
    else:
        keep2 = sort_and_cap(scores, int(max_det))

    keep2 = np.asarray(keep2, dtype=np.int32)
    if keep2.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            pre_nms_count,
        )
    return xyxy[keep2], scores[keep2], class_ids[keep2], pre_nms_count


def _save_outputs(
    output_dir: str,
    image_name: str,
    image_bgr,
    ratio: float,
    pad_w: float,
    pad_h: float,
    xyxy_640,
    scores,
    class_ids,
    classes: list[str],
    class_colors: list[tuple[int, int, int]],
) -> int:
    _ensure_runtime_deps()
    h0, w0 = image_bgr.shape[:2]
    out_img = image_bgr.copy()
    lines = []
    written = 0

    for i in range(scores.shape[0]):
        x1, y1, x2, y2 = map(float, xyxy_640[i])
        sc = float(scores[i])
        cid = int(class_ids[i])

        x1o = clamp((x1 - pad_w) / ratio, 0.0, w0 - 1.0)
        y1o = clamp((y1 - pad_h) / ratio, 0.0, h0 - 1.0)
        x2o = clamp((x2 - pad_w) / ratio, 0.0, w0 - 1.0)
        y2o = clamp((y2 - pad_h) / ratio, 0.0, h0 - 1.0)
        if (x2o - x1o) < 1.0 or (y2o - y1o) < 1.0:
            continue

        box_color = class_color(class_colors, cid)
        cv2.rectangle(out_img, (int(x1o), int(y1o)), (int(x2o), int(y2o)), box_color, 2)

        label = f"{classes[cid]}:{sc:.2f}"
        y_text = int(max(0.0, y1o - 5.0))
        cv2.putText(
            out_img,
            label,
            (int(x1o), y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out_img,
            label,
            (int(x1o), y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        xc = (x1o + x2o) / 2.0
        yc = (y1o + y2o) / 2.0
        bw = x2o - x1o
        bh = y2o - y1o
        lines.append(
            f"{cid} {xc / w0:.6f} {yc / h0:.6f} {bw / w0:.6f} {bh / h0:.6f} {sc:.6f}"
        )
        written += 1

    os.makedirs(output_dir, exist_ok=True)
    out_img_path = os.path.join(output_dir, image_name)
    out_txt_path = os.path.join(output_dir, Path(image_name).stem + ".txt")
    cv2.imwrite(out_img_path, out_img)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return written


def _has_meaningful_outputs(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    for p in output_dir.iterdir():
        if not p.is_file():
            continue
        if p.name == "classes.txt":
            continue
        return True
    return False


def _commit_output_dir(staging_dir: Path, final_output_dir: Path) -> None:
    if final_output_dir.exists():
        if not final_output_dir.is_dir():
            raise RuntimeError(
                f"Output path exists and is not a directory: {final_output_dir}"
            )
        raise RuntimeError(
            "Output directory already exists: "
            f"{final_output_dir}. Remove it or choose a different output path."
        )
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging_dir), str(final_output_dir))


def _prepare_image_input(
    image_dir: str | None, image_path: str | None
) -> tuple[str, tempfile.TemporaryDirectory | None]:
    if bool(image_dir) == bool(image_path):
        raise RuntimeError("Specify exactly one of image_dir or image_path")
    if image_dir:
        collect_image_files(image_dir)
        return image_dir, None

    img = Path(image_path).expanduser().resolve()
    if not img.is_file():
        raise RuntimeError(f"Image not found: {img}")

    tmp_obj = tempfile.TemporaryDirectory(prefix="onnx_single_image_")
    dst = Path(tmp_obj.name) / img.name
    shutil.copy2(str(img), str(dst))
    return tmp_obj.name, tmp_obj


def _run_test_directory(
    *,
    runner: "ORTRawModel",
    yaml_path: str,
    image_dir: str,
    output_dir: str,
    default_flow: str,
    conf_thres: float,
    iou_thres: float,
    topk: int,
    max_det: int,
    postprocess_flow: str,
    o2o_nms: bool,
) -> None:
    _ensure_runtime_deps()
    class_names = load_class_names(yaml_path)
    if len(class_names) != runner.meta.class_count:
        raise RuntimeError(
            "Class count mismatch: "
            f"YAML defines {len(class_names)} classes, "
            f"but model outputs {runner.meta.class_count}."
        )
    class_colors = build_class_colors(len(class_names))
    final_output_dir = Path(output_dir).expanduser().resolve()
    staging_tmp = tempfile.TemporaryDirectory(prefix="onnx_local_output_")
    staging_output_dir = Path(staging_tmp.name) / "output"
    staging_output_dir.mkdir(parents=True, exist_ok=True)
    flow = _resolve_effective_flow(postprocess_flow, default_flow)

    try:
        with open(staging_output_dir / "classes.txt", "w", encoding="utf-8") as f:
            for idx, name in enumerate(class_names):
                f.write(f"{idx} {name}\n")

        processed = 0
        total_time_s = 0.0
        invoke_time_s = 0.0
        for image_file in collect_image_files(image_dir):
            t0 = time.perf_counter()
            box_tensor, class_tensor, ratio, pad_w, pad_h, orig_size = runner.infer_raw(
                str(image_file)
            )
            image_bgr = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {image_file}")

            xyxy, scores, class_ids, pre_nms = postprocess_output(
                box_tensor,
                class_tensor,
                meta=runner.meta,
                conf_thres=float(conf_thres),
                iou_thres=float(iou_thres),
                topk=int(topk),
                max_det=int(max_det),
                flow=flow,
                o2o_nms=bool(o2o_nms),
            )
            written = _save_outputs(
                output_dir=str(staging_output_dir),
                image_name=image_file.name,
                image_bgr=image_bgr,
                ratio=ratio,
                pad_w=pad_w,
                pad_h=pad_h,
                xyxy_640=xyxy,
                scores=scores,
                class_ids=class_ids,
                classes=class_names,
                class_colors=class_colors,
            )
            t1 = time.perf_counter()
            processed += 1
            total_time_s += t1 - t0
            invoke_time_s += runner.last_invoke_time_s
            print(f"{image_file.name}: flow={flow} preNMS={pre_nms} kept={written}")

        if processed > 0:
            print("=== Inference Timing Summary ===")
            print(f"processed={processed}")
            print(f"avg_total_inference_ms={(total_time_s / processed) * 1000.0:.3f}")
            print(f"avg_model_invoke_ms={(invoke_time_s / processed) * 1000.0:.3f}")

        if not _has_meaningful_outputs(staging_output_dir):
            raise RuntimeError(
                "Inference finished but no output artifacts were produced."
            )
        _commit_output_dir(staging_output_dir, final_output_dir)
    finally:
        staging_tmp.cleanup()


class ORTRawModel:
    def __init__(
        self,
        model_path: str,
        model_type: str,
        precision: str,
        qnn_lib: str = "libQnnHtp.so",
        backend: str = "htp",
        no_qnn: bool = False,
        box_quant: QuantParams | None = None,
        class_quant: QuantParams | None = None,
    ):
        _warn_unused_onnx_backend_flag(backend)
        self.artifact = resolve_onnx_model_artifact(model_path)
        self.meta = load_onnx_model_metadata(
            model_path=self.artifact.model_path,
            model_type=model_type,
            precision=precision,
            box_quant=box_quant,
            class_quant=class_quant,
        )
        self.session = create_ort_session(self.meta.model_path, qnn_lib, no_qnn)
        self.last_invoke_time_s = 0.0

    def cleanup(self) -> None:
        self.artifact.cleanup()

    def infer_raw(self, image_path: str):
        _ensure_runtime_deps()
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        input_tensor, ratio, pad_w, pad_h = build_input_tensor_for_meta(
            image_bgr, self.meta
        )
        t_invoke0 = time.perf_counter()
        outputs = self.session.run(
            [self.meta.box_output_name, self.meta.class_output_name],
            {self.meta.input_name: input_tensor},
        )
        self.last_invoke_time_s = time.perf_counter() - t_invoke0
        return (
            outputs[0],
            outputs[1],
            ratio,
            pad_w,
            pad_h,
            (image_bgr.shape[1], image_bgr.shape[0]),
        )


class ADBORTRawModel:
    def __init__(
        self,
        model_path: str,
        model_type: str,
        precision: str,
        adb_serial: str | None,
        remote_workdir: str,
        remote_runner: str,
        remote_python: str,
        qnn_lib: str,
        backend: str,
        no_qnn: bool,
        shared_remote_input_dir: str,
        shared_meta: dict,
        artifact: ResolvedONNXArtifact | None = None,
    ):
        self.artifact = artifact or resolve_onnx_model_artifact(model_path)
        self.model_path = self.artifact.model_path
        self.precision = precision
        self.adb_serial = adb_serial
        self.remote_workdir = remote_workdir
        self.remote_runner = remote_runner
        self.remote_python = remote_python
        self.qnn_lib = qnn_lib
        self.backend = backend
        self.no_qnn = no_qnn
        self.shared_remote_input_dir = shared_remote_input_dir
        self.shared_meta = shared_meta
        self.meta = load_onnx_model_metadata(
            model_path=self.model_path,
            model_type=model_type,
            precision=precision,
        )

        self.remote_model = f"{self.remote_workdir}/{Path(self.model_path).name}"
        self.remote_output_dir = f"{self.remote_workdir}/outputs"
        adb_shell = _adb_shell
        adb_push = _adb_push
        adb_shell(self.adb_serial, f"mkdir -p {shlex.quote(self.remote_workdir)}")
        adb_push(self.adb_serial, self.model_path, self.remote_model)
        for sidecar in self.artifact.sidecars:
            adb_push(
                self.adb_serial,
                str(sidecar),
                f"{self.remote_workdir}/{sidecar.name}",
            )
        self.local_output_dir = None

    def prepare_batch(self, images) -> None:
        _warn_unused_onnx_backend_flag(self.backend)
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        local_output_dir = Path(self._tmpdir_obj.name) / "outputs"
        local_output_dir.mkdir(parents=True, exist_ok=True)
        self.local_output_dir = local_output_dir

        _adb_shell(
            self.adb_serial,
            f"rm -rf {shlex.quote(self.remote_output_dir)} && "
            f"mkdir -p {shlex.quote(self.remote_output_dir)}",
        )

        cmd = [
            self.remote_python,
            self.remote_runner,
            "--mode",
            "batch-raw",
            "--model",
            self.remote_model,
            "--model-type",
            self.meta.model_type,
            "--precision",
            self.precision,
            "--input-dir",
            self.shared_remote_input_dir,
            "--output-dir",
            self.remote_output_dir,
            "--qnn-lib",
            self.qnn_lib,
            "--backend",
            self.backend,
        ]
        if self.precision == "w8a16":
            if self.meta.box_quant is None or self.meta.class_quant is None:
                raise RuntimeError("Missing W8A16 quantization metadata.")
            cmd.extend(
                [
                    "--box-scale",
                    str(self.meta.box_quant.scale),
                    "--box-zero-point",
                    str(self.meta.box_quant.zero_point),
                    "--class-scale",
                    str(self.meta.class_quant.scale),
                    "--class-zero-point",
                    str(self.meta.class_quant.zero_point),
                ]
            )
        if self.no_qnn:
            cmd.append("--no-qnn")
        _adb_shell(
            self.adb_serial,
            _format_remote_onnx_command(cmd, self.remote_python),
        )
        _adb_pull(self.adb_serial, self.remote_output_dir, str(local_output_dir.parent))

    def infer_raw(self, image_path: str):
        raise RuntimeError(
            "ADBORTRawModel uses prepare_batch(images) + get_result_for_image(rec)"
        )

    def get_result_for_image(self, rec):
        if self.local_output_dir is None:
            raise RuntimeError(
                "prepare_batch(images) must be called before get_result_for_image(rec)"
            )

        stem = str(rec.image_id)
        boxes_path = self.local_output_dir / f"{stem}_boxes.npy"
        scores_path = self.local_output_dir / f"{stem}_scores.npy"
        if not boxes_path.exists():
            raise FileNotFoundError(boxes_path)
        if not scores_path.exists():
            raise FileNotFoundError(scores_path)

        boxes = np.load(boxes_path)
        scores = np.load(scores_path)
        meta = self.shared_meta[rec.image_id]
        return (
            boxes,
            scores,
            meta["ratio"],
            meta["padw"],
            meta["padh"],
            meta["orig_size"],
        )

    def cleanup(self):
        if hasattr(self, "_tmpdir_obj"):
            self._tmpdir_obj.cleanup()
        self.artifact.cleanup()


def prepare_shared_onnx_inputs(
    images,
    model_path: str,
    model_type: str,
    precision: str,
    adb_serial: str | None,
    remote_input_dir: str,
    artifact: ResolvedONNXArtifact | None = None,
):
    _ensure_runtime_deps()
    resolved_artifact = artifact or resolve_onnx_model_artifact(model_path)
    meta = load_onnx_model_metadata(
        resolved_artifact.model_path, model_type, precision
    )
    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = Path(tmpdir_obj.name)
    local_input_dir = tmpdir / "inputs"
    local_input_dir.mkdir(parents=True, exist_ok=True)

    meta_by_image = {}
    for rec in images:
        image_bgr = cv2.imread(rec.path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {rec.path}")
        tensor, ratio, padw, padh = build_input_tensor_for_meta(image_bgr, meta)
        np.save(local_input_dir / f"{rec.image_id}.npy", tensor)
        meta_by_image[rec.image_id] = {
            "ratio": ratio,
            "padw": padw,
            "padh": padh,
            "orig_size": (image_bgr.shape[1], image_bgr.shape[0]),
        }

    _adb_shell(
        adb_serial,
        f"rm -rf {shlex.quote(remote_input_dir)} && "
        f"mkdir -p {shlex.quote(remote_input_dir)}",
    )
    for input_file in sorted(local_input_dir.glob("*.npy")):
        _adb_push(adb_serial, str(input_file), f"{remote_input_dir}/{input_file.name}")

    return tmpdir_obj, meta_by_image, meta, resolved_artifact


def run_onnx_test_inference_local(
    model_path: str,
    yaml_path: str,
    output_dir: str,
    model_type: str,
    default_flow: str,
    conf_thres: float,
    iou_thres: float,
    topk: int,
    max_det: int,
    image_dir: str | None = None,
    image_path: str | None = None,
    postprocess_flow: str = "auto",
    o2o_nms: bool = False,
    disable_int8_prefilter: bool = False,
    no_qnn: bool = False,
    qnn_lib: str = "libQnnHtp.so",
    backend: str = "htp",
    runtime: str = "onnx",
    precision: str = "fp32",
) -> None:
    if runtime != "onnx":
        raise RuntimeError(f"Unsupported runtime for ONNX helpers: {runtime}")
    if disable_int8_prefilter:
        print(
            "[warn] --disable-int8-prefilter is not used by the ONNX Runtime path and will be ignored."
        )

    prepared_dir, tmp_obj = _prepare_image_input(
        image_dir=image_dir, image_path=image_path
    )
    runner = None
    try:
        runner = ORTRawModel(
            model_path=model_path,
            model_type=model_type,
            precision=precision,
            qnn_lib=qnn_lib,
            backend=backend,
            no_qnn=no_qnn,
        )
        _validate_class_name_count(runner.meta, yaml_path)
        _run_test_directory(
            runner=runner,
            yaml_path=yaml_path,
            image_dir=prepared_dir,
            output_dir=output_dir,
            default_flow=default_flow,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            topk=topk,
            max_det=max_det,
            postprocess_flow=postprocess_flow,
            o2o_nms=o2o_nms,
        )
    finally:
        if runner is not None:
            runner.cleanup()
        if tmp_obj is not None:
            tmp_obj.cleanup()


def run_onnx_test_inference_adb(
    model_path: str,
    yaml_path: str,
    output_dir: str,
    model_type: str,
    default_flow: str,
    conf_thres: float,
    iou_thres: float,
    topk: int,
    max_det: int,
    image_dir: str | None = None,
    image_path: str | None = None,
    postprocess_flow: str = "auto",
    o2o_nms: bool = False,
    disable_int8_prefilter: bool = False,
    adb_serial: str | None = None,
    remote_workdir: str = "/data/local/tmp/yolo_test",
    no_qnn: bool = False,
    qnn_lib: str = "libQnnHtp.so",
    backend: str = "htp",
    runtime: str = "onnx",
    precision: str = "fp32",
) -> None:
    _ensure_runtime_deps()
    if runtime != "onnx":
        raise RuntimeError(f"Unsupported runtime for ONNX helpers: {runtime}")
    if disable_int8_prefilter:
        print(
            "[warn] --disable-int8-prefilter is not used by the ONNX Runtime path and will be ignored."
        )

    from tool.adb_runtime_bootstrap import ensure_adb_runtime_venv

    artifact = resolve_onnx_model_artifact(model_path)
    meta = load_onnx_model_metadata(artifact.model_path, model_type, precision)
    _validate_class_name_count(meta, yaml_path)
    prepared_dir, tmp_obj = _prepare_image_input(
        image_dir=image_dir, image_path=image_path
    )
    final_output_dir = Path(output_dir).expanduser().resolve()
    pull_tmp = tempfile.TemporaryDirectory(prefix="onnx_adb_pull_")
    pull_output_dir = Path(pull_tmp.name) / "output"
    pull_output_dir.mkdir(parents=True, exist_ok=True)

    remote_root = _strip_trailing_slash(remote_workdir)
    if not remote_root:
        remote_root = "/data/local/tmp/yolo_test"
    remote_run_dir = f"{remote_root}/onnx_test_{os.getpid()}_{int(time.time() * 1000)}"
    remote_model = f"{remote_run_dir}/{Path(artifact.model_path).name}"
    remote_yaml = f"{remote_run_dir}/{Path(yaml_path).name}"
    remote_script = f"{remote_run_dir}/onnx_inference.py"
    remote_img_dir = f"{remote_run_dir}/images"
    remote_output_dir = f"{remote_run_dir}/output"

    local_model = artifact.model_path
    local_yaml = str(Path(yaml_path).expanduser().resolve())
    local_script = str(Path(__file__).resolve())
    local_target_requirements = str(
        Path(__file__).resolve().parent.parent / "requirements" / "target.txt"
    )
    remote_python = ensure_adb_runtime_venv(
        adb_serial=adb_serial,
        local_requirements_path=local_target_requirements,
        local_ort_qnn_wheel_path=DEFAULT_ORT_QNN_WHEEL,
    )

    try:
        _adb_shell(
            adb_serial,
            f"rm -rf {shlex.quote(remote_run_dir)} && mkdir -p {shlex.quote(remote_img_dir)}",
        )
        _adb_push(adb_serial, local_model, remote_model)
        for sidecar in artifact.sidecars:
            _adb_push(
                adb_serial,
                str(sidecar),
                f"{remote_run_dir}/{sidecar.name}",
            )
        _adb_push(adb_serial, local_yaml, remote_yaml)
        _adb_push(adb_serial, local_script, remote_script)
        for image in collect_image_files(prepared_dir):
            _adb_push(adb_serial, str(image), f"{remote_img_dir}/{image.name}")

        remote_cmd = [
            remote_python,
            remote_script,
            "--mode",
            "test",
            "--model",
            remote_model,
            "--model-type",
            model_type,
            "--precision",
            precision,
            "--yaml",
            remote_yaml,
            "--img-dir",
            remote_img_dir,
            "--output-dir",
            remote_output_dir,
            "--default-flow",
            default_flow,
            "--conf-thres",
            str(conf_thres),
            "--iou-thres",
            str(iou_thres),
            "--topk",
            str(topk),
            "--max-det",
            str(max_det),
            "--postprocess-flow",
            postprocess_flow,
            "--qnn-lib",
            qnn_lib,
            "--backend",
            backend,
        ]
        if precision == "w8a16":
            if meta.box_quant is None or meta.class_quant is None:
                raise RuntimeError("Missing W8A16 quantization metadata.")
            remote_cmd.extend(
                [
                    "--box-scale",
                    str(meta.box_quant.scale),
                    "--box-zero-point",
                    str(meta.box_quant.zero_point),
                    "--class-scale",
                    str(meta.class_quant.scale),
                    "--class-zero-point",
                    str(meta.class_quant.zero_point),
                ]
            )
        if o2o_nms:
            remote_cmd.append("--o2o-nms")
        if no_qnn:
            remote_cmd.append("--no-qnn")

        _adb_shell(
            adb_serial,
            _format_remote_onnx_command(remote_cmd, remote_python),
        )
        _adb_pull(adb_serial, f"{remote_output_dir}/.", str(pull_output_dir))
        if not _has_meaningful_outputs(pull_output_dir):
            raise RuntimeError(
                "Remote ONNX inference finished but no output artifacts were produced."
            )
        _commit_output_dir(pull_output_dir, final_output_dir)
    finally:
        artifact.cleanup()
        if tmp_obj is not None:
            tmp_obj.cleanup()
        pull_tmp.cleanup()
        try:
            _adb_shell(adb_serial, f"rm -rf {shlex.quote(remote_run_dir)}")
        except subprocess.CalledProcessError as cleanup_exc:
            print(f"[warn] remote cleanup failed: {cleanup_exc}")


def _run_batch_raw_mode(args: argparse.Namespace) -> None:
    _ensure_runtime_deps()
    box_quant = None
    class_quant = None
    if args.precision == "w8a16":
        if (
            args.box_scale is None
            or args.box_zero_point is None
            or args.class_scale is None
            or args.class_zero_point is None
        ):
            raise RuntimeError("W8A16 batch mode requires explicit output quant metadata.")
        box_quant = QuantParams(args.box_scale, args.box_zero_point)
        class_quant = QuantParams(args.class_scale, args.class_zero_point)

    runner = ORTRawModel(
        model_path=args.model,
        model_type=args.model_type,
        precision=args.precision,
        qnn_lib=args.qnn_lib,
        backend=args.backend,
        no_qnn=args.no_qnn,
        box_quant=box_quant,
        class_quant=class_quant,
    )
    try:
        input_dir = Path(args.input_dir).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        for input_file in sorted(input_dir.glob("*.npy")):
            tensor = np.load(input_file)
            outputs = runner.session.run(
                [runner.meta.box_output_name, runner.meta.class_output_name],
                {runner.meta.input_name: tensor},
            )
            stem = input_file.stem
            np.save(output_dir / f"{stem}_boxes.npy", outputs[0])
            np.save(output_dir / f"{stem}_scores.npy", outputs[1])
    finally:
        runner.cleanup()


def _run_test_mode_cli(args: argparse.Namespace) -> None:
    box_quant = None
    class_quant = None
    if args.precision == "w8a16":
        if (
            args.box_scale is None
            or args.box_zero_point is None
            or args.class_scale is None
            or args.class_zero_point is None
        ):
            raise RuntimeError("W8A16 test mode requires explicit output quant metadata.")
        box_quant = QuantParams(args.box_scale, args.box_zero_point)
        class_quant = QuantParams(args.class_scale, args.class_zero_point)
    runner = ORTRawModel(
        model_path=args.model,
        model_type=args.model_type,
        precision=args.precision,
        qnn_lib=args.qnn_lib,
        backend=args.backend,
        no_qnn=args.no_qnn,
        box_quant=box_quant,
        class_quant=class_quant,
    )
    try:
        _validate_class_name_count(runner.meta, args.yaml)
        _run_test_directory(
            runner=runner,
            yaml_path=args.yaml,
            image_dir=args.img_dir,
            output_dir=args.output_dir,
            default_flow=args.default_flow,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            topk=args.topk,
            max_det=args.max_det,
            postprocess_flow=args.postprocess_flow,
            o2o_nms=args.o2o_nms,
        )
    finally:
        runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(prog="onnx_inference.py")
    parser.add_argument("--mode", choices=["test", "batch-raw"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--precision", choices=["fp32", "w8a16"], required=True)
    parser.add_argument("--qnn-lib", default="libQnnHtp.so")
    parser.add_argument("--backend", default="htp")
    parser.add_argument("--no-qnn", action="store_true")
    parser.add_argument("--yaml")
    parser.add_argument("--img-dir")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--default-flow", choices=["default", "o2o", "o2m"], default="o2m")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument(
        "--postprocess-flow",
        choices=["auto", "default", "o2o", "o2m"],
        default="auto",
    )
    parser.add_argument("--o2o-nms", action="store_true")
    parser.add_argument("--box-scale", type=float, default=None)
    parser.add_argument("--box-zero-point", type=int, default=None)
    parser.add_argument("--class-scale", type=float, default=None)
    parser.add_argument("--class-zero-point", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "test":
        if not args.yaml or not args.img_dir:
            raise SystemExit("[error] test mode requires --yaml and --img-dir")
        _run_test_mode_cli(args)
        return
    if not args.input_dir:
        raise SystemExit("[error] batch-raw mode requires --input-dir")
    _run_batch_raw_mode(args)


if __name__ == "__main__":
    main()
