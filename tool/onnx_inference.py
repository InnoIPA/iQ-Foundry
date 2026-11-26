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
"""Shared ONNX Runtime inference helpers for test mode and eval mode."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Shared detection helpers live in tool/postprocess. Run from the repo they import as
# tool.postprocess; pushed to the target as a standalone script they sit next to it.
try:
    from tool.postprocess.detect import (
        IMG_H,
        IMG_W,
        _class_dim_for_anchor_count,
        _extract_box_layout,
        _normalize_output_layout,
        _output_order_key,
        _select_box_and_class_outputs,
        _should_run_nms,
        build_anchor_centers_8400,
        clamp,
        classwise_nms_xyxy,
        dfl_to_xyxy_pixels,
        letterbox_bgr,
        ltrb_grid_to_xyxy_pixels,
        sigmoid_clip,
        sort_and_cap,
    )
    from tool.postprocess.outputs import load_raw_outputs, save_avg_invoke_ms, save_raw_outputs
    from tool.postprocess.pose import postprocess_pose, save_pose_outputs
    from tool.postprocess.segment import postprocess_segment, save_segment_outputs
    from tool.postprocess.specs import MODEL_SPECS, TASK_SPECS
except ModuleNotFoundError:
    from postprocess.detect import (
        IMG_H,
        IMG_W,
        _class_dim_for_anchor_count,
        _extract_box_layout,
        _normalize_output_layout,
        _output_order_key,
        _select_box_and_class_outputs,
        _should_run_nms,
        build_anchor_centers_8400,
        clamp,
        classwise_nms_xyxy,
        dfl_to_xyxy_pixels,
        letterbox_bgr,
        ltrb_grid_to_xyxy_pixels,
        sigmoid_clip,
        sort_and_cap,
    )
    from postprocess.outputs import load_raw_outputs, save_avg_invoke_ms, save_raw_outputs
    from postprocess.pose import postprocess_pose, save_pose_outputs
    from postprocess.segment import postprocess_segment, save_segment_outputs
    from postprocess.specs import MODEL_SPECS, TASK_SPECS

cv2 = None
np = None
onnx = None
ort = None
yaml = None

SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
IQ9_ARCH_ALIASES = {"aarch64", "arm64"}
# Runtime change (ORT cp312 -> cp314): QLI2.0 ships Python 3.14, so the previous
# cp312 wheel could not be installed on target at all. This is the only place the
# bundled wheel filename appears; both ADB call sites read this constant.
DEFAULT_ORT_QNN_WHEEL = str(
    Path(__file__).resolve().parent.parent
    / "wheels"
    / "onnxruntime_qnn-1.25.1-cp314-cp314-linux_aarch64.whl"
)
EXPECTED_BOX_MODES = {name: spec.box_mode for name, spec in MODEL_SPECS.items()}
BOX_MODE_CHANNELS = {"dfl64": 64, "ltrb4": 4}
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
    task: str = "detect"
    # Pose / segmentation outputs after boxes and scores, in TaskSpec.outputs order.
    extra_output_names: tuple[str, ...] = ()
    extra_quants: tuple[QuantParams, ...] = ()
    # The model's input shape as the graph declares it (filled for classification, whose input
    # size is read from the model instead of being fixed at 640).
    input_shape: tuple[int, ...] = ()

    @property
    def output_names(self) -> list[str]:
        """Graph output names, in the role order of the task (see TaskSpec.outputs)."""
        if self.task == "classify":
            return [self.box_output_name]  # the only output, the logits (kept in the box slot)
        return [self.box_output_name, self.class_output_name, *self.extra_output_names]

    def to_float(self, outputs: dict) -> dict:
        """Pose / segmentation: all outputs as float (w8a16 outputs are dequantized)."""
        if self.precision != "w8a16":
            return {role: t.astype(np.float32, copy=False) for role, t in outputs.items()}
        quants = (self.box_quant, self.class_quant, *self.extra_quants)
        return {
            role: _dequant_w8a16_output(outputs[role], quant)
            for role, quant in zip(TASK_SPECS[self.task].outputs, quants)
        }


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


def _resolve_input_layout(shape: Sequence[int], any_square_size: bool = False) -> str:
    dims = [int(v) for v in shape]
    if any_square_size:  # classification: the input size is whatever the model was exported at
        if len(dims) == 4 and dims[0] == 1 and dims[1] == 3 and dims[2] == dims[3]:
            return "nchw"
        if len(dims) == 4 and dims[0] == 1 and dims[3] == 3 and dims[1] == dims[2]:
            return "nhwc"
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


def _iter_graph_tensors(graph):
    """Every TensorProto in a graph, including node attributes and subgraphs."""
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            yield from attr.tensors
            if attr.HasField("g"):
                yield from _iter_graph_tensors(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_graph_tensors(subgraph)


def _external_data_locations(model_path: Path) -> list[str]:
    _ensure_onnx()
    model = onnx.load(str(model_path), load_external_data=False)
    locations = set()
    for tensor in _iter_graph_tensors(model.graph):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                locations.add(entry.value)
    return sorted(locations)


def collect_model_sidecars(model_path: str) -> list[Path]:
    """External-data files the model references, to be shipped alongside it.

    qc names the sidecar `<model stem>.data`; older artifacts and AI Hub bundles
    use `model.data`. The graph is read when `onnx` is importable (host). On the
    target, where `onnx` is not installed, the known names are probed instead.
    """
    model = Path(model_path).expanduser().resolve()
    try:
        locations = _external_data_locations(model)
    except RuntimeError:
        for name in (f"{model.stem}.data", "model.data"):
            candidate = model.with_name(name)
            if candidate.is_file():
                return [candidate]
        return []

    sidecars: list[Path] = []
    for location in locations:
        candidate = model.parent / location
        if not candidate.is_file():
            raise RuntimeError(
                f"ONNX model {model} references external tensor data "
                f"'{location}', but {candidate} is missing."
            )
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
        resolved_sidecars = collect_model_sidecars(str(resolved_model))

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
    task: str = "detect",
    extra_quants: Sequence[QuantParams] | None = None,
) -> ORTModelMeta:
    _ensure_runtime_deps()
    _ensure_ort()
    model = str(Path(model_path).expanduser().resolve())
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    roles = TASK_SPECS[task].outputs

    if len(inputs) != 1:
        raise RuntimeError(f"Expected exactly one ONNX model input, got {len(inputs)}")
    if len(outputs) != len(roles):
        raise RuntimeError(
            f"--task {task} expects {len(roles)} raw ONNX outputs ({', '.join(roles)}), "
            f"got {len(outputs)}: {[tuple(o.shape) for o in outputs]}. "
            "Check that --task and --model match."
        )

    input_meta = inputs[0]
    input_layout = _resolve_input_layout(input_meta.shape, any_square_size=task == "classify")
    input_dtype = ort_type_to_numpy_dtype(input_meta.type)

    if task == "classify":
        # One output, the logits; it sits in the box fields (class_count = number of classes).
        logits_quant = box_quant
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
            if logits_quant is None:
                logits_quant = _load_w8a16_output_quant_params(model)[outputs[0].name]
        else:
            raise RuntimeError(f"Unsupported ONNX precision: {precision}")
        return ORTModelMeta(
            model_path=model,
            model_type=model_type,
            precision=precision,
            input_name=input_meta.name,
            input_layout=input_layout,
            input_dtype=input_dtype,
            box_output_name=outputs[0].name,
            class_output_name="",
            box_mode="",
            class_count=int(outputs[0].shape[-1]),
            box_quant=logits_quant,
            task=task,
            input_shape=tuple(int(v) for v in input_meta.shape),
        )

    selected = _select_box_and_class_outputs(
        [(o.name, o.shape) for o in outputs],
        expected_box_channels=BOX_MODE_CHANNELS.get(EXPECTED_BOX_MODES.get(model_type)),
    )
    if selected is None:
        raise RuntimeError(
            "Could not identify raw ONNX box/class outputs. "
            f"Outputs were: {[(o.name, tuple(int(v) for v in o.shape), o.type) for o in outputs]}"
        )
    box_meta = outputs[selected[0]]
    class_meta = outputs[selected[1]]
    box_mode = "dfl64" if _extract_box_layout(box_meta.shape)[0] == 64 else "ltrb4"
    class_count = int(_class_dim_for_anchor_count(class_meta.shape, anchor_count=8400))
    rest = [(i, o) for i, o in enumerate(outputs) if i not in selected]
    rest.sort(key=lambda item: _output_order_key(item[1].name, item[0]))
    extra_names = tuple(o.name for _, o in rest)

    resolved_box_quant = box_quant
    resolved_class_quant = class_quant
    resolved_extra_quants = tuple(extra_quants or ())
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
        if (
            resolved_box_quant is None
            or resolved_class_quant is None
            or len(resolved_extra_quants) != len(extra_names)
        ):
            output_quants = _load_w8a16_output_quant_params(model)
            names = (box_meta.name, class_meta.name, *extra_names)
            if any(name not in output_quants for name in names):
                raise RuntimeError("Missing W8A16 output quantization parameters.")
            resolved_box_quant = output_quants[box_meta.name]
            resolved_class_quant = output_quants[class_meta.name]
            resolved_extra_quants = tuple(output_quants[name] for name in extra_names)
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
        task=task,
        extra_output_names=extra_names,
        extra_quants=resolved_extra_quants,
    )
    _validate_model_type_compatibility(meta, model_type)
    return meta


def quant_cli_args(meta: ORTModelMeta) -> list[str]:
    """--task and w8a16 output quant flags for the target-side runner (it has no `onnx`)."""
    args = ["--task", meta.task] if meta.task != "detect" else []
    if meta.precision != "w8a16":
        return args
    if meta.box_quant is None or (meta.task != "classify" and meta.class_quant is None):
        raise RuntimeError("Missing W8A16 quantization metadata.")
    args += [
        "--box-scale", str(meta.box_quant.scale),
        "--box-zero-point", str(meta.box_quant.zero_point),
    ]
    if meta.class_quant is not None:
        args += [
            "--class-scale", str(meta.class_quant.scale),
            "--class-zero-point", str(meta.class_quant.zero_point),
        ]
    for quant in meta.extra_quants:
        args += ["--extra-quant", str(quant.scale), str(quant.zero_point)]
    return args


def quant_params_from_args(args: argparse.Namespace):
    """Inverse of quant_cli_args: (box_quant, class_quant, extra_quants) for the target runner."""
    if args.precision != "w8a16":
        return None, None, None
    if args.task == "classify":  # the logits quant travels in the box fields
        if None in (args.box_scale, args.box_zero_point):
            raise RuntimeError(f"W8A16 {args.mode} mode requires explicit output quant metadata.")
        return QuantParams(args.box_scale, args.box_zero_point), None, []
    if None in (args.box_scale, args.box_zero_point, args.class_scale, args.class_zero_point):
        raise RuntimeError(f"W8A16 {args.mode} mode requires explicit output quant metadata.")
    extra = [QuantParams(float(s), int(float(z))) for s, z in (args.extra_quant or [])]
    return (
        QuantParams(args.box_scale, args.box_zero_point),
        QuantParams(args.class_scale, args.class_zero_point),
        extra,
    )


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


def _accumulate_timing(runner, span_s, total_time_s, invoke_time_s):
    """Fold one image's timing in.

    Batched runners fetch results before the loop, so their device time is not
    inside the measured span and is added back here.
    """
    invoke_s = runner.last_invoke_time_s
    if not getattr(runner, "device_time_included", True):
        span_s += invoke_s or 0.0
    if invoke_s is None or invoke_time_s is None:
        return total_time_s + span_s, None
    return total_time_s + span_s, invoke_time_s + invoke_s


def _format_invoke_ms(invoke_time_s, processed: int) -> str:
    if invoke_time_s is None:
        return "n/a"
    return f"{(invoke_time_s / processed) * 1000.0:.3f}"


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
    task: str = "detect",
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
            outputs, ratio, pad_w, pad_h, orig_size = runner.infer_raw(
                str(image_file)
            )
            image_bgr = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {image_file}")

            if task == "pose":
                xyxy, scores, class_ids, kpts, pre_nms = postprocess_pose(
                    outputs["boxes"],
                    outputs["scores"],
                    outputs["kpts"],
                    conf_thres=float(conf_thres),
                    iou_thres=float(iou_thres),
                    topk=int(topk),
                    max_det=int(max_det),
                    flow=flow,
                    o2o_nms=bool(o2o_nms),
                )
                written = save_pose_outputs(
                    str(staging_output_dir), image_file.name, image_bgr,
                    ratio, pad_w, pad_h, xyxy, scores, class_ids, kpts, class_names,
                )
            elif task == "segment":
                xyxy, scores, class_ids, coefs, proto, pre_nms = postprocess_segment(
                    outputs["boxes"],
                    outputs["scores"],
                    outputs["mask_coef"],
                    outputs["proto"],
                    conf_thres=float(conf_thres),
                    iou_thres=float(iou_thres),
                    topk=int(topk),
                    max_det=int(max_det),
                    flow=flow,
                    o2o_nms=bool(o2o_nms),
                )
                written = save_segment_outputs(
                    str(staging_output_dir), image_file.name, image_bgr,
                    ratio, pad_w, pad_h, xyxy, scores, class_ids, coefs, proto, class_names,
                )
            else:
                xyxy, scores, class_ids, pre_nms = postprocess_output(
                    outputs["boxes"],
                    outputs["scores"],
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
            total_time_s, invoke_time_s = _accumulate_timing(
                runner, t1 - t0, total_time_s, invoke_time_s
            )
            print(f"{image_file.name}: flow={flow} preNMS={pre_nms} kept={written}")

        if processed > 0:
            print("=== Inference Timing Summary ===")
            print(f"processed={processed}")
            print(f"avg_total_inference_ms={(total_time_s / processed) * 1000.0:.3f}")
            print(f"avg_model_invoke_ms={_format_invoke_ms(invoke_time_s, processed)}")

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
        task: str = "detect",
        extra_quants: Sequence[QuantParams] | None = None,
    ):
        _warn_unused_onnx_backend_flag(backend)
        self.artifact = resolve_onnx_model_artifact(model_path)
        self.meta = load_onnx_model_metadata(
            model_path=self.artifact.model_path,
            model_type=model_type,
            precision=precision,
            box_quant=box_quant,
            class_quant=class_quant,
            task=task,
            extra_quants=extra_quants,
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
        results = self.session.run(
            self.meta.output_names,
            {self.meta.input_name: input_tensor},
        )
        self.last_invoke_time_s = time.perf_counter() - t_invoke0
        outputs = dict(zip(TASK_SPECS[self.meta.task].outputs, results))
        if self.meta.task != "detect":
            outputs = self.meta.to_float(outputs)
        return (
            outputs,
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
        task: str = "detect",
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
            task=task,
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
            *quant_cli_args(self.meta),
        ]
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

        outputs = load_raw_outputs(
            self.local_output_dir, str(rec.image_id), TASK_SPECS[self.meta.task].outputs
        )
        if self.meta.task != "detect":
            outputs = self.meta.to_float(outputs)
        meta = self.shared_meta[rec.image_id]
        return (
            outputs,
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
    task: str = "detect",
):
    _ensure_runtime_deps()
    resolved_artifact = artifact or resolve_onnx_model_artifact(model_path)
    meta = load_onnx_model_metadata(
        resolved_artifact.model_path, model_type, precision, task=task
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
    task: str = "detect",
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
            task=task,
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
            task=task,
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
    task: str = "detect",
) -> None:
    _ensure_runtime_deps()
    if runtime != "onnx":
        raise RuntimeError(f"Unsupported runtime for ONNX helpers: {runtime}")
    if disable_int8_prefilter:
        print(
            "[warn] --disable-int8-prefilter is not used by the ONNX Runtime path and will be ignored."
        )

    from tool.adb_runtime_bootstrap import (
        ensure_adb_runtime_venv,
        push_postprocess_package,
    )

    artifact = resolve_onnx_model_artifact(model_path)
    meta = load_onnx_model_metadata(artifact.model_path, model_type, precision, task=task)
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
        push_postprocess_package(adb_serial, remote_run_dir)
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
            *quant_cli_args(meta),
        ]
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
    box_quant, class_quant, extra_quants = quant_params_from_args(args)
    runner = ORTRawModel(
        model_path=args.model,
        model_type=args.model_type,
        precision=args.precision,
        qnn_lib=args.qnn_lib,
        backend=args.backend,
        no_qnn=args.no_qnn,
        box_quant=box_quant,
        class_quant=class_quant,
        task=args.task,
        extra_quants=extra_quants,
    )
    try:
        input_dir = Path(args.input_dir).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        invoke_seconds = []
        for input_file in sorted(input_dir.glob("*.npy")):
            tensor = np.load(input_file)
            t0 = time.perf_counter()
            results = runner.session.run(
                runner.meta.output_names,
                {runner.meta.input_name: tensor},
            )
            invoke_seconds.append(time.perf_counter() - t0)
            save_raw_outputs(
                output_dir,
                input_file.stem,
                dict(zip(TASK_SPECS[runner.meta.task].outputs, results)),
            )
        save_avg_invoke_ms(output_dir, invoke_seconds)
    finally:
        runner.cleanup()


def _run_test_mode_cli(args: argparse.Namespace) -> None:
    box_quant, class_quant, extra_quants = quant_params_from_args(args)
    runner = ORTRawModel(
        model_path=args.model,
        model_type=args.model_type,
        precision=args.precision,
        qnn_lib=args.qnn_lib,
        backend=args.backend,
        no_qnn=args.no_qnn,
        box_quant=box_quant,
        class_quant=class_quant,
        task=args.task,
        extra_quants=extra_quants,
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
            task=args.task,
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
    parser.add_argument(
        "--task", choices=["detect", "pose", "segment", "classify"], default="detect"
    )
    # w8a16 pose / segmentation: scale and zero point of each output after boxes and scores.
    parser.add_argument("--extra-quant", nargs=2, action="append", metavar=("SCALE", "ZP"))
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
