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
"""Shared QAIRT inference helpers for test mode and mAP evaluation.

QAIRT runs a pre-compiled HTP context binary through `qnn-net-run`. The target
already ships the QAIRT runtime, so nothing is installed on device: inputs are
pushed as raw tensors, the device's own binary runs them, and the raw outputs are
pulled back and decoded on the host.

Geometry, decode, NMS and drawing are imported from tool.onnx_inference so the
three runtimes produce identical detections from identical tensors.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from tool.onnx_inference import (
    IMG_H,
    IMG_W,
    QuantParams,
    _adb_pull,
    _adb_push,
    _adb_shell,
    _commit_output_dir,
    _has_meaningful_outputs,
    _is_iq9_native_runtime,
    _prepare_image_input,
    _run_test_directory,
    collect_image_files,
    letterbox_bgr,
)

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

ANCHOR_COUNT = 8400
# Box channel count per family; also selects the decoder downstream.
EXPECTED_BOX_CHANNELS = {"yolov10": 64, "yolov11": 64, "yolov26": 4}
EXPECTED_BOX_MODES = {"yolov10": "dfl64", "yolov11": "dfl64", "yolov26": "ltrb4"}

QAIRT_SDK_RELATIVE_PATH = "vendor/qairt/2.47.0.260601"
DEFAULT_QAIRT_BACKEND = "libQnnHtp.so"
DEFAULT_QAIRT_REMOTE_WORKDIR = "/data/local/tmp/yolo_test"


def _ensure_runtime_deps() -> None:
    global np, cv2
    if np is None:
        import numpy as _np

        np = _np
    if cv2 is None:
        import cv2 as _cv2

        cv2 = _cv2


# --- SDK location -------------------------------------------------------------
def qairt_sdk_root() -> str | None:
    """Vendored host SDK, or None when running on the target (tools are in PATH)."""
    root = Path(__file__).resolve().parent.parent / QAIRT_SDK_RELATIVE_PATH
    return str(root) if (root / "bin").is_dir() else None


def qairt_tool(name: str) -> str:
    """Prefer the vendored host tool, else the target's own copy on PATH."""
    sdk = qairt_sdk_root()
    if sdk and os.path.exists(os.path.join(sdk, "bin", name)):
        return os.path.join(sdk, "bin", name)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(
            f"QAIRT tool '{name}' not found. Expected the vendored SDK under "
            f"{QAIRT_SDK_RELATIVE_PATH} or the QAIRT runtime on PATH."
        )
    return found


def qairt_env() -> dict:
    sdk = qairt_sdk_root()
    env = dict(os.environ)
    if sdk:
        env["QNN_SDK_ROOT"] = sdk
        env["LD_LIBRARY_PATH"] = f"{sdk}/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["PYTHONPATH"] = f"{sdk}/lib/python:" + env.get("PYTHONPATH", "")
    return env


def run_qairt_tool(name: str, args: list[str], step: str) -> str:
    proc = subprocess.run(
        [qairt_tool(name), *args], env=qairt_env(), capture_output=True, text=True
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-12:])
        raise RuntimeError(f"QAIRT {step} failed:\n{tail}")
    return proc.stdout


# --- metadata -----------------------------------------------------------------
class QAIRTModelMeta:
    """Graph I/O description of a context binary.

    `precision` deliberately reports the tensor domain seen by postprocessing, not
    the user's precision: qnn-net-run returns already-dequantized float32, so the
    shared postprocess must not dequantize again. The requested precision is kept
    separately in `model_precision`.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str,
        model_precision: str,
        input_name: str,
        box_output_name: str,
        class_output_name: str,
        box_mode: str,
        class_count: int,
        box_quant: QuantParams | None,
        class_quant: QuantParams | None,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.model_precision = model_precision
        self.precision = "fp32"
        self.input_name = input_name
        self.input_layout = "nhwc"
        self.box_output_name = box_output_name
        self.class_output_name = class_output_name
        self.box_mode = box_mode
        self.class_count = class_count
        self.box_quant = box_quant
        self.class_quant = class_quant


def _quant_params_from_tensor(info: dict) -> QuantParams | None:
    """QNN reports offset = -zero_point; normalize to the repo's convention."""
    if "FIXED_POINT" not in str(info.get("dataType", "")):
        return None
    scale_offset = (info.get("quantizeParams") or {}).get("scaleOffset") or {}
    if "scale" not in scale_offset:
        return None
    return QuantParams(
        scale=float(scale_offset["scale"]),
        zero_point=-int(scale_offset.get("offset", 0)),
    )


def load_qairt_model_metadata(
    model_path: str, model_type: str, precision: str
) -> QAIRTModelMeta:
    """Read graph I/O from the context binary and validate it against --type."""
    if not os.path.exists(model_path):
        raise RuntimeError(f"Context binary not found: {model_path}")
    with tempfile.TemporaryDirectory(prefix="qairt_meta_") as tmp:
        info_path = os.path.join(tmp, "context.json")
        run_qairt_tool(
            "qnn-context-binary-utility",
            [f"--context_binary={model_path}", f"--json_file={info_path}"],
            "context-binary-utility",
        )
        with open(info_path) as handle:
            info = json.load(handle)

    graphs = (info.get("info") or {}).get("graphs") or []
    if not graphs:
        raise RuntimeError(f"No graphs found in context binary: {model_path}")
    graph = graphs[0]["info"]
    tensors = {
        t["info"]["name"]: t["info"]
        for key in ("graphInputs", "graphOutputs")
        for t in graph.get(key, [])
    }
    for required in ("image", "boxes", "scores"):
        if required not in tensors:
            raise RuntimeError(
                f"Context binary is missing the '{required}' tensor. "
                f"Found: {sorted(tensors)}. Check that --model is an iQ-Foundry "
                "QAIRT artifact."
            )

    box_dims = list(tensors["boxes"]["dimensions"])
    cls_dims = list(tensors["scores"]["dimensions"])
    expected_channels = EXPECTED_BOX_CHANNELS.get(model_type)
    if expected_channels is None:
        raise RuntimeError(f"Unsupported model_type: {model_type}")
    if box_dims != [1, expected_channels, ANCHOR_COUNT]:
        raise RuntimeError(
            f"Output shape mismatch: --type {model_type} expects boxes "
            f"[1,{expected_channels},{ANCHOR_COUNT}], but model exposes {box_dims}. "
            "Check that --type and --model match."
        )
    if len(cls_dims) != 3 or cls_dims[2] != ANCHOR_COUNT:
        raise RuntimeError(f"Unexpected scores shape {cls_dims}.")

    return QAIRTModelMeta(
        model_path=model_path,
        model_type=model_type,
        model_precision=precision,
        input_name="image",
        box_output_name="boxes",
        class_output_name="scores",
        box_mode=EXPECTED_BOX_MODES[model_type],
        class_count=int(cls_dims[1]),
        box_quant=_quant_params_from_tensor(tensors["boxes"]),
        class_quant=_quant_params_from_tensor(tensors["scores"]),
    )


# --- preprocessing ------------------------------------------------------------
def build_qairt_input_tensor(image_bgr):
    """Letterboxed NHWC float32 in [0,1]; qnn-net-run converts to the graph dtype."""
    _ensure_runtime_deps()
    padded_bgr, (ratio, pad_w, pad_h) = letterbox_bgr(image_bgr, (IMG_H, IMG_W))
    rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 255.0)[None, ...]
    return np.ascontiguousarray(tensor, dtype=np.float32), ratio, pad_w, pad_h


def _read_result_pair(result_dir: str, meta: QAIRTModelMeta):
    """Load one qnn-net-run result directory as (boxes, scores) float32 tensors."""
    _ensure_runtime_deps()
    box_channels = EXPECTED_BOX_CHANNELS[meta.model_type]
    box_path = os.path.join(result_dir, f"{meta.box_output_name}.raw")
    cls_path = os.path.join(result_dir, f"{meta.class_output_name}.raw")
    for path in (box_path, cls_path):
        if not os.path.exists(path):
            raise RuntimeError(f"qnn-net-run produced no output at {path}")
    boxes = np.fromfile(box_path, dtype=np.float32).reshape(1, box_channels, ANCHOR_COUNT)
    scores = np.fromfile(cls_path, dtype=np.float32).reshape(
        1, meta.class_count, ANCHOR_COUNT
    )
    return boxes, scores


# --- runners ------------------------------------------------------------------
class QAIRTRawModel:
    """Runs the context binary locally. Only valid on the IQ9 target itself."""

    def __init__(
        self,
        model_path: str,
        model_type: str,
        precision: str,
        qnn_lib: str = DEFAULT_QAIRT_BACKEND,
        backend: str = "htp",
        no_qnn: bool = False,
        box_quant: QuantParams | None = None,
        class_quant: QuantParams | None = None,
    ):
        _ = backend, box_quant, class_quant  # read from the context binary instead
        if no_qnn:
            raise RuntimeError(
                "--no-qnn is not supported by the QAIRT runtime: a context binary "
                "is pre-compiled for the HTP and has no CPU fallback."
            )
        self.meta = load_qairt_model_metadata(model_path, model_type, precision)
        self.backend_lib = qnn_lib or DEFAULT_QAIRT_BACKEND
        self.last_invoke_time_s = 0.0
        self._tmp = tempfile.TemporaryDirectory(prefix="qairt_local_")

    def cleanup(self) -> None:
        self._tmp.cleanup()

    def infer_raw(self, image_path: str):
        _ensure_runtime_deps()
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        tensor, ratio, pad_w, pad_h = build_qairt_input_tensor(image_bgr)

        work = os.path.join(self._tmp.name, "run")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work, exist_ok=True)
        raw_path = os.path.join(work, "input.raw")
        tensor.tofile(raw_path)
        list_path = os.path.join(work, "input_list.txt")
        with open(list_path, "w") as handle:
            handle.write(raw_path + "\n")

        out_dir = os.path.join(work, "out")
        t0 = time.perf_counter()
        run_qairt_tool(
            "qnn-net-run",
            [
                f"--backend={self.backend_lib}",
                f"--retrieve_context={self.meta.model_path}",
                f"--input_list={list_path}",
                f"--output_dir={out_dir}",
                "--perf_profile=burst",
            ],
            "net-run",
        )
        self.last_invoke_time_s = time.perf_counter() - t0
        boxes, scores = _read_result_pair(os.path.join(out_dir, "Result_0"), self.meta)
        return (
            boxes,
            scores,
            ratio,
            pad_w,
            pad_h,
            (image_bgr.shape[1], image_bgr.shape[0]),
        )


class ADBQAIRTRawModel:
    """Runs the context binary on the target over adb, one batch per model.

    The target already provides qnn-net-run, so only the binary and the input
    tensors are pushed; decoding stays on the host.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str,
        precision: str,
        adb_serial: str | None,
        remote_workdir: str,
        qnn_lib: str = DEFAULT_QAIRT_BACKEND,
        backend: str = "htp",
        no_qnn: bool = False,
        shared_remote_input_dir: str | None = None,
        shared_meta: dict | None = None,
        box_quant: QuantParams | None = None,
        class_quant: QuantParams | None = None,
    ):
        _ = backend, box_quant, class_quant
        if no_qnn:
            raise RuntimeError(
                "--no-qnn is not supported by the QAIRT runtime: a context binary "
                "is pre-compiled for the HTP and has no CPU fallback."
            )
        self.meta = load_qairt_model_metadata(model_path, model_type, precision)
        self.adb_serial = adb_serial
        self.backend_lib = qnn_lib or DEFAULT_QAIRT_BACKEND
        self.shared_remote_input_dir = shared_remote_input_dir
        self.shared_meta = shared_meta or {}
        self.last_invoke_time_s = 0.0
        self.remote_dir = (
            f"{remote_workdir.rstrip('/')}/qairt_run_{os.getpid()}_"
            f"{int(time.time() * 1000)}"
        )
        self._pulled = tempfile.TemporaryDirectory(prefix="qairt_adb_out_")
        self._results: dict[int, str] = {}
        _adb_shell(self.adb_serial, f"rm -rf {self.remote_dir}; mkdir -p {self.remote_dir}")
        self.remote_model = f"{self.remote_dir}/{Path(self.meta.model_path).name}"
        _adb_push(self.adb_serial, self.meta.model_path, self.remote_model)

    def prepare_batch(self, images) -> None:
        """Run every image in one qnn-net-run invocation and pull the outputs."""
        if not self.shared_remote_input_dir:
            raise RuntimeError("prepare_batch requires a shared remote input dir.")
        order = [rec.image_id for rec in images]
        listing = "\n".join(
            f"{self.shared_remote_input_dir}/{image_id}.raw" for image_id in order
        )
        list_local = os.path.join(self._pulled.name, "input_list.txt")
        with open(list_local, "w") as handle:
            handle.write(listing + "\n")
        _adb_push(self.adb_serial, list_local, f"{self.remote_dir}/input_list.txt")

        remote_out = f"{self.remote_dir}/out"
        t0 = time.perf_counter()
        _adb_shell(
            self.adb_serial,
            f"cd {self.remote_dir} && rm -rf out && qnn-net-run "
            f"--backend {self.backend_lib} "
            f"--retrieve_context {self.remote_model} "
            f"--input_list input_list.txt --output_dir out --perf_profile burst",
        )
        elapsed = time.perf_counter() - t0
        self.last_invoke_time_s = elapsed / max(len(order), 1)

        local_out = os.path.join(self._pulled.name, "out")
        shutil.rmtree(local_out, ignore_errors=True)
        os.makedirs(local_out, exist_ok=True)
        _adb_pull(self.adb_serial, remote_out, local_out)
        pulled_root = os.path.join(local_out, "out")
        if not os.path.isdir(pulled_root):
            pulled_root = local_out
        for index, image_id in enumerate(order):
            self._results[image_id] = os.path.join(pulled_root, f"Result_{index}")

    def infer_raw(self, image_path: str):
        raise RuntimeError(
            "ADBQAIRTRawModel uses prepare_batch() + get_result_for_image()"
        )

    def get_result_for_image(self, rec):
        result_dir = self._results.get(rec.image_id)
        if result_dir is None:
            raise RuntimeError(f"No QAIRT result for image_id {rec.image_id}")
        boxes, scores = _read_result_pair(result_dir, self.meta)
        info = self.shared_meta[rec.image_id]
        return (
            boxes,
            scores,
            info["ratio"],
            info["padw"],
            info["padh"],
            info["orig_size"],
        )

    def cleanup(self) -> None:
        try:
            _adb_shell(self.adb_serial, f"rm -rf {self.remote_dir}")
        except subprocess.CalledProcessError as exc:
            print(f"[warn] remote cleanup failed: {exc}")
        self._pulled.cleanup()


def prepare_shared_qairt_inputs(
    images,
    model_path: str,
    model_type: str,
    precision: str,
    adb_serial: str | None,
    remote_input_dir: str,
):
    """Letterbox every image once on the host and push the raw tensors to target.

    Returns (tmpdir, meta_by_image_id, model_meta, None). The trailing None keeps
    the 4-tuple shape the mAP caller already unpacks for the ONNX path.
    """
    _ensure_runtime_deps()
    meta = load_qairt_model_metadata(model_path, model_type, precision)
    tmp = tempfile.TemporaryDirectory(prefix="qairt_shared_inputs_")
    stage = os.path.join(tmp.name, "inputs")
    os.makedirs(stage, exist_ok=True)

    meta_by_image: dict[int, dict] = {}
    for rec in images:
        image_bgr = cv2.imread(str(rec.path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {rec.path}")
        tensor, ratio, pad_w, pad_h = build_qairt_input_tensor(image_bgr)
        tensor.tofile(os.path.join(stage, f"{rec.image_id}.raw"))
        meta_by_image[rec.image_id] = {
            "ratio": ratio,
            "padw": pad_w,
            "padh": pad_h,
            "orig_size": (image_bgr.shape[1], image_bgr.shape[0]),
        }

    _adb_shell(adb_serial, f"rm -rf {remote_input_dir}; mkdir -p {remote_input_dir}")
    _adb_push(adb_serial, stage, remote_input_dir)
    # adb push of a directory nests it under the destination; flatten if so.
    _adb_shell(
        adb_serial,
        f"if [ -d {remote_input_dir}/inputs ]; then "
        f"mv {remote_input_dir}/inputs/* {remote_input_dir}/ 2>/dev/null; "
        f"rmdir {remote_input_dir}/inputs; fi",
    )
    return tmp, meta_by_image, meta, None


# --- test mode entry points ---------------------------------------------------
def run_qairt_test_inference_local(
    *,
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
    qnn_lib: str = DEFAULT_QAIRT_BACKEND,
    backend: str = "htp",
    runtime: str = "qairt",
    precision: str = "int8",
    enforce_iq9_native: bool = False,
) -> None:
    """Run test mode directly on the target using its own qnn-net-run."""
    if runtime != "qairt":
        raise RuntimeError(f"run_qairt_test_inference_local got runtime={runtime}")
    if disable_int8_prefilter:
        print(
            "[warn] --disable-int8-prefilter is not used by the QAIRT path "
            "and will be ignored."
        )
    if enforce_iq9_native and not _is_iq9_native_runtime():
        raise RuntimeError(
            "QAIRT local test mode runs a pre-compiled HTP context binary and is "
            "only valid on the IQ9 target. Use --adb from an x86 host."
        )

    resolved_dir, tmp_obj = _prepare_image_input(image_dir, image_path)
    runner = QAIRTRawModel(
        model_path=model_path,
        model_type=model_type,
        precision=precision,
        qnn_lib=qnn_lib,
        backend=backend,
        no_qnn=no_qnn,
    )
    try:
        _run_test_directory(
            runner=runner,
            yaml_path=yaml_path,
            image_dir=resolved_dir,
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
        runner.cleanup()
        if tmp_obj is not None:
            tmp_obj.cleanup()


class _AdbBatchTestRunner:
    """Adapts the adb batch runner to the per-image loop used by test mode."""

    def __init__(self, inner: ADBQAIRTRawModel, results: dict, order: list):
        self.meta = inner.meta
        self._inner = inner
        self._results = results
        self._order = order
        self.last_invoke_time_s = inner.last_invoke_time_s

    def infer_raw(self, image_path: str):
        key = str(image_path)
        if key not in self._results:
            raise RuntimeError(f"No QAIRT result for {key}")
        self.last_invoke_time_s = self._inner.last_invoke_time_s
        return self._results[key]

    def cleanup(self) -> None:
        self._inner.cleanup()


def run_qairt_test_inference_adb(
    *,
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
    qnn_lib: str = DEFAULT_QAIRT_BACKEND,
    backend: str = "htp",
    runtime: str = "qairt",
    precision: str = "int8",
    adb_serial: str | None = None,
    remote_workdir: str = DEFAULT_QAIRT_REMOTE_WORKDIR,
) -> None:
    """Run test mode on the target over adb; decode and draw on the host."""
    if runtime != "qairt":
        raise RuntimeError(f"run_qairt_test_inference_adb got runtime={runtime}")
    if disable_int8_prefilter:
        print(
            "[warn] --disable-int8-prefilter is not used by the QAIRT path "
            "and will be ignored."
        )
    _ensure_runtime_deps()

    resolved_dir, tmp_obj = _prepare_image_input(image_dir, image_path)
    image_files = collect_image_files(resolved_dir)
    if not image_files:
        raise RuntimeError(f"No images found in {resolved_dir}")

    runner = ADBQAIRTRawModel(
        model_path=model_path,
        model_type=model_type,
        precision=precision,
        adb_serial=adb_serial,
        remote_workdir=remote_workdir,
        qnn_lib=qnn_lib,
        backend=backend,
        no_qnn=no_qnn,
    )
    try:
        # Push one raw tensor per image, run them as a single batch, pull results.
        remote_inputs = f"{runner.remote_dir}/inputs"
        stage = tempfile.TemporaryDirectory(prefix="qairt_test_inputs_")
        geometry = {}
        try:
            for index, image_file in enumerate(image_files):
                image_bgr = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"Failed to read image: {image_file}")
                tensor, ratio, pad_w, pad_h = build_qairt_input_tensor(image_bgr)
                tensor.tofile(os.path.join(stage.name, f"{index:06d}.raw"))
                geometry[str(image_file)] = (
                    ratio,
                    pad_w,
                    pad_h,
                    (image_bgr.shape[1], image_bgr.shape[0]),
                )
            _adb_shell(adb_serial, f"mkdir -p {remote_inputs}")
            _adb_push(adb_serial, stage.name, remote_inputs)
            _adb_shell(
                adb_serial,
                f"if [ -d {remote_inputs}/{Path(stage.name).name} ]; then "
                f"mv {remote_inputs}/{Path(stage.name).name}/* {remote_inputs}/; fi",
            )

            listing = "\n".join(
                f"{remote_inputs}/{index:06d}.raw" for index in range(len(image_files))
            )
            list_local = os.path.join(stage.name, "input_list.txt")
            with open(list_local, "w") as handle:
                handle.write(listing + "\n")
            _adb_push(adb_serial, list_local, f"{runner.remote_dir}/input_list.txt")

            t0 = time.perf_counter()
            _adb_shell(
                adb_serial,
                f"cd {runner.remote_dir} && rm -rf out && qnn-net-run "
                f"--backend {runner.backend_lib} "
                f"--retrieve_context {runner.remote_model} "
                f"--input_list input_list.txt --output_dir out --perf_profile burst",
            )
            runner.last_invoke_time_s = (time.perf_counter() - t0) / len(image_files)

            pulled = tempfile.TemporaryDirectory(prefix="qairt_test_out_")
            _adb_pull(adb_serial, f"{runner.remote_dir}/out", pulled.name)
            root = os.path.join(pulled.name, "out")
            if not os.path.isdir(root):
                root = pulled.name

            results = {}
            for index, image_file in enumerate(image_files):
                boxes, scores = _read_result_pair(
                    os.path.join(root, f"Result_{index}"), runner.meta
                )
                ratio, pad_w, pad_h, orig = geometry[str(image_file)]
                results[str(image_file)] = (boxes, scores, ratio, pad_w, pad_h, orig)

            _run_test_directory(
                runner=_AdbBatchTestRunner(runner, results, image_files),
                yaml_path=yaml_path,
                image_dir=resolved_dir,
                output_dir=output_dir,
                default_flow=default_flow,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                topk=topk,
                max_det=max_det,
                postprocess_flow=postprocess_flow,
                o2o_nms=o2o_nms,
            )
            pulled.cleanup()
        finally:
            stage.cleanup()
    finally:
        runner.cleanup()
        if tmp_obj is not None:
            tmp_obj.cleanup()
