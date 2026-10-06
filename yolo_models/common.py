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
"""QC helpers shared by the yolov10 / yolov11 / yolov26 pipelines."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import qai_hub as hub
except Exception:
    hub = None


def require_qc_deps(family: str, need_hub: bool = True) -> None:
    """Fail early with every missing QC package listed.

    qai_hub is only needed by the AI Hub flows (litert, onnx); qairt converts
    offline with the vendored SDK, so it passes need_hub=False.
    """
    missing = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if Image is None:
        missing.append("Pillow")
    if YOLO is None:
        missing.append("ultralytics")
    if need_hub and hub is None:
        missing.append("qai_hub")
    if missing:
        raise RuntimeError(
            f"Missing QC dependencies for {family} quantize_convert: "
            + ", ".join(missing)
        )


# ----------------------------
# Calibration loader
# ----------------------------
def load_calibration_images(
    images_dir: str, input_hw: int, max_images: int = 200
) -> list[np.ndarray]:
    """
    Loads up to max_images from images_dir and returns NHWC float32 arrays in [0,1]:
      each element: [1,H,W,3]
    """
    missing = [name for name, mod in (("numpy", np), ("Pillow", Image)) if mod is None]
    if missing:
        raise RuntimeError(
            "Missing calibration dependencies: " + ", ".join(missing)
        )
    if not os.path.isdir(images_dir):
        raise RuntimeError(f"Calibration dir not found: {images_dir}")

    sample_inputs: list[np.ndarray] = []
    for name in sorted(os.listdir(images_dir)):
        if len(sample_inputs) >= max_images:
            break
        p = os.path.join(images_dir, name)
        if not os.path.isfile(p):
            continue
        try:
            im = Image.open(p).convert("RGB").resize((input_hw, input_hw))
        except Exception:
            continue
        arr = (np.array(im).astype(np.float32) / 255.0)[None, ...]  # [1,H,W,3]
        sample_inputs.append(arr)

    if not sample_inputs:
        raise RuntimeError(f"No calibration images loaded from: {images_dir}")

    return sample_inputs


# ----------------------------
# ONNX artifact finalization
# ----------------------------
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


def _write_self_contained_onnx(src_model: Path, dst_model: Path) -> None:
    """Write src_model to dst_model with its external data at <dst stem>.data.

    AI Hub names the weights file `model.data`. Keeping that name would make
    every ONNX artifact in one output directory share a single sidecar, so each
    qc run would silently break the models written before it. The tensor bytes
    are copied untouched; only the location recorded in the graph is renamed.
    """
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Python package 'onnx' is required to finalize ONNX artifacts."
        ) from exc

    model = onnx.load(str(src_model), load_external_data=False)
    data_name = f"{dst_model.stem}.data"
    source_locations: set[str] = set()
    for tensor in _iter_graph_tensors(model.graph):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                source_locations.add(entry.value)
                entry.value = data_name

    if not source_locations:
        if src_model != dst_model:
            shutil.copy2(src_model, dst_model)
        return
    if len(source_locations) != 1:
        raise RuntimeError(
            f"Expected one external data file for {src_model}, "
            f"got {sorted(source_locations)}"
        )

    src_data = src_model.parent / source_locations.pop()
    if not src_data.is_file():
        raise RuntimeError(
            f"ONNX model {src_model} references missing external data {src_data}"
        )
    dst_data = dst_model.with_name(data_name)
    if src_data.resolve() != dst_data.resolve():
        shutil.copy2(src_data, dst_data)
    onnx.save(model, str(dst_model))


def finalize_downloaded_onnx_artifact(
    downloaded_path: str | None,
    requested_output_path: str,
) -> str:
    requested = Path(requested_output_path).expanduser().resolve()
    candidate = Path(downloaded_path).expanduser().resolve() if downloaded_path else requested
    requested.parent.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(candidate):
        with tempfile.TemporaryDirectory(prefix="iqf_onnx_artifact_") as tmpdir:
            tmp_root = Path(tmpdir)
            with zipfile.ZipFile(candidate) as zf:
                zf.extractall(tmp_root)

            extracted_model = next(tmp_root.rglob("model.onnx"), None)
            if extracted_model is None:
                raise RuntimeError(
                    "Downloaded ONNX bundle does not contain model.onnx"
                )
            _write_self_contained_onnx(extracted_model, requested)
        if candidate != requested:
            # The AI Hub download (<output>.onnx.zip) is fully unpacked above.
            candidate.unlink(missing_ok=True)
        return str(requested)

    _write_self_contained_onnx(candidate, requested)
    return str(requested)


# ----------------------------
# QAIRT offline conversion
# ----------------------------
# Builds a pre-compiled HTP context binary locally: ONNX -> DLC -> (quantize) ->
# context binary. No Qualcomm AI Hub and no device-side compilation.

QAIRT_SDK_RELATIVE_PATH = "vendor/qairt/2.47.0.260601"
QAIRT_DSP_ARCH = "v73"  # QCS9075 / SA8775P-class HTP
QAIRT_ONNX_OPSET = 17
# Repo scheme names map onto qairt-quantizer's spelling: it accepts "min-max",
# not the repo's "minmax".
QAIRT_CALIBRATION_METHODS = {"mse": "mse", "minmax": "min-max"}


def qairt_sdk_root() -> str:
    root = Path(__file__).resolve().parent.parent / QAIRT_SDK_RELATIVE_PATH
    if not (root / "bin").is_dir():
        raise RuntimeError(
            f"QAIRT SDK not found at {root}. "
            "The vendored SDK ships with the repository; re-clone or restore it."
        )
    return str(root)


def qairt_env() -> dict:
    """Environment for the vendored SDK tools."""
    sdk = qairt_sdk_root()
    env = dict(os.environ)
    env["QNN_SDK_ROOT"] = sdk
    env["PATH"] = f"{sdk}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{sdk}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = f"{sdk}/lib/python:" + env.get("PYTHONPATH", "")
    return env


def run_qairt_tool(args: list[str], step: str) -> str:
    """Run one vendored SDK tool, surfacing its tail on failure."""
    sdk = qairt_sdk_root()
    proc = subprocess.run(
        [f"{sdk}/bin/{args[0]}", *args[1:]],
        env=qairt_env(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-12:])
        raise RuntimeError(f"QAIRT {step} failed:\n{tail}")
    return proc.stdout


def qairt_graph_name(dlc_path: str) -> str:
    """Graph name recorded in the DLC; the HTP config must reference it exactly."""
    out = run_qairt_tool(["qairt-dlc-info", "-i", dlc_path], "dlc-info")
    for line in out.splitlines():
        if line.startswith("Info of graph: "):
            return line.split("Info of graph: ", 1)[1].strip()
    raise RuntimeError(f"Could not read graph name from {dlc_path}")


def write_qairt_calibration_inputs(
    calib_dir: str, input_hw: int, max_calib: int, work_dir: str
) -> str:
    """Dump calibration images as float32 NHWC .raw files and list them."""
    samples = load_calibration_images(calib_dir, input_hw, max_images=max_calib)
    if not samples:
        raise RuntimeError(f"No calibration images found in {calib_dir}")
    raw_dir = os.path.join(work_dir, "calib")
    os.makedirs(raw_dir, exist_ok=True)
    listed = []
    for index, sample in enumerate(samples):
        raw_path = os.path.join(raw_dir, f"calib_{index:04d}.raw")
        np.ascontiguousarray(sample, dtype=np.float32).tofile(raw_path)
        listed.append(raw_path)
    list_path = os.path.join(work_dir, "input_list.txt")
    with open(list_path, "w") as handle:
        handle.write("\n".join(listed) + "\n")
    return list_path


def export_traced_model_to_onnx(pt_model, input_shape, onnx_path: str) -> None:
    """Export the traced wrapper to ONNX with fixed shapes and named outputs."""
    assert torch is not None
    dummy = torch.zeros(input_shape, dtype=torch.float32)
    kwargs = {
        "opset_version": QAIRT_ONNX_OPSET,
        "input_names": ["image"],
        "output_names": ["boxes", "scores"],
        "dynamic_axes": None,
        "do_constant_folding": True,
    }
    # torch >= 2.6 defaults to the dynamo exporter, which cannot export these heads.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(pt_model, dummy, onnx_path, **kwargs)


def build_qairt_context_binary(
    dlc_path: str, output_path: str, work_dir: str, vtcm_mb: int = 8
) -> None:
    """Offline-prepare the DLC into an HTP context binary."""
    sdk = qairt_sdk_root()
    graph_name = qairt_graph_name(dlc_path)
    graph_cfg = os.path.join(work_dir, "htp_graph.json")
    ext_cfg = os.path.join(work_dir, "htp_ext.json")
    with open(graph_cfg, "w") as handle:
        json.dump(
            {
                "graphs": [
                    {
                        "graph_names": [graph_name],
                        "vtcm_mb": vtcm_mb,
                        "O": 3,
                        "dlbc": 1,
                    }
                ],
                "devices": [
                    {"dsp_arch": QAIRT_DSP_ARCH, "pd_session": "unsigned"}
                ],
            },
            handle,
        )
    with open(ext_cfg, "w") as handle:
        json.dump(
            {
                "backend_extensions": {
                    "shared_library_path": "libQnnHtpNetRunExtensions.so",
                    "config_file_path": graph_cfg,
                }
            },
            handle,
        )

    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(output_path).stem
    run_qairt_tool(
        [
            "qnn-context-binary-generator",
            "--backend", f"{sdk}/lib/libQnnHtp.so",
            "--dlc_path", dlc_path,
            "--binary_file", stem,
            "--config_file", ext_cfg,
            "--output_dir", out_dir,
            "--log_level", "error",
        ],
        "context-binary-generator",
    )
    produced = os.path.join(out_dir, f"{stem}.bin")
    if produced != os.path.abspath(output_path):
        shutil.move(produced, output_path)
    if not os.path.exists(output_path):
        raise RuntimeError(f"QAIRT context binary was not produced at {output_path}")

