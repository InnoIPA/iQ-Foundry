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

import os
from pathlib import Path
import shutil
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


YOLOV11_TEST_DEFAULTS = {
    "default_flow": "default",
    "conf_thres": 0.25,
    "iou_thres": 0.6,
    "topk": 300,
    "max_det": 100,
}


def _require_qc_deps() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if Image is None:
        missing.append("Pillow")
    if YOLO is None:
        missing.append("ultralytics")
    if hub is None:
        missing.append("qai_hub")
    if missing:
        raise RuntimeError(
            "Missing QC dependencies for yolov11 quantize_convert: "
            + ", ".join(missing)
        )


_TORCH_BASE = torch.nn.Module if torch is not None else object


# ----------------------------
# Export wrapper (YOLOv11 boxes + scores)
# ----------------------------
class Yolo11BoxesScoresWrapper(_TORCH_BASE):
    """
    Input:
      NHWC float32 [1,H,W,3] in [0,1]
    Output:
      boxes, scores from YOLOv11 output dict:
        out_dict["boxes"], out_dict["scores"]
    """

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

        # Keep export flags OFF if present (defensive)
        if hasattr(self.core, "export"):
            self.core.export = False
        if hasattr(self.core, "model") and len(self.core.model) > 0:
            last = self.core.model[-1]
            if hasattr(last, "export"):
                last.export = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2).contiguous()

        out = self.core(x)

        # Common patterns:
        #  - (something, dict)
        #  - dict
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            out_dict = out[1]
        elif isinstance(out, dict):
            out_dict = out
        else:
            raise RuntimeError(f"Unexpected YOLOv11 output structure: type={type(out)}")

        if "boxes" not in out_dict or "scores" not in out_dict:
            raise RuntimeError(
                f"Expected keys 'boxes' and 'scores'. Keys={list(out_dict.keys())}"
            )

        boxes = out_dict["boxes"]
        scores = out_dict["scores"]

        if not torch.is_tensor(boxes) or not torch.is_tensor(scores):
            raise RuntimeError("boxes/scores are not tensors")

        # Light sanity: require batch dimension
        if boxes.ndim < 2 or scores.ndim < 2:
            raise RuntimeError(
                "Unexpected shapes: "
                f"boxes={tuple(boxes.shape)} scores={tuple(scores.shape)}"
            )

        return boxes, scores


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
    _require_qc_deps()
    if not os.path.isdir(images_dir):
        raise RuntimeError(f"Calibration dir not found: {images_dir}")
    assert np is not None and Image is not None

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


def _finalize_downloaded_onnx_artifact(
    downloaded_path: str | None,
    requested_output_path: str,
) -> str:
    requested = Path(requested_output_path).expanduser().resolve()
    candidate = Path(downloaded_path).expanduser().resolve() if downloaded_path else requested
    requested.parent.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(candidate):
        with tempfile.TemporaryDirectory(prefix="yolov11_onnx_artifact_") as tmpdir:
            tmp_root = Path(tmpdir)
            with zipfile.ZipFile(candidate) as zf:
                zf.extractall(tmp_root)

            extracted_model = next(tmp_root.rglob("model.onnx"), None)
            if extracted_model is None:
                raise RuntimeError(
                    "Downloaded ONNX bundle does not contain model.onnx"
                )

            shutil.copy2(extracted_model, requested)
            extracted_data = extracted_model.with_name("model.data")
            if extracted_data.is_file():
                shutil.copy2(extracted_data, requested.with_name("model.data"))
        return str(requested)

    if candidate != requested:
        shutil.copy2(candidate, requested)
        candidate_data = candidate.with_name("model.data")
        if candidate_data.is_file():
            shutil.copy2(candidate_data, requested.with_name("model.data"))
    return str(requested)


# ----------------------------
# Pipeline
# ----------------------------
class YoloV11Pipeline:
    def convert(
        self,
        model_path: str,
        output_path: str,
        runtime: str,
        precision: str,
        calib_dir: str,
        max_calib: int = 200,
        qc_head: str = "default",
        qc_quant_scheme: str = "minmax",
    ) -> None:
        if runtime == "litert" and precision == "int8":
            self.quantize_convert(
                model_path=model_path,
                out_tflite=output_path,
                calib_dir=calib_dir,
                max_calib=max_calib,
                qc_head=qc_head,
                qc_quant_scheme=qc_quant_scheme,
            )
            return
        if runtime == "litert" and precision == "fp32":
            self.export_tflite_fp32(
                model_path=model_path,
                output_path=output_path,
            )
            return
        if runtime == "onnx" and precision == "fp32":
            self.export_onnx_fp32(
                model_path=model_path,
                output_path=output_path,
            )
            return
        if runtime == "onnx" and precision == "w8a16":
            self.export_onnx_w8a16(
                model_path=model_path,
                output_path=output_path,
                calib_dir=calib_dir,
                max_calib=max_calib,
                qc_quant_scheme=qc_quant_scheme,
            )
            return
        raise ValueError(f"Unsupported runtime/precision: {runtime}/{precision}")

    def _build_traced_model(self, model_path: str):
        _require_qc_deps()
        assert torch is not None and YOLO is not None

        input_hw = 640
        input_shape = (1, input_hw, input_hw, 3)
        y = YOLO(model_path)
        core = y.model.eval()
        torch_model = Yolo11BoxesScoresWrapper(core).eval()
        example = torch.rand(input_shape, dtype=torch.float32)
        pt_model = torch.jit.trace(
            torch_model, example, strict=False, check_trace=False
        )
        return pt_model, input_hw, input_shape

    def export_onnx_fp32(
        self,
        model_path: str,
        output_path: str,
    ) -> None:
        _require_qc_deps()
        assert hub is not None

        pt_model, _, input_shape = self._build_traced_model(model_path)
        device = hub.Device("Dragonwing IQ-9075 EVK")
        compile_job = hub.submit_compile_job(
            model=pt_model,
            device=device,
            input_specs={"image": input_shape},
            options="--target_runtime onnx",
        )
        downloaded_path = compile_job.download_target_model(str(output_path))
        final_path = _finalize_downloaded_onnx_artifact(downloaded_path, output_path)
        print(f"[yolov11] wrote onnx: {final_path}")

    def export_tflite_fp32(
        self,
        model_path: str,
        output_path: str,
    ) -> None:
        _require_qc_deps()
        assert hub is not None

        pt_model, _, input_shape = self._build_traced_model(model_path)
        device = hub.Device("Dragonwing IQ-9075 EVK")
        compile_onnx_job = hub.submit_compile_job(
            model=pt_model,
            device=device,
            input_specs={"image": input_shape},
            options="--target_runtime onnx",
        )
        unquantized_onnx_model = compile_onnx_job.get_target_model()

        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        compile_tflite_job = hub.submit_compile_job(
            model=unquantized_onnx_model,
            device=device,
            options="--target_runtime tflite",
        )
        compile_tflite_job.download_target_model(output_path)
        print(f"[yolov11] wrote tflite: {output_path}")

    def export_onnx_w8a16(
        self,
        model_path: str,
        output_path: str,
        calib_dir: str,
        max_calib: int = 200,
        qc_quant_scheme: str = "minmax",
    ) -> None:
        if qc_quant_scheme not in ("mse", "minmax"):
            raise ValueError(f"Unsupported qc_quant_scheme: {qc_quant_scheme}")
        _require_qc_deps()
        assert hub is not None

        pt_model, input_hw, input_shape = self._build_traced_model(model_path)
        device = hub.Device("Dragonwing IQ-9075 EVK")
        compile_onnx_job = hub.submit_compile_job(
            model=pt_model,
            device=device,
            input_specs={"image": input_shape},
            options="--target_runtime onnx",
        )
        unquantized_onnx_model = compile_onnx_job.get_target_model()

        sample_inputs = load_calibration_images(
            calib_dir, input_hw, max_images=max_calib
        )
        quantize_kwargs = {
            "model": unquantized_onnx_model,
            "calibration_data": {"image": sample_inputs},
            "weights_dtype": hub.QuantizeDtype.INT8,
            "activations_dtype": hub.QuantizeDtype.INT16,
        }
        if qc_quant_scheme == "minmax":
            quantize_kwargs["options"] = "--range_scheme min_max"
        quantize_job = hub.submit_quantize_job(**quantize_kwargs)
        quantized_onnx_model = quantize_job.get_target_model()

        compile_quantized_job = hub.submit_compile_job(
            model=quantized_onnx_model,
            device=device,
            options="--target_runtime onnx --quantize_io",
        )
        downloaded_path = compile_quantized_job.download_target_model(str(output_path))
        final_path = _finalize_downloaded_onnx_artifact(downloaded_path, output_path)
        print(f"[yolov11] wrote onnx: {final_path}")

    def quantize_convert(
        self,
        model_path: str,
        out_tflite: str,
        calib_dir: str,
        max_calib: int = 200,
        qc_head: str = "default",
        qc_quant_scheme: str = "minmax",
    ):
        """
        YOLOv11 .pt -> (trace wrapper) -> AI Hub compile ONNX -> quant INT8
        -> compile TFLite -> download

        Notes:
          - wrapper exports boxes + scores from YOLOv11 output dict
          - keeps --quantize_io
          - calib_dir is compulsory (enforced by cli.py)
        """
        if qc_quant_scheme not in ("mse", "minmax"):
            raise ValueError(f"Unsupported qc_quant_scheme: {qc_quant_scheme}")
        _ = qc_head  # intentionally ignored for yolov11
        _require_qc_deps()
        assert torch is not None and YOLO is not None and hub is not None

        input_hw = 640
        input_shape = (1, input_hw, input_hw, 3)  # NHWC
        device_name = "Dragonwing IQ-9075 EVK"
        images_dir = calib_dir

        out_tflite = str(out_tflite)
        Path(out_tflite).parent.mkdir(parents=True, exist_ok=True)

        # 1) Load Ultralytics model
        y = YOLO(model_path)
        core = y.model.eval()

        # 2) Wrap to export boxes + scores only
        torch_model = Yolo11BoxesScoresWrapper(core).eval()

        # 3) Trace
        example = torch.rand(input_shape, dtype=torch.float32)
        pt_model = torch.jit.trace(
            torch_model, example, strict=False, check_trace=False
        )

        # 4) Compile TorchScript -> ONNX (AI Hub)
        device = hub.Device(device_name)
        compile_onnx_job = hub.submit_compile_job(
            model=pt_model,
            device=device,
            input_specs={"image": input_shape},
            options="--target_runtime onnx",
        )
        unquantized_onnx_model = compile_onnx_job.get_target_model()

        # 5) Calibration data
        sample_inputs = load_calibration_images(
            images_dir, input_hw, max_images=max_calib
        )
        calibration_data = {"image": sample_inputs}

        # 6) Quantize INT8 (MSE default; minmax via explicit option)
        quantize_kwargs = {
            "model": unquantized_onnx_model,
            "calibration_data": calibration_data,
            "weights_dtype": hub.QuantizeDtype.INT8,
            "activations_dtype": hub.QuantizeDtype.INT8,
        }
        if qc_quant_scheme == "minmax":
            quantize_kwargs["options"] = "--range_scheme min_max"
        quantize_job = hub.submit_quantize_job(**quantize_kwargs)
        quantized_onnx_model = quantize_job.get_target_model()

        # 7) Compile to TFLite (keep quantized IO)
        compile_tflite_job = hub.submit_compile_job(
            model=quantized_onnx_model,
            device=device,
            options="--target_runtime tflite --quantize_io",
        )

        # 8) Download compiled model
        compile_tflite_job.download_target_model(out_tflite)
        print(f"[yolov11] wrote tflite: {out_tflite}")
