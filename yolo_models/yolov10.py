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


YOLOV10_TEST_DEFAULTS = {
    "default_flow": "o2m",
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
            "Missing QC dependencies for yolov10 quantize_convert: "
            + ", ".join(missing)
        )


_TORCH_BASE = torch.nn.Module if torch is not None else object


# ----------------------------
# RAW export wrapper (YOLO10 branch-selectable: one2many / one2one)
# ----------------------------
class Yolo10RawBranchWrapper(_TORCH_BASE):
    """
    Input:
      NHWC float32 [1,H,W,3] in [0,1]
    Output:
      boxes_dfl  [1,64,8400]  (DFL logits for 4 coords, REG_MAX=16 => 4*16=64)
      cls_logits [1,C,8400]   (class logits, dynamic class count)
    """

    def __init__(self, core: torch.nn.Module, branch_key: str):
        super().__init__()
        self.core = core
        self.branch_key = branch_key

        # Keep export flags OFF so dict branches remain
        if hasattr(self.core, "export"):
            self.core.export = False
        if hasattr(self.core, "model") and len(self.core.model) > 0:
            last = self.core.model[-1]
            if hasattr(last, "export"):
                last.export = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # NHWC -> NCHW (Ultralytics expects NCHW)
        x = x.permute(0, 3, 1, 2).contiguous()

        y = self.core(x)

        # Expected: (end2end, dict(one2many=..., one2one=...))
        if not (
            isinstance(y, (tuple, list)) and len(y) >= 2 and isinstance(y[1], dict)
        ):
            raise RuntimeError(f"Unexpected core output structure: type={type(y)}")

        d = y[1]
        if self.branch_key not in d:
            raise RuntimeError(
                f"Expected '{self.branch_key}' key. Keys={list(d.keys())}"
            )

        b = d[self.branch_key]
        if not (isinstance(b, dict) and "boxes" in b and "scores" in b):
            raise RuntimeError(
                f"{self.branch_key} missing boxes/scores. Keys={list(b.keys())}"
            )

        boxes_dfl = b["boxes"]  # [1,64,8400]
        cls_logits = b["scores"]  # [1,C,8400]

        # Shape guards (fail fast)
        if (
            boxes_dfl.ndim != 3
            or int(boxes_dfl.shape[1]) != 64
            or int(boxes_dfl.shape[2]) != 8400
        ):
            raise RuntimeError(
                "Unexpected boxes shape: "
                f"{tuple(boxes_dfl.shape)} (expected [1,64,8400])"
            )
        if (
            cls_logits.ndim != 3
            or int(cls_logits.shape[0]) != int(x.shape[0])
            or int(cls_logits.shape[1]) <= 0
            or int(cls_logits.shape[2]) != 8400
        ):
            raise RuntimeError(
                "Unexpected scores shape: "
                f"{tuple(cls_logits.shape)} (expected [B,C,8400])"
            )

        return boxes_dfl, cls_logits


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


# ----------------------------
# Pipeline
# ----------------------------
class YoloV10Pipeline:
    def quantize_convert(
        self,
        model_path: str,
        out_tflite: str,
        calib_dir: str,
        max_calib: int = 200,
        qc_head: str = "one2many",
        qc_quant_scheme: str = "mse",
    ):
        """
        YOLOv10 .pt -> (trace wrapper) -> AI Hub compile ONNX -> quant INT8
        -> compile TFLite -> download

        Notes:
          - wrapper exports RAW branch heads (one2many or one2one)
          - keeps --quantize_io
        """
        if qc_head not in ("one2many", "one2one"):
            raise ValueError(f"Unsupported qc_head for yolov10: {qc_head}")
        if qc_quant_scheme not in ("mse", "minmax"):
            raise ValueError(f"Unsupported qc_quant_scheme: {qc_quant_scheme}")
        _require_qc_deps()
        assert torch is not None and YOLO is not None and hub is not None

        # Defaults (later can be promoted to CLI args)
        input_hw = 640
        input_shape = (1, input_hw, input_hw, 3)  # NHWC
        device_name = "Dragonwing IQ-9075 EVK"
        images_dir = calib_dir

        out_tflite = str(out_tflite)
        Path(out_tflite).parent.mkdir(parents=True, exist_ok=True)

        # 1) Load Ultralytics model
        y = YOLO(model_path)
        core = y.model.eval()

        # 2) Wrap to export selected raw branch tensors
        torch_model = Yolo10RawBranchWrapper(core, branch_key=qc_head).eval()

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
        print(f"[yolov10] wrote tflite: {out_tflite}")
