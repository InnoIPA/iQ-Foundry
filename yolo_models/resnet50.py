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
"""ResNet50 image classification: torchvision state_dict -> litert / onnx / qairt.

The exported graph takes NHWC float32 [1, 224, 224, 3] in [0, 1] and returns the raw class
scores `logits` [1, num_classes]. The ImageNet mean / std are inside the graph, so the board
feeds plain 0..255 pixels, the same as for the YOLO models.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

try:
    import torch
except Exception:
    torch = None

try:
    import torchvision
except Exception:
    torchvision = None

try:
    import qai_hub as hub
except Exception:
    hub = None

from yolo_models.common import (
    QAIRT_CALIBRATION_METHODS,
    build_qairt_context_binary,
    export_traced_model_to_onnx,
    finalize_downloaded_onnx_artifact,
    load_calibration_images,
    run_qairt_tool,
    write_qairt_calibration_inputs,
)
from tool.postprocess.classify import preprocess_classify
from tool.postprocess.specs import TASK_SPECS

INPUT_HW = 224
INPUT_SHAPE = (1, INPUT_HW, INPUT_HW, 3)  # NHWC
# The torchvision ImageNet normalization; eval / test must resize + crop the same way
# (short side 256, center crop 224, see tool/postprocess/classify.py).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICE_NAME = "Dragonwing IQ-9075 EVK"

_TORCH_BASE = torch.nn.Module if torch is not None else object


class ResNet50LogitsWrapper(_TORCH_BASE):
    """NHWC [0, 1] input -> ImageNet normalization -> ResNet50 -> logits [1, num_classes]."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.core((x - self.mean) / self.std)


def load_resnet50(model_path: str) -> torch.nn.Module:
    """Build torchvision's ResNet50 from a state_dict (.pt / .pth); classes come from fc."""
    if torch is None or torchvision is None:
        raise RuntimeError("resnet50 qc needs torch and torchvision")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "fc.weight" not in state:
        raise ValueError(
            f"{model_path} is not a torchvision ResNet50 state_dict (no 'fc.weight' entry)"
        )
    model = torchvision.models.resnet50(weights=None, num_classes=state["fc.weight"].shape[0])
    model.load_state_dict(state)
    return model.eval()


class ResNet50Pipeline:
    def convert(
        self,
        model_path: str,
        output_path: str,
        runtime: str,
        precision: str,
        calib_dir: str | None = None,
        max_calib: int = 200,
        qc_head: str | None = None,
        qc_quant_scheme: str = "mse",
        task: str = "classify",
    ) -> None:
        if runtime == "qairt":
            self.export_qairt(
                model_path, output_path, precision, calib_dir, max_calib, qc_quant_scheme
            )
        else:
            self.export_with_ai_hub(
                model_path, output_path, runtime, precision, calib_dir, max_calib,
                qc_quant_scheme,
            )

    @staticmethod
    def _trace(model_path: str):
        wrapper = ResNet50LogitsWrapper(load_resnet50(model_path)).eval()
        example = torch.rand(INPUT_SHAPE, dtype=torch.float32)
        return torch.jit.trace(wrapper, example, strict=False, check_trace=False)

    def export_with_ai_hub(
        self,
        model_path: str,
        output_path: str,
        runtime: str,
        precision: str,
        calib_dir: str | None,
        max_calib: int,
        qc_quant_scheme: str,
    ) -> None:
        """litert / onnx: TorchScript -> AI Hub ONNX -> (quantize) -> ONNX or TFLite."""
        if hub is None:
            raise RuntimeError("resnet50 qc for litert / onnx needs qai_hub")
        pt_model = self._trace(model_path)
        device = hub.Device(DEVICE_NAME)
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        compile_onnx_job = hub.submit_compile_job(
            model=pt_model,
            device=device,
            input_specs={"image": INPUT_SHAPE},
            options="--target_runtime onnx",
        )
        model = compile_onnx_job.get_target_model()
        compile_options = f"--target_runtime {'tflite' if runtime == 'litert' else 'onnx'}"

        if precision != "fp32":
            # litert int8 is W8A8, onnx w8a16 is W8A16.
            activations = hub.QuantizeDtype.INT8 if precision == "int8" else hub.QuantizeDtype.INT16
            quantize_kwargs = {
                "model": model,
                "calibration_data": {
                    "image": load_calibration_images(
                        calib_dir, INPUT_HW, max_images=max_calib, preprocess=preprocess_classify
                    )
                },
                "weights_dtype": hub.QuantizeDtype.INT8,
                "activations_dtype": activations,
            }
            if qc_quant_scheme == "minmax":
                quantize_kwargs["options"] = "--range_scheme min_max"
            model = hub.submit_quantize_job(**quantize_kwargs).get_target_model()
            compile_options += " --quantize_io"

        # Everything but onnx fp32 above went through the intermediate ONNX; compile it to the
        # target. (onnx fp32 needs no further step: its first ONNX is the artifact.)
        if runtime == "onnx" and precision == "fp32":
            downloaded = compile_onnx_job.download_target_model(output_path)
        else:
            downloaded = hub.submit_compile_job(
                model=model, device=device, options=compile_options
            ).download_target_model(output_path)
        if runtime == "onnx":
            output_path = finalize_downloaded_onnx_artifact(downloaded, output_path)
        print(f"[resnet50] wrote {runtime}: {output_path}")

    def export_qairt(
        self,
        model_path: str,
        output_path: str,
        precision: str,
        calib_dir: str | None,
        max_calib: int,
        qc_quant_scheme: str,
    ) -> None:
        """Build an HTP context binary: ONNX -> DLC -> (quantize) -> .bin, all offline."""
        pt_model = self._trace(model_path)
        with tempfile.TemporaryDirectory(prefix="qairt_resnet50_") as work_dir:
            onnx_path = os.path.join(work_dir, "model.onnx")
            export_traced_model_to_onnx(
                pt_model, INPUT_SHAPE, onnx_path, TASK_SPECS["classify"].outputs
            )

            dlc_path = os.path.join(work_dir, "model.dlc")
            convert_args = [
                "qairt-converter",
                "--input_network", onnx_path,
                "--source_model_input_shape", "image", f"1,{INPUT_HW},{INPUT_HW},3",
                "--source_model_input_layout", "image", "NHWC",
                *[
                    arg
                    for name in TASK_SPECS["classify"].outputs
                    for arg in ("--out_tensor_name", name)
                ],
                "--target_backend", "HTP",
                "--output_path", dlc_path,
            ]
            if precision == "fp16":
                # HTP has no FP32 math; a float graph is built directly at FP16.
                convert_args += ["--float_bitwidth", "16"]
            run_qairt_tool(convert_args, "converter")

            graph_dlc = dlc_path
            if precision != "fp16":
                list_path = write_qairt_calibration_inputs(
                    calib_dir, INPUT_HW, max_calib, work_dir, preprocess=preprocess_classify
                )
                graph_dlc = os.path.join(work_dir, "model_quantized.dlc")
                run_qairt_tool(
                    [
                        "qairt-quantizer",
                        "--input_dlc", dlc_path,
                        "--input_list", list_path,
                        "--output_dlc", graph_dlc,
                        "--act_bitwidth", "16" if precision == "w8a16" else "8",
                        "--weights_bitwidth", "8",
                        "--bias_bitwidth", "32",
                        "--use_per_channel_quantization",
                        "--act_quantizer_calibration",
                        QAIRT_CALIBRATION_METHODS.get(qc_quant_scheme, qc_quant_scheme),
                        "--act_quantizer_schema", "asymmetric",
                        "--param_quantizer_calibration", "min-max",
                        "--param_quantizer_schema", "symmetric",
                        "--target_backend", "HTP",
                    ],
                    "quantizer",
                )

            build_qairt_context_binary(graph_dlc, output_path, work_dir)
        print(f"[resnet50] wrote qairt: {output_path}")
