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
import tempfile

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

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
    require_qc_deps,
    run_qairt_tool,
    write_qairt_calibration_inputs,
)


YOLOV26_TEST_DEFAULTS = {
    "default_flow": "o2m",
    "conf_thres": 0.25,
    "iou_thres": 0.6,
    "topk": 300,
    "max_det": 100,
}


def _require_qc_deps(need_hub: bool = True) -> None:
    require_qc_deps("yolov26", need_hub=need_hub)


_TORCH_BASE = torch.nn.Module if torch is not None else object


# ----------------------------
# RAW export wrapper (YOLO26 branch-selectable -> boxes[1,4,8400], cls[1,C,8400])
# ----------------------------
class Yolo26RawBranch8400Wrapper(_TORCH_BASE):
    """
    Input:
      NHWC float32 [1,H,W,3] in [0,1]
    Output:
      boxes [1,4,8400]
      cls   [1,C,8400]

    Notes:
      Ultralytics YOLO26 core returns `(tensor, dict)` where the dict has
      `"one2many"` and `"one2one"` keys. We recursively collect tensors
      inside the selected branch and pick the shapes we need.
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

    @staticmethod
    def _collect_tensors(obj, out_list: list[torch.Tensor]) -> None:
        if torch.is_tensor(obj):
            out_list.append(obj)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                Yolo26RawBranch8400Wrapper._collect_tensors(v, out_list)
        elif isinstance(obj, dict):
            for v in obj.values():
                Yolo26RawBranch8400Wrapper._collect_tensors(v, out_list)

    @staticmethod
    def _find_named_tensor(obj, candidate_keys: tuple[str, ...]) -> torch.Tensor | None:
        if isinstance(obj, dict):
            for key in candidate_keys:
                value = obj.get(key)
                if torch.is_tensor(value):
                    return value
            for value in obj.values():
                found = Yolo26RawBranch8400Wrapper._find_named_tensor(
                    value, candidate_keys
                )
                if found is not None:
                    return found
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                found = Yolo26RawBranch8400Wrapper._find_named_tensor(
                    value, candidate_keys
                )
                if found is not None:
                    return found
        return None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # NHWC -> NCHW
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        y = self.core(x_nchw)

        if not (
            isinstance(y, (tuple, list)) and len(y) >= 2 and isinstance(y[1], dict)
        ):
            raise RuntimeError(f"Unexpected core output structure: type={type(y)}")

        d = y[1]
        if self.branch_key not in d:
            raise RuntimeError(
                f"Expected '{self.branch_key}' key. Keys={list(d.keys())}"
            )

        branch = d[self.branch_key]

        boxes = self._find_named_tensor(branch, ("boxes", "box"))
        cls = self._find_named_tensor(branch, ("scores", "cls", "classes", "logits"))

        ts: list[torch.Tensor] = []
        self._collect_tensors(branch, ts)

        if boxes is None:
            box_candidates = [
                t
                for t in ts
                if t.ndim == 3
                and int(t.shape[0]) == int(x.shape[0])
                and int(t.shape[1]) == 4
                and int(t.shape[2]) == 8400
            ]
            if len(box_candidates) == 1:
                boxes = box_candidates[0]

        if cls is None:
            cls_candidates = [
                t
                for t in ts
                if t.ndim == 3
                and int(t.shape[0]) == int(x.shape[0])
                and int(t.shape[1]) > 0
                and int(t.shape[2]) == 8400
                and (boxes is None or t is not boxes)
            ]
            if len(cls_candidates) == 1:
                cls = cls_candidates[0]

        if (
            boxes is None
            or cls is None
            or boxes.ndim != 3
            or int(boxes.shape[0]) != int(x.shape[0])
            or int(boxes.shape[1]) != 4
            or int(boxes.shape[2]) != 8400
            or cls.ndim != 3
            or int(cls.shape[0]) != int(x.shape[0])
            or int(cls.shape[1]) <= 0
            or int(cls.shape[2]) != 8400
        ):
            shapes = [tuple(t.shape) for t in ts]
            raise RuntimeError(
                f"{self.branch_key} branch: could not find boxes [1,4,8400] "
                f"and cls [B,C,8400]. Shapes={shapes}"
            )

        return boxes, cls


# ----------------------------
# Pipeline
# ----------------------------
class YoloV26Pipeline:
    def convert(
        self,
        model_path: str,
        output_path: str,
        runtime: str,
        precision: str,
        calib_dir: str,
        max_calib: int = 200,
        qc_head: str = "one2many",
        qc_quant_scheme: str = "mse",
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
                qc_head=qc_head,
            )
            return
        if runtime == "onnx" and precision == "fp32":
            self.export_onnx_fp32(
                model_path=model_path,
                output_path=output_path,
                qc_head=qc_head,
            )
            return
        if runtime == "onnx" and precision == "w8a16":
            self.export_onnx_w8a16(
                model_path=model_path,
                output_path=output_path,
                calib_dir=calib_dir,
                max_calib=max_calib,
                qc_head=qc_head,
                qc_quant_scheme=qc_quant_scheme,
            )
            return
        if runtime == "qairt":
            self.export_qairt(
                model_path=model_path,
                output_path=output_path,
                precision=precision,
                calib_dir=calib_dir,
                max_calib=max_calib,
                qc_head=qc_head,
                qc_quant_scheme=qc_quant_scheme,
            )
            return
        raise ValueError(f"Unsupported runtime/precision: {runtime}/{precision}")

    def _build_traced_model(self, model_path: str, qc_head: str):
        _require_qc_deps(need_hub=False)
        assert torch is not None and YOLO is not None

        input_hw = 640
        input_shape = (1, input_hw, input_hw, 3)
        y = YOLO(model_path)
        core = y.model.eval()
        torch_model = Yolo26RawBranch8400Wrapper(core, branch_key=qc_head).eval()
        example = torch.rand(input_shape, dtype=torch.float32)
        pt_model = torch.jit.trace(
            torch_model, example, strict=False, check_trace=False
        )
        return pt_model, input_hw, input_shape

    def export_qairt(
        self,
        model_path: str,
        output_path: str,
        precision: str,
        calib_dir: str | None = None,
        max_calib: int = 200,
        qc_head: str = "one2many",
        qc_quant_scheme: str = "mse",
    ) -> None:
        """Build an HTP context binary: ONNX -> DLC -> (quantize) -> .bin."""
        _require_qc_deps(need_hub=False)
        if precision not in ("int8", "w8a16", "fp16"):
            raise ValueError(f"Unsupported QAIRT precision: {precision}")

        pt_model, input_hw, input_shape = self._build_traced_model(model_path, qc_head)

        with tempfile.TemporaryDirectory(prefix="qairt_yolov26_") as work_dir:
            onnx_path = os.path.join(work_dir, "model.onnx")
            export_traced_model_to_onnx(pt_model, input_shape, onnx_path)

            dlc_path = os.path.join(work_dir, "model.dlc")
            convert_args = [
                "qairt-converter",
                "--input_network", onnx_path,
                "--source_model_input_shape", "image", f"1,{input_hw},{input_hw},3",
                "--source_model_input_layout", "image", "NHWC",
                "--out_tensor_name", "boxes",
                "--out_tensor_name", "scores",
                "--target_backend", "HTP",
                "--output_path", dlc_path,
            ]
            if precision == "fp16":
                # HTP has no FP32 math; a float graph is built directly at FP16.
                convert_args += ["--float_bitwidth", "16"]
            run_qairt_tool(convert_args, "converter")

            graph_dlc = dlc_path
            if precision != "fp16":
                if not calib_dir:
                    raise ValueError(
                        f"--calib_dir is required for qairt/{precision}"
                    )
                list_path = write_qairt_calibration_inputs(
                    calib_dir, input_hw, max_calib, work_dir
                )
                graph_dlc = os.path.join(work_dir, "model_quantized.dlc")
                method = QAIRT_CALIBRATION_METHODS.get(
                    qc_quant_scheme, qc_quant_scheme
                )
                quant_args = [
                    "qairt-quantizer",
                    "--input_dlc", dlc_path,
                    "--input_list", list_path,
                    "--output_dlc", graph_dlc,
                    "--act_bitwidth", "16" if precision == "w8a16" else "8",
                    "--weights_bitwidth", "8",
                    "--bias_bitwidth", "32",
                    "--use_per_channel_quantization",
                    "--act_quantizer_calibration", method,
                    "--act_quantizer_schema", "asymmetric",
                    "--param_quantizer_calibration", "min-max",
                    "--param_quantizer_schema", "symmetric",
                    "--target_backend", "HTP",
                ]
                if precision == "w8a16":
                    # The attention MatMuls take two dynamic activations, so at
                    # 16-bit they land on HTP's A16W16 path, which requires a
                    # symmetric B input and otherwise fails to finalize.
                    quant_args.append("--disable_dynamic_16_bit_weights")
                run_qairt_tool(quant_args, "quantizer")

            build_qairt_context_binary(graph_dlc, output_path, work_dir)

    def export_onnx_fp32(
        self,
        model_path: str,
        output_path: str,
        qc_head: str = "one2many",
    ) -> None:
        if qc_head not in ("one2many", "one2one"):
            raise ValueError(f"Unsupported qc_head for yolov26: {qc_head}")
        _require_qc_deps()
        assert hub is not None

        pt_model, _, input_shape = self._build_traced_model(model_path, qc_head)
        device = hub.Device("Dragonwing IQ-9075 EVK")
        compile_job = hub.submit_compile_job(
            model=pt_model,
            device=device,
            input_specs={"image": input_shape},
            options="--target_runtime onnx",
        )
        downloaded_path = compile_job.download_target_model(str(output_path))
        final_path = finalize_downloaded_onnx_artifact(downloaded_path, output_path)
        print(f"[yolov26] wrote onnx: {final_path}")

    def export_tflite_fp32(
        self,
        model_path: str,
        output_path: str,
        qc_head: str = "one2many",
    ) -> None:
        if qc_head not in ("one2many", "one2one"):
            raise ValueError(f"Unsupported qc_head for yolov26: {qc_head}")
        _require_qc_deps()
        assert hub is not None

        pt_model, _, input_shape = self._build_traced_model(model_path, qc_head)
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
        print(f"[yolov26] wrote tflite: {output_path}")

    def export_onnx_w8a16(
        self,
        model_path: str,
        output_path: str,
        calib_dir: str,
        max_calib: int = 200,
        qc_head: str = "one2many",
        qc_quant_scheme: str = "mse",
    ) -> None:
        if qc_head not in ("one2many", "one2one"):
            raise ValueError(f"Unsupported qc_head for yolov26: {qc_head}")
        if qc_quant_scheme not in ("mse", "minmax"):
            raise ValueError(f"Unsupported qc_quant_scheme: {qc_quant_scheme}")
        _require_qc_deps()
        assert hub is not None

        pt_model, input_hw, input_shape = self._build_traced_model(model_path, qc_head)
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
        final_path = finalize_downloaded_onnx_artifact(downloaded_path, output_path)
        print(f"[yolov26] wrote onnx: {final_path}")

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
        YOLOv26 .pt -> (trace wrapper) -> AI Hub compile ONNX -> quant INT8
        -> compile TFLite -> download

        Notes:
          - wrapper exports RAW selected branch tensors: boxes[1,4,8400], cls[1,C,8400]
          - keeps --quantize_io
        """
        if qc_head not in ("one2many", "one2one"):
            raise ValueError(f"Unsupported qc_head for yolov26: {qc_head}")
        if qc_quant_scheme not in ("mse", "minmax"):
            raise ValueError(f"Unsupported qc_quant_scheme: {qc_quant_scheme}")
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

        # 2) Wrap to export selected raw branch tensors
        torch_model = Yolo26RawBranch8400Wrapper(core, branch_key=qc_head).eval()

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
        print(f"[yolov26] wrote tflite: {out_tflite}")
