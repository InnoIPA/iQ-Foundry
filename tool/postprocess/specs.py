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
"""What each model type's and each task's outputs look like, defined once.

This is what inference and eval need to read a model's raw outputs (box encoding, heads, the
output tensors of each task, post-processing defaults). What the command line accepts (types,
tasks, runtimes, precisions and their valid combinations) is iqf_registry.py, not here.

This module is pure data (standard library only) and lives in tool/postprocess/ on purpose: that
package is pushed to the target next to inference_tflite.py / onnx_inference.py, which need
these facts on the device as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Box tensor channel count for each box encoding the exports produce.
BOX_MODE_CHANNELS = {"dfl64": 64, "ltrb4": 4}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    # Default activation range scheme for quantized qc runs ("mse" or "minmax").
    default_quant_scheme: str
    # How the exported box tensor is encoded: "dfl64" (DFL, 64 channels) or "ltrb4" (4 channels).
    # None for models without boxes (resnet50).
    box_mode: str | None = None
    # Output heads the model can be exported / evaluated with. A single entry means the model has
    # no head choice (yolov11 only has "default"); empty means the concept does not apply.
    heads: tuple[str, ...] = ()
    default_head: str | None = None
    # Defaults for `test` mode post-processing.
    test_defaults: dict = field(default_factory=dict)

    @property
    def box_channels(self) -> int | None:
        return BOX_MODE_CHANNELS.get(self.box_mode)

    @property
    def single_head(self) -> bool:
        return len(self.heads) == 1


_TWO_HEADS = ("one2many", "one2one")


def _test_defaults(default_flow: str) -> dict:
    return {
        "default_flow": default_flow,
        "conf_thres": 0.25,
        "iou_thres": 0.6,
        "topk": 300,
        "max_det": 100,
    }


MODEL_SPECS = {
    "yolov10": ModelSpec(
        name="yolov10",
        box_mode="dfl64",
        heads=_TWO_HEADS,
        default_head="one2many",
        default_quant_scheme="mse",
        test_defaults=_test_defaults("o2m"),
    ),
    "yolov11": ModelSpec(
        name="yolov11",
        box_mode="dfl64",
        heads=("default",),
        default_head="default",
        default_quant_scheme="minmax",
        test_defaults=_test_defaults("default"),
    ),
    "yolov26": ModelSpec(
        name="yolov26",
        box_mode="ltrb4",
        heads=_TWO_HEADS,
        default_head="one2many",
        default_quant_scheme="mse",
        test_defaults=_test_defaults("o2m"),
    ),
    # Classifier: no boxes, no heads, none of the detection thresholds.
    "resnet50": ModelSpec(name="resnet50", default_quant_scheme="mse"),
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    # Raw model outputs in the order the exported graph returns them. These role names are the
    # keys of the output dict that travels through the code, and the <image>_<role>.npy file
    # names used between the target and the host.
    outputs: tuple[str, ...]


TASK_SPECS = {
    "detect": TaskSpec(name="detect", outputs=("boxes", "scores")),
    # kpts: raw keypoint head output [1, 3 * num_keypoints, 8400] (x, y, visibility per point,
    # not yet decoded against anchors / strides).
    "pose": TaskSpec(name="pose", outputs=("boxes", "scores", "kpts")),
    # mask_coef: per-anchor mask coefficients [1, 32, 8400]; proto: mask prototypes
    # [1, 32, 160, 160] (laid out NHWC by some converters; the decoder accepts both).
    "segment": TaskSpec(name="segment", outputs=("boxes", "scores", "mask_coef", "proto")),
    # logits: raw class scores [1, num_classes], before softmax.
    "classify": TaskSpec(name="classify", outputs=("logits",)),
}

DETECT_OUTPUTS = TASK_SPECS["detect"].outputs
