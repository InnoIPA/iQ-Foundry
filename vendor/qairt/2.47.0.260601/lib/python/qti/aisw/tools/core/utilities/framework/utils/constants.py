# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from enum import IntEnum
from typing import Dict, List, TypeAlias, Union

# ruff: noqa: F821

# Model types
OnnxFrameworkModel: TypeAlias = "ModelProto"
TorchFrameworkModel: TypeAlias = "torch.nn.Module"
TFLiteFrameworkModel: TypeAlias = "tf.lite.Interpreter"

FrameworkModels: TypeAlias = Union[OnnxFrameworkModel, TorchFrameworkModel, TFLiteFrameworkModel]

# Execute input data types
ExecuteInputData: TypeAlias = Union["List[np.ndarray]", "Dict[str, np.ndarray]"]

# Execute return types
OnnxExecuteReturn: TypeAlias = Union["SparseTensorProto", List, Dict]
PytorchExecuteReturn: TypeAlias = "torch.Tensor"
TFLiteExecuteReturn: TypeAlias = "tf.Tensor"
FrameworkExecuteReturn: TypeAlias = Union["np.ndarray", OnnxExecuteReturn, PytorchExecuteReturn, TFLiteExecuteReturn]


class MaxLimits(IntEnum):
    """Maximum limits enum."""

    max_file_name_size = 255
    max_model_size_with_intermediates = 3840  # Model has to fit in one DSP of 3.75 GB = 3840 MB


class ModelFrameworkInfo:
    """Model framework info class."""

    name: str
    extensions: List[str]

    @classmethod
    def check_framework(cls, extension: str) -> bool:
        """Validates framework extension."""
        return True if extension in cls.extensions else False


class OnnxFrameworkInfo(ModelFrameworkInfo):
    """Onnx framework info class."""

    name = "onnx"
    extensions = [".onnx"]


class TFLiteFrameworkInfo(ModelFrameworkInfo):
    """TFLite framework info class."""

    name = "tflite"
    extensions = [".tflite"]


class TensorflowFrameworkInfo(ModelFrameworkInfo):
    """Tensorflow framework info class."""

    name = "tensorflow"
    extensions = [".pb"]


class PytorchFrameworkInfo(ModelFrameworkInfo):
    """Pytorch framework info class."""

    name = "pytorch"
    extensions = [".pt"]


SUPPORTED_EXTENSIONS = [
    *OnnxFrameworkInfo.extensions,
    *PytorchFrameworkInfo.extensions,
    *TensorflowFrameworkInfo.extensions,
    *TFLiteFrameworkInfo.extensions,
]
