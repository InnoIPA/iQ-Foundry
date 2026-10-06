# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import List
from enum import IntEnum


class ModelFrameworkInfo:
    name: str
    extensions: List[str]

    @classmethod
    def check_framework(cls, extension: str) -> bool:
        return True if extension in cls.extensions else False


class OnnxFrameworkInfo(ModelFrameworkInfo):
    name = "onnx"
    extensions = [".onnx"]


class TFLiteFrameworkInfo(ModelFrameworkInfo):
    name = "tflite"
    extensions = [".tflite"]


class TensorflowFrameworkInfo(ModelFrameworkInfo):
    name = "tensorflow"
    extensions = [".pb"]


class PytorchFrameworkInfo(ModelFrameworkInfo):
    name = "pytorch"
    extensions = [".pt"]


SUPPORTED_EXTENSIONS = [*OnnxFrameworkInfo.extensions, *PytorchFrameworkInfo.extensions,
                        *TensorflowFrameworkInfo.extensions, *TFLiteFrameworkInfo.extensions]


class LOGLEVEL(IntEnum):
    VERBOSE = 4
    DEBUG_3 = 3
    DEBUG_2 = 2
    DEBUG_1 = 1
    DEBUG = 0
    INFO = -1

