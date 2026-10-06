#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from abc import ABC, abstractmethod
from enum import Enum
from qti.aisw.core.model_level_api.utils.qnn_sdk import qnn_sdk_root


class WorkflowMode(Enum):
    INFERENCE = 1
    CONTEXT_BINARY_GENERATION = 2


class Workflow(ABC):
    @abstractmethod
    def __init__(self, backend, model, executor=None, sdk_path=None):
        self._backend = backend
        self._model = model
        self._executor = executor
        self._profiling_data = []
        if sdk_path is None:
            sdk_path = qnn_sdk_root()
        self._sdk_path = sdk_path

    def get_profiling_data(self):
        return self._profiling_data

    def clear_profiling_data(self):
        self._profiling_data = []
