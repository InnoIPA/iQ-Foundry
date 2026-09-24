#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from qti.aisw.core.model_level_api.executor.android_subprocess_executor import (
    AndroidSubprocessExecutor,
)
from qti.aisw.core.model_level_api.utils.qnn_profiling import ProfilingData
from qti.aisw.tools.core.modules.api.backend.backend import Backend

if TYPE_CHECKING:
    from qti.aisw.tools.core.modules.context_bin_gen import GenerateConfig
    from qti.aisw.tools.core.modules.net_runner import InferenceConfig
from qti.aisw.tools.core.modules.api.definitions import ModelConfig, ModelType

NamedTensorData = Dict[str, np.ndarray]

logger = logging.getLogger(__name__)


class OELinuxSubprocessExecutor(AndroidSubprocessExecutor):
    """
    Subprocess Executor class  for OELinux devices, contains pubilc methods to setup an instance,
    run an inference and generate a context_binary
    """

    def __init__(self, device_temp_dir_prefix='/etc/'):
        super().__init__()
        self._device_temp_dir_prefix = device_temp_dir_prefix

    def setup(self, workflow_mode, backend, model, sdk_root, config, output_dir):
        """
        This method is used to load the model and the artifacts to device

        Args:
            workflow_mode: can be WorkflowMode.INFERENCE or WorkflowMode.CONTEXT_BINARY_GENERATOR
            backend: Backend class
            model: Class containing path to the model library or context binary
            sdk_root: Path to SDK
            config: Config file, given to inference

        Returns: None

        Raises:
            ValueError: An error occurred when DLC file path given
            FileNotFoundError: An error occured when a executable or backend lib not found
            RuntimeError: An error occured when wrong workflow_mode given
        """

        return super().setup(workflow_mode, backend, model, sdk_root, config, output_dir)

    @staticmethod
    def _get_model_artifacts(model: ModelConfig, sdk_root: Union[str, os.PathLike]):
        try:
            model_path = Path(model.path)
            if not model_path.is_file():
                raise FileNotFoundError(f"Could not find model: {model_path}")
            return [model_path]
        except AttributeError:
            raise ValueError("Could not retrieve path to provided model")
