# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import logging
import shutil
import weakref
import os

from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from qti.aisw.core.model_level_api.executor.device_subprocess_executor import (
    DeviceSubprocessExecutor,
    DeviceTempDirectory,
)
from qti.aisw.core.model_level_api.executor.executor import Executor
from qti.aisw.core.model_level_api.utils.exceptions import (
    ContextBinaryGenerationError,
    InferenceError,
    return_code_to_netrun_error_enum,
)
from qti.aisw.core.model_level_api.utils.qnn_profiling import (
    ProfilingData,
    get_backend_profiling_data,
)
from qti.aisw.core.model_level_api.utils.subprocess_executor import (
    create_op_package_argument,
    create_set_output_tensors_argument,
    generate_config_file,
    generate_input_list,
    output_dir_to_np_array,
    update_run_config_native_inputs,
)
from qti.aisw.core.model_level_api.workflow.workflow import WorkflowMode

if TYPE_CHECKING:
    from qti.aisw.tools.core.modules.context_bin_gen import GenerateConfig
    from qti.aisw.tools.core.modules.net_runner import InferenceConfig
from qti.aisw.tools.core.modules.api.backend.backend import Backend
from qti.aisw.tools.core.modules.api.definitions import ModelConfig

NamedTensorData = Dict[str, np.ndarray]
InputTensorData = Union[NamedTensorData, Path, List[NamedTensorData], List[np.ndarray]]

logger = logging.getLogger(__name__)


class QNXSubprocessExecutor(DeviceSubprocessExecutor):
    def __init__(self):
        super().__init__()
        self._artifact_directory = None
        self._run_config = None
        self._setup = False
        self._architecture = 'aarch64-qnx'
    def run_inference(
        self,
        config: Optional["InferenceConfig"],
        backend: Backend,
        model: ModelConfig,
        sdk_root: str,
        input_data: InputTensorData,
        output_dir: str,
        graph_name: Optional[str] = "",
    ) -> Tuple[List[np.ndarray], ProfilingData | None]:
        """
        Runs the inference workflow, including setting up, processing, and retrieving results.

        Args:
            config: Configuration for running inference.
            backend: The backend to run inference on.
            model: The model to be used in the inference.
            sdk_root: SDK root path.
            input_data: Input data for inference.
            output_dir: Directory to store the output.
            graph_name: Name of graph for inference.

        Returns:
            The inference output NumPy array and profiling data.
        """

        if not self._setup:
            self.setup(
                WorkflowMode.INFERENCE, backend, model, sdk_root, config, output_dir
            )

        if config is None:
            config = self._run_config

        if config is not None and getattr(config, "use_mmap", None) is not None:
            raise ValueError("use_mmap option is not supported for QNX Target")

        temp_directory, qnx_temp_directory = self.prepare_temp_directories(backend)

        with qnx_temp_directory as qnx_temp_directory:
            backend.before_run_hook(qnx_temp_directory, sdk_root)
            input_list_filename, input_files, config_file_arg, config_file_artifacts, op_package_args = \
                self.generate_input_output_files(backend, sdk_root, temp_directory,
                                                 qnx_temp_directory, input_data=input_data,
                                                 is_execution=True)

            device_output_dir = PurePosixPath(qnx_temp_directory, "output")

            self.push_artifacts_to_device(backend, qnx_temp_directory, input_list_filename,
                                          input_files, config_file_artifacts, op_package_args,
                                          is_execution=True)

            # Create set_output_tensors argument if specified
            set_output_tensors_arg = create_set_output_tensors_argument(
                config, backend, temp_directory.name, qnx_temp_directory
            )

            command_env = {
                'LD_LIBRARY_PATH': f'$(pwd):{self._artifact_directory.name}:/mnt/lib64/dll/',
                'CDSP_LIBRARY_PATH': f'"$(pwd);{self._artifact_directory.name};/mnt/etc/images;'
                                     f'/dsplib/image/dsp/cdsp0;/dsplib/image/dsp;'
                                     f'/dspfw/image/dsp/cdsp0;/dspfw/image/dsp"',
                'CDSP1_LIBRARY_PATH': f'"$(pwd);{self._artifact_directory.name};/mnt/etc/images;'
                                      f'/dsplib/image/dsp/cdsp1;/dsplib/image/dsp;'
                                      f'/dspfw/image/dsp/cdsp1;/dspfw/image/dsp"'
            }

            return_code, stdout, stderr = self.execute_inference(backend, model, config,
                                                                 config_file_arg, op_package_args,
                                                                 set_output_tensors_arg,
                                                                 qnx_temp_directory,
                                                                 device_output_dir, command_env,
                                                                 input_data, graph_name)

            if return_code != 0:
                self.handle_context_bin_or_net_run_error(return_code, stdout, stderr,
                                                         is_execution=True)

            backend.target.pull(str(device_output_dir), temp_directory.name)

            host_output_dir = Path(temp_directory.name)

            backend.after_run_hook(qnx_temp_directory, sdk_root)

            profiling_data = None
            if config and config.profiling_level:
                profiling_data = get_backend_profiling_data(
                    backend, output_dir, host_output_dir
                )

            native_outputs = config and config.use_native_output_data
            return output_dir_to_np_array(host_output_dir, native_outputs), profiling_data

    def generate_context_binary(self,
                                config: Optional['GenerateConfig'],
                                backend: Backend,
                                model: ModelConfig,
                                sdk_root: str,
                                output_path: str,
                                output_filename: str,
                                backend_specific_filename: str = '') -> Tuple[ModelConfig,
                                Optional[str], Optional[ProfilingData]]:
        """
        Main function to orchestrate context binary generation for the backend, model, and target device.

        Args:
            config: Configuration object for context binary generation.
            backend: The backend object which includes the target device and backend-specific configurations.
            model: The model for which the context binary is generated.
            sdk_root: SDK root directory path.
            output_path: Path where the final output binary will be stored.
            output_filename: Filename for the generated binary.
            backend_specific_filename: Filename for the generated backend-specific binary

        Returns:
            Tuple: A tuple containing:
                ModelConfig: An object storing the name and path of the generated binary
                str: Path of the generated backend-specific binary
                Profiling_data: An object containing the profiling log and any backend profiling artifacts

        Raises:
            NotImplementedError: QNX Devices don't support on device context bin generation
        """
        raise NotImplementedError(
            "QNX Devices do not support online prepare context bin generation")

    @staticmethod
    def _get_model_artifacts(model: ModelConfig, sdk_root: Union[str, os.PathLike]):
        try:
            model_path = Path(model.path)
            if not model_path.is_file():
                raise FileNotFoundError(f"Could not find model: {model_path}")
            return [model_path]
        except AttributeError:
            raise ValueError("Could not retrieve path to provided model")
