#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

import logging
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from uuid import uuid4
import yaml

if TYPE_CHECKING:
    from qti.aisw.tools.core.modules.context_bin_gen import GenerateConfig
    from qti.aisw.tools.core.modules.net_runner import InferenceConfig
import os

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
from qti.aisw.tools.core.modules.api.definitions import ModelConfig
from qti.aisw.tools.core.modules.api import generate_context_bin_cli_args
from qti.aisw.tools.core.modules.api.backend.backend import Backend

NamedTensorData = Dict[str, np.ndarray]

logger = logging.getLogger(__name__)

class AndroidSubprocessExecutor(DeviceSubprocessExecutor):
    def __init__(self):
        super().__init__()
        # a directory which will be used to store backend, model, and executables that are relevant
        # for the entire lifetime of the Executor. Must be created during setup() below since the
        # remote device has not been selected when the Executor is created.
        self._artifact_directory = None
        self._run_config = None
        self._setup = False

    def run_inference(
        self,
        config: Optional["InferenceConfig"],
        backend: Backend,
        model: ModelConfig,
        sdk_root: str,
        input_data: Union[
            NamedTensorData, Path, List[NamedTensorData], List[np.ndarray]
        ],
        output_dir: str,
        graph_name: Optional[str] = None,
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
            self.setup(WorkflowMode.INFERENCE, backend, model, sdk_root, config, output_dir)

        if config is None:
            config = self._run_config

        temp_directory, android_temp_directory = self.prepare_temp_directories(backend)

        with android_temp_directory as android_temp_directory:
            backend.before_run_hook(android_temp_directory, sdk_root)

            input_list_filename, input_files, config_file_arg, config_file_artifacts, op_package_args = \
                self.generate_input_output_files(backend, sdk_root, temp_directory,
                                                 android_temp_directory,
                                                 input_data=input_data, is_execution=True)

            device_output_dir = PurePosixPath(android_temp_directory, 'output')

            self.push_artifacts_to_device(backend, android_temp_directory, input_list_filename,
                                          input_files, config_file_artifacts, op_package_args,
                                          is_execution=True)

            # Create set_output_tensors argument if specified
            set_output_tensors_arg = create_set_output_tensors_argument(
                config, backend, temp_directory.name, android_temp_directory
            )

            adapter_names = []
            if config.binary_updates:
                binary_update_file = Path(self._run_config.binary_updates)
                binary_updates_temp_file = Path(temp_directory.name, 'binary_updates.yaml')

                # Load and validate YAML data
                with binary_update_file.open() as f:
                    yaml_data = yaml.safe_load(f)

                if not yaml_data or not yaml_data.get("graphs"):
                    raise ValueError(f"Invalid binary_updates file: {binary_update_file}. "
                                   "Missing or empty 'graphs' section.")

                # Process binary updates and collect artifacts
                adapter_binaries = []

                for graph in yaml_data["graphs"]:
                    graph_name = next(iter(graph))
                    update_list = graph[graph_name]

                    for i, update in enumerate(update_list):
                        if isinstance(update, str):
                            # Handle direct adapter path
                            adapter_path = Path(update)
                            adapter_binaries.append(adapter_path)
                            adapter_names.append(adapter_path.stem)
                            update_list[i] = str(android_temp_directory) + '/' + adapter_path.name
                        else:
                            # Handle grouped adapters
                            group_name = next(iter(update))
                            adapter_group = update[group_name]

                            for j, adapter in enumerate(adapter_group):
                                adapter_path = Path(adapter)
                                adapter_binaries.append(adapter_path)

                                # Only track weight adapter names for output processing
                                if "weight" in adapter_path.stem:
                                    adapter_names.append(adapter_path.stem)

                                adapter_group[j] = str(android_temp_directory) + '/' + adapter_path.name

                # Write updated YAML to temp file
                with open(binary_updates_temp_file, 'w') as f:
                    yaml.dump(yaml_data, f)

                # Update config to point to device path
                config.binary_updates = PurePosixPath(android_temp_directory, "binary_updates.yaml")

                # Push all artifacts to device in batch
                backend.target.push(str(binary_updates_temp_file), android_temp_directory)
                for adapter_binary in adapter_binaries:
                    backend.target.push(str(adapter_binary), android_temp_directory)

            command_env = {
                'LD_LIBRARY_PATH': f'$(pwd):{self._artifact_directory.name}',
                'ADSP_LIBRARY_PATH': f'$(pwd);{self._artifact_directory.name}'
            }

            return_code, stdout, stderr = self.execute_inference(backend, model, config,
                                                                 config_file_arg,
                                                                 op_package_args,
                                                                 set_output_tensors_arg,
                                                                 android_temp_directory,
                                                                 device_output_dir, command_env,
                                                                 input_data, graph_name)

            if return_code != 0:
                self.handle_context_bin_or_net_run_error(return_code, stdout, stderr,
                                                         is_execution=True)

            backend.target.pull(str(device_output_dir), temp_directory.name)

            host_output_dir = Path(temp_directory.name, 'output')
            if not Path(host_output_dir).is_dir():
                raise RuntimeError("Failed to pull outputs from device")

            backend.after_run_hook(android_temp_directory, sdk_root)

            profiling_data = None
            if config and config.profiling_level:
                profiling_data = get_backend_profiling_data(backend, output_dir, host_output_dir)

            native_outputs = config and config.use_native_output_data

            return output_dir_to_np_array(host_output_dir, native_outputs, adapter_names), profiling_data

    def generate_context_binary(self,
                                config: "GenerateConfig",
                                backend: Backend,
                                model: ModelConfig | List[ModelConfig],
                                sdk_root: str, output_path: str,
                                output_filename: str,
                                backend_specific_filename: str = '') -> Tuple[ModelConfig,
                                                                              Optional[str],
                                                                              Optional[ProfilingData]]:
        """
        Main function to orchestrate context binary generation for the backend, model, and target
        device.

        Args:
            config: Configuration object for context binary generation.
            backend: The backend object which includes the target device and backend-specific
                     configurations.
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
        """
        temp_directory = TemporaryDirectory()
        logger.debug(f"created temp dir: {temp_directory.name}")

        # create temp directory on device based on host temp directory name to make the directory
        # name on device unique
        temp_dir_name = Path(temp_directory.name).name
        with DeviceTempDirectory(temp_dir_name, backend.target,
                                 prefix=self._device_temp_dir_prefix) as android_temp_directory:
            backend.before_generate_hook(android_temp_directory, sdk_root)

            device_backend_lib = PurePosixPath(self._artifact_directory.name,
                                               backend.backend_library)

            device_context_bin_generator = PurePosixPath(self._artifact_directory.name,
                                                         'qnn-context-binary-generator')

            config_file_arg, config_file_artifacts = generate_config_file(backend,
                                                                          temp_directory.name,
                                                                          sdk_root,
                                                                          android_temp_directory)
            op_package_arg = create_op_package_argument(
                backend,
                android_temp_directory)

            model_arg = self._create_context_binary_generator_model_argument(model,
                                                                             self._artifact_directory.name)

            output_tensors_arg = create_set_output_tensors_argument(config, backend,
                                                                    temp_directory.name,
                                                                    android_temp_directory)

            if output_filename:
                binary_file = output_filename
            elif model.name:
                binary_file = model.name
            else:
                binary_file = 'context'

            self.push_artifacts_to_device(backend=backend,
                                          device_temp_directory=android_temp_directory,
                                          config_file_artifacts=config_file_artifacts,
                                          is_execution=False)

            context_bin_command = f'{device_context_bin_generator} ' \
                                  f'--backend {device_backend_lib} ' \
                                  f'{model_arg} --binary_file {binary_file} ' \
                                  f'--backend_binary {backend_specific_filename} ' \
                                  f'{config_file_arg} {op_package_arg} {output_tensors_arg} '
            if config:
                context_bin_command += generate_context_bin_cli_args(config)

            command_env = {'LD_LIBRARY_PATH': f'$(pwd):{self._artifact_directory.name}',
                           'ADSP_LIBRARY_PATH': f'$(pwd);{self._artifact_directory.name}'}
            logger.debug("running command " + context_bin_command)
            return_code, stdout, stderr = backend.target.run_command(context_bin_command,
                                                                     cwd=android_temp_directory,
                                                                     env=command_env)

            if return_code != 0:
                self.handle_context_bin_or_net_run_error(return_code, stdout, stderr,
                                                         is_execution=False)

            device_output_dir = PurePosixPath(android_temp_directory, 'output')

            backend.target.pull(str(device_output_dir), temp_directory.name)
            host_temp_dir = Path(temp_directory.name, 'output')
            if not Path(host_temp_dir).is_dir():
                raise RuntimeError("Failed to pull artifacts from device")

            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            temp_output_bin = host_temp_dir / (binary_file + '.bin')
            if not temp_output_bin.is_file():
                raise RuntimeError("Failed to pull context binary from device")

            output_filename = binary_file + '.bin'
            output_bin = (output_path / output_filename).resolve()
            shutil.copy(temp_output_bin, output_bin)

            if backend_specific_filename:
                temp_output_binary_bin = host_temp_dir / (backend_specific_filename + '.bin')
                if not temp_output_binary_bin.is_file():
                    raise RuntimeError("Failed to pull backend binary from device")

                output_binary_bin = (output_path / (backend_specific_filename + '.bin')).resolve()
                shutil.copy(temp_output_binary_bin, output_binary_bin)


            backend.after_generate_hook(android_temp_directory, sdk_root)

            profiling_data = None
            if config and config.profiling_level:
                profiling_data = get_backend_profiling_data(backend, output_path, host_temp_dir)

            context_binary = ModelConfig(path=str(output_bin))
            backend_binary_path = None
            if backend_specific_filename and output_binary_bin.is_file():
                backend_binary_path = ModelConfig(path=output_binary_bin)

            return context_binary, backend_binary_path, profiling_data

    @staticmethod
    def _get_model_artifacts(model: ModelConfig, sdk_root: Union[str, os.PathLike]):
        try:
            model_path = Path(model.path)
            if not model_path.is_file():
                raise FileNotFoundError(f"Could not find model: {model_path}")
            return [model_path]
        except AttributeError:
            raise ValueError("Could not retrieve path to provided model")
