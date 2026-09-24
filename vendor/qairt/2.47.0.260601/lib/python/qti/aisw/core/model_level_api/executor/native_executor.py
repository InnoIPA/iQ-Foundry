#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================
import os
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import yaml

import numpy as np
from qti.aisw.core.model_level_api.executor.executor import Executor
from qti.aisw.core.model_level_api.utils import py_net_run
from qti.aisw.core.model_level_api.utils.exceptions import (
    ContextBinaryGenerationError,
    InferenceError,
    NetRunErrorCode,
)
from qti.aisw.core.model_level_api.utils.native_executor import (
    create_adapter_weight_config_file_argument,
    create_async_execute_queue_depth_argument,
    create_backend_extension_argument,
    create_batch_multiplier_argument,
    create_cache_compatibility_argument,
    create_context_priority_argument,
    create_debug_argument,
    create_enable_graphs_argument,
    create_enable_intermediate_outputs_argument,
    create_input_output_tensor_mem_type_argument,
    create_keep_num_outputs_argument,
    create_log_level_argument,
    create_memory_limit_argument,
    create_num_inferences_argument,
    create_op_package_argument,
    create_output_datatype_argument,
    create_perf_profile_argument,
    create_persistent_binary_argument,
    create_platform_options_argument,
    create_profile_level_argument,
    create_profile_option_argument,
    create_set_output_tensors_argument,
    create_total_duration_argument,
    create_use_mmap_argument,
    input_list_to_in_memory_input,
    run_config_requires_reinit,
    temporaryDirectoryChange,
)
from qti.aisw.core.model_level_api.utils.yaml_utils import (
    extract_and_load_input_tensors,
)
from qti.aisw.core.model_level_api.utils.qnn_profiling import (
    ProfilingData,
    default_profiling_log_name,
    move_backend_profiling_artifacts,
)
from qti.aisw.core.model_level_api.workflow.workflow import WorkflowMode
from qti.aisw.tools.core.modules.api.backend.backend import Backend
from qti.aisw.tools.core.modules.api.backend.utility import HexagonEnvironmentManager
from qti.aisw.tools.core.modules.api.backend.htp_backend import HtpBackend
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.tools.core.utilities.devices.api.device_factory import DeviceFactory
if TYPE_CHECKING:
    from qti.aisw.tools.core.modules.context_bin_gen import GenerateConfig
    from qti.aisw.tools.core.modules.net_runner import InferenceConfig
from qti.aisw.tools.core.modules.api.definitions import ModelConfig, ModelType

NamedTensorData = Dict[str, np.ndarray]

logger = logging.getLogger(__name__)


def _check_pnr_error(err, msg, exception_type, netrun_error_enum):
    if err != py_net_run.StatusCode.SUCCESS:
        raise exception_type(netrun_error_enum, msg)


class NativeExecutor(Executor):
    _model_lib_key = "model_lib_key"
    _backend_lib_key = "backend_lib_key"
    _backend_key = "backend_key"
    _logger_key = "logger_key"
    _device_key = "device_key"
    _context_key = "context_key"

    def __init__(self):
        super().__init__()
        self._run_temp_directory = None
        self._pbm = None
        self._created_from_qnn_model = False
        self._created_from_binary = False
        self._previous_backend_config = None
        self._setup_inference_config = None
        self._run_index = 0
        self._oneshot_execute = False

    def setup(self, workflow_mode: WorkflowMode, backend: Backend, model: ModelConfig, sdk_root: str,
              config: Any, output_dir: str | os.PathLike) -> \
            Optional[ProfilingData]:

        if ( isinstance(backend, HtpBackend) and
            backend.target.target_platform_type == DevicePlatformType.WOS
        ):
            soc_dsp_arch = ""
            soc_model = backend.target.device.device_info.identifier.soc_model if backend.target.device.device_info and \
            backend.target.device.device_info.identifier.soc_model else None

            if soc_model:
                device_info = DeviceFactory.get_device_soc_details("HTP", soc_model)
                if device_info and device_info.dsp_arch:
                    soc_dsp_arch = "v" + str(device_info.dsp_arch)

            if soc_dsp_arch:
                if soc_dsp_arch != "v73":
                    raise RuntimeError(f"HTP architecture {soc_dsp_arch} is not supported in {DevicePlatformType.WOS}")
            else:
                logger.info("HTP architecture is not detected. Defaulting to v73.")
                soc_dsp_arch = "v73"

            HexagonEnvironmentManager.activate_hexagon_env(soc_dsp_arch)

        # setup is only required for inference, no-op for context binary generation
        if workflow_mode == WorkflowMode.CONTEXT_BINARY_GENERATION:
            return None

        output_dir = Path(output_dir).resolve()

        self._setup_inference_config = config

        self._run_temp_directory = TemporaryDirectory()
        logger.debug(f'created temp directory: {self._run_temp_directory.name}')

        backend.before_run_hook(self._run_temp_directory.name, sdk_root)

        log_level_arg = create_log_level_argument(config)
        op_package_arg = create_op_package_argument(backend)
        profile_level_arg = create_profile_level_argument(config)
        profile_option_arg = create_profile_option_argument(config)
        debug_arg = create_debug_argument(config)
        platform_options_arg = create_platform_options_argument(config)
        use_mmap_arg = create_use_mmap_argument(config)

        context_config = backend.get_context_config()
        context_priority = create_context_priority_argument(context_config)
        async_execute_queue_depth, configure_async_execute_queue_depth = (
            create_async_execute_queue_depth_argument(context_config)
        )
        enable_graphs = create_enable_graphs_argument(context_config)
        memory_limit_hint, memory_limit_hint_present = create_memory_limit_argument(
            context_config
        )
        is_persistent_binary = create_persistent_binary_argument(context_config)
        cache_compatibility_mode, cache_compatibility_mode_present = (
            create_cache_compatibility_argument(context_config)
        )

        # if the model is a context binary, its path must be provided during PythonBackendManager
        # construction
        if model.model_type == ModelType.QNN_CONTEXT_BINARY:
            binary_path_arg = model.path
            self._created_from_binary = True
        else:
            binary_path_arg = ''
        if not self._created_from_binary:
            if model.model_type == ModelType.DLC:
                dlc_path = Path(model.path).resolve()
        else:
            dlc_path = ''
        if not dlc_path:
            model_path = Path(model.path).resolve()
            if not model_path.is_file():
                    raise FileNotFoundError(f"Could not locate {str(model_path)}")


        with temporaryDirectoryChange(self._run_temp_directory.name):
            self._pbm = py_net_run.PythonBackendManager(logLevel=log_level_arg,
                                                        opPackagePaths=op_package_arg,
                                                        cachedBinaryPath=str(binary_path_arg),
                                                        profilingLevel=profile_level_arg,
                                                        debug=debug_arg,
                                                        profilingOption=profile_option_arg)
            if not self._created_from_binary and not dlc_path:
                err = self._pbm.loadModelLib(str(model_path),
                                             self._model_lib_key)
                _check_pnr_error(err,
                                 f'Failed to load model library: {str(model_path)}',
                                 InferenceError,
                                 NetRunErrorCode.INITIALIZE)
                self._created_from_qnn_model = True
            if backend.target.target_platform_type == DevicePlatformType.LINUX_EMBEDDED:
                backend_lib_path = str(Path(sdk_root, 'lib', backend.target.target_name,
                                        backend.backend_library))
            else:
                backend_lib_path = backend.backend_library
            err = self._pbm.loadBackendLib(self._backend_lib_key, backend_lib_path)
            _check_pnr_error(err,
                             f'Failed to load backend library: {backend.backend_library}',
                             InferenceError,
                             NetRunErrorCode.INITIALIZE)

            err = self._pbm.createLogHandle(self._backend_lib_key, self._logger_key, log_level_arg)
            _check_pnr_error(err,
                             'Failed to initialize logging in the backend',
                             InferenceError,
                             NetRunErrorCode.INITIALIZE)

            self._previous_backend_config = backend.get_config_json()

            [extension_lib_path, json_path] = \
                create_backend_extension_argument(backend,
                                                  self._run_temp_directory.name,
                                                  sdk_root)
            if extension_lib_path:
                perf_profile_arg = create_perf_profile_argument(config)
                err = self._pbm.initializeBackendExtension(self._backend_lib_key,
                                                           extension_lib_path,
                                                           json_path,
                                                           py_net_run.AppType.QNN_APP_NETRUN,
                                                           perf_profile_arg)
                _check_pnr_error(err,
                                 'Failed to initialize backend extensions',
                                 InferenceError,
                                 NetRunErrorCode.INITIALIZE)

            if self._oneshot_execute:
                profile_log_name = default_profiling_log_name
            else:
                profile_log_name = 'qnn-profiling-data-load.log'
            if profile_level_arg != py_net_run.ProfilingLevel.OFF:
                err = self._pbm.initializeProfileLogger(self._backend_lib_key,
                                                        str(output_dir),
                                                        profile_log_name)
                _check_pnr_error(err,
                                 'Failed to initialize the profile logger',
                                 InferenceError,
                                 NetRunErrorCode.INITIALIZE)

            err = self._pbm.createBackendHandle(self._backend_lib_key,
                                                self._logger_key,
                                                self._backend_key,
                                                platform_options_arg)
            _check_pnr_error(err,
                             'Failed to create a backend handle',
                             InferenceError,
                             NetRunErrorCode.CREATE_BACKEND)
            dlc_has_executable_cache = False
            if dlc_path:
                dlc_has_executable_cache, err = self._pbm.checkDlcHasExecutableCache(str(dlc_path),
                                                                                     self._backend_lib_key,
                                                                                     self._backend_key)

                _check_pnr_error(err,
                                 'Failed to check DLC has cache or not',
                                 InferenceError,
                                 NetRunErrorCode.INITIALIZE)

            err = self._pbm.createDeviceHandle(self._backend_lib_key,
                                               self._logger_key,
                                               self._device_key)
            _check_pnr_error(err,
                             'Failed to create a device handle',
                             InferenceError,
                             NetRunErrorCode.CREATE_DEVICE)

            err = self._pbm.registerOpPackage(self._backend_lib_key, self._backend_key)
            _check_pnr_error(err,
                             'Failed to register op packages',
                             InferenceError,
                             NetRunErrorCode.REGISTER_OPPACKAGE)

            if self._created_from_binary:
                err = self._pbm.createContextFromBinaryFile(self._backend_lib_key,
                                                            self._backend_key,
                                                            self._device_key,
                                                            self._context_key,
                                                            enable_graphs,
                                                            use_mmap_arg,
                                                            context_priority,
                                                            async_execute_queue_depth,
                                                            configure_async_execute_queue_depth,
                                                            cache_compatibility_mode_present,
                                                            cache_compatibility_mode,
                                                            memory_limit_hint_present,
                                                            memory_limit_hint,
                                                            is_persistent_binary)
                _check_pnr_error(err,
                                 f'Failed to create a context from the provided binary file: '
                                 f'{binary_path_arg}',
                                 InferenceError,
                                 NetRunErrorCode.CREATE_FROM_BINARY)

            else:
                # DLC cache init
                if dlc_path and dlc_has_executable_cache and not debug_arg:
                    err = self._pbm.createContextFromBinaryCache(self._backend_lib_key,
                                                                 self._backend_key,
                                                                 self._device_key,
                                                                 self._context_key,
                                                                 str(dlc_path))
                    _check_pnr_error(err,
                                     f'Failed to create a context from the selected cache in the DLC,'
                                     f'or no runnable cache detected',
                                     InferenceError,
                                     NetRunErrorCode.CREATE_FROM_CACHE)
                # irGraph init
                else:
                    err = self._pbm.createContext(self._backend_lib_key,
                                                  self._backend_key,
                                                  self._device_key,
                                                  self._context_key,
                                                  context_priority,
                                                  async_execute_queue_depth,
                                                  configure_async_execute_queue_depth)
                    _check_pnr_error(err,
                                     'Failed to create a context',
                                     InferenceError,
                                     NetRunErrorCode.CREATE_CONTEXT)

                    output_tensor_names_arg = create_set_output_tensors_argument(config)
                    if dlc_path:
                        err = self._pbm.composeGraphsFromDlc(self._backend_lib_key,
                                                             self._backend_key,
                                                             self._context_key,
                                                             str(dlc_path),
                                                             log_level_arg,
                                                             output_tensor_names_arg,
                                                             str(output_dir))
                    else:
                        err = self._pbm.composeGraphs(self._backend_lib_key,
                                                      self._backend_key,
                                                      self._context_key,
                                                      self._model_lib_key,
                                                      output_tensor_names_arg)
                    _check_pnr_error(err,
                                     'Failed to compose graphs',
                                     InferenceError,
                                     NetRunErrorCode.COMPOSE_GRAPHS)

                    err = self._pbm.finalizeGraphs(self._backend_lib_key,
                                                   self._backend_key,
                                                   self._context_key,
                                                   profilingLevel=profile_level_arg,
                                                   profilingOption=profile_option_arg)
                    _check_pnr_error(err,
                                     'Failed to finalize graphs',
                                     InferenceError,
                                     NetRunErrorCode.FINALIZE_GRAPHS)

            profiling_log = None
            if profile_level_arg != py_net_run.ProfilingLevel.OFF and not self._oneshot_execute:
                err = self._pbm.disposeProfileLogger(self._backend_lib_key)
                _check_pnr_error(err,
                                 'Failed to dispose profile logger',
                                 InferenceError,
                                 NetRunErrorCode.TERMINATE)

                profiling_log = Path(output_dir, profile_log_name).resolve()
                if not profiling_log.is_file():
                    raise RuntimeError(f"Could not locate profiling log at {profiling_log}")

            profiling_data = None
            if profiling_log is not None:
                profiling_data = ProfilingData(profiling_log=profiling_log,
                                               backend_profiling_artifacts=None)

        return profiling_data

    def get_graph_input_name_dtype_pairs(
        self, input_yaml_file: Union[Path, str]
    ) -> List[Dict[str, List[Tuple[str, str]]]]:
        """
        Get the input name and data type pairs for each graph in the parsed_yaml file.

        Args:
            input_yaml_file (Union[Path, str]): The path to the input yaml file.

        Returns:
            List[Dict[str, List[Tuple[str, str]]]]: A list of dictionaries,
            where each dictionary maps a graph name to a list of input name and data type pairs.

        Sample output:
        [
            {
                "model_configuration_1": [
                    ("input_ids", "QNN_DATATYPE_INT_64"),
                    ("cache_index", "QNN_DATATYPE_INT_64"),
                    ("attention_mask", "QNN_DATATYPE_BOOL_8"),
                    ("position_ids", "QNN_DATATYPE_INT_64")
                ]
            },
            {
                "model_configuration_2": [
                    ("cache_index", "QNN_DATATYPE_INT_64"),
                    ("input_ids", "QNN_DATATYPE_INT_64"),
                    ("position_ids", "QNN_DATATYPE_INT_64")
                ]
            }
        ]
        """

        with open(input_yaml_file, "r") as stream:
            parsed_yaml_data = yaml.safe_load(stream)

        graph_input_name_dtype_pairs_list = []
        for graph in parsed_yaml_data["graphs"]:
            graph_name = graph["name"]
            input_name_dtype_pairs = self._pbm.getGraphInputNameDtypePairs(graph_name)
            graph_input_name_dtype_pairs_list.append(
                {graph_name: input_name_dtype_pairs}
            )

        return graph_input_name_dtype_pairs_list


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
        graph_name: Optional[str] = "",
        optional_output_tensor_names: Optional[List[str]] = None,
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
            optional_output_tensor_names: optional tensor names that user wants to see in output
            including the mandatory ones.

        Returns:
            The inference output NumPy array and profiling data.
        """
        output_dir = Path(output_dir).resolve()

        # if self._pbm is None, setup() was not called. This means that we should initialize and
        # de-initialize in this function. We should also ensure that all profiling data is returned
        # at once in a single entity instead of separated into 3.
        if self._pbm is None:
            self._oneshot_execute = True
            logger.info("Executor was not setup, setting up now")
            self.setup(WorkflowMode.INFERENCE, backend, model, sdk_root, config, output_dir)

        if config is None:
            # if a config was not provided, we should use whatever was provided during setup
            config = self._setup_inference_config
        elif not self._oneshot_execute:
            # otherwise we should clear the setup config at this point, if a user calls:
            # 1. setup() with config
            # 2. run_inference() with config
            # 3. run_inference() without config
            # the 3rd function should be a run with an empty config rather than retrieving the
            # config given to setup(). Essentially, providing a config to run_inference() at any
            # point should disable retrieving the config given during setup() for any future calls
            # to run_inference()
            self._setup_inference_config = None

        reinit_required = False
        backend_config = backend.get_config_json()
        if backend_config:
            if self._previous_backend_config is None:
                # if a config was not provided to setup() but one was provided here, we must reinit
                reinit_required = True
            elif backend_config != self._previous_backend_config:
                # if the config differs to the one provided during setup(), we must reinit
                reinit_required = True
        if run_config_requires_reinit(config, self._setup_inference_config):
            reinit_required = True

        if reinit_required:
            self.teardown(backend, sdk_root, config, output_dir)
            self.setup(WorkflowMode.INFERENCE, backend, model, sdk_root, config, output_dir)


        # if input data is an input list (Path or str), resolve the path before entering the temp
        # directory in case a relative path was provided
        if isinstance(input_data, Path) or isinstance(input_data, str):
            input_data = Path(input_data).resolve()

        with temporaryDirectoryChange(self._run_temp_directory.name):
            # if input_data is an input list, read it into memory so it can be passed via pybind
            if isinstance(input_data, Path):
                native_inputs = None
                native_input_tensor_names = None
                graph_input_name_dtype_pairs = None
                if config:
                    native_inputs = config.use_native_input_data
                    native_input_tensor_names = config.native_input_tensor_names
                    if native_inputs or native_input_tensor_names:
                        try:
                            graph_input_name_dtype_pairs = (
                                self._pbm.getGraphInputNameDtypePairs(graph_name)
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to get Graph input dtype pairs with error {e}"
                            )

                if str(input_data).endswith((".yaml", ".yml")):
                    graph_input_name_dtype_pairs_list = (
                        self.get_graph_input_name_dtype_pairs(input_data)
                    )
                    input_data = extract_and_load_input_tensors(
                        input_data,
                        native_inputs,
                        native_input_tensor_names,
                        graph_input_name_dtype_pairs_list,
                    )
                    if isinstance(input_data, dict):
                        raise RuntimeError(
                            "For multi-graph execution, inference must be run from the macro-module"
                        )
                else:
                    input_data = input_list_to_in_memory_input(input_data,
                                                        native_inputs,
                                                        native_input_tensor_names,
                                                        graph_input_name_dtype_pairs)

            # the pybind layer supports 2 forms of input data:
            # - list[list[np.ndarray]], where the length of the inner list must be the # of inputs to
            #   the network, and the length of the outer list is the # of inferences to run
            # - list[dict[str, np.ndarray]], where the inner list provides name -> input mappings for
            #   all network inputs, and the outer list is the # of inferences to run

            # if input_data is a single np array, wrap it in in a 2d list as it can be assumed the user
            # is running one (possibly batched) inference of a single input network
            elif isinstance(input_data, np.ndarray):
                input_data = [[input_data]]

            # if input data is a list of numpy arrays, assume the user is running multiple inferences of
            # a single input network and wrap each np array in its own list
            elif isinstance(input_data, list) and isinstance(input_data[0], np.ndarray):
                input_data = [[input_arr] for input_arr in input_data]

            # if input data is a dict of name -> np.array mappings, the user is running a single
            # inference of a multi-input network so wrap the dict in a list
            elif isinstance(input_data, dict):
                input_data = [input_data]

            batch_multiplier_arg = create_batch_multiplier_argument(config)
            output_datatype_arg = create_output_datatype_argument(config)
            profile_level_arg = create_profile_level_argument(config)
            profile_option_arg = create_profile_option_argument(config)
            num_inferences_arg = create_num_inferences_argument(config)
            total_duration_arg = create_total_duration_argument(config)
            keep_num_outputs_arg = create_keep_num_outputs_argument(config)

            profile_log_name = f'qnn-profiling-data-run-{self._run_index}.log'
            if profile_level_arg != py_net_run.ProfilingLevel.OFF and not self._oneshot_execute:
                err = self._pbm.initializeProfileLogger(self._backend_lib_key,
                                                        str(output_dir),
                                                        profile_log_name)
                _check_pnr_error(err,
                                 'Failed to initialize the profile logger',
                                 InferenceError,
                                 NetRunErrorCode.INITIALIZE)

            synchronous_arg = config and config.synchronous
            try:
                output_data = self._pbm.executeGraphs(
                    self._backend_lib_key,
                    self._backend_key,
                    self._context_key,
                    graph_name,
                    input_data,
                    profilingLevel=profile_level_arg,
                    batchMultiplier=batch_multiplier_arg,
                    outputDataType=output_datatype_arg,
                    synchronous=synchronous_arg,
                    profilingOption=profile_option_arg,
                    numInferences=num_inferences_arg,
                    totalDuration=total_duration_arg,
                    keepNumOutputs=keep_num_outputs_arg,
                    optionalOutputTensorNames=optional_output_tensor_names,
                )
            except Exception as e:
                error_str = f'({NetRunErrorCode.EXECUTE_GRAPHS}) Exception occurred during graph ' \
                            f'execution: {e}'
                logger.error(error_str)
                raise InferenceError(NetRunErrorCode.EXECUTE_GRAPHS, error_str)

            if self._oneshot_execute:
                self.teardown(backend, sdk_root, config, output_dir)
                profile_log_name = default_profiling_log_name

            profiling_log = None
            if profile_level_arg != py_net_run.ProfilingLevel.OFF:
                # the profile logger is disposed during teardown() so it does not need to be
                # disposed again here in the one shot execute case
                if not self._oneshot_execute:
                    err = self._pbm.disposeProfileLogger(self._backend_lib_key)
                    _check_pnr_error(err,
                                     'Failed to dispose profile logger',
                                     InferenceError,
                                     NetRunErrorCode.TERMINATE)

                profiling_log = Path(output_dir, profile_log_name).resolve()
                if not profiling_log.is_file():
                    raise RuntimeError(f"Could not locate profiling log at {profiling_log}")

            profiling_data = None
            if profiling_log is not None:
                profiling_data = ProfilingData(profiling_log=profiling_log,
                                               backend_profiling_artifacts=None)

            self._run_index += 1
            return output_data, profiling_data

    def generate_context_binary(self,
                                config: Optional['GenerateConfig'],
                                backend: Backend,
                                model: Union[ModelConfig, List[ModelConfig]],
                                sdk_root: str,
                                output_path: str,
                                output_filename: str,
                                backend_specific_filename: str = '') -> Tuple[ModelConfig,
                                                               Optional[str],
                                                               Optional[ProfilingData]]:

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
        """
        output_path = Path(output_path).resolve()

        temp_directory = TemporaryDirectory()

        logger.debug(f'Created temp directory: {temp_directory.name}')

        backend.before_generate_hook(temp_directory.name, sdk_root)

        log_level_arg = create_log_level_argument(config)
        op_package_arg = create_op_package_argument(backend)
        profile_level_arg = create_profile_level_argument(config)
        profile_option_arg = create_profile_option_argument(config)
        enable_intermediate_outputs_arg = create_enable_intermediate_outputs_argument(config)
        platform_options_arg = create_platform_options_argument(config)
        adapter_weight_config_file_arg = create_adapter_weight_config_file_argument(config)

        # Resolution order here is:
        # 2 DLCs (weight sharing) or single DLC or qnn model library (.so)

        dlc_paths = []
        if isinstance(model, list):
            if not all(model_.model_type == ModelType.DLC for model_ in model):
                raise AttributeError("Only multiple DLCs are supported")

            dlc_paths = [str(Path(model_.path).resolve()) for model_ in model]
            logger.debug("Multiple DLCs passed to context binary generator.")

        elif isinstance(model, ModelConfig):
            if model.model_type == ModelType.DLC:
                dlc_paths = [str(model.path.resolve())]
            elif model.model_type == ModelType.QNN_MODEL_LIBRARY:
                model_path = Path(model.path).resolve()
                if not model_path.is_file():
                    raise FileNotFoundError(f"Could not locate {model.path}")
            else:
                raise TypeError(f"Unsupported model type: {model.model_type}")

        model_name = model[0].name if isinstance(model, list) else model.name

        # resolve generated context binary file name
        if output_filename:
            output_name = output_filename
        elif model_name:
            output_name = model_name
        else:
            output_name = 'context'

        output_filepath = output_path / (output_name + '.bin')
        context_binary_path = str(Path(output_filepath.parent, output_filepath.stem))

        backend_binary_path = ''
        if backend_specific_filename:
            output_binary_filepath = output_path / (backend_specific_filename + '.bin')
            backend_binary_path = str(Path(output_binary_filepath.parent, output_binary_filepath.stem))

        with temporaryDirectoryChange(temp_directory.name):
            data_format_config_path = ""
            if config and config.data_format_config is not None:
                path_obj = Path(config.data_format_config)
                data_format_config_path = str(path_obj.resolve())

            pbm = py_net_run.PythonBackendManager(logLevel=log_level_arg,
                                                  opPackagePaths=op_package_arg,
                                                  debug=enable_intermediate_outputs_arg,
                                                  profilingLevel=profile_level_arg,
                                                  profilingOption=profile_option_arg,
                                                  dataFormatConfigPath=data_format_config_path)
            if not dlc_paths:
                err = pbm.loadModelLib(str(model_path), self._model_lib_key)
                _check_pnr_error(err,
                                f'Failed to load model library: {model_path}',
                                ContextBinaryGenerationError,
                                NetRunErrorCode.INITIALIZE)
                self._created_from_qnn_model = True
            if backend.target.target_platform_type == DevicePlatformType.LINUX_EMBEDDED:
                backend_lib_path = str(Path(sdk_root, 'lib', backend.target.target_name,
                                        backend.backend_library))
            else:
                backend_lib_path = backend.backend_library
            err = pbm.loadBackendLib(self._backend_lib_key, backend_lib_path)
            _check_pnr_error(err,
                             f'Failed to load backend library: {backend.backend_library}',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.INITIALIZE)

            err = pbm.createLogHandle(self._backend_lib_key, self._logger_key, log_level_arg)
            _check_pnr_error(err,
                             'Failed to initialize logging in the backend',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.INITIALIZE)

            [extension_lib_path, json_path] = create_backend_extension_argument(backend,
                                                                                temp_directory.name,
                                                                                sdk_root)
            if extension_lib_path:
                err = pbm.initializeBackendExtension(self._backend_lib_key,
                                                     extension_lib_path,
                                                     json_path,
                                                     py_net_run.AppType.QNN_APP_CONTEXT_BINARY_GENERATOR)
                _check_pnr_error(err,
                                 'Failed to initialize backend extensions',
                                 ContextBinaryGenerationError,
                                 NetRunErrorCode.INITIALIZE)

            if profile_level_arg != py_net_run.ProfilingLevel.OFF:
                err = pbm.initializeProfileLogger(self._backend_lib_key,
                                                  str(output_path),
                                                  default_profiling_log_name)
                _check_pnr_error(err,
                                 'Failed to initialize the profile logger',
                                 ContextBinaryGenerationError,
                                 NetRunErrorCode.INITIALIZE)

            err = pbm.createBackendHandle(self._backend_lib_key, self._logger_key,
                                          self._backend_key, platform_options_arg)
            _check_pnr_error(err,
                             'Failed to create a backend handle',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.CREATE_BACKEND)

            err = pbm.createDeviceHandle(self._backend_lib_key, self._logger_key, self._device_key)
            _check_pnr_error(err,
                             'Failed to create a device handle',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.CREATE_DEVICE)

            err = pbm.registerOpPackage(self._backend_lib_key, self._backend_key)
            _check_pnr_error(err,
                             'Failed to register op packages',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.REGISTER_OPPACKAGE)

            err = pbm.createContext(self._backend_lib_key,
                                    self._backend_key,
                                    self._device_key,
                                    self._context_key)
            _check_pnr_error(err,
                             'Failed to create a context',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.CREATE_CONTEXT)

            output_tensor_names_arg = create_set_output_tensors_argument(config)
            input_output_tensor_mem_type_arg = create_input_output_tensor_mem_type_argument(config)
            if dlc_paths:
                for dlc_path in dlc_paths:
                    err = pbm.composeGraphsFromDlc(self._backend_lib_key,
                                                   self._backend_key,
                                                   self._context_key,
                                                   str(dlc_path),
                                                   log_level_arg,
                                                   output_tensor_names_arg,
                                                   input_output_tensor_mem_type_arg)
                    _check_pnr_error(err,
                                     'Failed to compose graphs',
                                     ContextBinaryGenerationError,
                                     NetRunErrorCode.COMPOSE_GRAPHS)
            else:
                err = pbm.composeGraphs(self._backend_lib_key,
                                        self._backend_key,
                                        self._context_key,
                                        self._model_lib_key,
                                        output_tensor_names_arg,
                                        input_output_tensor_mem_type_arg)

                _check_pnr_error(err,
                                 'Failed to compose graphs',
                                 ContextBinaryGenerationError,
                                 NetRunErrorCode.COMPOSE_GRAPHS)

            err = pbm.finalizeGraphs(self._backend_lib_key,
                                     self._backend_key,
                                     self._context_key,
                                     profilingLevel=profile_level_arg,
                                     profilingOption=profile_option_arg)
            _check_pnr_error(err,
                             'Failed to finalize graphs',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.FINALIZE_GRAPHS)
            if adapter_weight_config_file_arg:
                err = pbm.createBinarySections(self._backend_lib_key,
                                self._backend_key,
                                self._context_key,
                                adapter_weight_config_file_arg,
                                str(output_path))
                _check_pnr_error(err,
                                'Failed to create binary sections',
                                ContextBinaryGenerationError,
                                NetRunErrorCode.BINARY_SECTION)

            logger.info(f'Writing context binary: {output_filepath}')

            err = pbm.saveContextToBinaryFile(self._backend_lib_key,
                                              self._backend_key,
                                              self._context_key,
                                              context_binary_path,
                                              backend_binary_path
                                            )
            _check_pnr_error(err,
                             'Failed to write context binary to a file',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.FINALIZE_GRAPHS)

            if not output_filepath.is_file():
                raise RuntimeError(f'Failed to generate binary file: {output_filepath}')

            if backend_specific_filename and not output_binary_filepath.is_file():
                raise RuntimeError(f'Failed to generate backend-specific binary file: {output_binary_filepath}')

            err = pbm.freeGraphsInfo(self._backend_lib_key, self._backend_key, self._context_key)
            _check_pnr_error(err,
                             'Failed to free graph info(s)',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.TERMINATE)

            err = pbm.freeContext(self._backend_lib_key,
                                  self._backend_key,
                                  self._context_key,
                                  profilingLevel=profile_level_arg)
            _check_pnr_error(err,
                             'Failed to free context',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.FREE_CONTEXT)

            err = pbm.freeDeviceHandle(self._backend_lib_key, self._device_key)
            _check_pnr_error(err,
                             'Failed to free device',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.FREE_DEVICE)

            profiling_log = None
            if profile_level_arg != py_net_run.ProfilingLevel.OFF:
                err = pbm.disposeProfileLogger(self._backend_lib_key)
                _check_pnr_error(err,
                                 'Failed to dispose profile logger',
                                 ContextBinaryGenerationError,
                                 NetRunErrorCode.TERMINATE)
                profiling_log = Path(output_path, default_profiling_log_name).resolve()
                if not profiling_log.is_file():
                    raise RuntimeError(f"Could not locate profiling log at {profiling_log}")

            err = pbm.freeBackendHandle(self._backend_lib_key, self._backend_key)
            _check_pnr_error(err,
                             'Failed to free backend handle',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.FREE_BACKEND)

            err = pbm.freeLogHandle(self._backend_lib_key, self._logger_key)
            _check_pnr_error(err,
                             'Failed to free log handle',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.TERMINATE)

            err = pbm.unloadBackendLib(self._backend_lib_key)
            _check_pnr_error(err,
                             'Failed to unload backend library',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.TERMINATE)

            if self._created_from_qnn_model:
                err = pbm.unloadModelLib(self._model_lib_key)
                _check_pnr_error(err,
                                'Failed to unload model library',
                                ContextBinaryGenerationError,
                                NetRunErrorCode.TERMINATE)

            backend.after_generate_hook(temp_directory.name, sdk_root)

            moved_profiling_artifacts = move_backend_profiling_artifacts(backend, output_path)
            profiling_data = None
            if profiling_log is not None:
                profiling_data = ProfilingData(profiling_log=profiling_log,
                                               backend_profiling_artifacts=moved_profiling_artifacts)

            context_binary = ModelConfig(path=output_filepath)
            backend_binary_path = None
            if backend_specific_filename and output_binary_filepath.is_file():
                backend_binary_path = ModelConfig(path=output_binary_filepath)

            return (context_binary, backend_binary_path, profiling_data)

    def teardown(self, backend, sdk_root, config, output_dir) -> Optional[ProfilingData]:
        if not self._pbm:
            # already torn down, no action needed
            return None

        output_dir = Path(output_dir).resolve()

        with temporaryDirectoryChange(self._run_temp_directory.name):
            profile_level_arg = create_profile_level_argument(config)
            profile_log_name = 'qnn-profiling-data-unload.log'
            if profile_level_arg != py_net_run.ProfilingLevel.OFF and not self._oneshot_execute:
                err = self._pbm.initializeProfileLogger(self._backend_lib_key,
                                                        str(output_dir),
                                                        profile_log_name)
                _check_pnr_error(err,
                                 'Failed to initialize the profile logger',
                                 InferenceError,
                                 NetRunErrorCode.INITIALIZE)

            err = self._pbm.freeGraphsInfo(self._backend_lib_key, self._backend_key,
                                           self._context_key)
            _check_pnr_error(err,
                             'Failed to free graph info(s)',
                             InferenceError,
                             NetRunErrorCode.TERMINATE)

            err = self._pbm.freeContext(self._backend_lib_key,
                                        self._backend_key,
                                        self._context_key,
                                        profilingLevel=profile_level_arg)
            _check_pnr_error(err,
                             'Failed to free context',
                             InferenceError,
                             NetRunErrorCode.FREE_CONTEXT)

            err = self._pbm.freeDeviceHandle(self._backend_lib_key, self._device_key)
            _check_pnr_error(err,
                             'Failed to free device',
                             InferenceError,
                             NetRunErrorCode.FREE_DEVICE)

            if profile_level_arg != py_net_run.ProfilingLevel.OFF:
                err = self._pbm.disposeProfileLogger(self._backend_lib_key)
                _check_pnr_error(err,
                                 'Failed to dispose profile logger',
                                 InferenceError,
                                 NetRunErrorCode.TERMINATE)

            err = self._pbm.freeBackendHandle(self._backend_lib_key, self._backend_key)
            _check_pnr_error(err,
                             'Failed to free backend handle',
                             InferenceError,
                             NetRunErrorCode.FREE_BACKEND)

            err = self._pbm.freeLogHandle(self._backend_lib_key, self._logger_key)
            _check_pnr_error(err,
                             'Failed to free log handle',
                             InferenceError,
                             NetRunErrorCode.TERMINATE)

            err = self._pbm.unloadBackendLib(self._backend_lib_key)
            _check_pnr_error(err,
                             'Failed to unload backend library',
                             InferenceError,
                             NetRunErrorCode.TERMINATE)

            if self._created_from_qnn_model:
                err = self._pbm.unloadModelLib(self._model_lib_key)
                _check_pnr_error(err,
                                 'Failed to unload model library',
                                 InferenceError,
                                 NetRunErrorCode.TERMINATE)

            self._pbm = None

            profiling_log = None
            if not self._oneshot_execute and profile_level_arg != py_net_run.ProfilingLevel.OFF:
                profiling_log = Path(output_dir, profile_log_name).resolve()
                if not profiling_log.is_file():
                    raise RuntimeError(f"Could not locate profiling log at {profiling_log}")

            backend.after_run_hook(self._run_temp_directory.name, sdk_root)

            moved_profiling_artifacts = move_backend_profiling_artifacts(backend, output_dir)
            profiling_data = None
            if profiling_log is not None:
                profiling_data = ProfilingData(profiling_log=profiling_log,
                                               backend_profiling_artifacts=moved_profiling_artifacts)

            return profiling_data
