# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import logging
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import qti.aisw.core.model_level_api as mlapi
from pydantic import Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    BackendType,
    ModelConfig,
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
    OpPackageIdentifier,
    ProfilingData,
    Target,
    expect_module_compliance,
)
from qti.aisw.tools.core.modules.api.backend.backend import Backend
from qti.aisw.tools.core.modules.api.definitions.common import (
    QNNCommonConfig,
    QNNContextConfig,
)
from qti.aisw.tools.core.modules.api.utils.configure_backend import (
    create_backend,
    get_supported_backends,
)
from qti.aisw.tools.core.modules.net_runner.utils import (
    NamedTensorMapping,
    NetRunnerInputData,
    is_required_tensor_data_provided,
)

from qti.aisw.tools.core.utilities.qairt_logging import QAIRTLogger


InferencerHandle = int


class InferenceConfig(QNNCommonConfig):
    """Defines supported inference parameters that are implemented by backend-agnostic tools, therefore
    are applicable to all backends (subject to feature support as generally these correspond to
    optional API capabilities).

    Args:
        batch_multiplier (Optional[str]): Specifies the value with which the batch value in the input
                and output tensors will be specified
        use_native_output_data (Optional[bool]): Specifies that the output data should be returned
            as a native data type (e.g. numpy array) instead of a list of tensors.
        use_native_input_data (Optional[bool]): Specifies that the input data should be returned
            as a native data type (e.g. numpy array) instead of a list of tensors.
        native_input_tensor_names (Optional[List[str]]): Provide a comma-separated list of input tensor names,
            for which the input files would be read/parsed in native format.
        synchronous (Optional[bool]): Specifies that the inference should be performed synchronously.
        debug (Optional[bool]): Specifies that output from all layers of the network
                                             will be saved.
        perf_profile (Optional[str]): Specifies the performance profile to be used for the inference.
        op_packages (Optional[List[OpPackageIdentifier]]): specifies the list of op packages, interface
            providers, and, optionally, targets to register.
        use_mmap (Optional[bool]): Specifies that the context binary should be loaded
            using Memory-mapped (MMAP) file I/O.
        num_inferences: Specifies the number of inferences. Loops over the input_list until
            the specified number of inferences has transpired.
        duration: Specifies the duration of the graph execution in seconds.
        keep_num_outputs: Specifies the number of outputs to be saved. Once the number of outputs reach the
            limit, subsequent outputs would be discarded.
        binary_updates: A yaml file containing the lora usecase adapters to be used in execution
    """

    batch_multiplier: Optional[str] = None
    use_native_output_data: Optional[bool] = None
    use_native_input_data: Optional[bool] = None
    native_input_tensor_names: Optional[List[str]] = None
    synchronous: Optional[bool] = None
    debug: Optional[bool] = None
    perf_profile: Optional[str] = None
    op_packages: Optional[List[OpPackageIdentifier]] = None
    use_mmap: Optional[bool] = None
    duration: Optional[float] = None
    num_inferences: Optional[int] = None
    keep_num_outputs: Optional[int] = None
    binary_updates: Optional[Union[str, PathLike]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_mutual_exclusion(cls, values):
        """Validates 'duration' and 'num_inferences' should be mutually exclusive"""
        if values.get("duration") and values.get("num_inferences"):
            raise ValueError(
                "Only one of 'duration' or 'num_inferences' can be specifed, not both"
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_keep_num_outputs(cls, values):
        """Validates 'keep_num_outputs' is a positive integer if provided"""
        keep_num_outputs = values.get("keep_num_outputs")
        if keep_num_outputs is not None and keep_num_outputs <= 0:
            raise ValueError("'keep_num_outputs' must be a positive integer")
        return values


class InferenceIdentifier(AISWBaseModel):
    """Defines the model, target, and backend to be used when running inferences.

    If target is not provided, the backend will choose a sane default based on its typical
    workflows, e.g. QNN HTP will run on Android by default, but QNN CPU will run on the host.
    """

    model: ModelConfig
    target: Optional[Target] = None
    backend: BackendType


class NetRunnerLoadArgConfig(AISWBaseModel):
    """Defines arguments for loading artifacts for an inference.

    Backend-specific parameters are passed via config file or config dict. Any duplicate
    arguments specified in the config dict will override identical arguments in the config file.

    Any artifacts generated during loading (e.g. profiling logs) will be output in the location
    specified by the output_dir field.
    """

    identifier: InferenceIdentifier
    backend_config_file: Optional[Union[str, PathLike]] = None
    backend_config_dict: Optional[Dict[str, Any]] = None
    context_config: Optional[QNNContextConfig] = None
    inference_config: Optional[InferenceConfig] = None
    output_dir: Union[PathLike, str] = "./output/"


class NetRunnerLoadOutputConfig(AISWBaseModel):
    """Defines loading output data. The returned handle is used in subsequent module functions to
    identify the corresponding inference configuration.
    """

    handle: InferencerHandle
    profiling_data: Optional[ProfilingData] = None


class NetRunnerRunArgConfig(AISWBaseModel, arbitrary_types_allowed=True):
    """Defines arguments for an inference.

    The identifier refers to the desired model, target, and backend to run inferences with. This can
    be provided directly to run(), or it can be passed to load() and the returned handle can be
    provided instead.

    Backend-specific parameters are passed via config file or config dict. Any duplicate
    arguments specified in the config dict will override identical arguments in the config file.

    If a backend config is given via file or dictionary, or an InferenceConfig is provided, it will
    override any configs provided during load().

    Any artifacts generated during inference (e.g. profiling logs) will be output in the location
    specified by the output_dir field. Note that inference output data will not be output in this
    directory, it will be returned in-memory.
    """

    identifier: Union[InferenceIdentifier, InferencerHandle]
    backend_config_file: Optional[Union[str, PathLike]] = None
    backend_config_dict: Optional[Dict[str, Any]] = None
    context_config: Optional[QNNContextConfig] = None
    graph_name: Optional[str] = ""
    input_data: SkipJsonSchema[NetRunnerInputData] = Field(exclude=True)
    inference_config: Optional[InferenceConfig] = None
    output_dir: Union[PathLike, str] = "./output/"
    input_tensor_names: Optional[List[str]] = None
    optional_output_tensor_names: Optional[List[str]] = None


class NetRunnerRunOutputConfig(AISWBaseModel, arbitrary_types_allowed=True):
    """Defines inference output data.

    Output_data will be returned as:
    1) list of tensor name -> numpy array mappings (one mapping per inference), or
    2) Dict mapping LoRA adapter name to list of tensor name -> numpy array mapping. This is used
    in case LoRA usecase adapters are provided in the InferenceConfig.

    Profiling data will be returned if it was enabled in the InferenceConfig
    """

    output_data: Union[List[NamedTensorMapping], Dict[str, List[NamedTensorMapping]]]
    profiling_data: Optional[ProfilingData] = None


class NetRunnerUnloadArgConfig(AISWBaseModel):
    """Defines arguments for unloading artifacts for an inference.

    Any artifacts generated during unloading (e.g. profiling logs) will be output in the location
    specified by the output_dir field.
    """

    handle: InferencerHandle
    output_dir: Union[PathLike, str] = "./output/"


class NetRunnerUnloadOutputConfig(AISWBaseModel):
    """Defines unloading output data."""

    profiling_data: Optional[ProfilingData] = None


class NetRunnerModuleSchema(ModuleSchema):
    _BACKENDS = get_supported_backends()
    _VERSION = ModuleSchemaVersion(major=0, minor=4, patch=0)

    name: Literal["NetRunnerModule"] = "NetRunnerModule"
    path: Path = Path(__file__)
    arguments: NetRunnerRunArgConfig
    outputs: SkipJsonSchema[Optional[NetRunnerRunOutputConfig]] = None
    backends: List[str] = _BACKENDS


@expect_module_compliance
class NetRunner(Module):
    _SCHEMA = NetRunnerModuleSchema

    def __init__(self, logger: logging.Logger = None):
        """Initializes a NetRunner module instance

        Args:
            logger (any): A logger instance to be used by the NetRunner module
        """
        if logger:
            self._logger = QAIRTLogger.get_logger(
                "NetRunnerLogger", parent_logger=logger
            )
        else:
            self._logger = QAIRTLogger.get_logger(
                "NetRunnerLogger",
                level="INFO",
                formatter_val="extended",
                handler_list=["dev_console"],
            )
        self._inferencer_cache: Dict[InferencerHandle, mlapi.Inferencer] = {}
        self._backend_cache: Dict[InferencerHandle, Backend] = {}
        self._next_inferencer_handle = 0

    def properties(self) -> Dict[str, Any]:
        return self._SCHEMA.model_json_schema()

    def get_logger(self) -> Any:
        return self._logger

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        pass

    def load(self, config: NetRunnerLoadArgConfig) -> NetRunnerLoadOutputConfig:
        """Loads required artifacts for running inferences using the specified backend on the provided
        target.

        This function is optional, and should be used if you intend to call run() multiple times
        with the  same model, backend, and target. This allows fine grained control over the
        lifetime of the artifacts.

        Artifacts loaded during load() will persist until unload() is called with the associated
        handle. However, if the model, backend, and target are given directly to run(), the
        artifacts will be unloaded at the end of run(). So providing the same model, backend, and
        target to multiple calls to run() will result in redundant setup and teardown steps being
        performed. This redundant setup and teardown can be avoided by explicitly calling
        load()/unload().

        Args:
            config (NetRunnerLoadArgConfig): Arguments which indicate what artifacts need to be
            loaded, and which device they should be loaded onto (if applicable).

        Returns:
            NetRunnerLoadOutputConfig: A handle to identify the corresponding inference
            configuration in subsequent module functions, as well as profiling data if profiling
            was enabled and was generated during the function.

        Examples: The example below shows how to load a dlc on an X86 linux target
            >>> model = ModelConfig(path="/path/to/dlc")
            >>> x86_linux_target = Target(type=DevicePlatformType.X86_64_LINUX)
            >>> inference_identifier = InferenceIdentifier(model=model,
            >>>                                            target=x86_linux_target,
            >>>                                            backend=BackendType.HTP)
            >>> load_arg_config = NetRunnerLoadArgConfig(identifier=inference_identifier)
            >>> net_runner = NetRunner()
            >>> load_output_config = net_runner.load(load_arg_config)
        """
        self._logger.debug(f"Loading with config: {config}")
        backend_instance = create_backend(
            config.identifier.backend,
            config.identifier.target,
            config.backend_config_file,
            config.backend_config_dict,
            config.context_config,
        )
        mlapi_model = config.identifier.model
        if config.inference_config and config.inference_config.op_packages:
            for op_package in config.inference_config.op_packages:
                backend_instance.register_op_package(
                    op_package.package_path,
                    op_package.interface_provider,
                    op_package.target_name,
                    op_package.cpp_stl_path,
                )
        inferencer = mlapi.Inferencer(backend_instance, mlapi_model)

        run_config = config.inference_config
        inferencer.setup(run_config, output_dir=config.output_dir)

        profiling_data = inferencer.get_profiling_data()
        profiling_data = profiling_data[0] if profiling_data else None
        inferencer.clear_profiling_data()

        profiling_output = None
        if profiling_data is not None:
            profiling_output = ProfilingData(
                profiling_log=profiling_data.profiling_log,
                backend_profiling_artifacts=profiling_data.backend_profiling_artifacts,
            )

        inferencer_handle = self._next_inferencer_handle
        self._inferencer_cache[inferencer_handle] = inferencer
        self._backend_cache[inferencer_handle] = backend_instance

        self._next_inferencer_handle += 1

        return NetRunnerLoadOutputConfig(
            handle=inferencer_handle, profiling_data=profiling_output
        )

    def run(self, config: NetRunnerRunArgConfig) -> NetRunnerRunOutputConfig:
        """Runs inferences on a backend, model, and target based on the provided identifier.

        If the identifier is an InferencerHandle, it will be used to lookup the backend, model, and
        target which were given to load(). Artifacts will persist until unload() is called.

        If the identifier is an InferenceIdentifier, the provided backend, model, and target will be
        used, and artifacts will be unloaded at the end of the function.

        Args:
            config (NetRunnerRunArgConfig): Arguments of the inference where input data and an
            identifier for a model, backend, and target are specified, as well as any other
            miscellaneous parameters.

        Returns:
            NetRunnerRunOutputConfig: The output data as a list of tensor name -> np array mappings,
            as well as profiling data if profiling was enabled.

        Examples: The example below shows how to run a dlc on an X86 linux target identified by
                  a handle returned from load()
            >>> input_data = np.fromfile("/path/to/numpy_raw_data").astype(np.float32)
            >>> inf_config = InferenceConfig()
            >>> run_arg_config = NetRunnerRunArgConfig(identifier=load_output_config.handle,
            >>>                                        config=inf_config,
            >>>                                        input_data=input_data)
            >>> run_output_config = net_runner.run(run_arg_config)

                  The example below shows how to run a dlc on an Android target provided directly
                  to run() instead of using a handle returned from load()
            >>> model = ModelConfig(path="/path/to/dlc")
            >>> android_serial = RemoteDeviceIdentifier(serial_id='abcd1234')
            >>> android_target = Target(type=DevicePlatformType.ANDROID, identifier=android_serial)
            >>> inference_identifier = InferenceIdentifier(model=model,
            >>>                                            target=android_target,
            >>>                                            backend=BackendType.HTP)
            >>> input_data = np.fromfile("/path/to/numpy_raw_data").astype(np.float32)
            >>> inf_config = InferenceConfig()
            >>> run_arg_config = NetRunnerRunArgConfig(identifier=inference_identifier,
            >>>                                        config=inf_config,
            >>>                                        input_data=input_data)
            >>> run_output_config = net_runner.run(run_arg_config)
        """
        self._logger.debug(f"Running inference with config: {config}")

        # TODO: AISW-147318 Enable this validation
        # Check if all non optional input tensors have data provided
        # if config.input_tensor_names:
        #     if not is_required_tensor_data_provided(config.input_tensor_names, config.input_data):
        #         raise ValueError("Please provide input data for all input tensors")

        # if the identifier is a handle, look up the Inferencer created during load()
        if isinstance(config.identifier, InferencerHandle):
            inferencer = self._inferencer_cache.get(config.identifier)
            if inferencer is None:
                raise KeyError("Unrecognized inferencer handle {config.identifier}")

            if config.backend_config_dict or config.backend_config_file:
                self._backend_cache[config.identifier].update_config(
                    config.backend_config_dict, config.backend_config_file
                )
        # otherwise create the Inferencer now
        elif isinstance(config.identifier, InferenceIdentifier):
            backend_instance = create_backend(
                config.identifier.backend,
                config.identifier.target,
                config.backend_config_file,
                config.backend_config_dict,
                config.context_config,
            )
            if config.inference_config and config.inference_config.op_packages:
                for op_package in config.inference_config.op_packages:
                    backend_instance.register_op_package(
                        op_package.package_path,
                        op_package.interface_provider,
                        op_package.target_name,
                        op_package.cpp_stl_path,
                    )
            mlapi_model = config.identifier.model
            inferencer = mlapi.Inferencer(backend_instance, mlapi_model)

        else:
            raise TypeError(f"Unknown identifier {config.identifier}")

        run_config = config.inference_config
        output_data = inferencer.run(
            input_data=config.input_data,
            graph_name=config.graph_name,
            config=run_config,
            output_dir=config.output_dir,
            optional_output_tensor_names=config.optional_output_tensor_names,
        )

        # profiling output is a list of logs since it can be accumulated, need to extract the first
        # element, or return None if the list is empty (i.e. if profiling is disabled)
        profiling_data = inferencer.get_profiling_data()
        profiling_data = profiling_data[0] if profiling_data else None
        inferencer.clear_profiling_data()

        profiling_output = None
        if profiling_data is not None:
            profiling_output = ProfilingData(
                profiling_log=profiling_data.profiling_log,
                backend_profiling_artifacts=profiling_data.backend_profiling_artifacts,
            )

        return NetRunnerRunOutputConfig(
            output_data=output_data, profiling_data=profiling_output
        )

    def unload(self, config: NetRunnerUnloadArgConfig) -> NetRunnerUnloadOutputConfig:
        """Unloads any artifacts that were loaded during load() based on the provided handle.

        Args:
            config (NetRunnerUnloadArgConfig): Provides a handle to identify which artifacts should
            be unloaded resulting from a previous call to load().

        Returns:
            NetRunnerUnloadOutputConfig: Profiling data if profiling was enabled and was generated
            during the function.

        Examples: The example below shows how to unload a dlc on an X86 linux target identified by
                  a handle returned from load()
            >>> unload_arg_config = NetRunnerUnloadArgConfig(handle=load_output_config.handle)
            >>> unload_output_config = net_runner.unload(unload_arg_config)
        """
        self._logger.debug(f"Unloading with config: {config}")

        inferencer = self._inferencer_cache.get(config.handle)
        if inferencer is None:
            raise KeyError(f"Unrecognized inferencer handle {config.handle}")

        inferencer.teardown(output_dir=config.output_dir)

        profiling_data = inferencer.get_profiling_data()
        profiling_data = profiling_data[0] if profiling_data else None

        profiling_output = None
        if profiling_data is not None:
            profiling_output = ProfilingData(
                profiling_log=profiling_data.profiling_log,
                backend_profiling_artifacts=profiling_data.backend_profiling_artifacts,
            )

        del self._backend_cache[config.handle]
        del self._inferencer_cache[config.handle]
        return NetRunnerUnloadOutputConfig(profiling_data=profiling_output)
