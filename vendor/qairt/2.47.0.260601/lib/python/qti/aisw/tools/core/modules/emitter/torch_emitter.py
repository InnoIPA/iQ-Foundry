# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import qti.aisw.emitter.utils.ir_graph_utils as ir_graph_utils
from pydantic import DirectoryPath, Field, FilePath, model_validator
from qti.aisw.converters.common import ir_graph as ir_graph_lib
from qti.aisw.converters.common.backend_aware_configs.backend_awareness_utils import (
    get_path_for_target_config,
)

# Converter Imports
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.emitter.ir_graph_op_handler import qnn_numpy_type_to_actual_numpy_dtype
from qti.aisw.emitter.ir_to_torch import TorchEmitter
from qti.aisw.emitter.utils.config import CustomOpInfo

# Module Imports
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
    expect_module_compliance,
)
from qti.aisw.tools.core.modules.api.definitions.common import BackendType
from qti.aisw.tools.core.modules.converter.constants import LOGLEVEL

try:
    from qti.aisw.converters.common import ir_quantizer
except ImportError as ie:
    print("Failed to find necessary quantization packages:")
    print(str(ie))
    print("Please ensure that $QNN_SDK_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

IrGraph = ir_graph_lib.IrGraph


class EmitterInputConfig(AISWBaseModel):
    """Pydantic class for input config of TorchEmitterAndConfigGenerator"""

    input_graph: Any = Field(description="Optimized IR Grapg of type IrGraph or path of the non "
                                         "quantized DLC.")
    backend_name: Optional[str] = Field(default=None, description="Backend information.")
    path: Optional[DirectoryPath] = Field(default=".", description="Path to save newly built "
                                                                   "pytorch model definition")
    filename: Optional[str] = Field(default="converted_model", description="Filename to save "
                                                                           "newly built pytorch "
                                                                           "model definition")
    model_name: Optional[str] = Field(default="ConvertedModel", description="Name of the "
                                                                            "converted model")
    dummy_model_input: Optional[Any] = Field(default=None, description="Dummy model input")
    dummy_model_output: Optional[Any] = Field(default=None, description="Dummy model output")
    keep_linear_without_bias: Optional[bool] = Field(default=False, description="Flag variable "
                                                                                "whether to keep "
                                                                                "the original "
                                                                                "linear module "
                                                                                "after preparation")
    keep_original_model_structure: Optional[bool] = Field(default=False, description="Flag for "
                                                                                     "keeping "
                                                                                     "original "
                                                                                     "model "
                                                                                     "structure "
                                                                                     "in emitter "
                                                                                     "model")
    block_names_mapping: Optional[dict] = Field(default=None, description="Block name to class "
                                                                          "name mapping")
    ignore_encodings: Optional[bool] = Field(default=False, description="Flag to determine "
                                                                        "whether to ignore "
                                                                        "encodings, set to False "
                                                                        "by default")
    ir_graph_input_names: Optional[List] = Field(default=None, description="List of the ir_graph "
                                                                           "input names in order")
    ir_graph_output_names: Optional[List] = Field(default=None, description="List of the ir_graph "
                                                                            "output names in "
                                                                            "order")
    custom_op_info: Optional[CustomOpInfo] = Field(default=None, description="CustomIrOp to "
                                                                             "TorchImplementation "
                                                                             "mapping and their "
                                                                             "paths")

    @model_validator(mode="after")
    def validate_input_arguments(self):
        """Validates the input argument of this modular API"""
        if self.block_names_mapping and not self.keep_original_model_structure:
            raise ValueError("block_names_mapping cannot be passed without enabling "
                             "keep_original_model_structure flag.")
        return self


class EmitterOutputConfig(AISWBaseModel):
    """Pydantic class for output config of TorchEmitterAndConfigGenerator"""

    model_definition_path: FilePath = Field(description="Path of the torch script file")
    state_dict_path: FilePath = Field(description="Path of the SafeTensor file with state dict")
    backend_base_config_path: Optional[FilePath] = Field(default=None, description="Backend aware "
                                                                                   "base config "
                                                                                   "path")


class EmitterModuleSchemaV1(ModuleSchema):
    """Schema class for TorchEmitterAndConfigGenerator"""
    _VERSION = ModuleSchemaVersion(major=0, minor=3, patch=0)
    _BACKENDS = None
    name: Literal["EmitterModule"] = "EmitterModule"
    path: Path = Path(__file__)
    arguments: EmitterInputConfig
    outputs: Optional[EmitterOutputConfig] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


@expect_module_compliance
class TorchEmitterAndConfigGenerator(Module):
    """User interface class for Preparer Pro API"""
    _SCHEMA = EmitterModuleSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger=None):
        """Constructor method"""
        if not logger:
            logger = logging.getLogger("EmitterLogger")
        converter_utils.LOGGER = logger
        super().__init__(logger)
        self._debug_level = LOGLEVEL.INFO

        self._ir_graph_input_shapes = None
        self._ir_graph_input_dtypes = None

    def _filter_and_create_backend_aware_json_config_for_aimet_usecase(self, config_file_name: str,
                                                                       backend_aware_config_path:
                                                                       str):
        """This method filters backend aware config used in QAIRT and create backend aware JSON
        config using backend provided

        Args:
            config_file_name (str): Backend file name
            backend_aware_config_path (str): Output file path where JSON would be created
        """
        backend_aware_json_path = get_path_for_target_config(config_file_name)
        with open(backend_aware_json_path, 'r') as file:
            backend_config_json = json.load(file)

        for op in list(backend_config_json['op_type'].keys()):
            if 'supported_kernels' in backend_config_json['op_type'][op]:
                supported_kernels = backend_config_json['op_type'][op]['supported_kernels']
                kernels_for_op_with_multiple_outputs = []
                for kernel in supported_kernels:
                    for key in list(kernel.keys()):
                        if key.startswith('activation_'):
                            activation_kernel = {}
                            activation_kernel['activation'] = kernel[key]
                            kernels_for_op_with_multiple_outputs.append(activation_kernel)
                            del kernel[key]

                supported_kernels.extend(kernels_for_op_with_multiple_outputs)

                for kernel in supported_kernels:
                    kernel.pop('bias', None)
                    kernel.pop('inputs', None)
                    kernel['activation'].pop('is_signed', None)
                    kernel['activation'].pop('is_fixed_point', None)
                    kernel['activation'].pop('output_id', None)

                    if 'param' in kernel:
                        kernel['param'].pop('is_signed', None)
                        kernel['param'].pop('is_fixed_point', None)

                unique_supported_kernels = []
                for kernel in supported_kernels:
                    if ((kernel['activation']['dtype'] != 'bool') and
                            (kernel['activation']['bitwidth'] != 64)):
                        if 'param' in kernel:
                            if ((kernel['param']['dtype'] != 'bool') and
                                    (kernel['param']['bitwidth'] != 64)):
                                if kernel not in unique_supported_kernels:
                                    unique_supported_kernels.append(kernel)
                        else:
                            if kernel not in unique_supported_kernels:
                                unique_supported_kernels.append(kernel)

                if ((len(backend_config_json['op_type'][op].keys()) == 1) and
                        ('supported_kernels' in backend_config_json['op_type'][op]) and
                        (not unique_supported_kernels)):
                    del backend_config_json['op_type'][op]

                if unique_supported_kernels:
                    backend_config_json['op_type'][op]['supported_kernels'] = unique_supported_kernels

        with open(backend_aware_config_path, 'w') as file:
            json.dump(backend_config_json, file, indent=4)

    def _get_backend_aware_base_config(self, backend_name: str = None,
                                       backend_aware_config_dir_path: str = None) -> Optional[str]:
        """Returns path of backend aware config.

        Args:
            backend_name: Backend name (str): Allowed values are 'HTP', 'CPU', 'AIC' and 'LPAI'
            backend_aware_config_dir_path (str): Directory where backend aware JSON files are
            present or would get created

        Returns:
            str: The return value, path of backend aware JSON config
        """
        if backend_name and backend_aware_config_dir_path:
            backend_to_json_file_name = {BackendType.HTP.value: 'htp',
                                         BackendType.CPU.value: 'cpu',
                                         BackendType.LPAI.value: 'lpai',
                                         BackendType.AIC.value: 'aic'
                                         }
            config_file_name = backend_to_json_file_name.get(backend_name, None)

            if config_file_name is None:
                raise ValueError(f"Backend {backend_name} does not have backend aware config.")

            backend_specific_file_name = config_file_name + '.json'
            backend_aware_config_path = os.path.join(backend_aware_config_dir_path,
                                                     backend_specific_file_name)

            if not os.path.exists(backend_aware_config_path):
                self._filter_and_create_backend_aware_json_config_for_aimet_usecase(
                    config_file_name, backend_aware_config_path)

            return backend_aware_config_path

        else:
            return None

    def _get_ir_graph_input_shapes(self) -> Dict:
        """Method to get IR graph input shapes, once it is created
        Returns:
            A dict where key is the input name and value is the shape of the input tensor
        """
        return self._ir_graph_input_shapes

    def _get_ir_graph_input_dtypes(self) -> Dict:
        """Method to get IR graph input dtypes, once it is created
        Returns:
             A dict where key is the input name and value is the numpy dtype of the input tensor
        """
        return self._ir_graph_input_dtypes

    def prepare_model(self, config: EmitterInputConfig) -> EmitterOutputConfig:
        """This is quantizer API method. It accepts input arguments in "AISWBaseModel" object type and
        returns outputs in "AISWBaseModel" object type.

        Args:
            config: "EmitterInputConfig" object containing converter module input arguments.

        Returns:
            EmitteroOutputConfig object containing model definition path, model weight file path,
            backend aware config path.

        Examples:
            >>> from qti.aisw.tools.core.modules.emitter.torch_emitter import
            >>>TorchEmitterAndConfigGenerator, EmitterInputConfig
            >>> emitter_input = EmitterInputConfig(input_graph='path/to/dlc/file')
            >>> emitter_config = TorchEmitterAndConfigGenerator().prepare_from_dlc(emitter_input)

        """
        # Convert pydantic model to dictionary
        args_dict = config.model_dump()
        custom_op_info = args_dict.pop('custom_op_info')
        args_dict['custom_op_info'] = CustomOpInfo(**custom_op_info) if (custom_op_info is not
                                                                         None) else CustomOpInfo()
        try:
            input_graph = args_dict.pop('input_graph')

            if isinstance(input_graph, IrGraph):
                ir_graph = input_graph
            elif os.path.exists(input_graph):
                ir_graph, _ = ir_graph_utils.get_ir_graph_from_dlc(input_graph)
            else:
                raise ValueError("Invalid input graph provided!")

            opts = ir_quantizer.IrQuantizerOpts()
            quantizer = ir_quantizer.IrQuantizer(opts, ir_graph)
            quantizer.translate_quant_to_float_graph()

            self._ir_graph_input_shapes = {input_tensor.name(): input_tensor.dims()
                                           for input_tensor in
                                           ir_graph.get_input_tensors_to_graph()}
            self._ir_graph_input_dtypes = {tensor.name(): qnn_numpy_type_to_actual_numpy_dtype(tensor)
                                           for tensor in ir_graph.get_input_tensors_to_graph()}

            backend_name = args_dict.pop('backend_name')
            base_config_path = self._get_backend_aware_base_config(backend_name, args_dict['path'])

            emitter_obj = TorchEmitter(**args_dict)
            model_definition_path, state_dict_file = emitter_obj.generate_torch_artifacts(ir_graph)

            output_config = EmitterOutputConfig(model_definition_path=model_definition_path,
                                                state_dict_path=state_dict_file,
                                                backend_base_config_path=base_config_path
                                                )
        except Exception as e:
            self._logger.error("Emitter Model generation failed.")
            raise e
        return output_config

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        """Sets optimizer log level.

        Args:
            debug_level: LOGLEVEL.VERBOSE enables VERBOSE and higher level messages.
               LOGLEVEL.DEBUG enables DEBUG and higher level messages.
               LOGLEVEL.DEBUG_3 enables DEBUG_3 and higher level messages.
               LOGLEVEL.DEBUG_2 enables DEBUG_2 and higher level messages.
               LOGLEVEL.DEBUG_1 enables DEBUG_1 and higher level messages.
               LOGLEVEL.INFO enables INFO and higher level messages.
            **kwargs: keyword args

        Returns:
            bool: 'True' if debugging is enabled else return 'False'.
        """
        if debug_level < LOGLEVEL.INFO or debug_level > LOGLEVEL.VERBOSE:
            return False
        self._debug_level = debug_level
        converter_utils.setup_logging(self._debug_level)
        return True

    @property
    def _schema(self):
        return self._SCHEMA

    def get_logger(self) -> Any:
        """Returns the logger"""
        return self._logger

    def properties(self) -> Dict[str, Any]:
        """Returns the properties of the schema"""
        return self._schema.model_json_schema()
