# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated

from pydantic import ConfigDict, DirectoryPath, Field, FilePath, field_validator, model_validator
from pydantic.functional_validators import AfterValidator

# Converter Imports
from qti.aisw.converters.common.converter_ir.op_graph import InputEncodings, InputLayout
from qti.aisw.converters.common.input_shape import InputShapeArgParser, InputShapeInfo
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.converters.common import ir_quantizer

# Module Imports
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
    expect_module_compliance,
)
from qti.aisw.tools.core.modules.converter.common import DLCBackendConfig
from qti.aisw.tools.core.modules.converter.constants import *
from qti.aisw.tools.core.modules.converter.utils import *
from qti.aisw.tools.core.utilities.qairt_logging import QAIRTLogger


def check_layout(layout_str: str) -> str:
    if not InputLayout.is_valid_type(layout_str):
        raise AssertionError(f"The I/O layout: {layout_str} is not valid")
    return layout_str


def check_encoding(encoding_str: str) -> str:
    if not InputEncodings.is_valid_encoding(encoding_str):
        raise AssertionError(f"Input Encoding: {encoding_str} is not valid")
    return encoding_str


# Define string types for encoding and layouts
InputOutputLayoutStr = Annotated[str, AfterValidator(check_layout)]
InputEncodingStr = Annotated[str, AfterValidator(check_encoding)]


class QuantParam(AISWBaseModel):
    """Required when the desired model data type is a quantized data type.
    Has two sub fields: Scale and Offset.
    """

    scale: Optional[float] = Field(
        default=None, description="Float value for the scale of the buffer as desired by the user."
    )
    offset: Optional[int] = Field(
        default=None, description="Integer value for the offset as desired by the user."
    )


class InputTensorConfig(AISWBaseModel):
    """Model for input tensor configuration. Any of the keys can be omitted, except name."""

    name: str = Field(description="Name of tensor.")
    source_model_input_datatype: Optional[str] = Field(
        default=None, description="Source data type of input tensor. If not specified, will be inferred from the model."
    )
    source_model_input_layout: Optional[InputOutputLayoutStr] = Field(
        default=None, description="Source layout of each input tensor."
    )
    source_model_input_shape: Union[str, List[str]] = Field(default="", description="Shape of input tensor.")
    desired_model_input_datatype: Optional[str] = Field(
        default=None, description="Desired data type of input tensor."
    )
    desired_model_input_layout: Optional[InputOutputLayoutStr] = Field(
        default=None, description="Desired layout of each input tensor."
    )
    desired_input_color_encoding: Optional[InputEncodingStr] = Field(
        default="bgr", description="Input encoding of the network inputs. Default is bgr."
    )
    optional: Optional[bool] = Field(default=False, description="If True, the input tensor is optional.")
    quant_param: Optional[QuantParam] = Field(
        default=None, description="QuantParam field (optional), has two sub fields: Scale and Offset."
    )

    @model_validator(mode="before")
    def validate_quant_param(cls, values):
        """Validates that the quant_param is provided when quantized inputs (int8 or uint8) are used"""
        datatype = values.get("desired_model_input_datatype")
        if datatype in ["int8", "uint8"]:
            quant_param = values.get("quant_param")
            if quant_param is None or quant_param.scale is None or quant_param.offset is None:
                raise ValueError(
                    "Both scale and offset must be provided for quantized outputs (int8 or uint8)"
                )
        return values


class OutputTensorConfig(AISWBaseModel):
    """Model for ouput tensor configuration. Any of the keys can be omitted, except name."""

    name: str = Field(description="Name of tensor.")
    source_model_output_layout: Optional[InputOutputLayoutStr] = Field(
        default=None, description="Source layout of each output tensor."
    )
    desired_model_output_datatype: Optional[str] = Field(
        default=None, description="Desired data type of ouput tensor."
    )
    desired_model_output_layout: Optional[InputOutputLayoutStr] = Field(
        default=None, description="Desired layout of each output tensor."
    )
    optional: Optional[bool] = Field(default=False, description="If True, the output tensor is optional.")
    quant_param: Optional[QuantParam] = Field(
        default=None, description="QuantParam field (optional) has two sub fields: Scale and Offset."
    )

    @model_validator(mode="before")
    def validate_quant_param(cls, values):
        """Validates that the quant_param is provided when quantized outputs (int8 or uint8) are used"""
        datatype = values.get("desired_model_output_datatype")
        if datatype in ["int8", "uint8"]:
            quant_param = values.get("quant_param")
            if quant_param is None or quant_param.scale is None or quant_param.offset is None:
                raise ValueError(
                    "Both scale and offset must be provided for quantized outputs (int8 or uint8)"
                )
        return values


class ConverterInputConfig(AISWBaseModel):
    _model_framework: str
    input_network: FilePath | DirectoryPath = Field(description="Path to the source  framework model.")
    io_config: Optional[FilePath] = Field(default=None, description="Path to the custom IO config file.")
    input_tensors: Optional[List[InputTensorConfig]] = Field(
        default=None, description="Input tensors config."
    )
    output_tensors: Optional[List[OutputTensorConfig]] = Field(
        default=None, description="Output tensors config."
    )
    preserve_io_datatype: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Use this option to preserve IO datatype of the "
        "source model, when sequences of unsupported "
        "static operations are present. "
        "To preserve datatype for all inputs and outputs, "
        "provide option as, preserve_io_datatype='all'."
        "To preserve datatype for select inputs and outputs,"
        "preserve_io_datatype=[input1,input2,output1]",
    )
    float_bitwidth: Literal[32, 16] = Field(
        default=32, description="Convert the graph to specified float bitwidth."
    )
    float_bias_bitwidth: Literal[16, 32] = Field(
        default=32, description="Option to select the bit-width to use for float bias tensor."
    )
    quantization_overrides: Optional[FilePath] = Field(
        default=None, description="Option to specify a json file with parameters to use for quantization."
    )
    dry_run: bool = Field(
        default=False,
        description="Evaluates the model without actually converting any ops"
        ", and return unsupported ops/attributes as well as"
        " unused inputs and/or outputs if any.",
    )
    copyright_file: str = Field(
        default="",
        description="Path to copyright file. If provided, "
        "the content of the file will be added "
        "to the output model",
    )
    model_version: str = Field(
        default="",
        description="User-defined ASCII string to identify the model, only first 64 bytes will be stored.",
    )
    onnx_simplification: bool = Field(
        default=True,
        description="Do not attempt to simplify the model "
        "automatically.This may prevent some models from "
        "converting when sequences of unsupported static "
        "operations are present.",
    )
    onnx_batch: Optional[int] = Field(
        default=None,
        description="The batch dimension override.This will take the "
        "first dimension of all inputs and treat it as "
        "a batch dim, overriding it with the value "
        "provided here.",
    )
    onnx_define_symbol: Optional[List[Tuple[str, int]]] = Field(
        default=None, description="Option to override specific input dimension symbols."
    )
    onnx_defer_loading: bool = Field(
        default=False,
        description="Option to have the model not load weights. If False, the model will be loaded eagerly.",
    )
    tf_show_unconsumed_nodes: bool = Field(
        default=False, description="Displays a list of unconsumed nodes, if there any are found."
    )
    tf_saved_model_tag: Optional[str] = Field(
        default="serve", description="Specify the tag to select a MetaGraph from saved model."
    )
    tf_saved_model_signature_key: str = Field(
        default="serving_default",
        description="Specify signature key to select input and output of the model.",
    )
    tf_no_optimization: bool = Field(
        default=False, description="Do not attempt to optimize the model automatically."
    )
    tf_validate_models: bool = Field(
        default=False, description="Validate the original TF model against optimized TF model."
    )
    tflite_signature_name: str = Field(
        default="", description="Use this option to specify a specific Subgraph signature to convert."
    )
    enable_framework_trace: bool = Field(
        default=False,
        description="Use this option to enable converter to trace the o/p tensor change information.",
    )
    remove_unused_inputs: bool = Field(
        default=False,
        description="Use this option to remove the disconnected graph input nodes after the conversion",
    )
    output_path: str = Field(
        default="",
        description="Path where the converted output model should be saved.If"
        " not specified, the converter model will be written to a "
        "file with same name as the input model",
    )
    export_format: Literal["DLC_DEFAULT", "DLC_STRIP_QUANT"] = Field(default="DLC_DEFAULT",
        description="DLC_DEFAULT (default)\n"
        "- Produce a Float graph given a Float Source graph\n"
        "- Produce a Quant graph given a Source graph with provided Encodings\n"
        "DLC_STRIP_QUANT\n"
        "- Produce a Float Graph with discarding Quant data\n")
    calc_static_encodings: bool = Field(
        default=False,
    )
    quantizer_log: str = Field(
        default="",
        description="Valid for use with v2.0.0 JSON schema for quantization "
                    "overrides or when --use_quantize_v2 is provided. "
    )
    quantizer_log_level: ir_quantizer.LogLevel = Field(
        default=ir_quantizer.LogLevel.NONE,
        description="Sets the logging level in the quantizer. See --quantizer_log.\n"
                    "INFO: Emits a file in the CSV format. Requires --quantizer_log <file_name.csv> to be set. "
                    "TRACE: Emits a file in the TXT format. Requires --quantizer_log "
                    "<file_name.txt> to be set. "
                    "NONE: Default value. No file is emitted. "
                    "Warnings and errors are emitted to the console.")
    use_quantize_v2: bool = Field(
        default=False,
        description="Enables IrQuantizerV2 flow."
    )
    pack_4_bit_weights: bool = Field(
        default=False,
    )
    disable_dynamic_16_bit_weights: bool = Field(
        default=False,
    )
    # Custom op related arguments
    op_package_config: Optional[List[FilePath]] = Field(
        default=None,
        description="List of absolute paths to a Qnn Op Package XML configuration file that "
        "contains user defined custom operations.",
    )
    converter_op_package_lib: Optional[List[FilePath]] = Field(
        default=None,
        description="List of absolute paths to converter op package library compiled by "
        "the OpPackage generator.",
    )
    package_name: str = Field(
        default="",
        description="A global package name to be used for each node in "
        "the Model.cpp file. Defaults to Qnn header defined "
        "package name.",
    )
    lora_weight_list: Optional[FilePath] = Field(
        default=None,
        description="Path to a file specifying a list of tensor names that should be updatable."
    )
    quant_updatable_mode: Optional[Literal["none", "adapter_only", "all"]] = Field(
        default=None,
        description="Specify whether/for which tensors the quantization encodings change " \
                    "across use-cases. In none mode, no quantization encodings are updatable. " \
                    "In adapter_only mode quantization encodings for " \
                    "only lora/adapter branch (Conv->Mul->Conv) change across use-case, "
                    "the base branch quantization encodings remain the same. " \
                    "In all mode, all quantization encodings are updatable.")
    graph_specific_input_dims: Optional[List[List[str]]] = Field(
        default=None,
        description="Graph-specific input dimensions are only required when"
        "performing conversion for network specialization",
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=(),
                              arbitrary_types_allowed=True)

    @cached_property
    def input_dim(self) -> list:
        if not self.input_tensors:
            return []
        return [[tensor.name, tensor.source_model_input_shape]
                for tensor in self.input_tensors if tensor.source_model_input_shape]

    @model_validator(mode="before")
    def check_float_bitwidth(cls, values):
        if values.get("float_bitwidth") == 16 and values.get("float_bias_bitwidth") is None:
            """
            NOTE: float_bias_bitwidth is set to 16 in this case to have parity with
            qairt-converter CLI. Currently, in qairt-converter, if float_bitwidth is set to 16,
            all float32 tensors are converted to float16.
            If we default float_bias_bitwidth to 32 in this case, the bias tensors get set to FP32
            which breaks parity with qairt-converter.
            """
            values["float_bias_bitwidth"] = 16
        return values

    @field_validator("input_network")
    @classmethod
    def validate_framework(cls, v: FilePath | DirectoryPath) -> FilePath | DirectoryPath:
        cls._model_framework = infer_framework(input_network=v)
        return v

    @field_validator("quantizer_log_level", mode="before")
    @classmethod
    def parse_quantizer_log_level(cls, value):
        if isinstance(value, str):
            return ir_quantizer.LogLevel.from_string(value)
        return value

    @model_validator(mode="after")
    def validate_input_arguments(self):
        if (
            self._model_framework == TensorflowFrameworkInfo.name
            or self._model_framework == PytorchFrameworkInfo.name
        ):
            if not self.input_tensors:
                raise ValueError(
                    "Input tensor name(s) and dimension(s) details required for '{}' model.".format(
                        self._model_framework
                    )
                )
            for input_tensor in self.input_tensors:
                if not input_tensor.source_model_input_shape:
                    raise ValueError("Dimension is needed for input tensor {}.".format(input_tensor.name))
        if self._model_framework == TensorflowFrameworkInfo.name:
            if not self.output_tensors:
                raise ValueError(
                    "Output tensor name(s) and dimension(s) details required for '{}' model.".format(
                        self._model_framework
                    )
                )

        # Only one of the 'package_name', 'op_package_config' can be specified
        if self.op_package_config and self.package_name:
            raise ValueError(
                "'package_name' and 'op_package_config' are "
                "mutually exclusive options."
                "Only one of them can be specified."
            )

        # Check if quantization override file is loadable.
        if self.quantization_overrides:
            import json

            try:
                with open(self.quantization_overrides) as override_file:
                    _ = json.load(override_file)
            except Exception as e:
                raise Exception("Failed to load quantization override file with error: {}".format(str(e)))

        if self.lora_weight_list:
            input_shape_info: InputShapeInfo = InputShapeArgParser(self.input_dim).input_shape_info
            if input_shape_info.has_dynamic_shapes:
                raise Exception("Dynamic tensors are not supported with LoRA with this tool.")

        if self.quant_updatable_mode and not self.lora_weight_list:
            raise ValueError("Quant updatable mode requires LoRA weight list value.")

        return self

    def get_model_framework(self):
        return self._model_framework


class ConverterOutputConfig(AISWBaseModel):
    ir_graph: Any
    framework: str
    dlc_backend_config: DLCBackendConfig


class ConverterModuleSchemaV1(ModuleSchema):
    _VERSION = ModuleSchemaVersion(major=0, minor=4, patch=0)
    _BACKENDS = None
    name: Literal["ConverterModule"] = "ConverterModule"
    path: Path = Path(__file__)
    arguments: ConverterInputConfig
    outputs: Optional[ConverterOutputConfig] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


@expect_module_compliance
class QAIRTConverter(Module):
    """User interface class for model conversion API."""

    _SCHEMA = ConverterModuleSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger: logging.Logger = None) -> None:
        """Args:
        logger: A logger instance to be used by the Converter module
        """
        if logger:
            self._logger = QAIRTLogger.get_logger("ConverterLogger", parent_logger=logger)
        else:
            self._logger = QAIRTLogger.get_logger(
                "ConverterLogger", level="INFO", formatter_val="extended", handler_list=["dev_console"]
            )
        converter_utils.LOGGER = self._logger
        # Change this log level to debug Converter related issues
        converter_utils.LOG_LEVEL = logging.ERROR
        self._debug_level = QAIRTLogger.get_default_logging_level("ConverterLogger")

    @classmethod
    def _get_args(cls, config: ConverterInputConfig) -> dict:
        """This method accepts converter input config and return arguments in a dictionary.
        1. Converts arguments names to converter internal names.
        2. Sets default values for suppressed arguments.
        3. Converts v1 args to v2. See Converter API for more details.

        Args:
            config: Converter input arguments config.

        Returns:
            Return dictionary containing arguments.
        """
        option_dict = config.model_dump()
        option_dict["input_network"] = str(option_dict["input_network"])

        option_dict["custom_op_config_paths"] = option_dict.pop("op_package_config")
        if option_dict["custom_op_config_paths"]:
            option_dict["custom_op_config_paths"] = [
                str(item) for item in option_dict["custom_op_config_paths"]
            ]

        if option_dict["converter_op_package_lib"]:
            option_dict["converter_op_package_lib"] = ",".join(
                [str(item) for item in option_dict["converter_op_package_lib"]]
            )
        else:
            option_dict["converter_op_package_lib"] = ""

        option_dict["copyright_file"] = option_dict.pop("copyright_file")
        option_dict["float_bitwidth"] = option_dict.pop("float_bitwidth")
        option_dict["float_bias_bitwidth"] = option_dict.pop("float_bias_bitwidth")
        option_dict["model_version"] = option_dict.pop("model_version")
        option_dict["output_path"] = option_dict.pop("output_path")
        option_dict["package_name"] = option_dict.pop("package_name")
        option_dict["quantization_overrides"] = option_dict.pop("quantization_overrides")
        option_dict["lora_weight_list"] = option_dict.pop("lora_weight_list")
        option_dict["quant_updatable_mode"] = option_dict.pop("quant_updatable_mode")
        option_dict["export_format"] = option_dict.pop("export_format")
        option_dict["calc_static_encodings"] = option_dict.pop("calc_static_encodings")
        option_dict["quantizer_log"] = option_dict.pop("quantizer_log")
        option_dict["quantizer_log_level"] = option_dict.pop("quantizer_log_level")
        option_dict["use_quantize_v2"] = option_dict.pop("use_quantize_v2")
        option_dict["pack_4_bit_weights"] = option_dict.pop("pack_4_bit_weights")
        option_dict["disable_dynamic_16_bit_weights"] = option_dict.pop("disable_dynamic_16_bit_weights")

        if option_dict["preserve_io_datatype"] is not None:
            preserve_io_datatype = option_dict.pop("preserve_io_datatype")
            if preserve_io_datatype == "all":
                option_dict["preserve_io_datatype"] = [[]]
            else:
                option_dict["preserve_io_datatype"] = [preserve_io_datatype]

        # unpack input/output tensors.
        output_tensors = option_dict.pop("output_tensors")
        input_tensors = option_dict.pop("input_tensors")
        graph_specific_input_dims = option_dict.pop("graph_specific_input_dims")
        out_names = []
        output_layout = []
        desired_output_layout = []
        input_layout = []
        input_dtype = []
        input_encoding = []

        if input_tensors:
            for input_tensor in input_tensors:
                # Only add input_dtype if explicitly provided by user (not None)
                if input_tensor["source_model_input_datatype"] is not None:
                    input_dtype.append([input_tensor["name"], input_tensor["source_model_input_datatype"]])
                if input_tensor["source_model_input_layout"]:
                    input_layout.append([input_tensor["name"], input_tensor["source_model_input_layout"]])
                if input_tensor["desired_input_color_encoding"]:
                    input_encoding.append(
                        [input_tensor["name"], input_tensor["desired_input_color_encoding"]]
                    )

        option_dict["input_dim"] = graph_specific_input_dims if graph_specific_input_dims else config.input_dim

        option_dict["input_dtype"] = input_dtype
        option_dict["input_layout"] = input_layout
        option_dict["input_encoding"] = input_encoding

        io_config = option_dict.pop("io_config")
        option_dict["io_config"] = (
            io_config
            if io_config
            else [
                tensor
                for tensor in (input_tensors or []) + (output_tensors or [])
                if (
                    tensor.get("desired_model_input_datatype")
                    or tensor.get("desired_model_input_layout")
                    or tensor.get("optional")
                    or tensor.get("desired_model_output_datatype")
                    or tensor.get("desired_model_output_layout")
                )
            ]
        )

        if output_tensors:
            for output_tensor in output_tensors:
                out_names.append(output_tensor["name"])
                if output_tensor["source_model_output_layout"]:
                    output_layout.append([output_tensor["name"], output_tensor["source_model_output_layout"]])
                if output_tensor["desired_model_output_layout"]:
                    desired_output_layout.append(
                        [output_tensor["name"], output_tensor["desired_model_output_layout"]]
                    )

        option_dict["out_names"] = out_names
        option_dict["output_layout"] = output_layout
        option_dict["desired_io_layout"] = desired_output_layout

        # Map user input arguments name to converter internal names.
        option_dict["no_simplification"] = not option_dict.pop("onnx_simplification")
        option_dict["batch"] = option_dict.pop("onnx_batch")
        option_dict["validate_models"] = option_dict.pop("tf_validate_models")
        option_dict["define_symbol"] = option_dict.pop("onnx_define_symbol")
        option_dict["defer_loading"] = option_dict.pop("onnx_defer_loading")
        option_dict["disable_defer_loading"] = not option_dict["defer_loading"]
        option_dict["no_optimization"] = option_dict.pop("tf_no_optimization")
        option_dict["show_unconsumed_nodes"] = option_dict.pop("tf_show_unconsumed_nodes")
        option_dict["saved_model_tag"] = option_dict.pop("tf_saved_model_tag")
        option_dict["saved_model_signature_key"] = option_dict.pop("tf_saved_model_signature_key")
        option_dict["signature_name"] = option_dict.pop("tflite_signature_name")

        # Suppressed arguments
        option_dict["dump_inferred_model"] = False
        option_dict["dump_io_config_template"] = False
        option_dict["dump_value_info"] = False
        option_dict["disable_preserve_io"] = False
        option_dict["keep_int64_inputs"] = False
        option_dict["keep_quant_nodes"] = False
        option_dict["use_onnx_relay"] = False
        option_dict["disable_match_lstms"] = False
        option_dict["expand_lstm_op_structure"] = False
        option_dict["pytorch_custom_op_lib"] = ""
        option_dict["dump_relay"] = ""
        option_dict["partial_graph_input_name"] = []
        option_dict["qairt_converter_api"] = True

        args = convert_args_v2_to_v1(option_dict)

        return args

    def convert(self, config: ConverterInputConfig) -> ConverterOutputConfig:
        """This method accepts a converter input config containing a framework model and produces a
        converter output config containing an IRGraph.

        Args:
            config: "ConverterInputConfig" object containing converter module input arguments.

        Returns:
            "ConverterOutputConfig" object containing IRgraph and framework.

        Examples:
            >>> input_network = "path/to/model"
            >>> converter_module  = QAIRTConverter()
            >>> out_config = converter_module.convert(ConverterInputConfig(input_network=input_network))
        """
        args = self._get_args(config)

        # Add log level to args so that internal libs use same log level as API.
        args["debug"] = self._debug_level
        dlc_backend_config = DLCBackendConfig(
            copyright_file=args["copyright_file"],
            float_bitwidth=args["float_bitwidth"],
            float_bias_bitwidth=args["float_bias_bitwidth"],
            model_version=args["model_version"],
            output_path=args["output_path"],
            package_name=args["package_name"],
            quantization_overrides=args["quantization_overrides"],
            export_format=args["export_format"],
            calc_static_encodings=args["calc_static_encodings"],
            quantizer_log=args["quantizer_log"],
            quantizer_log_level=args["quantizer_log_level"],
            use_quantize_v2=args["use_quantize_v2"],
            pack_4_bit_weights=args["pack_4_bit_weights"],
            disable_dynamic_16_bit_weights=args["disable_dynamic_16_bit_weights"],
        )

        try:
            num_graph_configs = get_num_graph_configs(args)
            if num_graph_configs != 1:
                raise NotImplementedError(
                    "Multi configuration is not directly supported in the Converter Module,"
                    "Please refer example_network_specialization.py in qairt-tools for Network Specialization Implementation"
                )

            framework = config.get_model_framework()

            if framework == PytorchFrameworkInfo.name:
                ts_to_onnx(args)
                framework = OnnxFrameworkInfo.name

            converter = get_frontend_converter(framework, args)
            converter.qairt_converter = True
            ir_graph = converter.convert()
            # TODO: Generate dlc here when AISW-91995: Enable serialization of IRgraph is enabled.
            output_config = ConverterOutputConfig(
                ir_graph=ir_graph, framework=framework, dlc_backend_config=dlc_backend_config
            )

            # validate original model vs framework optimized model. See Converter
            # help for more details.
            self._validate_framework_optimizations(converter)

            return output_config

        except Exception as e:
            self._logger.error("Model conversion failed.")
            raise e

    def _validate_framework_optimizations(self, converter: ConverterFrontend) -> None:
        """Validates the original framework model against a model that has undergone
        frontend optimizations.

        #TODO: Need a better description of this from the converter team

        Args:
            converter: Frontend converter instance of

        Returns: None
        """
        # Validate output of original model with frontend optimized model
        if not hasattr(converter, "validator"):
            self._logger.warning(
                f"Validation is not supported for this frontend: {converter.__class__.__name__}"
            )
            return

        if validator := getattr(converter, "validator", None):
            try:
                results = validator.validate()
                for result in results:
                    self._logger.info(result)
            except Exception as e:
                self._logger.warning(
                    "Model conversion is completed but error encountered during validation : {}".format(
                        str(e)
                    )
                )

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        """Sets converter log level.
        Args:
            debug_level: LOGLEVEL.DEBUG enables DEBUG and higher level messages.
               LOGLEVEL.INFO enables INFO and higher level messages.
            **kwargs:

        Returns:
            bool: 'True' if debugging is enabled else return 'False'.
        """
        if debug_level < LOGLEVEL.INFO:
            return False
        self._debug_level = debug_level
        converter_utils.setup_logging(self._debug_level)
        return True

    @property
    def _schema(self):
        return self._SCHEMA

    def get_logger(self) -> Any:
        return self._logger

    def properties(self) -> Dict[str, Any]:
        return self._schema.model_json_schema()


def get_input_dimensions(
        input_tensors: Optional[List[InputTensorConfig]] = None,
        io_config: Optional[Union[str, os.PathLike]] = None,
    ) -> List[Tuple[str, Union[Tuple[int, ...], List[Tuple[int, ...]]]]]:
    """Retrieves input tensor dimensions from either a list of input tensor or a YAML I/O config file.
    
    When both `io_config` and `input_tensors` are provided, `io_config` takes precedence for determining
    the input dimensions.

    Args:
        input_tensors (Optional[List[InputTensorConfig]]): List of input tensor configuration objects.
        io_config (Optional[Union[str, os.PathLike]]): Path to a YAML file containing I/O configuration.

    Returns:
        List[Tuple[str, Union[Tuple[int, ...], List[Tuple[int, ...]]]]]: A list of tuples, each containing
        the tensor nameand its shape as a tuple or list of tuples.

    Raises:
        ValueError: If shape parsing fails for any input tensor or YAML configuration.
    """
    logger = QAIRTLogger.get_logger("ConverterLogger")
    input_dims = []

    if io_config:
        logger.info(f"Attempting to retrieve input dimensions from I/O config")
        try:
            input_dims = get_input_dims_from_yaml(io_config)
        except Exception as e:
            raise ValueError(f"Error parsing YAML I/O config: {e}")
    elif input_tensors:
        logger.info(f"Attempting to retrieve input dimensions from Input Tensors")
        for input_tensor in input_tensors:
            if input_tensor.source_model_input_shape:
                shapes = input_tensor.source_model_input_shape
                if isinstance(shapes, str):
                    shapes = [shapes]

                parsed_shapes = []
                for shape_str in shapes:
                    shape_parts = [s.strip() for s in shape_str.split("),")]
                    for part in shape_parts:
                        if not part.endswith(")"):
                            part += ")"
                        try:
                            parsed_shape = parse_shape(part)
                            parsed_shapes.append(parsed_shape)
                        except Exception as e:
                            raise ValueError(
                                f"Error parsing shape '{part}' for tensor '{input_tensor.name}': {e}"
                            )

                shape = parsed_shapes if len(parsed_shapes) > 1 else parsed_shapes[0]
                input_dims.append((input_tensor.name, shape))

    return input_dims
