# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import logging
import inspect
from typing import Literal, Optional, List, Dict, Any, Union
from pydantic import model_validator, FilePath, Field, ConfigDict, field_serializer, field_validator

# Module Imports
from qti.aisw.tools.core.modules.api import Module, ModuleSchema, AISWBaseModel, \
    ModuleSchemaVersion, \
    expect_module_compliance
from qti.aisw.tools.core.modules.converter.common import BackendInfoConfig
from qti.aisw.tools.core.modules.converter.aimet_config_definition import (QuantSimConfig, AdaRoundConfig,
                                                                           AMPConfig, AutoQuantConfig)

# Converter Imports
from qti.aisw.converters.common import ir_quantizer
from qti.aisw.converters.common.dlc_quantizer import DLCQuantizer
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.converters.common.backend_awareness import BackendInfo
from qti.aisw.tools.core.modules.converter.constants import LOGLEVEL


quant_calibration = Literal["min-max", "sqnr", "entropy", "mse", "percentile"]
quant_schema = Literal["asymmetric", "symmetric", "unsignedsymmetric", "signedasymmetric"]


class QuantizerInputConfig(AISWBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_dlc: FilePath = Field(description="Path to input dlc file.")
    output_dlc: str = Field(default="",
                            description="Path to the dlc container containing the model for which "
                                        "fixed-point encoding metadata should be generated.")
    input_list: Optional[FilePath] = Field(default=None, description="Path to a file specifying "
                                                                     "the input raw data.")
    preserve_io_datatype: Optional[Union[str, List[str]]] = Field(default=None,
                                                                  description="Use this option to preserve "
                                                                  "IO datatype of the source model."
                                                                  "To preserve datatype for all inputs and "
                                                                  "outputs, provide option as, "
                                                                  "preserve_io_datatype='all'."
                                                                  "To preserve datatype for select inputs "
                                                                  "and outputs,"
                                                                  "preserve_io_datatype=[input1,input2,output1]")
    float_fallback: bool = Field(default=False,
                                 description="Option to enable fallback to floating point (FP) "
                                             "instead of fixed point.")
    algorithms: Optional[List[str]] = Field(default=[], description="Option to select "
                                                                    "optimization algorithm.")
    bias_bitwidth: Literal[8, 32] = Field(default=8,
                                          description="Option to select the bitwidth to use when "
                                                      "quantizing the biases, either 8 (default) "
                                                      "or 32.")
    act_bitwidth: Literal[8, 16] = Field(default=8,
                                         description="Option to select the bitwidth to use when "
                                                     "quantizing the activations, either 8 ("
                                                     "default) or 16.")
    weights_bitwidth: Literal[4, 8, 16] = Field(default=8,
                                                description="Option to select the bitwidth to use when"
                                                            "quantizing the weights, either 4, 8 ("
                                                            "default) or 16.")
    float_bitwidth: Literal[32, 16] = Field(default=32,
                                            description="Convert the graph to specified float "
                                                        "bitwidth.")
    float_bias_bitwidth: Optional[Literal[16, 32]] = Field(default=None,
                                                           description="Option to select the "
                                                                       "bitwidth to use for float "
                                                                       "bias tensor.")

    keep_weights_quantized: bool = Field(default=False, description="Use this option to keep the "
                                                                    "weights quantized even"
                                                                    " when the output of the op is "
                                                                    "in floating point.")

    adjust_bias_encoding: bool = Field(default=False, description="Use --adjust_bias_encoding option to modify bias encoding"
                                                                  "and weight encoding to ensure that the bias value is in"
                                                                  "the range of the bias encoding. This option is only applicable"
                                                                  "for per-channel quantized weights. \n"
                                                                  "NOTE: This may result in clipping of the weight values")

    advanced_quantizer_config: Optional[Union[AdaRoundConfig, AMPConfig, AutoQuantConfig, QuantSimConfig]] \
        = Field(default=None, description="Config options for advance AIMET quantizers")
    ignore_encodings: bool = Field(default=False,
                                   description="Use only quantizer generated encodings, "
                                               "ignoring any user"
                                               " or model provided encodings.")
    use_per_channel_quantization: bool = Field(default=False,
                                               description="option to enable per-channel "
                                                           "quantization for convolution-based op "
                                                           "weights.")
    use_per_row_quantization: bool = Field(default=False,
                                           description="option to enable row wise quantization of"
                                                       " Matmul and "
                                                       "FullyConnected ops.")
    enable_per_row_quantized_bias: bool = Field(default=False,
                                           description="Use this option to enable rowwise "
                                           "quantization of bias for FullyConnected ops, when "
                                           "weights are per-row quantized.")
    use_native_input_files: bool = Field(default=False,
                                         description="Boolean flag to indicate how to read input files.\n"
                                                     "False(default): Reads inputs as floats and "
                                                     "quantizes if necessary based on "
                                                     " quantization parameters in the model.\n"
                                                     "True: Reads inputs assuming the data type to "
                                                     "be native to the Model.")
    use_native_output_files: bool = Field(default=False,
                                          description="Boolean flag to indicate the data type of the"
                                                      "output files.\n False(default): Output the "
                                                      "file as floats.\n True: Outputs the file "
                                                      "that is native to the model.")
    restrict_quantization_steps: Optional[List[str]] = Field(default=[],
                                                             description="Specifies the number of "
                                                                         "steps to use for computing"
                                                                         " quantization encodings")
    act_quantizer_calibration: quant_calibration = Field(default="min-max",
                                                         description="Specify which quantization "
                                                                     "calibration method to "
                                                                     "use for activations supported"
                                                                     " values: min-max (default), "
                                                                     "sqnr, entropy, mse, "
                                                                     "percentile.")
    param_quantizer_calibration: quant_calibration = Field(default="min-max",
                                                           description="Specify which quantization "
                                                                       "calibration method "
                                                                       " to use for activations"
                                                                       " supported values: min-max "
                                                                       "(default), sqnr, entropy,"
                                                                       " mse, percentile.")
    act_quantizer_schema: quant_schema = Field(default="asymmetric",
                                               description="Specify which quantization schema to use"
                                                           " for activations. Supported values: "
                                                           "asymmetric (default), symmetric, unsignedsymmetric, signedasymmetric")
    param_quantizer_schema: quant_schema = Field(default="asymmetric",
                                                 description="Specify which quantization schema to "
                                                             "use for parameters. Supported values:"
                                                             " asymmetric (default), symmetric, unsignedsymmetric, signedasymmetric")
    percentile_calibration_value: float = Field(default=99.99, ge=90.0, le=100,
                                                description=" Specify the percentile value to be "
                                                            " used with Percentile calibration "
                                                            "method. "
                                                            "The specified float value must lie "
                                                            "within 90 and 100, default: 99.99")
    op_package_lib: Optional[List[str]] = Field(default=[],
                                                description="Option to pass list of op package "
                                                            "library for quantization.")
    backend_info: Optional[BackendInfoConfig] = Field(default=None,
                                                      description="Determines the backend to use "
                                                                  " for optimizations")
    dump_encoding_json: bool = Field(default=False,
                                     description="Dump encoding of all the tensors in a json file")
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

    @field_serializer("advanced_quantizer_config")
    def serialize_advanced_quantizer_config(self, advanced_quantizer_config):
        if advanced_quantizer_config is not None:
            config_dict = advanced_quantizer_config.model_dump()
            return config_dict
        return None

    @field_validator("quantizer_log_level", mode="before")
    @classmethod
    def parse_quantizer_log_level(cls, value):
        if isinstance(value, str):
            return ir_quantizer.LogLevel.from_string(value)
        return value

    @model_validator(mode="after")
    def validate_input_arguments(self):

        if self.float_fallback and (self.input_list or self.ignore_encodings):
            raise ValueError("'input_list' or 'ignore_encodings' option must not be provided when "
                             "'float_fallback' option enabled.")

        if not (self.float_fallback or self.advanced_quantizer_config) and not self.input_list:
            raise ValueError("Please supply input_list to quantize model.")

        return self


class QuantizerOutputConfig(AISWBaseModel):
    dlc_output: str
    encoding_json: Optional[FilePath] = None


class QuantizerModuleSchemaV1(ModuleSchema):
    _VERSION = ModuleSchemaVersion(major=0, minor=2, patch=0)
    _BACKENDS = None
    name: Literal["QuantizerModule"] = "QuantizerModule"
    path: Literal[str(inspect.getfile(os))] = str(inspect.getfile(os))
    arguments: QuantizerInputConfig
    outputs: Optional[QuantizerOutputConfig] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


@expect_module_compliance
class QAIRTQuantizer(Module):
    """
    User interface class for quantizer API
    """
    _SCHEMA = QuantizerModuleSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger=None):
        if not logger:
            logger = logging.getLogger("QuantizerLogger")
        converter_utils.LOGGER = logger
        super().__init__(logger)
        self._debug_level = LOGLEVEL.INFO

    @staticmethod
    def _get_args(config: QuantizerInputConfig) -> Dict[str, Any]:
        """
        This method accepts quantizer input config and return arguments in dictionary object.
        1. Converter arguments names to quantizer internal names.
        2. Sets default values for suppressed arguments.
        Args:
            config: Converter input arguments config.
        Returns:
            Dictionary containing arguments.
        """
        option_dict = config.model_dump()
        option_dict["input_dlc"] = str(option_dict["input_dlc"])
        option_dict["input_list"] = str(option_dict["input_list"]) if config.input_list else None
        # Unpack backend options
        backend_info = option_dict.pop("backend_info")
        if backend_info:
            option_dict["backend"] = backend_info["backend"]
            option_dict["soc_model"] = backend_info["soc_model"]
        else:
            option_dict["backend"] = None
            option_dict["soc_model"] = None

        if option_dict["float_bias_bitwidth"] is None:
            option_dict["float_bias_bitwidth"] = 0

        advanced_quantizer_config = option_dict.pop("advanced_quantizer_config")
        if os.environ.get("AIMET_ENV_PYTHON") is not None:
            option_dict["use_aimet_quantizer"] = True
            if advanced_quantizer_config:
                option_dict["aimet_config"] = advanced_quantizer_config
            elif os.environ.get("AIMET_VARIANT") == "torch-gpu":
                option_dict["aimet_config"] = {'quantsim': {'dataloader': None,
                                                            'iteration_size': None}}
            else:
                option_dict["use_aimet_quantizer"] = False
        else:
            option_dict["use_aimet_quantizer"] = False

        if option_dict["op_package_lib"]:
            option_dict["op_package_lib"] = ",".join(config.op_package_lib)
        else:
            option_dict["op_package_lib"] = ""

        # Check if output dlc path is supplied. If not dump output dlc at input dlc location.
        if not ("output_dlc" in option_dict and option_dict["output_dlc"]):
            input_dlc_dir, input_dlc_file_name = os.path.split(
                                                        os.path.abspath(option_dict["input_dlc"]))
            output_dlc = os.path.join(input_dlc_dir, "quantized_"+input_dlc_file_name)
            option_dict["output_dlc"] = output_dlc

        # Disable legacy quantizer
        option_dict["disable_legacy_quantizer"] = True

        option_dict["backend_info_obj"] = BackendInfo.get_instance(option_dict.pop("backend"),
                                                                   option_dict.pop("soc_model"))

        if option_dict["preserve_io_datatype"] is not None:
            preserve_io_datatype = option_dict.pop("preserve_io_datatype")
            if preserve_io_datatype == "all":
                option_dict["preserve_io_datatype"] = [[]]
            else:
                option_dict["preserve_io_datatype"] = [preserve_io_datatype]

        option_dict["should_compute_statics"] = option_dict.pop("calc_static_encodings")
        option_dict["log_file_name"] = option_dict.pop("quantizer_log")
        option_dict["log_level"] = option_dict.pop("quantizer_log_level")

        return option_dict

    def quantize(self, config: QuantizerInputConfig) -> QuantizerOutputConfig:
        """
          Quantizes an input DLC contained in the config and produces and output DLC path.

          Args:
              config (QuantizerInputConfig): contains quantizer module input arguments

          Returns:
              config (QuantizerOutputConfig): contains quantizer module output arguments

          Examples:
            >>> input_dlc = "/path/to/dlc"
            >>> quantizer = QAIRTQuantizer()
            >>> out_config = quantizer.quantize(QuantizerInputConfig(input_dlc=input_dlc))

        """
        # Convert pydantic model to dictionary
        args_dict = self._get_args(config)

        try:
            dlc_quantizer = DLCQuantizer(**args_dict)
            dlc_quantizer.quantize()

            quantizer_command = converter_utils.sanitize_args(args_dict,
                                                              args_to_ignore=['input_dlc', 'i',
                                                                              'output_dlc', 'o'])
            dlc_quantizer.save(quantizer_command)
        except Exception as e:
            self._logger.error("Quantization failed")
            raise e
        if config.dump_encoding_json:
            output_encoding_file = os.path.splitext(args_dict["output_dlc"])[0] + "_encoding.json"
            output_config = QuantizerOutputConfig(dlc_output=args_dict["output_dlc"],
                                                  encoding_json=output_encoding_file)
        else:
            output_config = QuantizerOutputConfig(dlc_output=args_dict["output_dlc"])

        return output_config

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        """Sets quantizer log level.

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
