# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


import logging
import os
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any
from pydantic import DirectoryPath, FilePath, Field

# Module Imports
from qti.aisw.tools.core.modules.api import Module, ModuleSchema, AISWBaseModel, \
    ModuleSchemaVersion, expect_module_compliance

# Converter Imports
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.tools.core.modules.converter.constants import LOGLEVEL
from qti.aisw.tools.light_weight_quantizer import LWQWrapper


class LightWeightQuantizerInputConfig(AISWBaseModel):
    """
    Input config class for the Light Weight Quantizer Module
    """

    path: DirectoryPath = Field(description="Path of the directory to save exported DLC file.")
    filename_prefix: str = Field(description="Filename to save exported DLC file with. The file will end in .dlc extension")
    dlc_path: FilePath = Field(description="Path to the input DLC file generated during model preparation.")
    weight_file_path: Optional[FilePath] = Field(default=None, description="Path to safetensors file dumped by Emitter, "
                                                                 "which contains weights and some model metadata.")
    encoding_path: Optional[FilePath] = Field(default=None, description="Encoding file path. This contains the encoding to be applied on ir_graph.")
    quantize_dlc: bool = Field(default=True, description="True if the exported dlc should be quantized, False otherwise")
    activation_bitwidth: Optional[Literal[32, 16, 8]] = Field(default=8,
                                                              description="activation bitwidth to be used to quantize scalar values.")
    float_bias_bitwidth: Optional[Literal[0, 16, 32]] = Field(default=32,
                                                              description="float bias bitwidth, 0, 16 or 32 (default 32)")


class LightWeightQuantizerOutputConfig(AISWBaseModel):
    """
    Output config class for the Light Weight Quantizer Module
    """
    dlc_path: str = Field(default=None, description="Path to the output DLC file.")


class LightWeightQuantizerModuleSchemaV1(ModuleSchema):
    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)
    _BACKENDS = None
    name: Literal["LightWeightQuantizerModule"] = "LightWeightQuantizerModule"
    path: Path = Path(__file__)
    arguments: LightWeightQuantizerInputConfig
    outputs: Optional[LightWeightQuantizerOutputConfig] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


@expect_module_compliance
class QAIRTLightWeightQuantizer(Module):
    """
    User interface class for model conversion API.
    """

    _SCHEMA = LightWeightQuantizerModuleSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger: Any = None) -> None:
        """
        Args:
            logger:
        """
        if not logger:
            logger = logging.getLogger("LWQLogger")
        converter_utils.LOGGER = logger
        super().__init__(logger)
        self._debug_level = LOGLEVEL.INFO

    def export(self, input_config: LightWeightQuantizerInputConfig) -> LightWeightQuantizerOutputConfig:
        """
        Applies the encoding file  and updated the weight on top of a given DLC file. Optionally can be quantized as well.

        Args:
            input_config (LightWeightQuantizerInputConfig): Input Config for the Light Weight Quantizer.

        Returns:
            LightWeightQuantizerOutputConfig: object containing the path of the exported DLC file.

        Example:
            >>> # Example to quantize a model using Light Weight Quantizer
            >>> from qti.aisw.tools.core.modules.light_weight_quantizer import LightWeightQuantizerInputConfig,\
                    QAIRTLightWeightQuantizer
            >>> lwq_input_config = LightWeightQuantizerInputConfig()
            >>> lwq_input_config.path = '/path/to/output_dir'
            >>> lwq_input_config.filename_prefix = 'lwq_quantized_dlc'
            >>> lwq_input_config.dlc_path = '/path/to/input/float_model.dlc'
            >>> lwq_input_config.weight_file_path = '/path/to/input/weight.safetensors'
            >>> lwq_input_config.encoding_path = '/path/to/input/encoding.json'
            >>> lwq_module = QAIRTLightWeightQuantizer()
            >>> lwq_output_config = lwq_module.export(lwq_input_config)
        """
        weight_file_path = str(input_config.weight_file_path) if input_config.weight_file_path is not None else None
        encoding_path = str(input_config.encoding_path) if input_config.encoding_path is not None else None
        LWQWrapper.export(input_config.path, input_config.filename_prefix, str(input_config.dlc_path),
                          weight_file_path, encoding_path, input_config.quantize_dlc,
                          input_config.activation_bitwidth, input_config.float_bias_bitwidth)

        output_dlc_path = os.path.join(input_config.path, input_config.filename_prefix + '.dlc')
        output = LightWeightQuantizerOutputConfig(dlc_path=output_dlc_path)
        return output

    def enable_debug(self, debug_level: int) -> Optional[bool]:
        """
        Sets converter log level.
        Args:
            debug_level: LOGLEVEL.VERBOSE enables VERBOSE and higher level messages.
               LOGLEVEL.DEBUG enables DEBUG and higher level messages.
               LOGLEVEL.DEBUG_3 enables DEBUG_3 and higher level messages.
               LOGLEVEL.DEBUG_2 enables DEBUG_2 and higher level messages.
               LOGLEVEL.DEBUG_1 enables DEBUG_1 and higher level messages.
               LOGLEVEL.INFO enables INFO and higher level messages.

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
        return self._logger

    def properties(self) -> Dict[str, Any]:
        return self._schema.model_json_schema()
