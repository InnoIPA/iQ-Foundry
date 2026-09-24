# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging
import os
from argparse import Namespace
from pathlib import Path

import onnx
from pydantic import DirectoryPath, Field, FilePath


try:
    import torch
    from qti.aisw.emitter.utils.torch_utils import TorchModelMetadata, get_model_metadata
except ImportError:
    pass

from typing import Any, Dict, Literal, Optional, Tuple

from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.emitter.utils import onnx_utils
from qti.aisw.emitter.utils.model_preparer_utils import count_inputs
from qti.aisw.emitter.utils import onnx_saver

# Module Imports
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
    expect_module_compliance,
)
from qti.aisw.tools.core.modules.converter.constants import LOGLEVEL


class TorchModelInfo(AISWBaseModel):
    """Pydantic class for source model information"""

    order_inputs: bool = Field(default=False, description='Flag to indicate if input order should '
                                                          'be same in prepared model as in the '
                                                          'original model.')
    order_outputs: bool = Field(default=False, description='Flag to indicate if output order '
                                                           'should be same in prepared model as '
                                                           'in the original model.')
    input_names: Optional[list[str]] = Field(default=None, description='Input names that need to '
                                                                       'appear in the generated '
                                                                       'onnx model.')
    output_names: Optional[list[str]] = Field(default=None, description='Output names that needs'
                                                                        'to appear in the '
                                                                        'generated onnx model.')
    block_names: Optional[list[str]] = Field(default=None, description='Name of the blocks in the '
                                                                       'source model for which '
                                                                       'class mapping needs to be '
                                                                       'extracted')
    return_prepare_model: bool = Field(default=False, description='Flag to indicate if in-memory '
                                                                  'prepared model needs to be '
                                                                  'returned in MPP flow.')


class SourceModelConverterInputConfig(AISWBaseModel):
    """Pydantic class for input config of SourceModelConverter"""

    source_model: Any = Field(description="Source Model in memory")
    dummy_input: Any = Field(description="Dummy model input")
    path: DirectoryPath = Field(default=".", description="Path to save the exported onnx model")
    filename: str = Field(default="onnx_model", description="Filename to save exported onnx model")
    export_args: Optional[Dict[str, Any]] = Field(default=None, description='Optional export '
                                                                            'arguments for '
                                                                            'onnx export')
    skipped_optimizers: Optional[list[str]] = Field(default=None, description="optimizer names to "
                                                                              "disable during "
                                                                              "onnx "
                                                                              "simplification")
    model_info: TorchModelInfo = Field(default_factory=TorchModelInfo, description='Dataclass '
                                                                                   'with '
                                                                                   'source model info.')


class SourceModelConverterOutputConfig(AISWBaseModel):
    """Pydantic class for output config of SourceModelConverter"""

    onnx_path: FilePath = Field(description="Path of the onnx model file")
    source_model_metadata: TorchModelMetadata = Field(description="Metadata with "
                                                                  "information extracted "
                                                                  "from source model")


class SourceModelConverterModuleSchemaV1(ModuleSchema):
    """Schema class for SourceModelConverter"""

    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)
    _BACKENDS = None
    name: Literal["SourceModelConverterModule"] = "SourceModelConverterModule"
    path: Path = Path(__file__)
    arguments: SourceModelConverterInputConfig
    outputs: Optional[SourceModelConverterOutputConfig] = None
    backends: Optional[list[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


@expect_module_compliance
class SourceModelConverter(Module):
    """User interface class for Source Model Converter API"""

    _SCHEMA = SourceModelConverterModuleSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger=None):
        """Constructor method"""
        if not logger:
            logger = logging.getLogger("SourceModelConverterModule")
        converter_utils.LOGGER = logger
        super().__init__(logger)
        self._debug_level = LOGLEVEL.INFO

    @classmethod
    def _get_args(cls, config: SourceModelConverterInputConfig) -> Tuple[Namespace, Namespace]:
        """This method accepts converter input config and return arguments in namespace object.

        Args:
            config: Converter input arguments config.

        Returns:
            Return namespace object containing arguments.
        """
        option_dict = config.model_dump()
        model_info = option_dict.pop('model_info')
        args = Namespace(**option_dict)
        model_info = Namespace(**model_info)

        return args, model_info

    def _validate_inputs_for_onnx_model(self, dummy_input: Any,
                                        onnx_model: onnx.ModelProto, order_inputs: bool):
        """Validate dummy inputs to check that they match with the number of inputs in the onnx
        graph. In the case that order_inputs is False, the number of dummy inputs may differ from
        the number of ir graph inputs if optional arguments are present in the original model
        definition.

        Args:
            dummy_input: Inputs to validate
            onnx_model: ONNX model to check inputs for
            order_inputs: Flag to specify if ordering of input is required.
        """
        dummy_input_count = count_inputs(dummy_input, True)
        if len(onnx_model.graph.input) != dummy_input_count:
            if order_inputs:
                error_msg = (f'Number of onnx graph inputs ({len(onnx_model.graph.input)}) does '
                             f'not match number of dummy inputs provided ({dummy_input_count}). '
                             f'Unable to align dummy inputs with onnx graph inputs. '
                             f'This may be a result of optional arguments not specified in dummy '
                             f'inputs. '
                             f'Include any such inputs in dummy inputs.')
                self._logger.error(error_msg)
                raise AssertionError(error_msg)

            self._logger.warning('Number of onnx graph inputs (%s) does not match number of '
                                 'flattened dummy inputs provided (%s). This may be a result of '
                                 'optional arguments not specified in dummy inputs. The prepared '
                                 'model will expect any such optional arguments in the original '
                                 'model as required arguments.', len(onnx_model.graph.input),
                                 dummy_input_count)

    def torch2onnx(self, config: SourceModelConverterInputConfig) -> SourceModelConverterOutputConfig:
        """This method accepts a converter input config containing a framework model and produces a
        converter output config containing an onnx model path.

        Args:
            config: "SourceModelConverterInputConfig" object containing converter module input
            arguments.

        Returns:
            "SourceModelConverterOutputConfig" object containing onnx model path and source model
            metadata.

        Examples:
            >>> model = torch_model
            >>> dummy_input = torch.randn(1, 3, 224, 224)
            >>> converter_module  = SourceModelConverter()
            >>> out_config = converter_module.convert(SourceModelConverterInputConfig(
            >>> source_model=model, dummy_input=dummy_input))
        """
        config, model_info = SourceModelConverter._get_args(config)
        try:
            export_path = os.path.join(config.path, config.filename + '.onnx')
            source_model = config.source_model
            onnx_export_args = onnx_utils.prepare_torch_to_onnx_export_args(config.export_args,
                                                                            model_info.input_names,
                                                                            model_info.output_names)
            model_metadata = get_model_metadata(source_model, config.dummy_input, model_info)

            # Allows exporting layers/model weights with same naming convention as that of Pytorch
            self._logger.info("Using OnnxSaver for torch2onnx export, which creates "
                              "onnx model with pytorch layer/weight names")
            onnx_saver.OnnxSaver.create_onnx_model_with_pytorch_layer_names(export_path, source_model,
                                                                            config.dummy_input,
                                                                            onnx_export_args=onnx_export_args)

            '''
            NOTE: Below logic is basically to support model preparation for multiple LoRA adapter
            model correctly. For multiple LoRA adapter case, each LoRA_B has zero weight,
            and it causes initializer pruning in ONNX graph. With initializer pruned ONNX
            model, prepared model is not generated as we expect (Introduce DynamicLinear instead
            of nn.Linear). To prevent DynamicLinear, we are restoring pruned initializer if
            explicit disablement of `eliminate_duplicate_initializer` is passed
            '''
            if (config.skipped_optimizers and "eliminate_duplicate_initializer" in
                    config.skipped_optimizers):
                self._logger.info("Save ONNX Model after restoring pruned initializers")
                onnx_utils.save_initializer_restored_onnx_graph(export_path, export_path)

            onnx_model = onnx.load(export_path)

            self._validate_inputs_for_onnx_model(config.dummy_input, onnx_model,
                                                 model_info.order_inputs)

            # get onnx model input names
            model_metadata.onnx_model_input_name = [node.name for node in onnx_model.graph.input]

            del onnx_model

            output_config = SourceModelConverterOutputConfig(onnx_path=export_path,
                                                             source_model_metadata=model_metadata)
        except Exception as e:
            self._logger.error("Onnx model generation failed.")
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
            **kwargs: keyword arguments

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
