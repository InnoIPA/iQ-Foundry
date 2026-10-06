# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Light Weight Quantizer Wrapper  """
# pylint: disable=import-error

import os
import json
from typing import Any, Dict
from qti.aisw.converters.common.utils.converter_utils import *
from .ir_graph_updater import IrGraphUpdater
from qti.aisw.converters.common import modeltools


class LWQWrapper:
    """
    Class for exporting a DLC with embedded encodings and updated weights. Optionally the DLC can be quantized as well.
    """

    # pylint:disable=too-many-locals
    @staticmethod
    def export(path: str, filename_prefix: str, dlc_path: str,
               weight_file_path: str = None, encoding_path: str = None, quantize_dlc: bool = True,
               activation_bw: int = 8, float_bias_bw: int = 32):
        """
        Export model to DLC. Requires a DLC file as input, generated from model preparer pro.
        The exported DLC will contain encodings and weights associated with the model in quantsim.

        :param path: Path to save exported DLC file
        :param filename_prefix: Filename to save exported DLC file with. The file will end in .dlc extension
        :param dlc_path: Path to the input DLC file generated during model preparation.
        :param weight_file_path: Path to weight file in safetensors file.
        :param encoding_path: Encoding file path. This contains the encoding to be applied on ir_graph.
        :param quantize_dlc: True if the exported dlc should be quantized, False otherwise
        :param activation_bw: activation bw to be used to quantized scalar values.
        :param float_bias_bw: float bias bitwidth, 0, 16 or 32 (default 32)
        """
        # below two lines cannot be merged into one because of a known QNN bug. model_reader must be kept alive when
        # updating the ir_graph using ir_graph_updater

        if (weight_file_path is None and encoding_path is not None) or (weight_file_path is not None and encoding_path is None):
            raise RuntimeError('Provide both weight_file_path and encoding_path (or) provide None!')

        model_reader = LWQWrapper.get_model_reader_for_dlc(dlc_path)
        dlc_metadata = {'model_version': model_reader.custom_model_version(), 'copyright_str': model_reader.copyright(),
                        'converter_command': model_reader.converter_command()}

        ir_graph = model_reader.get_ir_graph()

        quantized_ir_graph = LWQWrapper._light_weight_quantize(ir_graph, weight_file_path, encoding_path,
                                                               quantize_dlc, activation_bw, float_bias_bw)

        LWQWrapper.export_dlc(quantized_ir_graph, os.path.join(path, filename_prefix + '.dlc'), dlc_metadata)

    @staticmethod
    def _light_weight_quantize(ir_graph, weight_file_path: str, encoding_path: str,
                               quantize_dlc: bool, activation_bw: int,
                               float_bias_bw: int):
        activation_encodings = None
        param_encodings = None

        ir_graph_updater = IrGraphUpdater(ir_graph, weight_file_path)

        if encoding_path is None:
            # Rely on embedded encodings in IR Graph when no weight file path or encoding path is given
            log_info('encoding_path is not provided. Expecting IRGraph to have embedded encodings in order to quantize!')
        else:
            with open(encoding_path, 'r') as f:
                enc = json.load(f)
            activation_encodings, param_encodings = enc['activation_encodings'], enc['param_encodings']
            # clears previously filled encoding before applying new encodings
            ir_graph_updater.reset_encodings()

        if weight_file_path is not None:
            # Update tensor data is only needed when weight file path is given
            ir_graph_updater.update_tensor_data()

        ir_graph_updater.set_encodings(activation_encodings, param_encodings, activation_bw)

        assert float_bias_bw in [0, 16, 32], (f'float_bias_bw of {float_bias_bw} is not supported!'
                                              f' float_bias_bw can be either 0, 16 or 32')
        # pylint: disable = unused-variable
        ir_graph_updater.quantize_ir_graph(quantize_dlc=quantize_dlc, float_bias_bw=float_bias_bw)

        return ir_graph_updater.ir_graph

    @staticmethod
    def get_model_reader_for_dlc(dlc_path: str) -> 'IrDlcReader':
        """
        Get a model reader for a dlc file.

        :param dlc_path: Path to dlc file
        :return: Model reader
        """
        if not os.path.isfile(dlc_path):
            log_error('Invalid dlc path specified: %s. Ensure that dlc_path is the full path to the .dlc file, '
                      'including the filename and extension.', dlc_path)
            raise AssertionError(f'Invalid dlc path specified: {dlc_path}. Ensure that dlc_path is the full path to '
                                 f'the .dlc file, including the filename and extension.')
        model_reader = modeltools.IrDlcReader()
        model_reader.open(dlc_path)
        return model_reader

    @staticmethod
    def export_dlc(ir_graph, dlc_path: str, dlc_metadata: Dict[str, Any]):
        """
        Generate a dlc from ir_graph.

        :param ir_graph: Ir graph to serialize
        :param dlc_path: Path to export dlc
        :return: Dlc serialized from ir_graph
        """

        dlc_serializer = modeltools.IrDlcSerializer(dlc_path,
                                                    dlc_metadata['copyright_str'],
                                                    dlc_metadata['model_version'],
                                                    dlc_metadata['converter_command'])
        dlc_serializer.initialize()
        dlc_serializer.serialize(ir_graph)
        dlc_serializer.finish()

