# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Class for updating IR graph based on weights file and encodings """
# pylint: disable=import-error
# pylint: disable=no-name-in-module
import json
import os
from typing import Dict, Set

import numpy as np
from safetensors.numpy import load as load_safetensor
from .qnn_defs import (is_qnn_data_movement_op, is_integer_dtype, is_fxp_qnn_dtype, has_dummy_bias,
                       VALID_ENCODING_TYPES, VALID_AXIS_ENCODING_TYPES)

from qti.aisw.emitter.ir_graph_op_handler import get_op_name
from qti.aisw.emitter.utils.axis_tracking_utils import get_transpose_order
from qti.aisw.emitter.utils.model_preparer_utils import PreparedModelInfo
from qti.aisw.converters.common.converter_ir.op_graph_optimizations import IROptimizations
from qti.aisw.converters.common import ir_graph as ir_graph_lib
from qti.aisw.converters.common import light_weight_ir_quantizer
from qti.aisw.converters.common import ir_quantizer

from qti.aisw.converters.common.utils.converter_utils import *
ir_graph_utils_helper = ir_graph_lib.PyIrGraph('utils')


class IrGraphUpdater:
    """
    Class for applying model weights and encodings from QuantSim to a corresponding IR graph.
    """
    def __init__(self, ir_graph, weight_file_path: str = None):
        """
        Constructor for IrGraphUpdater.

        :param ir_graph: Ir graph to update
        :param weight_file_path: Path to safetensor format weight file. Also contains the prepared model info.
        """
        self.ir_graph = ir_graph
        self.state_dict = None
        self.prepared_model_info = None
        self.sanitized_name_to_node_name_map = {}
        self.stateful_ops_to_sanitized_name_map = {}

        if weight_file_path is not None:
            if not os.path.isfile(weight_file_path):
                log_error('Invalid prepared model info path specified: %s. Ensure that prepared_model_info_path is the '
                          'full path to the .pkl file, including the filename and extension.', weight_file_path)
                raise AssertionError(f'Invalid prepared model info path specified: {weight_file_path}. Ensure that '
                                     f'prepared_model_info_path is the full path to the .pkl file, including the filename '
                                     f'and extension.')

            self.state_dict, metadata = self._get_metadata_and_state_dict(weight_file_path)
            self.prepared_model_info = PreparedModelInfo.load_from_metadata(metadata)
            self.populate_sanitized_name_to_node_name_map(self.ir_graph)
            # Dequantize only if the graph is a quantized one
            if self.prepared_model_info.is_quantized_graph:
                self._dequantize_ir_graph()
        else:
            self._dequantize_ir_graph()

    def _dequantize_ir_graph(self):
        opts = ir_quantizer.IrQuantizerOpts()
        quantizer = ir_quantizer.IrQuantizer(opts, self.ir_graph)
        quantizer.translate_quant_to_float_graph()

    def _get_metadata_and_state_dict(self, weight_file_path):
        """
        Gets the mpp_metadata and the state_dic from the safetensors weight flie.

        :param weight_file_path: Weight file in safetensors format

        """
        with open(weight_file_path, "rb") as f:
            data = f.read()

        # Get the header length to extract the metadata
        header_length = int.from_bytes(data[:8], "little", signed=False)
        meta_data = json.loads(data[8:8 + header_length].decode()).get('__metadata__', {})

        # Load the state dict and convert it to torch tensor
        state_dict = load_safetensor(data)

        return state_dict, meta_data

    def update_tensor_data(self):
        """
        Update static tensors in self.ir_graph using weights from model.
        """
        for param_name, data in self.state_dict.items():
            IrGraphUpdater.update_data_for_tensor(data, param_name, self.ir_graph,
                                                  self.prepared_model_info)

    @staticmethod
    def update_data_for_tensor(tensor_to_replace: np.ndarray, param_name: str, ir_graph, prepared_model_info):
        """
        Update static data for a particular tensor in ir_graph.

        :param tensor_to_replace: Numpy Data to replace with in ir_graph
        :param param_name: Name of parameter
        :param ir_graph: Ir graph to set encodings for
        :param prepared_model_info: PreparedModelInfo object generated during model preparation
        """
        if param_name in prepared_model_info.param_name_mapping:
            param_tensor_name, transform_order = prepared_model_info.param_name_mapping[param_name]
            ir_static_tensor = ir_graph.get_tensor(param_tensor_name)
            ir_tensor_data = ir_static_tensor.get_data()

            if transform_order is not None:
                # get the transpose order need to be applied to get it in ir graph axis format
                transform_order = get_transpose_order(transform_order, None)
                tensor_to_replace = tensor_to_replace.transpose(transform_order)

            if tensor_to_replace.shape == ir_tensor_data.shape and tensor_to_replace.dtype in [np.float16, np.float32]:
                ir_static_tensor.update_data(np.ascontiguousarray(tensor_to_replace.astype(np.float32)))
            elif tensor_to_replace.dtype not in [np.float16, np.float32]:
                log_warning(f'Weight tensor {param_name} in prepared model is neither float16 nor float32.')
            else:
                log_str = 'Weight tensors %s have different shapes in both models.' % (param_name,)
                log_error(log_str)
                raise Exception(log_str)
        else:
            log_warning(f'Param/buffer {param_name} not found in ir graph.')

    def reset_encodings(self):
        """
        Clears all the previously filled encodings on the ir_graph
        """
        for tensor_name, ir_tensor in self.ir_graph.get_tensor_map().items():
            ir_tensor.reset_encoding()

    # pylint:disable = too-many-locals, too-many-branches
    def set_encodings(self, activation_encodings: Dict, param_encodings: Dict, act_bw: int):
        """
        Update tensor encodings in self.ir_graph.

        :param activation_encodings: Activation encodings to set
        :param param_encodings: Param encodings to set
        :param act_bw: Default activation bitwidth to be applied to the constant tensor
        """
        static_tensor_encoding_added = set()

        is_embedded_ir = activation_encodings is None and param_encodings is None

        if not is_embedded_ir:
            self.set_given_encodings(activation_encodings, param_encodings, static_tensor_encoding_added)

        self.fill_missing_encodings(act_bw, static_tensor_encoding_added, is_embedded_ir)
        self.propagate_data_movement_encodings()

    @staticmethod
    def set_encoding_for_tensor(ir_tensor: ir_graph_lib.IrTensor, quant_info: ir_graph_lib.IrQuantizationInfo):
        """
        set encoding for tensor the same way QNN does
        :param ir_tensor: IR tensor for which the encoding must be set
        :param quant_info: Quant info to be set for the given IR tensor
        """
        if quant_info.type != ir_graph_lib.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED and ir_tensor.is_quantizable():
            ir_tensor.set_encoding(ir_tensor.data_type(), quant_info)
            return

        if not ir_tensor.is_quantizable() or not is_fxp_qnn_dtype(ir_tensor.data_type()):
            err_str = (f'Cannot set encoding for tensor {ir_tensor.name()}, with data type '
                       f'{ir_tensor.data_type_string()} and quantizable flag {ir_tensor.is_quantizable()}')
            log_error(err_str)
            raise Exception(err_str)
        elif quant_info.type == ir_graph_lib.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED:
            err_str = (f'QuantInfo not defined for quantizable tensor, {ir_tensor.name()}. Must set valid '
                       f'quantization info w/quantized data types')
            raise AssertionError(err_str)

    def fill_quant_info_and_set_tensor_encoding(self, ir_tensor, encoding):
        aimet_encoding = IROptimizations.extract_encoding_dict(ir_tensor.name(), [encoding], dump_dtype=False)
        quant_info = ir_graph_utils_helper.fill_quant_info(aimet_encoding)
        self.set_encoding_for_tensor(ir_tensor, quant_info)

    def set_stateful_op_act_encodings(self, activation_encodings: Dict, static_tensor_encoding_added: set):
        """
        Update stateful LSTM quantization info
        :param activation_encodings:
        :param param_encodings:
        :param static_tensor_encoding_added:
        :return:
        """
        prep_inputs_to_ir_inputs_idx_map = {
            "Lstm": {
                0: 0,
                1: 10,
                2: 11
            },
            "Gru": {
                0: 0,
                1: 13
            }
        }
        for op, sanitized_op_name in self.stateful_ops_to_sanitized_name_map.items():
            op_inputs = op.inputs()
            op_outputs = op.outputs()
            op_encoding = activation_encodings[sanitized_op_name]

            if "input" in op_encoding:
                for idx, encoding in op_encoding['input'].items():
                    enc = op_encoding["input"][idx]
                    idx = int(idx)
                    ir_tensor = op_inputs[prep_inputs_to_ir_inputs_idx_map[op.type][idx]]
                    self.fill_quant_info_and_set_tensor_encoding(ir_tensor, enc)
                    if ir_tensor.is_static_tensor():
                        static_tensor_encoding_added.add(ir_tensor.name())

            if "output" in op_encoding:
                if op.type == "Lstm":
                    # Set hidden state encodings
                    H_t_encoding = op_encoding["output"]["0"]
                    h_t_encoding = op_encoding["output"]["2"]
                    assert h_t_encoding == H_t_encoding, \
                        "Output and final hidden state encodings are not matching for Lstm!"
                    if "input" in op_encoding and "1" in op_encoding["input"]:
                        h_0_encoding = op_encoding["input"]["1"]
                        assert h_0_encoding == h_t_encoding, \
                            "Output hidden state encoding is not matching with initial_h encoding for Lstm!"
                    self.fill_quant_info_and_set_tensor_encoding(op_outputs[0], h_t_encoding)
                    self.fill_quant_info_and_set_tensor_encoding(op_outputs[2], h_t_encoding)
                    op.attrs.addFloat(ir_graph_lib.QNN_OP_LSTM_PARAM_HIDDEN_STATE_QSCALE, h_t_encoding["scale"])
                    op.attrs.addInt32(ir_graph_lib.QNN_OP_LSTM_PARAM_HIDDEN_STATE_OFFSET, h_t_encoding["offset"])

                    # Set cell state encoding
                    c_t_encoding = op_encoding["output"]["1"]
                    if "input" in op_encoding and "2" in op_encoding["input"]:
                        c_0_encoding = op_encoding["input"]["2"]
                        assert c_0_encoding == c_t_encoding, \
                            "Output cell state encoding is not matching with initial_c encoding!"
                    self.fill_quant_info_and_set_tensor_encoding(op_outputs[1], c_t_encoding)

                    # Set intermediate gate outputs
                    # Input gate
                    input_gate_output_encoding = op_encoding["output"]["3"]
                    op.attrs.addFloat(ir_graph_lib.QNN_OP_LSTM_PARAM_INPUT_GATE_QSCALE, input_gate_output_encoding["scale"])
                    # Forget gate
                    forget_gate_output_encoding = op_encoding["output"]["4"]
                    op.attrs.addFloat(ir_graph_lib.QNN_OP_LSTM_PARAM_FORGET_GATE_QSCALE, forget_gate_output_encoding["scale"])
                    # Cell gate
                    cell_gate_output_encoding = op_encoding["output"]["5"]
                    op.attrs.addFloat(ir_graph_lib.QNN_OP_LSTM_PARAM_CELL_GATE_QSCALE, cell_gate_output_encoding["scale"])
                    # Output gate
                    output_gate_output_encoding = op_encoding["output"]["6"]
                    op.attrs.addFloat(ir_graph_lib.QNN_OP_LSTM_PARAM_OUTPUT_GATE_QSCALE, output_gate_output_encoding["scale"])
                elif op.type == "Gru":
                    h_t_encoding = op_encoding["output"]["1"]
                    Y_encoding = op_encoding["output"]["0"]
                    assert h_t_encoding == Y_encoding, "Output and final hidden state encodings are not matching for Gru!"
                    if "input" in op_encoding and "1" in op_encoding["input"]:
                        h_0_encoding = op_encoding["input"]["1"]
                        assert h_0_encoding == h_t_encoding, \
                            "Output hidden state encoding is not matching with initial_h encoding!"
                    self.fill_quant_info_and_set_tensor_encoding(op_outputs[0], h_t_encoding)
                    self.fill_quant_info_and_set_tensor_encoding(op_outputs[1], h_t_encoding)

    # pylint: disable=too-many-statements
    def set_given_encodings(self, activation_encodings: Dict, param_encodings: Dict, static_tensor_encoding_added: set):
        """
        Update tensor encodings in self.ir_graph given encodings from QuantSim.

        :param activation_encodings: Activation encodings to set
        :param param_encodings: Param encodings to set
        :param static_tensor_encoding_added: Set of static tensors for which encodings were already added
        """
        for op_name, encodings in activation_encodings.items():
            input_index = 0
            op_mapped = False
            # In case of additional transpose op is added explicitly for the model's input tensor
            if op_name in self.prepared_model_info.additional_transpose_info:
                log_msg = f"Transpose Op Node '{op_name}' mapped to '{self.prepared_model_info.additional_transpose_info[op_name]}'"
                log_info(log_msg)
                op_name, input_index = self.prepared_model_info.additional_transpose_info[op_name]
                op_mapped = True

            # If the encoding for the explicitly added pad node is found then the corresponding op will not
            # be present in the IrOp as it was added in emitter graph only. So need to fetch the name of the
            # op for which the pad was added.
            if op_name in self.prepared_model_info.additional_pad_to_node_mapping:
                log_msg = f"Pad Node '{op_name}' mapped to '{self.prepared_model_info.additional_pad_to_node_mapping[op_name]}'"
                log_info(log_msg)
                op_name = self.prepared_model_info.additional_pad_to_node_mapping[op_name]
                op_mapped = True

            if op_name in self.stateful_ops_to_sanitized_name_map.values():
                continue
            else:
                assert op_name in self.sanitized_name_to_node_name_map, f"{op_name} not in sanitized name mapping"
                ir_op = self.ir_graph.get_op(self.sanitized_name_to_node_name_map[op_name])
                if op_mapped:
                    # In case of mapped op only set the input encodings
                    # As mapped input are only the transpose and pad ops
                    if 'input' in encodings:
                        encoding = encodings['input']['0']
                        ir_tensor = ir_op.inputs()[input_index]
                        assert isinstance(ir_tensor, ir_graph_lib.IrTensor)
                        aimet_encoding = IROptimizations.extract_encoding_dict(ir_tensor.name(), [encoding], dump_dtype=False)
                        quant_info = ir_graph_utils_helper.fill_quant_info(aimet_encoding)
                        self.set_encoding_for_tensor(ir_tensor, quant_info)

                else:
                    if 'input' in encodings:
                        for idx, encoding in encodings['input'].items():
                            idx = int(idx)
                            if ir_op.type == "Conv3d" and ir_op.attrs_dict['reuse_sparse_indicies'] and \
                                    ir_op.inputs()[0].get_producer() is not None and \
                                    ir_op.inputs()[0].get_producer().type == "CreateSparse":
                                # Custom handling of SpConv Op whose input coming from CreateSparse Op
                                ir_tensor = ir_op.inputs()[0].get_producer().inputs()[idx]
                            else:
                                ir_tensor = ir_op.inputs()[idx]
                            assert isinstance(ir_tensor, ir_graph_lib.IrTensor)
                            aimet_encoding = IROptimizations.extract_encoding_dict(ir_tensor.name(), [encoding], dump_dtype=False)
                            quant_info = ir_graph_utils_helper.fill_quant_info(aimet_encoding)
                            self.set_encoding_for_tensor(ir_tensor, quant_info)
                            # Static input tensors which are not weights or biases are quantized as activations in AIMET,
                            # such tensors must also be added to 'static_tensor_encoding_added' set
                            if ir_tensor.is_static_tensor():
                                static_tensor_encoding_added.add(ir_tensor.name())
                    if 'output' in encodings:
                        for idx, encoding in encodings['output'].items():
                            idx = int(idx)
                            ir_tensor = ir_op.outputs()[idx]
                            aimet_encoding = IROptimizations.extract_encoding_dict(ir_tensor.name(), [encoding], dump_dtype=False)
                            quant_info = ir_graph_utils_helper.fill_quant_info(aimet_encoding)
                            self.set_encoding_for_tensor(ir_tensor, quant_info)

        self.set_stateful_op_act_encodings(activation_encodings, static_tensor_encoding_added)

        for param_name, encodings in param_encodings.items():
            assert param_name in self.prepared_model_info.param_name_mapping, f"No param name mapping for {param_name}"
            param_tensor_name, _ = self.prepared_model_info.param_name_mapping[param_name]
            ir_tensor = self.ir_graph.get_tensor(param_tensor_name)
            assert isinstance(ir_tensor, ir_graph_lib.IrStaticTensor)
            if not isinstance(encodings, list):
                encodings = [encodings]
            aimet_encoding = IROptimizations.extract_encoding_dict(ir_tensor.name(), encodings, dump_dtype=False)
            quant_info = ir_graph_utils_helper.fill_quant_info(aimet_encoding)
            self.set_encoding_for_tensor(ir_tensor, quant_info)
            static_tensor_encoding_added.add(param_tensor_name)

    def propagate_data_movement_encodings(self):
        """
        Propagate encodings for data movement ops
        """
        for op in self.ir_graph.get_ops():
            if is_qnn_data_movement_op(op):
                if op.type == "CreateSparse":
                    input_tensor = op.inputs()[1]
                else:
                    input_tensor = op.inputs()[0]
                encoding = input_tensor.get_encoding()

                # Skip encoding propagation when the datatype of input is an integer data type
                if is_integer_dtype(input_tensor.data_type()):
                    continue

                if encoding.encInfo.is_fixed_point:
                    # Input encoding should be a valid encoding
                    assert encoding.encInfo.scale != 0.0, f"Input encoding for Op {op.name} is invalid"

                for output_tensor in op.outputs():
                    if not is_integer_dtype(output_tensor.data_type()):
                        out_encoding = output_tensor.get_encoding()
                        # Propagate encodings only if the encoding is not overridden and when the tensor is quantizable
                        if output_tensor.is_quantizable() and not out_encoding.overridden:
                            self.set_encoding_for_tensor(output_tensor, encoding)

    @staticmethod
    def is_valid_fxp_encoding(encoding):
        if encoding.type in VALID_ENCODING_TYPES:
            return encoding.encInfo.is_fixed_point and encoding.encInfo.scale != 0.0
        elif encoding.type in VALID_AXIS_ENCODING_TYPES:
            is_valid_encoding = True
            for enc_info in encoding.axisEncInfo.encInfos:
                is_valid_encoding = is_valid_encoding and (enc_info.is_fixed_point and enc_info.scale != 0.0)
            return is_valid_encoding

    def fill_missing_encodings(self, act_bw: int, static_tensor_encoding_added: Set, is_embedded_ir: bool):
        """
        Fill missing encodings for StaticTensors

        :param act_bw: Activation bw to use when generating encodings
        :param static_tensor_encoding_added: Set of static tensors for which encodings were already added
        :param is_embedded_ir: Flag that signals whether to assume that encodings are embedded in IRGraph
        """
        ir_graph_to_model_param_mapping = None
        if not is_embedded_ir:
            ir_graph_to_model_param_mapping = {v[0]: k for k, v in self.prepared_model_info.param_name_mapping.items()}

        for op in self.ir_graph.get_ops():
            for idx, tensor in enumerate(op.inputs()):
                # Need not deal with bias tensors, as LWQ takes care of those
                if ir_quantizer.is_input_tensor_bias(op, tensor):
                    continue

                tensor_name = tensor.name()

                # Need to generate encodings for the static tensors only
                # whose encodings are not generated. Except for bias
                if tensor.is_static_tensor():
                    data = tensor.get_data()
                    # Generating encoding for float tensors only
                    if data.dtype not in [np.float32, np.float16]:
                        continue

                    if is_embedded_ir:
                        encoding = tensor.get_encoding()
                        is_encoding_added = self.is_valid_fxp_encoding(encoding) or tensor.is_overridden_float()
                    else:
                        is_encoding_added = tensor_name in static_tensor_encoding_added

                    if not is_encoding_added:
                        param_name = ir_graph_to_model_param_mapping.get(tensor_name) if (
                                ir_graph_to_model_param_mapping is not None) else tensor_name
                        if param_name and (param_name.endswith('.bias') or has_dummy_bias(op, tensor)):
                            continue

                        log_str = f'Filling missing encoding for static tensor {tensor_name} of size {data.size}'
                        if data.size > 1:
                            log_error(log_str)
                            raise ValueError("Filling missing encoding for tensor with size more than 1 is not supported.")
                        else:
                            log_info(log_str)

                        encoding = IrGraphUpdater.__get_encoding_for_scalar_tensor(act_bw, data)

                        aimet_encoding = IROptimizations.extract_encoding_dict(tensor.name(), encoding, dump_dtype=False)
                        quant_info = ir_graph_utils_helper.fill_quant_info(aimet_encoding)
                        self.set_encoding_for_tensor(tensor, quant_info)

    @staticmethod
    def __get_encoding_for_scalar_tensor(act_bw, data):

        enc_min = min(0, data.min())
        enc_max = max(0, data.max())

        if enc_min == enc_max:
            enc_max += 0.0001

        enc_scale = (enc_max - enc_min) / (2**act_bw - 1)
        enc_offset = round(enc_min/enc_scale)

        encoding = [{'min': enc_min,
                     'max': enc_max,
                     'scale': enc_scale,
                     'offset': int(enc_offset),
                     'bitwidth': act_bw,
                     'is_symmetric': 'False',
                     'dtype': "int"}]
        return encoding

    # pylint: disable=inconsistent-return-statements
    def quantize_ir_graph(self, quantize_dlc: bool = False, float_bias_bw: int = 32):
        """
        Quantize self.ir_graph using QNN lightweight quantizer.
        :param quantize_dlc: Flag to decide whether or not to actually quantize the DLC
        :param float_bias_bw: float fallback bitwidth for bias
        """
        # pylint: disable=broad-except
        try:
            light_weight_opts = light_weight_ir_quantizer.LightWeightIrQuantizerOpts()
            light_weight_opts.float_bias_bw = float_bias_bw
            if quantize_dlc:
                light_weight_opts.enable_qnn_quantizer = False
            else:
                return None
            light_weight_quantizer = light_weight_ir_quantizer.LightWeightIrQuantizer(light_weight_opts, self.ir_graph)
            light_weight_quantizer.quantize()

        except Exception as e:
            log_msg = f"Failed to create quantized IR Graph :: {str(e)}"
            log_error(log_msg)
            raise Exception(log_msg)

    def populate_sanitized_name_to_node_name_map(self, ir_graph) -> Dict[str, str]:
        """
        Get a mapping of sanitized node names to original node names in the ir_graph.

        :param ir_graph: IR graph to get node name mappings for
        :return: Map of sanitized node names to original node names
        """
        for op in ir_graph.get_ops():
            if op.type in ["Lstm", "Gru"]:
                op_name = get_op_name(op.name)
                self.stateful_ops_to_sanitized_name_map[op] = op_name
            self.sanitized_name_to_node_name_map[get_op_name(op.name)] = op.name

    @staticmethod
    def get_qnn_dtype_for_act(dtype: str, bw: int) -> ir_graph_lib.Qnn_DataType_t:
        """
        return the qnn dtypes corresponding to aimet representation

        :param dtype: data type of type str
        :param bw: bitwidth of type int
        :return: QNN data type corresponding to AIMET representation
        """
        # TODO Currently assume to be UFIXED always. Need to verify with QNN team if this changes based on input
        if dtype == 'int' and bw == 4:
            qnn_type = ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_4
        elif dtype == 'int' and bw == 8:
            qnn_type = ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8
        elif dtype == 'int' and bw == 16:
            qnn_type = ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_16
        else:
            log_error("Unsupported dtype:%s and bw:%d encountered", dtype, bw)
            raise NotImplementedError
        return qnn_type

    @staticmethod
    def get_backend_op_onnx_type(op_name: str) -> str:
        """
        Backend op "type names" are not stored for all the ops. For example: relu op returns "Neuron" op.
        Op type can instead be retrieved from the "name" of the op. For example: relu op name returns '/relu1/Relu'
        which can be used to retrieve "Relu"

        :param op_name: name of the op generated using backend_op.name
        :return: op onnx type
        """
        return op_name.split("/")[-1]
