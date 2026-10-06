# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

from qti.aisw.converters.common import ir_graph
from .onnx_translations import *
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders, AxisTracker
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_graph import TraceType
import qti.aisw.converters.common.converter_ir.op_properties.stateful_lstm as lstm_op_props
import qti.aisw.converters.common.converter_ir.op_properties.stateful_gru as gru_op_props

# ------------------------------------------------------------------------------
#  RNNTranslationBase
# ------------------------------------------------------------------------------
OPTIONAL_INPUTS = NamedDict(initial_h='', initial_c='', norm_weights='', cell_state_weights='', proj_weights='', proj_bias='')
RNN_INPUT_TYPES = ('_initial_h', '_initial_c')
QNN_LSTM_OUTPUT_TYPES = ('_all_hidden', '_final_cell', '_final_hidden')
QNN_GRU_OUTPUT_TYPES = ('_all_hidden', '_final_hidden')



class OnnxRnnTranslationsBase(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []
        self.weights = []
        self.rec_weights = []
        self.bias = None
        self.cell_state_weights = None
        self.params = NamedDict()
        self.no_of_gates = 1
        self.direction = ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD
        self.output_names = []
        self.num_directions = 1
        self.weights_src_info = []
        self.rec_weights_src_info = []
        self.bias_src_info = []
        self.src_input_dic = {}

    def extract_params_for_type(self, src_op, converter_context, rnn_type='Rnn'):
        graph = converter_context.ir_graph

        self.input_names = list(map(str, src_op.input))
        self.output_names = self.extract_output_names(src_op, converter_context)
        self.params.direction = str(self.params.direction).lower()
        self.direction = ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD if self.params.direction == 'forward' else ir_graph.QNN_OP_LSTM_DIRECTION_REVERSE
        if rnn_type == 'Gru':
            self.direction = ir_graph.QNN_OP_GRU_DIRECTION_FORWARD if self.params.direction == 'forward' else ir_graph.QNN_OP_GRU_DIRECTION_REVERSE
        self.num_directions = 1

        # Ensure weights or rec_weights are not passed in dynamically
        weights_input_name = self.input_names[1]
        if not converter_context.weights.has(weights_input_name) or \
            (graph.has_buffer(weights_input_name) and not isinstance(graph.get_producer_op(weights_input_name), op_adapter.ConstantOp)):
                raise ValueError("Unsupported dynamic weights for input {} of RNN node {}".format(weights_input_name,
                                                                                                  src_op.name))
        self.weights = converter_context.weights.fetch(weights_input_name)
        self.weights_src_info.append((self.input_names[1], TraceType.TENSOR))

        rec_weights_input_name = self.input_names[2]
        if not converter_context.weights.has(rec_weights_input_name) or \
            (graph.has_buffer(rec_weights_input_name) and not isinstance(graph.get_producer_op(rec_weights_input_name), op_adapter.ConstantOp)):
                raise ValueError("Unsupported dynamic weights for input {} of RNN node {}".format(rec_weights_input_name,
                                                                                                  src_op.name))
        self.rec_weights = converter_context.weights.fetch(rec_weights_input_name)
        self.rec_weights_src_info.append((self.input_names[2], TraceType.TENSOR))

        # ONNX may use empty string as a placeholder
        # So add an and-condition to further check it.
        if len(self.input_names) >= 4 and self.input_names[3]:
            bias_input_name = self.input_names[3]
            if not converter_context.weights.has(bias_input_name) or \
                (graph.has_buffer(bias_input_name) and not isinstance(graph.get_producer_op(bias_input_name), op_adapter.ConstantOp)):
                    raise ValueError("Unsupported dynamic bias for input {} of RNN node {}".format(bias_input_name,
                                                                                                   src_op.name))
            self.bias = converter_context.weights.fetch(bias_input_name)
            self.bias_src_info.append((self.input_names[3], TraceType.TENSOR))

        # Fetch cell_state_weights by source op input names if provided
        self.cell_state_weights = None

        # If it is bi-directional check that weights and rec_weights
        # have the right shape
        if self.params.direction == "bidirectional":
            log_assert(self.weights.shape[0] == 2 and self.rec_weights.shape[0] == 2,
                       "Node {}: Bidirectional input requires two sets of weights and recurrent "
                       "weights each. Got only {} set of weights",
                       src_op.name, self.weights.shape[0])

    def prepare_params_as_constants(self, graph, params):
        for param_name, tensor in params.items():
            if not graph.has_buffer(param_name):
                constant_node = graph.add(op_adapter.ConstantOp(name=param_name, tensor=tensor),
                                          input_names=[], output_names=[param_name],
                                          axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                graph.add_src_op_info(param_name, [], [param_name])
                src_info = self.src_input_dic[param_name] if param_name in self.src_input_dic.keys() else []
                if len(src_info) > 0:
                    graph.set_trace_info([constant_node, graph.get_output_buffers(constant_node)[0]], src_info)
            elif isinstance(graph.get_producer_op(param_name), op_adapter.ConstantOp):
                cur_constant_node = graph.get_producer_op(param_name)
                updated_constant_node = op_adapter.ConstantOp(param_name, tensor)
                graph.replace(cur_constant_node, updated_constant_node)
                graph.update_trace_info(cur_constant_node, updated_constant_node, remove_source=True)
            else:
                raise ValueError("lstm requires weights and bias to be constant, got dynamic tensor from {}".format(
                    graph.get_producer_op(param_name).name))

    def add_constant_reset_node(self, converter_context, input_names, rnn_type='Rnn'):
        graph = converter_context.ir_graph
        if rnn_type == 'Gru':
            if len(input_names) == gru_op_props.IR_NUM_INPUTS and input_names[gru_op_props.IR_RESET_IDX]:
                if converter_context.weights.has(input_names[gru_op_props.IR_RESET_IDX]) and not graph.has_buffer(input_names[gru_op_props.IR_RESET_IDX]):
                    tensor = np.squeeze(converter_context.weights.fetch(input_names[gru_op_props.IR_RESET_IDX], prunable=False))
                    constant_node = graph.add(op_adapter.ConstantOp(input_names[gru_op_props.IR_RESET_IDX], tensor),
                                              [],
                                              input_names[gru_op_props.IR_RESET_IDX])
                    self.insert_constant_trace_info(input_names[gru_op_props.IR_RESET_IDX], constant_node, converter_context)
        elif rnn_type == 'LSTM':
            IR_NUM_INPUTS = lstm_op_props.IR_NUM_INPUTS
            IR_RESET_IDX = lstm_op_props.IR_RESET_IDX
            if getattr(graph, 'qairt_converter', False) and OnnxTranslations.is_cpp_ir_pass('MULTI_TIME_STEPS_LSTM'):
                # Converted IR is LSTM op
                IR_NUM_INPUTS = lstm_op_props.LSTM_IR_NUM_INPUTS
                IR_RESET_IDX = lstm_op_props.LSTM_IR_RESET_IDX

            if len(input_names) == IR_NUM_INPUTS and input_names[IR_RESET_IDX]:
                if converter_context.weights.has(input_names[IR_RESET_IDX]) and \
                                    not graph.has_buffer(input_names[IR_RESET_IDX]):
                    # Convert reset constant tensor from 1D to 0D to align with backend validation requirements
                    tensor = np.squeeze(converter_context.weights.fetch(input_names[IR_RESET_IDX],
                                                                                 prunable=False))
                    constant_node = graph.add(op_adapter.ConstantOp(input_names[IR_RESET_IDX], tensor),
                                      [],
                                      input_names[IR_RESET_IDX])
                    self.insert_constant_trace_info(input_names[IR_RESET_IDX], constant_node, converter_context)

    def extract_output_names(self, src_op, converter_context):
        graph = converter_context.ir_graph
        onnx_output_names = [output for i, output in enumerate(src_op.output)]
        dummy_name_list = ["_all_output_dummy", "_hidden_output_dummy", "_cell_output_dummy"]

        # If the length of onnx_output_names less then 3
        # Firstly, add '' to the onnx_output_names to make it has 3 names
        # Then, use the dummy_name to fill with the empty names
        # Cause we can't make sure the first output name is alway not empty
        # we use the op.name to distinguish different empty name buffer in different op.
        # ---------------------------------------------------------------------------------------------------------------------------------
        # | src_op.outputs(src_op.name='Lstm') |    onnx_output_names - 1   |                   onnx_output_names - 2                     |
        # ---------------------------------------------------------------------------------------------------------------------------------
        # | ['Y'], ['Y', ''], ['Y', '', '']    |    ['Y', '', '']           | ['Y','Lstm_cell_output_dummy','Lstm_hidden_output_dummy']   |
        # | ['', 'Y_h'],['', 'Y_h', '']        |    ['', 'Y_h', '']         | ['Lstm_all_output_dummy', 'Lstm_cell_output_dummy', 'Y_h']  |
        # | ['', '', 'Y_c']                    |    ['', '', 'Y_c']         | ['Lstm_all_output_dummy', 'Y_c', 'Lstm_hidden_output_dummy']|
        # | ['Y', 'Y_h'], ['Y', 'Y_h', '']     |    ['Y', 'Y_h', '']        | ['Y','Lstm_cell_output_dummy','Y_h']                        |
        # | ['Y', '', 'Y_c']                   |    ['Y', '', 'Y_c']        | ['Y','Y_c','Lstm_hidden_output_dummy']                      |
        # | ['', 'Y_h', 'Y_c']                 |    ['', 'Y_h', 'Y_c']      | ['Lstm_all_output_dummy','Y_c','Y_h']                       |
        # | ['Y', 'Y_h', 'Y_c']                |    ['Y', 'Y_h', 'Y_c']     | ['Y','Y_c','Y_h']                                           |
        # ---------------------------------------------------------------------------------------------------------------------------------
        while len(onnx_output_names) < 3:
            onnx_output_names.append('')

        src_name = src_op.name if len(src_op.name) != 0 else \
                onnx_output_names[0] if onnx_output_names[0] else \
                    graph.naming_policy.get_op_name_by_type(op_adapter.LstmOp.TRANSLATION_KEY,
                                                            op_adapter.LstmOp.LEGACY_TRANSLATION_KEY)
        for idx, name in enumerate(onnx_output_names):
            if not name:
                onnx_output_names[idx] = src_name + dummy_name_list[idx]

        # ONNX LSTM output order: Y, Y_h, Y_c
        # SNPE LSTM output order: Y, Y_c, Y_h
        output_names = onnx_output_names
        output_names[1], output_names[2] = onnx_output_names[2], onnx_output_names[1]
        return output_names

    def extract_src_output_tuples(self, src_op):
        # Since extract_output_names may change the outputs names and their orders. Follow the changes
        # to construct a source output names for framework trace
        src_onnx_output_tuples = []
        for output in src_op.output:
            if output:
                src_onnx_output_tuples.append((output, TraceType.TENSOR))
            else:
                src_onnx_output_tuples.append((src_op.name, TraceType.OP))
        while len(src_onnx_output_tuples) < 3:
            src_onnx_output_tuples.append((src_op.name, TraceType.OP))

        src_onnx_output_tuples[1], src_onnx_output_tuples[2] = src_onnx_output_tuples[2], src_onnx_output_tuples[1]
        return src_onnx_output_tuples

    def add_h_c_inputs_reshape_if_needed(self, converter_context, src_op, module_name, input_names):
        """
        Ensure initial hidden (h_0) and cell (c_0) inputs are 2D tensors of shape [batch_size, hidden_size].
        - Injects Reshape ops when provided inputs have different rank or shape.
        - Only reshapes inputs that are present. Missing inputs are added later with the correct shape.
        - If c_0 reuses the same tensor as h_0, the reshaped h_0 output is reused.
        """
        graph = converter_context.ir_graph

        # Derive batch size from input sequence and hidden size from recurrent weights
        seq_input_shape = graph.get_buffer(src_op.input[0]).shape[:]
        batch_size, _, _ = AxisOrders.ONNX.extract_time_series_dims(seq_input_shape)
        hidden_size = self.rec_weights.shape[-1]

        # Indices for initial states in the IR
        initial_h_idx = lstm_op_props.LSTM_IR_INITIAL_H_IDX
        initial_c_idx = lstm_op_props.LSTM_IR_INITIAL_C_IDX

        initial_h_name = input_names[initial_h_idx]
        initial_c_name = input_names[initial_c_idx]

        # If no initial hidden state is provided, nothing to reshape here
        if not initial_h_name:
            return input_names

        # Reshape h_0 to [batch_size, hidden_size] if needed
        h_buf = graph.get_buffer(initial_h_name)
        h_shape = list(h_buf.shape) if hasattr(h_buf, 'shape') else []
        h_needs_reshape = len(h_shape) != 2 or h_shape != [batch_size, hidden_size]

        # Reshape c_0 to [batch_size, hidden_size] if needed
        c_buf = graph.get_buffer(initial_c_name)
        c_shape = list(c_buf.shape) if hasattr(c_buf, 'shape') else []
        c_needs_reshape = len(c_shape) != 2 or c_shape != [batch_size, hidden_size]

        h_reshape_name = f"{module_name}_{initial_h_name}_reshape"
        c_reshape_name = f"{module_name}_{initial_c_name}_reshape"
        c_input_source = initial_c_name

        # Maintaining name conversation for EXPAND_LSTM_OP_STRUCTURE
        if OnnxTranslations.is_cpp_ir_pass("EXPAND_LSTM_OP_STRUCTURE"):
            h_reshape_name = f"{initial_h_name}_reshape"
            c_reshape_name = f"{initial_c_name}_reshape"

        if graph.has_buffer(h_reshape_name):
            input_names[initial_h_idx] = h_reshape_name
        elif h_needs_reshape:
            h_target_shape = [batch_size, hidden_size]
            h_reshape_op = op_adapter.ReshapeOp(name=h_reshape_name, shape=h_target_shape)
            graph.inject(h_reshape_op, input_name=initial_h_name, output_name=h_reshape_name, consumer_names=[])
            input_names[initial_h_idx] = h_reshape_name

        if initial_c_name == initial_h_name:
            # EXPAND_LSTM_OP_STRUCTURE Python pass use the same reshape generated by h_needs_reshape
            if OnnxTranslations.is_cpp_ir_pass("EXPAND_LSTM_OP_STRUCTURE"):
                input_names[initial_c_idx] = h_reshape_name
                OPTIONAL_INPUTS.initial_h = OPTIONAL_INPUTS.initial_c = h_reshape_name
                return input_names
            # For MULTI_TIME_STEPS_LSTM and UNROLL_LSTM_TIME_STEPS
            # Identity reshape will be created
            c_input_source = input_names[initial_h_idx]
            c_reshape_name = f"{module_name}_{h_reshape_name}_reshape"

        if graph.has_buffer(c_reshape_name):
            input_names[initial_c_idx] = c_reshape_name
        elif c_needs_reshape:
            c_target_shape = [batch_size, hidden_size]
            c_reshape_op = op_adapter.ReshapeOp(name=c_reshape_name, shape=c_target_shape)
            graph.inject(c_reshape_op, input_name=c_input_source, output_name=c_reshape_name, consumer_names=[])
            input_names[initial_c_idx] = c_reshape_name

        return input_names


    def ensure_h_c_inputs_present(self, converter_context, module_name, src_op, input_names):
        """
        Ensures initial hidden (h_0) and cell (c_0) inputs exist for LSTM.
        - If either is missing, inserts a zero-initialized ConstantOp with shape [batch, hidden_size].
        - Leaves provided inputs untouched (reshaping, if needed, is handled earlier).
        """
        graph = converter_context.ir_graph

        # Derive batch size from input sequence tensor
        input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
        batch_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])[0]
        hidden_size = self.params.hidden_size

        # MultiTimeSteps LSTM initial_h and initial_c indices
        IR_INITIAL_H_IDX = lstm_op_props.LSTM_IR_INITIAL_H_IDX
        IR_INITIAL_C_IDX = lstm_op_props.LSTM_IR_INITIAL_C_IDX

        # Add missing initial hidden state
        if not input_names[IR_INITIAL_H_IDX]:
            initial_hidden_state_name = module_name + '_initial_hidden_state'
            initial_hidden_state_tensor = np.zeros((batch_size, hidden_size), dtype=np.float32)
            initial_hidden_state_op = op_adapter.ConstantOp(name=initial_hidden_state_name,
                                                            tensor=initial_hidden_state_tensor)
            graph.add(initial_hidden_state_op,
                      input_names=[],
                      output_names=[initial_hidden_state_name],
                      axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            self.insert_constant_trace_info(input_names[IR_INITIAL_H_IDX], initial_hidden_state_op, converter_context)
            input_names[IR_INITIAL_H_IDX] = initial_hidden_state_name

        # Add missing initial cell state
        if not input_names[IR_INITIAL_C_IDX]:
            initial_cell_state_name = module_name + '_initial_cell_state'
            initial_cell_state_tensor = np.zeros((batch_size, hidden_size), dtype=np.float32)
            initial_cell_state_op = op_adapter.ConstantOp(name=initial_cell_state_name,
                                                          tensor=initial_cell_state_tensor)
            graph.add(initial_cell_state_op,
                      input_names=[],
                      output_names=[initial_cell_state_name],
                      axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            self.insert_constant_trace_info(input_names[IR_INITIAL_C_IDX], initial_cell_state_op, converter_context)
            input_names[IR_INITIAL_C_IDX] = initial_cell_state_name

        return input_names

    def create_rnn(self, src_op, converter_context, create_unidirectional_func, create_bidirectional_func, rnn_type='Rnn'):
        graph = converter_context.ir_graph
        QNN_RNN_OUTPUT_TYPES = QNN_LSTM_OUTPUT_TYPES
        if rnn_type == 'Gru':
            QNN_RNN_OUTPUT_TYPES = QNN_GRU_OUTPUT_TYPES

        if self.params.direction == "bidirectional":
            create_bidirectional_func(src_op, converter_context)
        else:
            # set up naming so that the buffers are all different and tagged correctly
            input_names = self.extract_input_names(src_op, converter_context)
            output_names = self.extract_output_names(src_op, converter_context)
            trace_src_onnx_output_tuples = self.extract_src_output_tuples(src_op)
            # QNN GruOp has two mandatory outputs. If the model has only one output then add
            # an additional output which signifies the final hidden output.
            if rnn_type == 'Gru' and input_names[1] and len(output_names) == 1:
                final_hidden_output_name = output_names[0] + '_unidirection'
                output_names.append(final_hidden_output_name)
            output_names_len = len(output_names)
            module_name = src_op.name if len(src_op.name) != 0 else \
                output_names[0] if output_names_len != 0 else \
                    src_op.op_type

            # set up rnn ops
            if rnn_type=="Gru":
                rnn_op = create_unidirectional_func(graph, module_name, input_names, converter_context=converter_context)
                initial_state_input_names = [rnn_op.h_0_input_name]
            else:
                IR_INITIAL_H_IDX = lstm_op_props.IR_INITIAL_H_IDX
                IR_INITIAL_C_IDX = lstm_op_props.IR_INITIAL_C_IDX
                if getattr(graph, 'qairt_converter', False) and rnn_type == 'LSTM' and OnnxTranslations.is_cpp_ir_pass('MULTI_TIME_STEPS_LSTM'):
                    IR_INITIAL_H_IDX = lstm_op_props.LSTM_IR_INITIAL_H_IDX
                    IR_INITIAL_C_IDX = lstm_op_props.LSTM_IR_INITIAL_C_IDX
                initial_state_input_names = input_names[IR_INITIAL_H_IDX:IR_INITIAL_C_IDX+1]

            for input_name in initial_state_input_names:
                # Add constant op if one of the initial c/h inputs is Initializer and not added to graph
                if converter_context.weights.has(input_name) and not graph.has_buffer(input_name):
                    tensor = converter_context.weights.fetch(input_name, prunable=False)
                    constant_node = graph.add(op_adapter.ConstantOp(input_name, tensor),
                                            [],
                                            input_name,
                                            axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                    self.insert_constant_trace_info(input_name, constant_node, converter_context)
                elif graph.has_buffer(input_name):
                    # set axis format as NONTRIVIAL for dynamic c/h inputs
                    input_buf = graph.get_buffer(input_name)
                    input_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

            # Add constant node for reset input if it is static
            self.add_constant_reset_node(converter_context, input_names, rnn_type)

            # If ONNX doesn't have initial_H and initial_C, we will manually add 0 values
            # If initial_H and initial_c shape is not 2D, we will insert Reshape
            if getattr(graph, 'qairt_converter', False) and rnn_type == 'LSTM' and OnnxTranslations.is_cpp_ir_pass('MULTI_TIME_STEPS_LSTM'):
                input_names = self.add_h_c_inputs_reshape_if_needed(converter_context, src_op, module_name, input_names)
                input_names = self.ensure_h_c_inputs_present(converter_context, module_name, src_op, input_names)
                OPTIONAL_INPUTS.initial_h = input_names[IR_INITIAL_H_IDX]
                OPTIONAL_INPUTS.initial_c = input_names[IR_INITIAL_C_IDX]

            # set up rnn ops
            if rnn_type == 'LSTM':
                rnn_op = create_unidirectional_func(graph, module_name, input_names, src_op = src_op)

            # set up reshape ops
            reshape_ops = []
            input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
            batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
            hidden_size = self.rec_weights.shape[-1]

            output_shapes = []
            for i in range(0, output_names_len):
                if i == 0:
                    output_shapes.append([time_steps, self.num_directions, batch_size, hidden_size])
                else:
                    output_shapes.append([self.num_directions, batch_size, hidden_size])
            for i in range(0, output_names_len):
                # the outputs are reshaped to be ONNX format
                reshape_ops.append(op_adapter.ReshapeOp(output_names[i] + QNN_RNN_OUTPUT_TYPES[i] + '_reshape',
                                                        shape=output_shapes[i]))

            if getattr(graph, 'qairt_converter', False) and \
                    rnn_type == 'LSTM' and \
                    OnnxTranslations.is_cpp_ir_pass('MULTI_TIME_STEPS_LSTM') and \
                    not OnnxTranslations.is_cpp_ir_pass('UNROLL_LSTM_TIME_STEPS') and \
                    not OnnxTranslations.is_cpp_ir_pass('EXPAND_LSTM_OP_STRUCTURE'):
                # Add a suffix after tensor name only if MULTI_TIME_STEPS_LSTM is enabled
                rnn_output_names = [str(name) + QNN_RNN_OUTPUT_TYPES[i] + "_rnn_multi_time_step"
                                    for i, name in enumerate(output_names)]
            else:
                rnn_output_names = [str(name) + QNN_RNN_OUTPUT_TYPES[i] + "_rnn"
                                    for i, name in enumerate(output_names)]

            reshape_output_names = [str(name) for name in output_names]

            # set override encodings for rnn op
            # ONNX LSTM output order: Y, Y_h, Y_c
            if graph.user_quantization_overrides:
                for id in range(0, len(src_op.output)):
                    encoding = graph.get_overridden_encoding(src_op.output[id], False)
                    if encoding is not None and rnn_type == 'Gru':
                        graph.set_overridden_encoding(rnn_output_names[id], encoding, False)
                    elif encoding is not None and id == 1 and len(rnn_output_names) > 2:
                        graph.set_overridden_encoding(rnn_output_names[2], encoding, False)
                    elif encoding is not None and id == 2:
                        graph.set_overridden_encoding(rnn_output_names[1], encoding, False)
                    elif encoding is not None:
                        graph.set_overridden_encoding(rnn_output_names[id], encoding, False)

            self.add_src_op_info(rnn_op.name, src_op, graph)
            input_x_name = input_names[0]
            input_axis_format = graph.get_buffer(input_x_name).axis_format
            rnn_node = graph.add(rnn_op, input_names, rnn_output_names)
            if graph.has_buffer(input_x_name):
                # After adding rnn_op to the graph, original layout info about the
                # main input of rnn node will be lost and main input data axis format will
                # be changed to TNF. Reverting the main input data axis format to
                # original input layout
                rnn_node.op.data_axis_formats[0] = input_axis_format
                input_x_buf = graph.get_buffer(input_x_name)
                input_x_buf.set_axis_format(AxisTracker.AxisFormat.TNF)
            else:
                raise KeyError("Graph has no buffer {} for {} Op {} input x", input_x_name, rnn_type, src_op.name)

            self.insert_trace_info(rnn_node, [(src_op.name, TraceType.OP)], converter_context)
            for i, rnn_output_name in enumerate(rnn_output_names):
                self.insert_trace_info(graph.get_buffer(rnn_output_name), [trace_src_onnx_output_tuples[i]], converter_context)

            for i in reversed(range(0, output_names_len)):
                # add reshape for outputs to make shape to be ONNX's format
                log_debug("Adding reshape op {} while creating unidirectional RNN unit".format(reshape_ops[i].name))
                reshape_node = graph.add(reshape_ops[i], [rnn_output_names[i]], [reshape_output_names[i]])
                graph.add_src_op_info(reshape_ops[i].name, [rnn_output_names[i]], [reshape_output_names[i]])
                self.insert_trace_info([reshape_node, graph.get_buffer(rnn_output_names[i]), graph.get_buffer(reshape_output_names[i])],
                                       [(src_op.name, TraceType.OP), trace_src_onnx_output_tuples[i]], converter_context)

            return rnn_node

    def create_bidirectional_module(self, src_op, converter_context, weights, rec_weights, bias, params,
                                    create_rnn_type, rnn_type = 'Rnn'):
        graph = converter_context.ir_graph
        # set up naming so that the buffers are all different and tagged correctly
        src_op_inputs = list(src_op.input)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        module_name = src_op.name if len(src_op.name) != 0 else output_names[0]
        initial_h_c_inputs_present = True if OPTIONAL_INPUTS.initial_h and OPTIONAL_INPUTS.initial_c else False

        output_names_len = len(output_names)
        if output_names_len > 3:
            raise ValueError("Unsupported number of outputs for source bidirectional LSTM op {}. Expected number of outputs between 0 and 3, got {}".
                             format(src_op.name, output_names_len))

        # bidirectional lstm shares the same source weights name of forward and backward, so we need to specify them in graph
        forward_op_input_names = input_names[:]
        backward_op_input_names = input_names[:]
        # Do not create separate tensors for reset input, it is shared between forward and backwards
        MAX_INPUTS = lstm_op_props.NUM_INPUTS
        INITIAL_H_IDX = lstm_op_props.IR_INITIAL_H_IDX
        INITIAL_C_IDX = lstm_op_props.IR_INITIAL_C_IDX
        if getattr(graph, 'qairt_converter', False) and rnn_type == 'LSTM' and OnnxTranslations.is_cpp_ir_pass('MULTI_TIME_STEPS_LSTM'):
            MAX_INPUTS = lstm_op_props.LSTM_IR_NUM_INPUTS
            INITIAL_H_IDX = lstm_op_props.LSTM_IR_INITIAL_H_IDX
            INITIAL_C_IDX = lstm_op_props.LSTM_IR_INITIAL_C_IDX

        last_idx_to_split = min(len(input_names), MAX_INPUTS  - 1)
        for i in range(1, last_idx_to_split):
            if i == INITIAL_H_IDX or i == INITIAL_C_IDX:  # Skip ONNX_INITIAL_H_IDX and ONNX_INITIAL_C_IDX
                continue
            curr_input_name = input_names[i]
            if curr_input_name:
                forward_op_input_names[i] = curr_input_name + '_forward'
                backward_op_input_names[i] = curr_input_name + '_backward'
                # Set override param encodings for new name input_tensor
                if graph.user_quantization_overrides:
                    encoding = graph.get_overridden_encoding(curr_input_name)
                    if encoding is not None:
                        graph.set_overridden_encoding(forward_op_input_names[i], encoding, False)
                        graph.set_overridden_encoding(backward_op_input_names[i], encoding, False)

        h_0_forward_split_name, h_0_backward_split_name = '', ''
        c_0_forward_split_name, c_0_backward_split_name = '', ''
        # set up split ops for initial_h and initial_c along `num_directions` dimension if present
        if initial_h_c_inputs_present:
            src_h_c_input_names = src_op_inputs[5:7]
            h_c_split_output_names = list()
            for i, name in enumerate(src_h_c_input_names):
                # add initial_h and initial_c to graph
                if converter_context.weights.has(name) and not graph.has_buffer(name):
                    tensor = converter_context.weights.fetch(name, prunable=False)
                    initial_h_c_node = graph.add(op_adapter.ConstantOp(name, tensor),
                                                [],
                                                name,
                                                axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                    self.insert_constant_trace_info(name, initial_h_c_node, converter_context)
                elif graph.has_buffer(name):
                    input_buf = graph.get_buffer(name)
                    input_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

                split_op_name = f"{module_name}_{name}_{RNN_INPUT_TYPES[i]}_split"
                # If split_index is not given, split equally along the specified axis
                split_op = op_adapter.SplitOp(split_op_name, axis=0)
                split_output_names = [f"{split_op_name}_forward",
                                      f"{split_op_name}_backward"]
                h_c_split_output_names.append(split_output_names)

                # Skip adding split node if it has been added by the previous RNN node
                # which share the same initial hidden status buffer with this one.
                if not graph.has_node(split_op_name):
                    log_debug("Adding split op {} while creating bidirectional RNN unit".format(split_op.name))
                    split_node = graph.add(split_op, [name], split_output_names)
                    graph.add_src_op_info(split_op.name, [name], split_output_names)
                    split_targets = [split_node].extend(graph.get_output_buffers(split_node))
                    self.insert_trace_info(split_targets, (name, TraceType.TENSOR), converter_context)

                    # Set override param encodings for split op
                    if graph.user_quantization_overrides:
                        encoding = graph.get_overridden_encoding(name)
                        if encoding is not None:
                            for output_name in split_output_names:
                                graph.set_overridden_encoding(output_name, encoding, False)

            h_0_forward_split_name, h_0_backward_split_name = h_c_split_output_names[0]
            c_0_forward_split_name, c_0_backward_split_name = h_c_split_output_names[1]

        # set up forward op
        forward_op = create_rnn_type(graph,
                                     module_name + '_forward',
                                     forward_op_input_names,
                                     src_op = src_op,
                                     weights=weights[0, :, :],
                                     rec_weights=rec_weights[0, :, :],
                                     bias=bias[0, :] if bias is not None else bias,
                                     hidden_size=params.hidden_size,
                                     direction=ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD,
                                     h_0_input_name=h_0_forward_split_name,
                                     c_0_input_name=c_0_forward_split_name)

        # set up backward op
        backward_op = create_rnn_type(graph,
                                      module_name + '_backward',
                                      backward_op_input_names,
                                      src_op = src_op,
                                      weights=weights[1, :, :],
                                      rec_weights=rec_weights[1, :, :],
                                      bias=bias[1, :] if bias is not None else bias,
                                      hidden_size=params.hidden_size,
                                      direction=ir_graph.QNN_OP_LSTM_DIRECTION_REVERSE,
                                      h_0_input_name=h_0_backward_split_name,
                                      c_0_input_name=c_0_backward_split_name)

        # set up reshape ops
        reshape_ops = []
        input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
        batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
        hidden_size = params.hidden_size
        output_shapes = []
        for i in range(0, output_names_len):
            # The shape is for unidirectional lstm, so we set num_directions to 1 here
            # Outputs from forward and backward lstm will then be concatenated together
            if i == 0:
                output_shapes.append([time_steps, 1, batch_size, hidden_size])
            else:
                output_shapes.append([1, batch_size, hidden_size])
        for i in range(0, output_names_len):
            # the outputs are reshaped to be ONNX format
            reshape_ops.append([op_adapter.ReshapeOp(output_names[i] + QNN_LSTM_OUTPUT_TYPES[i] + '_reshape_forward',
                                                     shape=output_shapes[i]),
                                op_adapter.ReshapeOp(output_names[i] + QNN_LSTM_OUTPUT_TYPES[i] + '_reshape_backward',
                                                     shape=output_shapes[i])])
        # set up concat ops
        concat_ops = []
        for i in range(0, output_names_len):
            # The first output is used to concat all hidden output values at last axis
            axis = 1 if i == 0 else 0
            concat_ops.append(op_adapter.ConcatOp(output_names[i] + '_concat', axis=axis))

        # Add constant node for reset input if it is static
        self.add_constant_reset_node(converter_context, input_names, 'LSTM')

        forward_output_names = [module_name + str(name) + QNN_LSTM_OUTPUT_TYPES[i] + "_forward" for i, name in
                                enumerate(output_names)]
        backward_output_names = [module_name + str(name) + QNN_LSTM_OUTPUT_TYPES[i] + "_backward" for i, name in
                                 enumerate(output_names)]
        reshape_output_names = [[module_name + str(name) + QNN_LSTM_OUTPUT_TYPES[i] + "_reshape_forward",
                                 module_name + str(name) + QNN_LSTM_OUTPUT_TYPES[i] + "_reshape_backward"]
                                for i, name in enumerate(output_names)]
        concat_output_names = [str(name) for name in output_names]

        is_qairt_lstm = getattr(graph, 'qairt_converter', False) and rnn_type == 'LSTM' and OnnxTranslations.is_cpp_ir_pass('MULTI_TIME_STEPS_LSTM')
        if is_qairt_lstm:
            # Plug split initial states into per-direction inputs, then expand full input list (preserving x)
            forward_op_input_names[INITIAL_H_IDX] = h_0_forward_split_name
            forward_op_input_names[INITIAL_C_IDX] = c_0_forward_split_name
            backward_op_input_names[INITIAL_H_IDX] = h_0_backward_split_name
            backward_op_input_names[INITIAL_C_IDX] = c_0_backward_split_name

            forward_input_names = [input_names[0]] + forward_op_input_names[1:]
            backward_input_names = [input_names[0]] + backward_op_input_names[1:]

            # Reshape h_0/c_0 if needed and create zero-initial tensors when missing
            forward_input_names = self.add_h_c_inputs_reshape_if_needed(converter_context, src_op, forward_op.name, forward_input_names)
            forward_input_names = self.ensure_h_c_inputs_present(converter_context, forward_op.name, src_op, forward_input_names)
            backward_input_names = self.add_h_c_inputs_reshape_if_needed(converter_context, src_op, backward_op.name, backward_input_names)
            backward_input_names = self.ensure_h_c_inputs_present(converter_context, backward_op.name, src_op, backward_input_names)
        else:
            # For RolledLstm, h_0 and c_0 are second and third inputs respectively
            forward_input_names = [input_names[0], h_0_forward_split_name, c_0_forward_split_name] + forward_op_input_names[3:]
            backward_input_names = [input_names[0], h_0_backward_split_name, c_0_backward_split_name] + backward_op_input_names[3:]


        log_debug("Adding forward RNN op {} while creating bidirectional RNN unit".format(forward_op.name))
        forward_node = graph.add(forward_op, forward_input_names, forward_output_names)
        graph.add_src_op_info(forward_op.name, forward_input_names, forward_output_names)
        forward_targets = [forward_node].extend(graph.get_output_buffers(forward_node))
        self.insert_trace_info(forward_targets, [(src_op.name, TraceType.OP)], converter_context)

        log_debug("Adding backward RNN op {} while creating bidirectional RNN unit".format(backward_op.name))
        backward_node = graph.add(backward_op, backward_input_names, backward_output_names)
        graph.add_src_op_info(backward_op.name, backward_input_names, backward_output_names)
        backward_targets = [backward_node].extend(graph.get_output_buffers(backward_node))
        self.insert_trace_info(backward_targets, [(src_op.name, TraceType.OP)], converter_context)

        # add concat op to the graph, should end up as a child of both forward and backward ops.
        # we need more than one concat node, depending on the number of outputs.
        for id in range(0, output_names_len):
            # add reshape for outputs to make shape to be ONNX's format
            log_debug("Adding reshape op {} and {} while creating bidirectional RNN unit".format(reshape_ops[id][0].name, reshape_ops[id][1].name))
            backward_reshape_node = graph.add(reshape_ops[id][0], [forward_output_names[id]], [reshape_output_names[id][0]])
            forward_reshape_node = graph.add(reshape_ops[id][1], [backward_output_names[id]], [reshape_output_names[id][1]])
            graph.add_src_op_info(reshape_ops[id][0].name, [forward_output_names[id]], [reshape_output_names[id][0]])
            graph.add_src_op_info(reshape_ops[id][1].name, [backward_output_names[id]], [reshape_output_names[id][1]])
            self.insert_trace_info([backward_reshape_node, graph.get_output_buffers(backward_reshape_node)[0]], [(src_op.name, TraceType.OP)], converter_context)
            self.insert_trace_info([forward_reshape_node, graph.get_output_buffers(forward_reshape_node)[0]], [(src_op.name, TraceType.OP)], converter_context)

            log_debug("Adding concat op {} while creating bidirectional RNN unit".format(concat_ops[id].name))
            concat_node = graph.add(concat_ops[id], [reshape_output_names[id][0], reshape_output_names[id][1]], [concat_output_names[id]])
            graph.add_src_op_info(concat_ops[id].name, [reshape_output_names[id][0], reshape_output_names[id][1]], [concat_output_names[id]])
            self.insert_trace_info([concat_node], [(src_op.name, TraceType.OP)], converter_context)
            self.insert_trace_info(graph.get_output_buffers(concat_node), [(concat_output_names[id], TraceType.TENSOR)], converter_context)

            # Set override output encodings for rnn op
            if id < len(src_op.output) and graph.user_quantization_overrides:
                encoding = graph.get_overridden_encoding(src_op.output[id], False)
                if encoding is not None and id == 1 and len(forward_output_names) > 2:
                    graph.set_overridden_encoding(forward_output_names[2], encoding, False)
                    graph.set_overridden_encoding(backward_output_names[2], encoding, False)
                elif encoding is not None and id == 2:
                    graph.set_overridden_encoding(forward_output_names[1], encoding, False)
                    graph.set_overridden_encoding(backward_output_names[1], encoding, False)
                elif encoding is not None:
                    graph.set_overridden_encoding(forward_output_names[id], encoding, False)
                    graph.set_overridden_encoding(backward_output_names[id], encoding, False)


# ------------------------------------------------------------------------------
#  GRU
# ------------------------------------------------------------------------------

class OnnxGruTranslation(OnnxRnnTranslationsBase):
    def __init__(self):
        OnnxRnnTranslationsBase.__init__(self)
        schema_dict = self.register_op_schema('GRU', [1, 7, 14, 22], [['clip',
                                                               'activation_alpha',
                                                               'activation_beta',
                                                               'output_sequence']])
        schema_dict.replace_default_values(activations=['Sigmoid','Tanh'])
        schema_dict.register_method(self.validate_attribute_values)
        self.no_of_gates = 3
        self.linear_before_reset = 0
        self.NUM_INPUTS = gru_op_props.NUM_INPUTS - 1
        self.IR_NUM_INPUTS = gru_op_props.IR_NUM_INPUTS - 1
        self.NUM_REQUIRED_INPUTS = gru_op_props.NUM_REQUIRED_INPUTS

    def _preprocess_gru_weights(self, graph, gru_node_name, input_size, hidden_size,
                                gate_weights, gate_rec_weights, gate_bias, direction=ir_graph.QNN_OP_GRU_DIRECTION_FORWARD):
        # This function is adapted from OptimizeMergedWeightsGruTranslation.preprocess_merged_weights_gru_node
        # in op_graph_optimizations.py to be used within the translation class.
        def add_split_tensor_to_graph(tensor_name, tensor, desired_shape=None, src_name=None):
            # Share the tensor if they are already added in the graph
            if not graph.has_buffer(tensor_name):
                tensor = np.resize(tensor, desired_shape) if desired_shape is not None else tensor
                const_op = op_adapter.ConstantOp(name=tensor_name, tensor=tensor)
                graph.add(const_op, input_names=[], output_names=[tensor_name])
                if not src_name:
                    return
                encoding = graph.get_overridden_encoding(src_name)
                if encoding is not None:
                    graph.set_overridden_encoding(tensor_name, encoding, False)
            elif graph.get_producer_op(tensor_name).type != op_adapter.ConstantOp.TRANSLATION_KEY:
                raise ValueError("GruOp requires weights and biases to be constant, got dynamic tensor from {}".format(
                    graph.get_producer_op(tensor_name).name))

        # Split weights so that they can be indexed by gate
        input_split_weights = np.split(gate_weights, 3, axis=0)
        rec_split_weights = np.split(gate_rec_weights, 3, axis=0)
        gate_split_biases = np.split(gate_bias, 6, axis=0)

        # base names for weights and biases
        suffix = ''
        if self.params.direction == "bidirectional":
            if direction == ir_graph.QNN_OP_GRU_DIRECTION_FORWARD:
                suffix = '_forward'
            elif direction == ir_graph.QNN_OP_GRU_DIRECTION_REVERSE:
                suffix = '_backward'
        base_input_weights_name, base_rec_weights_name, base_bias_name = self.input_names[1:4] if len(self.input_names) >= 4 and self.input_names[3] else self.input_names[1:3] + ['gru_zero_bias']

        w_names = [base_input_weights_name + suffix + s for s in
                   ['_input_w_to_update_gate', '_input_w_to_reset_gate', '_input_w_to_new_gate']]
        r_names = [base_rec_weights_name + suffix + s for s in
                   ['_recurrent_w_to_update_gate', '_recurrent_w_to_reset_gate', '_recurrent_w_to_new_gate']]
        b_names = [base_bias_name + suffix + s for s in
                   ['_input_b_to_update_gate', '_input_b_to_reset_gate', '_input_b_to_new_gate',
                    '_recurrent_b_to_update_gate', '_recurrent_b_to_reset_gate', '_recurrent_b_to_new_gate']]

        # Add input weights and copy quantization parameters
        add_split_tensor_to_graph(w_names[0], input_split_weights[0], desired_shape=(hidden_size, input_size), src_name=base_input_weights_name)
        add_split_tensor_to_graph(w_names[1], input_split_weights[1], desired_shape=(hidden_size, input_size), src_name=base_input_weights_name)
        add_split_tensor_to_graph(w_names[2], input_split_weights[2], desired_shape=(hidden_size, input_size), src_name=base_input_weights_name)

        # Add recurrent weights and copy quantization parameters
        add_split_tensor_to_graph(r_names[0], rec_split_weights[0], desired_shape=(hidden_size, hidden_size), src_name=base_rec_weights_name)
        add_split_tensor_to_graph(r_names[1], rec_split_weights[1], desired_shape=(hidden_size, hidden_size), src_name=base_rec_weights_name)
        add_split_tensor_to_graph(r_names[2], rec_split_weights[2], desired_shape=(hidden_size, hidden_size), src_name=base_rec_weights_name)

        # Add biases and copy quantization parameters
        add_split_tensor_to_graph(b_names[0], gate_split_biases[0], desired_shape=(hidden_size,), src_name=base_bias_name)
        add_split_tensor_to_graph(b_names[1], gate_split_biases[1], desired_shape=(hidden_size,), src_name=base_bias_name)
        add_split_tensor_to_graph(b_names[2], gate_split_biases[2], desired_shape=(hidden_size,), src_name=base_bias_name)
        add_split_tensor_to_graph(b_names[3], gate_split_biases[3], desired_shape=(hidden_size,), src_name=base_bias_name)
        add_split_tensor_to_graph(b_names[4], gate_split_biases[4], desired_shape=(hidden_size,), src_name=base_bias_name)
        add_split_tensor_to_graph(b_names[5], gate_split_biases[5], desired_shape=(hidden_size,), src_name=base_bias_name)

        return w_names + r_names + b_names

    def extract_parameters(self, src_op, converter_context):
        self.params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type))
        self.extract_params_for_type(src_op, converter_context, rnn_type='Gru')

        if len(self.input_names) >= 6:
            # check if initial_h is included
            OPTIONAL_INPUTS.initial_h = self.input_names[5]

        self.linear_before_reset = self.params.linear_before_reset

    def extract_input_names(self, src_op, converter_context):
        input_weights_name = self.input_names[1]
        rec_weights_name = self.input_names[2]
        # Onnx may use empty string as a placeholder, so add an and-condition to further check it
        bias_name = self.input_names[3] if len(self.input_names) >= 4 and self.input_names[3] else 'gru_zero_bias'
        formatted_input_names = [self.input_names[0], OPTIONAL_INPUTS.initial_h, input_weights_name,
                                 rec_weights_name, bias_name]
        return formatted_input_names

    def extract_output_names(self, src_op, converter_context):
        graph = converter_context.ir_graph
        onnx_output_names = [output for i, output in enumerate(src_op.output)]
        dummy_name_list = ["_all_output_dummy", "_hidden_output_dummy"]

        # If the length of onnx_output_names less then 2
        # Firstly, add '' to the onnx_output_names to make it has 2 names
        # Then, use the dummy_name to fill with the empty names
        # Cause we can't make sure the first output name is alway not empty
        # we use the op.name to distinguish different empty name buffer in different op.
        # -------------------------------------------------------------------------------------------------------------
        # | src_op.outputs(src_op.name='Gru') |       onnx_output_names - 1       |      onnx_output_names - 2        |
        # -------------------------------------------------------------------------------------------------------------
        # |         ['Y'], ['Y', '']          |             ['Y', '']             |  ['Y', 'Gru_hidden_output_dummy'] |
        # |         ['', 'Y_h']               |             ['', 'Y_h']           |  ['Gru_all_output_dummy', 'Y_h']  |
        # |         ['Y', 'Y_h']              |             ['Y', 'Y_h']          |  ['Y', 'Y_h']                     |
        # -------------------------------------------------------------------------------------------------------------
        while len(onnx_output_names) < 2:
            onnx_output_names.append('')

        src_name = src_op.name if len(src_op.name) != 0 else \
                onnx_output_names[0] if onnx_output_names[0] else \
                    graph.naming_policy.get_op_name_by_type(op_adapter.GruOp.TRANSLATION_KEY,
                                                            op_adapter.GruOp.LEGACY_TRANSLATION_KEY)

        for idx, name in enumerate(onnx_output_names):
            if not name:
                onnx_output_names[idx] = src_name + dummy_name_list[idx]

        return onnx_output_names

    def extract_src_output_tuples(self, src_op):
        # Since extract_output_names may change the outputs names and their orders. Follow the changes
        # to construct a source output names for framework trace
        src_onnx_output_tuples = []
        for output in src_op.output:
            if output:
                src_onnx_output_tuples.append((output, TraceType.TENSOR))
            else:
                src_onnx_output_tuples.append((src_op.name, TraceType.OP))
        if len(src_onnx_output_tuples) < 2 :
            src_onnx_output_tuples.append((src_op.name, TraceType.OP))
        return src_onnx_output_tuples

    def convert_params_to_snpe(self, weights, rec_weights, bias, hidden_size):
        if bias is None:
            # Bias tensor was not provided, so create an all zeroes bias
            bias = np.zeros((6 * hidden_size), dtype=np.float32)
        else:
            # Flatten incoming tensors for consistency
            if len(bias.shape) > 1:
                bias = np.reshape(bias, (-1,))
            # Due to the linear_before_reset parameter, we may need the
            # recurrent biases split out from the other biases. We need the bias
            # tensor to hold the concatenation of Wb and Rb.
            if bias.shape[0] != 6*hidden_size:
                raise ValueError(f'GRU node bias has incorrect shape. Expected ({6*hidden_size}, ), '
                                 f'got {bias.shape}')

        # weights and rec_weights are also laid out as (no_of_gates*hidden_size, input_size)
        # and (no_of_gates*hidden_size, hidden_size) respectively. We need to reshape
        # to SNPE format depending on the rnn type.
        weights = np.reshape(weights, (self.no_of_gates, hidden_size, weights.shape[-1]))
        rec_weights = np.reshape(rec_weights, (self.no_of_gates, hidden_size, hidden_size))

        return weights, rec_weights, bias

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.extract_parameters(src_op, converter_context)
        self.add_src_op_info(src_op.name, src_op, graph)
        return self.create_rnn(src_op, converter_context, self.create_unidirectional_gru,
                               self.create_bidirectional_gru, rnn_type='Gru')

    def create_unidirectional_gru(self, graph, name, input_name_list, converter_context=None, src_op=None, **kwargs):

        if kwargs:
            [weights, rec_weights, bias] = self.convert_params_to_snpe(kwargs['weights'],
                                                                       kwargs['rec_weights'],
                                                                       kwargs['bias'],
                                                                       kwargs['hidden_size'])
            h_0_input_name = kwargs['h_0_input_name']
            hidden_size = kwargs['hidden_size']
        else:
            [weights, rec_weights, bias] = self.convert_params_to_snpe(self.weights,
                                                                       self.rec_weights,
                                                                       self.bias,
                                                                       self.params.hidden_size)
            h_0_input_name = OPTIONAL_INPUTS.initial_h
            hidden_size = self.params.hidden_size

        activations = self.params.activations
        gate_bias = bias.reshape(-1, )
        gate_rec_weights = rec_weights.reshape(self.no_of_gates * hidden_size, -1)
        gate_weights = weights.reshape(self.no_of_gates * hidden_size, -1)

        g2g_passes = OnnxTranslations.optimization_pass_mode_manager.get_cpp_ir_pass_list()
        if getattr(graph, 'qairt_converter', False) and 'MULTI_TIME_STEPS_GRU' in g2g_passes:
            if not h_0_input_name:
                input_shape = graph.get_buffer(input_name_list[0]).shape[:]
                batch_size, _, _ = AxisOrders.ONNX.extract_time_series_dims(input_shape)
                initial_hidden_state_name = name + '_initial_hidden_state'
                if not graph.has_buffer(initial_hidden_state_name):
                    initial_hidden_state_tensor = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
                    initial_hidden_state_op = op_adapter.ConstantOp(name=initial_hidden_state_name, tensor=initial_hidden_state_tensor)
                    constant_node = graph.add(initial_hidden_state_op, [], [initial_hidden_state_name], axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                    self.insert_constant_trace_info(initial_hidden_state_name, constant_node, converter_context)
                h_0_input_name = initial_hidden_state_name
            input_size = graph.get_buffer(self.input_names[0]).shape[-1]
            shared_input_names = self._preprocess_gru_weights(graph, name, input_size, hidden_size,
                                                              gate_weights, gate_rec_weights, gate_bias,
                                                              direction=kwargs.get('direction', self.direction))

            gru_op_inputs = [input_name_list[0]]
            gru_op_inputs.extend(shared_input_names)
            gru_op_inputs.append(h_0_input_name)
            reset_input_name = ''
            if len(input_name_list) > 5 and input_name_list[5]:
                reset_input_name = input_name_list[5]
                gru_op_inputs.append(reset_input_name)
            input_name_list[:] = gru_op_inputs

            return op_adapter.GruOp(name,
                                    activation=activations[0],
                                    gate_activation=activations[0],
                                    rec_gate_activation=activations[1],
                                    h_0_input_name=h_0_input_name,
                                    reset_input_name=kwargs.get('reset_input_name', ''),
                                    direction=kwargs.get('direction', self.direction),
                                    hidden_size=hidden_size,
                                    linear_before_reset=self.params.linear_before_reset,
                                    time_major=True)
        else:
            params_dict = {}
            # Extract the input names for weights and bias.
            if input_name_list is not None:
                input_weights_name, rec_weights_name, bias_name = input_name_list[2:5]
                params_dict[input_weights_name] = gate_weights
                params_dict[rec_weights_name] = gate_rec_weights
                params_dict[bias_name] = gate_bias
                self.prepare_params_as_constants(graph, params_dict)

            return op_adapter.MergedWeightsGruOp(name,
                                                 activation=activations[0],
                                                 gate_activation=activations[0],
                                                 rec_gate_activation=activations[1],
                                                 h_0_input_name=h_0_input_name,
                                                 direction=kwargs.get('direction', self.direction),
                                                 hidden_size=hidden_size,
                                                 linear_before_reset = self.params.linear_before_reset,
                                                 time_major=True)

    def create_bidirectional_gru(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        initial_h_input_present = True if OPTIONAL_INPUTS.initial_h else False
        src_op_inputs = list(src_op.input)
        src_op_inputs_len = len(src_op_inputs)
        if src_op_inputs_len > self.NUM_INPUTS or src_op_inputs_len < self.NUM_REQUIRED_INPUTS:
            raise ValueError("Unsupported number of inputs on source op {}. Expected between {} and {}, got {}".
                                format(src_op.name, self.NUM_REQUIRED_INPUTS, self.NUM_INPUTS, src_op_inputs_len))

        input_names_len = len(input_names)
        if input_names_len > self.IR_NUM_INPUTS:
            raise ValueError("Unsupported number of inputs for bidirectional unit on op {}. Expected <= {}, got {}".
                                format(src_op.name, self.IR_NUM_INPUTS, input_names_len))

        output_names_len = len(output_names)
        if output_names_len > 2:
            raise ValueError("Unsupported number of outputs for bidirectional unit on op {}. Expected < 2, got {}".
                                format(src_op.name, output_names_len))
        if input_names[1] and len(output_names) == 1:
            final_hidden_output_name = output_names[0] + '_bidirection'
            output_names.append(final_hidden_output_name)
        module_name = src_op.name if len(src_op.name) != 0 else output_names[0]

        def add_split_for_initial_h():
            h_0_forward_split_name, h_0_backward_split_name = '', ''
            if not initial_h_input_present:
                return h_0_forward_split_name, h_0_backward_split_name

            initial_h_name = str(input_names[1])
            if converter_context.weights.has(initial_h_name) and not graph.has_buffer(initial_h_name):
                tensor = converter_context.weights.fetch(initial_h_name, prunable=False)
                constant_node = graph.add(op_adapter.ConstantOp(initial_h_name, tensor),
                                          [],
                                          initial_h_name,
                                          axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                self.insert_constant_trace_info(initial_h_name, constant_node, converter_context)
            elif graph.has_buffer(initial_h_name):
                # set axis format as NONTRIVIAL for dynamic h input
                input_buf = graph.get_buffer(initial_h_name)
                input_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

            split_op_name = str(input_names[1]) + RNN_INPUT_TYPES[0] + "_split"
            split_op = op_adapter.SplitOp(split_op_name, axis=0)
            split_output_names = [f"{split_op_name}_forward", f"{split_op_name}_backward"]
            h_split_output_names = [split_output_names]

            if not graph.has_node(split_op.name):
                split_node = graph.add(split_op, [input_names[1]], split_output_names)
                graph.add_src_op_info(split_op.name, [input_names[1]], split_output_names)
                split_targets = [split_node]
                split_targets.extend(graph.get_output_buffers(split_node))
                self.insert_trace_info(split_targets, (input_names[1], TraceType.TENSOR), converter_context)
                # Set override param encodings for split op
                if graph.user_quantization_overrides:
                    encoding = graph.get_overridden_encoding(input_names[1])
                    if encoding is not None:
                        for output_name in split_output_names:
                            graph.set_overridden_encoding(output_name, encoding, False)

            return h_split_output_names[0]

        def create_reshape_and_concat_outputs(forward_output_names, backward_output_names, output_shapes):
            reshape_ops = []
            for i in range(output_names_len):
                reshape_ops.append([op_adapter.ReshapeOp(output_names[i] + QNN_GRU_OUTPUT_TYPES[i] + '_reshape_forward',
                                                        shape=output_shapes[i]),
                                    op_adapter.ReshapeOp(output_names[i] + QNN_GRU_OUTPUT_TYPES[i] + '_reshape_backward',
                                                        shape=output_shapes[i])])
            concat_ops = []
            for i in range(output_names_len):
                axis = 1 if i == 0 else 0
                concat_ops.append(op_adapter.ConcatOp(output_names[i] + '_concat', axis=axis))

            reshape_output_names = [[str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_reshape_forward",
                                     str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_reshape_backward"]
                                    for i, name in enumerate(output_names)]

            for i in range(output_names_len):
                graph.add(reshape_ops[i][0], [forward_output_names[i]], [reshape_output_names[i][0]])
                graph.add(reshape_ops[i][1], [backward_output_names[i]], [reshape_output_names[i][1]])
                graph.add(concat_ops[i], reshape_output_names[i], [output_names[i]])

                # Set override output encodings for gru ops
                if i < len(src_op.output) and graph.user_quantization_overrides:
                    encoding = graph.get_overridden_encoding(src_op.output[i], False)
                    if encoding is not None:
                        graph.set_overridden_encoding(forward_output_names[i], encoding, False)
                        graph.set_overridden_encoding(backward_output_names[i], encoding, False)

        QNN_GRU_OUTPUT_TYPES = ('_all_hidden', '_final_hidden')
        h_0_forward_split_name, h_0_backward_split_name = add_split_for_initial_h()

        forward_op_input_names = input_names[:]
        backward_op_input_names = input_names[:]

        g2g_passes = OnnxTranslations.optimization_pass_mode_manager.get_cpp_ir_pass_list()
        if getattr(graph, 'qairt_converter', False) and 'MULTI_TIME_STEPS_GRU' in g2g_passes:
            reset_input_name = ''
            if len(input_names) > 5 and input_names[5]:
                reset_input_name = input_names[5]

            forward_op = self.create_unidirectional_gru(graph,
                                                        module_name + '_forward',
                                                        forward_op_input_names,
                                                        converter_context=converter_context,
                                                        src_op=src_op,
                                                        weights=self.weights[0, :, :],
                                                        rec_weights=self.rec_weights[0, :, :],
                                                        bias=self.bias[0, :] if self.bias is not None else self.bias,
                                                        hidden_size=self.params.hidden_size,
                                                        direction=ir_graph.QNN_OP_GRU_DIRECTION_FORWARD,
                                                        h_0_input_name=h_0_forward_split_name,
                                                        reset_input_name=reset_input_name)

            # set up backward op
            backward_op = self.create_unidirectional_gru(graph,
                                                        module_name + '_backward',
                                                        backward_op_input_names,
                                                        converter_context=converter_context,
                                                        src_op=src_op,
                                                        weights=self.weights[1, :, :],
                                                        rec_weights=self.rec_weights[1, :, :],
                                                        bias=self.bias[1, :] if self.bias is not None else self.bias,
                                                        hidden_size=self.params.hidden_size,
                                                        direction=ir_graph.QNN_OP_GRU_DIRECTION_REVERSE,
                                                        h_0_input_name=h_0_backward_split_name,
                                                        reset_input_name=reset_input_name)

            self.add_src_op_info(forward_op.name, src_op, graph)
            self.add_src_op_info(backward_op.name, src_op, graph)

            QNN_GRU_OUTPUT_TYPES = ('_all_hidden', '_final_hidden')

            forward_output_names = [str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_forward" for i, name in
                                    enumerate(output_names)]
            backward_output_names = [str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_backward" for i, name in
                                    enumerate(output_names)]

            graph.add(forward_op, forward_op_input_names, forward_output_names)
            graph.add(backward_op, backward_op_input_names, backward_output_names)

            input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
            batch_size, time_steps, _ = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
            hidden_size = self.params.hidden_size
            output_shapes = []
            for i in range(output_names_len):
                if i == 0:
                    output_shapes.append([time_steps, 1, batch_size, hidden_size])
                else:
                    output_shapes.append([1, batch_size, hidden_size])

            create_reshape_and_concat_outputs(forward_output_names, backward_output_names, output_shapes)
        else:

            # Rename the weights, rec_weights, bias for 2 different directions by adding '_forward' or '_backward'.
            # Later, each above tensor will splite into 2 tensors. Forward uses the 1st batch and the backward 2nd.
            # Do not create separate tensors for reset input, it is shared between forward and backward
            last_idx_to_split = min(input_names_len, gru_op_props.IR_NUM_INPUTS - 1)
            for i in range(2, last_idx_to_split):
                curr_input_name = input_names[i]
                if curr_input_name:
                    forward_op_input_names[i] = curr_input_name + '_forward'
                    backward_op_input_names[i] = curr_input_name + '_backward'
                    # Set override param encodings for new name input_tensor
                    if graph.user_quantization_overrides:
                        encoding = graph.get_overridden_encoding(curr_input_name)
                        if encoding is not None:
                            graph.set_overridden_encoding(forward_op_input_names[i], encoding, False)
                            graph.set_overridden_encoding(backward_op_input_names[i], encoding, False)

            forward_op = self.create_unidirectional_gru(graph,
                                                       module_name + '_forward',
                                                       forward_op_input_names,
                                                       converter_context=converter_context,
                                                       src_op=src_op,
                                                       weights=self.weights[0, :, :],
                                                       rec_weights=self.rec_weights[0, :, :],
                                                       bias=self.bias[0, :] if self.bias is not None else self.bias,
                                                       hidden_size=self.params.hidden_size,
                                                       direction=ir_graph.QNN_OP_GRU_DIRECTION_FORWARD,
                                                       h_0_input_name=h_0_forward_split_name)

            backward_op = self.create_unidirectional_gru(graph,
                                                        module_name + '_backward',
                                                        backward_op_input_names,
                                                        converter_context=converter_context,
                                                        src_op=src_op,
                                                        weights=self.weights[1, :, :],
                                                        rec_weights=self.rec_weights[1, :, :],
                                                        bias=self.bias[1, :] if self.bias is not None else self.bias,
                                                        hidden_size=self.params.hidden_size,
                                                        direction=ir_graph.QNN_OP_GRU_DIRECTION_REVERSE,
                                                        h_0_input_name=h_0_backward_split_name)

            # Add constant node for reset input if it is static
            self.add_constant_reset_node(converter_context, input_names, 'Gru')

            forward_output_names = [str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_forward" for i, name in
                                    enumerate(output_names)]
            backward_output_names = [str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_backward" for i, name in
                                     enumerate(output_names)]
            reshape_output_names = [[str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_reshape_forward",
                                      str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_reshape_backward"]
                                    for i, name in enumerate(output_names)]
            concat_output_names = [str(name) for name in output_names]

            # Modify input names to be different according to split
            forward_input_names = [input_names[0], h_0_forward_split_name] + forward_op_input_names[2:]
            backward_input_names = [input_names[0], h_0_backward_split_name] + backward_op_input_names[2:]

            log_debug("Adding forward RNN op {} while creating bidirectional RNN unit".format(forward_op.name))
            forward_node = graph.add(forward_op, forward_input_names, forward_output_names)
            graph.add_src_op_info(forward_op.name, forward_input_names, forward_output_names)

            forward_targets = [forward_node]
            forward_targets.extend(graph.get_output_buffers(forward_node))
            self.insert_trace_info(forward_targets, [(src_op.name, TraceType.OP)], converter_context)

            log_debug("Adding backward RNN op {} while creating bidirectional RNN unit".format(backward_op.name))
            backward_node = graph.add(backward_op, backward_input_names, backward_output_names)
            graph.add_src_op_info(backward_op.name, backward_input_names, backward_output_names)

            backward_targets = [backward_node].extend(graph.get_output_buffers(backward_node))
            self.insert_trace_info(backward_targets, [(src_op.name, TraceType.OP)], converter_context)

            input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
            batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
            hidden_size = self.params.hidden_size
            output_shapes = []
            for i in range(0, output_names_len):
                if i == 0:
                    output_shapes.append([time_steps, self.num_directions, batch_size, hidden_size])
                else:
                    output_shapes.append([self.num_directions, batch_size, hidden_size])

            create_reshape_and_concat_outputs(forward_output_names, backward_output_names, output_shapes)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'linear_before_reset':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)
        elif attr_name == "activations":
            gru_supported_activations = ["Sigmoid", "Tanh"]
            if len(attr_value) == 4:
                # bidirectional case has double the number of activations
                gru_supported_activations = gru_supported_activations*2
            log_assert(attr_value == gru_supported_activations,
                       "Received activations list: {} differs from GRU op {} supported activations list: {}",
                       attr_value, src_op.name, gru_supported_activations)



OnnxTranslations.register_translation(OnnxGruTranslation(),
                                      converter_type('GRU', 'onnx'),
                                      op_adapter.GruOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#  LSTM
# ------------------------------------------------------------------------------
class OnnxLSTMTranslation(OnnxRnnTranslationsBase):
    from qti.aisw.converters.common.converter_ir.op_properties.stateful_lstm\
        import IR_INPUT_IDX, IR_INPUT_WEIGHTS_IDX, IR_HIDDEN_STATE_WEIGHTS_IDX,\
        IR_GATE_BIASES_IDX, IR_NORM_WEIGHTS_IDX, IR_CELL_STATE_WEIGHTS_IDX,\
        IR_PROJ_WEIGHTS_IDX, IR_PROJ_BIAS_IDX, IR_RESET_IDX,\
        IR_INITIAL_H_IDX, IR_INITIAL_C_IDX, IR_NUM_INPUTS,\
        IR_TO_ONNX_INDICES, \
        LSTM_IR_NUM_INPUTS, LSTM_IR_DATA_INPUT_IDX, \
        LSTM_INPUT_WEIGHT_IDX, LSTM_INPUT_WEIGHT_TO_FORGET_GATE_IDX, LSTM_INPUT_WEIGHT_TO_CELL_GATE_IDX, LSTM_INPUT_WEIGHT_TO_OUTPUT_GATE_IDX, \
        LSTM_HIDDEN_WEIGHT_IDX, LSTM_RECURRENT_WEIGHT_TO_FORGET_GATE_IDX, LSTM_RECURRENT_WEIGHT_TO_CELL_GATE_IDX, LSTM_RECURRENT_WEIGHT_TO_OUTPUT_GATE_IDX, \
        LSTM_BIAS_IDX, LSTM_BIAS_TO_FORGET_GATE_IDX, LSTM_BIAS_TO_CELL_GATE_IDX, LSTM_BIAS_TO_OUTPUT_GATE_IDX, \
        LSTM_IR_INITIAL_H_IDX, LSTM_IR_INITIAL_C_IDX, \
        LSTM_NORM_WEIGHT_TO_INPUT_GATE_IDX, LSTM_NORM_WEIGHT_TO_FORGET_GATE_IDX, LSTM_NORM_WEIGHT_TO_CELL_GATE_IDX, LSTM_NORM_WEIGHT_TO_OUTPUT_GATE_IDX, \
        LSTM_INPUT_WEIGHT_TO_INPUT_GATE_IDX, LSTM_RECURRENT_WEIGHT_TO_INPUT_GATE_IDX, \
        LSTM_CELL_WEIGHT_TO_INPUT_GATE_IDX, LSTM_CELL_WEIGHT_TO_FORGET_GATE_IDX, LSTM_CELL_WEIGHT_TO_OUTPUT_GATE_IDX, \
        LSTM_BIAS_TO_INPUT_GATE_IDX, \
        LSTM_PROJ_W_IDX, LSTM_PROJ_B_IDX, LSTM_IR_RESET_IDX, \
        LSTM_IR_TO_ONNX_INDICES, LSTM_IR_TO_GATE_INDICES

    def __init__(self):
        OnnxRnnTranslationsBase.__init__(self)
        schema_dict = self.register_op_schema('LSTM', [1, 7, 14, 22], [['clip',
                                                                'activation_alpha',
                                                                'activation_beta',
                                                                'output_sequence']])
        schema_dict.replace_default_values(activations=['Sigmoid', 'Tanh', 'Tanh'])
        schema_dict.register_method(self.validate_attribute_values)
        self.no_of_gates = 4
        self.peephole_weights = []
        # Number of inputs for standard ONNX LSTM just excludes the reset input
        # TODO: Update when op properties are refactored into C++
        self.NUM_INPUTS = lstm_op_props.NUM_INPUTS - 1

    def extract_parameters(self, src_op, converter_context):
        schema = self.op_schema(op_type=src_op.op_type)
        self.params = extract_attributes(src_op, schema=schema, validate=True)

        if "layout" in self.params and self.params.layout == 1:
            raise ValueError("ERROR: Unsupported value - 1 for layout attribute for {} op".format(src_op.name))

        # set parameters
        self.extract_params_for_type(src_op, converter_context)

        # check if initial_h and initial_c are included
        # snpe requires that if they are included, then all
        # 3 outputs will be returned.
        ONNX_INITIAL_H_IDX = self.IR_TO_ONNX_INDICES[self.IR_INITIAL_H_IDX]
        ONNX_INITIAL_C_IDX = self.IR_TO_ONNX_INDICES[self.IR_INITIAL_C_IDX]
        if len(self.input_names) > ONNX_INITIAL_H_IDX:
            OPTIONAL_INPUTS.initial_h = self.input_names[ONNX_INITIAL_H_IDX]
        else:
            OPTIONAL_INPUTS.initial_h = ''
        if len(self.input_names) > ONNX_INITIAL_C_IDX:
            OPTIONAL_INPUTS.initial_c = self.input_names[ONNX_INITIAL_C_IDX]
        else:
            OPTIONAL_INPUTS.initial_c = ''

    def reshape_split_tensor(self, tensor, desired_shape):
        # Share the tensor if they are already added in the graph
        tensor = np.resize(tensor, desired_shape) if desired_shape else tensor
        return tensor

    def split_lstm_tensor_per_gate(self, tensor, split_axis=0):
        split_sections = int(tensor.shape[split_axis] / self.params.hidden_size)
        param_split_tensor = np.split(tensor, indices_or_sections=split_sections, axis=split_axis)
        return param_split_tensor

    def extract_input_names_for_qairt(self, src_op, converter_context):
        """
        Build the expanded LSTM input list (25 entries) for the QAIRT multi-time-step flavor.
        This function assumes common validation and source-info bookkeeping have already been
        performed in extract_input_names. It focuses solely on expanding ONNX inputs into
        per-gate IR inputs and wiring optional tensors.
        """
        # ONNX index mapping
        ONNX_DATA_INPUT = self.LSTM_IR_TO_ONNX_INDICES[self.LSTM_IR_DATA_INPUT_IDX]
        ONNX_WEIGHT_INPUT = self.LSTM_IR_TO_ONNX_INDICES[self.LSTM_INPUT_WEIGHT_IDX]
        ONNX_HIDDEN_STATE_WEIGHT = self.LSTM_IR_TO_ONNX_INDICES[self.LSTM_HIDDEN_WEIGHT_IDX]
        ONNX_BIAS_INPUT = self.LSTM_IR_TO_ONNX_INDICES[self.LSTM_BIAS_IDX]
        ONNX_RESET_INPUT = self.LSTM_IR_TO_ONNX_INDICES[self.LSTM_IR_RESET_IDX]

        # Base ONNX tensor names
        data_input_name = self.input_names[ONNX_DATA_INPUT]
        weights_input_name = self.input_names[ONNX_WEIGHT_INPUT]
        hidden_state_weights_name = self.input_names[ONNX_HIDDEN_STATE_WEIGHT]
        gate_biases_base_name = self.input_names[ONNX_BIAS_INPUT] if len(self.input_names) > ONNX_BIAS_INPUT and self.input_names[ONNX_BIAS_INPUT] else 'lstm_dummy_bias_all_zeros'

        # Optional tensors (layer norm, peephole, projection) come via OPTIONAL_INPUTS
        norm_weights_base_name = OPTIONAL_INPUTS.norm_weights
        cell_state_weights_base_name = OPTIONAL_INPUTS.cell_state_weights
        proj_w_name = OPTIONAL_INPUTS.proj_weights
        proj_b_name = OPTIONAL_INPUTS.proj_bias

        # Reset input may exist for stateful flavor
        reset_input_name = self.input_names[ONNX_RESET_INPUT] if len(self.input_names) > ONNX_RESET_INPUT and self.input_names[ONNX_RESET_INPUT] else ''

        # Gate-specific derived names (ifoc order)
        input_w_to_forget_gate_name = weights_input_name + '_input_w_to_forget_gate'
        input_w_to_cell_gate_name   = weights_input_name + '_input_w_to_cell_gate'
        input_w_to_output_gate_name = weights_input_name + '_input_w_to_output_gate'
        input_w_to_input_gate_name  = weights_input_name + '_input_w_to_input_gate'

        recurrent_w_to_forget_gate_name = hidden_state_weights_name + '_recurrent_w_to_forget_gate'
        recurrent_w_to_cell_gate_name   = hidden_state_weights_name + '_recurrent_w_to_cell_gate'
        recurrent_w_to_output_gate_name = hidden_state_weights_name + '_recurrent_w_to_output_gate'
        recurrent_w_to_input_gate_name  = hidden_state_weights_name + '_recurrent_w_to_input_gate'

        b_to_forget_gate_name = gate_biases_base_name + '_b_to_forget_gate'
        b_to_cell_gate_name   = gate_biases_base_name + '_b_to_cell_gate'
        b_to_output_gate_name = gate_biases_base_name + '_b_to_output_gate'
        b_to_input_gate_name  = gate_biases_base_name + '_b_to_input_gate'

        # Optional layer-norm gate names (populate only when base provided)
        norm_w_to_input_gate_name  = norm_weights_base_name + '_norm_w_to_input_gate'  if len(norm_weights_base_name) > 0 else ''
        norm_w_to_forget_gate_name = norm_weights_base_name + '_norm_w_to_forget_gate' if len(norm_weights_base_name) > 0 else ''
        norm_w_to_cell_gate_name   = norm_weights_base_name + '_norm_w_to_cell_gate'   if len(norm_weights_base_name) > 0 else ''
        norm_w_to_output_gate_name = norm_weights_base_name + '_norm_w_to_output_gate' if len(norm_weights_base_name) > 0 else ''

        # Optional peephole weights (populate only when base provided)
        cell_w_to_input_gate_name  = cell_state_weights_base_name + '_cell_w_to_input_gate'  if len(cell_state_weights_base_name) > 0 else ''
        cell_w_to_forget_gate_name = cell_state_weights_base_name + '_cell_w_to_forget_gate' if len(cell_state_weights_base_name) > 0 else ''
        cell_w_to_output_gate_name = cell_state_weights_base_name + '_cell_w_to_output_gate' if len(cell_state_weights_base_name) > 0 else ''

        # Build formatted list aligned to LSTM IR indices
        formatted_input_names = [''] * self.LSTM_IR_NUM_INPUTS
        formatted_input_names[self.LSTM_IR_DATA_INPUT_IDX]                         = data_input_name

        formatted_input_names[self.LSTM_INPUT_WEIGHT_TO_FORGET_GATE_IDX]           = input_w_to_forget_gate_name
        formatted_input_names[self.LSTM_INPUT_WEIGHT_TO_CELL_GATE_IDX]             = input_w_to_cell_gate_name
        formatted_input_names[self.LSTM_INPUT_WEIGHT_TO_OUTPUT_GATE_IDX]           = input_w_to_output_gate_name

        formatted_input_names[self.LSTM_RECURRENT_WEIGHT_TO_FORGET_GATE_IDX]       = recurrent_w_to_forget_gate_name
        formatted_input_names[self.LSTM_RECURRENT_WEIGHT_TO_CELL_GATE_IDX]         = recurrent_w_to_cell_gate_name
        formatted_input_names[self.LSTM_RECURRENT_WEIGHT_TO_OUTPUT_GATE_IDX]       = recurrent_w_to_output_gate_name

        formatted_input_names[self.LSTM_BIAS_TO_FORGET_GATE_IDX]                   = b_to_forget_gate_name
        formatted_input_names[self.LSTM_BIAS_TO_CELL_GATE_IDX]                     = b_to_cell_gate_name
        formatted_input_names[self.LSTM_BIAS_TO_OUTPUT_GATE_IDX]                   = b_to_output_gate_name

        formatted_input_names[self.LSTM_IR_INITIAL_H_IDX]                          = OPTIONAL_INPUTS.initial_h
        formatted_input_names[self.LSTM_IR_INITIAL_C_IDX]                          = OPTIONAL_INPUTS.initial_c

        formatted_input_names[self.LSTM_NORM_WEIGHT_TO_INPUT_GATE_IDX]             = norm_w_to_input_gate_name
        formatted_input_names[self.LSTM_NORM_WEIGHT_TO_FORGET_GATE_IDX]            = norm_w_to_forget_gate_name
        formatted_input_names[self.LSTM_NORM_WEIGHT_TO_CELL_GATE_IDX]              = norm_w_to_cell_gate_name
        formatted_input_names[self.LSTM_NORM_WEIGHT_TO_OUTPUT_GATE_IDX]            = norm_w_to_output_gate_name

        formatted_input_names[self.LSTM_INPUT_WEIGHT_TO_INPUT_GATE_IDX]            = input_w_to_input_gate_name
        formatted_input_names[self.LSTM_RECURRENT_WEIGHT_TO_INPUT_GATE_IDX]        = recurrent_w_to_input_gate_name

        formatted_input_names[self.LSTM_CELL_WEIGHT_TO_INPUT_GATE_IDX]             = cell_w_to_input_gate_name
        formatted_input_names[self.LSTM_CELL_WEIGHT_TO_FORGET_GATE_IDX]            = cell_w_to_forget_gate_name
        formatted_input_names[self.LSTM_CELL_WEIGHT_TO_OUTPUT_GATE_IDX]            = cell_w_to_output_gate_name

        formatted_input_names[self.LSTM_BIAS_TO_INPUT_GATE_IDX]                    = b_to_input_gate_name
        formatted_input_names[self.LSTM_PROJ_W_IDX]                                = proj_w_name
        formatted_input_names[self.LSTM_PROJ_B_IDX]                                = proj_b_name
        formatted_input_names[self.LSTM_IR_RESET_IDX]                              = reset_input_name

        return formatted_input_names

    def extract_input_names(self, src_op, converter_context):
        graph = converter_context.ir_graph

        ONNX_INPUT_IDX = self.IR_TO_ONNX_INDICES[self.IR_INPUT_IDX]
        ONNX_INPUT_WEIGHTS_IDX = self.IR_TO_ONNX_INDICES[self.IR_INPUT_WEIGHTS_IDX]
        ONNX_HIDDEN_STATE_WEIGHTS_IDX = self.IR_TO_ONNX_INDICES[self.IR_HIDDEN_STATE_WEIGHTS_IDX]
        ONNX_GATE_BIASES_IDX = self.IR_TO_ONNX_INDICES[self.IR_GATE_BIASES_IDX]

        src_op_inputs_len = len(list(src_op.input))
        if src_op_inputs_len < lstm_op_props.NUM_REQUIRED_INPUTS or src_op_inputs_len > self.NUM_INPUTS:
            raise ValueError(f"Unsupported number of inputs for source LSTM op {src_op.name}. "
                             f"Expected number of inputs between {lstm_op_props.NUM_REQUIRED_INPUTS} "
                             f"and {self.NUM_INPUTS}, got {src_op_inputs_len}")

        input_weights_name = self.input_names[ONNX_INPUT_WEIGHTS_IDX]
        rec_weights_name = self.input_names[ONNX_HIDDEN_STATE_WEIGHTS_IDX]
        if len(self.input_names) > ONNX_GATE_BIASES_IDX:
            bias_name = self.input_names[ONNX_GATE_BIASES_IDX]
            self.bias_src_info = [(self.input_names[ONNX_GATE_BIASES_IDX], TraceType.TENSOR)]
        else:
            bias_name = ''
            self.bias_src_info = []
        # If no bias_name input provided, replace with dummy all-zero bias
        bias_name = bias_name if bias_name else 'lstm_dummy_bias_all_zeros'

        self.weights_src_info = [(self.input_names[ONNX_INPUT_WEIGHTS_IDX], TraceType.TENSOR)]
        self.rec_weights_src_info = [(self.input_names[ONNX_HIDDEN_STATE_WEIGHTS_IDX], TraceType.TENSOR)]

        # If QAIRT multi-time-step converter is active, build expanded gate list after common setup
        if getattr(graph, 'qairt_converter', False) and OnnxTranslations.is_cpp_ir_pass('MULTI_TIME_STEPS_LSTM'):
            return self.extract_input_names_for_qairt(src_op, converter_context)

        # Base ONNX LSTM (RolledLstm) inputs excluding reset
        formatted_input_names = [''] * (self.IR_NUM_INPUTS - 1)
        formatted_input_names[self.IR_INPUT_IDX] = self.input_names[ONNX_INPUT_IDX]
        formatted_input_names[self.IR_INITIAL_H_IDX] = OPTIONAL_INPUTS.initial_h
        formatted_input_names[self.IR_INITIAL_C_IDX] = OPTIONAL_INPUTS.initial_c
        formatted_input_names[self.IR_INPUT_WEIGHTS_IDX] = input_weights_name
        formatted_input_names[self.IR_HIDDEN_STATE_WEIGHTS_IDX] = rec_weights_name
        formatted_input_names[self.IR_GATE_BIASES_IDX] = bias_name
        formatted_input_names[self.IR_NORM_WEIGHTS_IDX] = OPTIONAL_INPUTS.norm_weights
        formatted_input_names[self.IR_CELL_STATE_WEIGHTS_IDX] = OPTIONAL_INPUTS.cell_state_weights
        formatted_input_names[self.IR_PROJ_WEIGHTS_IDX] = OPTIONAL_INPUTS.proj_weights
        formatted_input_names[self.IR_PROJ_BIAS_IDX] = OPTIONAL_INPUTS.proj_bias

        return formatted_input_names

    def convert_params_to_snpe(self, weights, rec_weights, bias, hidden_size):

        no_of_gates = self.no_of_gates

        if bias is None:
            bias = np.zeros((no_of_gates, hidden_size), dtype=np.float32)
        else:
            # for probably vendor specific reasons, ONNX defines LSTM bias to be
            # separated into forward and recurrent parts, that are always added
            # together So we will always combine.  We need to reshape bias which
            # is in (2*no_of_gates*hidden_size) into (2, no_of_gates *
            # hidden_size).
            bias = np.reshape(bias, (2, no_of_gates * hidden_size))
            new_bias = np.empty((no_of_gates * hidden_size), dtype=np.float32)
            # Elements are stored in [weights, rec_weights] where each column
            # represents the gate and the number of rows is the hidden size
            np.add(bias[0, :], bias[1, :], out=new_bias[:])
            bias = new_bias.reshape(no_of_gates, hidden_size)

        # weights and rec_weights are also laid out as (no_of_gates*hidden_size, input_size)
        # and (no_of_gates*hidden_size, hidden_size)respectively. We need to reshape
        # to SNPE format depending on the rnn type.
        weights = np.reshape(weights, (no_of_gates, hidden_size, weights.shape[-1]))
        rec_weights = np.reshape(rec_weights, (no_of_gates, hidden_size, hidden_size))

        return weights, rec_weights, bias

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.extract_parameters(src_op, converter_context)
        self.add_src_op_info(src_op.name, src_op, graph)
        return self.create_rnn(src_op, converter_context, self.create_unidirectional_lstm,
                               self.create_bidirectional_lstm, rnn_type='LSTM')

    def _copy_encoding(self, src_name, new_names, graph):
        if not src_name:
            return
        encoding = graph.get_overridden_encoding(src_name)
        if encoding is not None:
            for n in new_names:
                if n:
                    graph.set_overridden_encoding(n, encoding, False)

    def is_const_postprocessed(self, const_tensor, target_tensor):
        """
        This function is to check whether the post-processed weights/biases is updated to the graph.
        For weights and biases, they could be shared across multiple RolledLstm nodes.
        """
        return (
            const_tensor.shape == target_tensor.shape and np.allclose(const_tensor, target_tensor)
        )

    def create_unidirectional_lstm(self, graph, op_name, input_name_list, converter_context = None, src_op = None, **kwargs):

        if getattr(graph, 'qairt_converter', False) and OnnxTranslations.is_cpp_ir_pass('MULTI_TIME_STEPS_LSTM'):
            suffix = '' if OnnxTranslations.is_cpp_ir_pass('UNROLL_LSTM_TIME_STEPS') or OnnxTranslations.is_cpp_ir_pass('EXPAND_LSTM_OP_STRUCTURE') else '_multi_time_step'
            return self.create_unidirectional_lstm_qairt(graph, op_name + suffix, input_name_list, src_op, **kwargs)

        direction = kwargs['direction'] if kwargs else self.direction
        h_0_input_name = kwargs['h_0_input_name'] if kwargs else OPTIONAL_INPUTS.initial_h
        c_0_input_name = kwargs['c_0_input_name'] if kwargs else OPTIONAL_INPUTS.initial_c
        input_weights_name = input_name_list[self.IR_INPUT_WEIGHTS_IDX]
        rec_weights_name = input_name_list[self.IR_HIDDEN_STATE_WEIGHTS_IDX]
        bias_name = input_name_list[self.IR_GATE_BIASES_IDX]
        cell_state_weights_name = input_name_list[self.IR_CELL_STATE_WEIGHTS_IDX]

        if kwargs:
            weights = kwargs['weights']
            rec_weights = kwargs['rec_weights']
            bias = kwargs['bias']
            hidden_size = kwargs['hidden_size']
            src_infos = {
                input_weights_name: (op_name, TraceType.OP),
                rec_weights_name: (op_name, TraceType.OP),
                bias_name: (op_name, TraceType.OP)
            }
        else:
            weights = self.weights
            rec_weights = self.rec_weights
            bias = self.bias
            hidden_size = self.params.hidden_size
            src_infos = {
                input_weights_name: self.weights_src_info,
                rec_weights_name: self.rec_weights_src_info,
                bias_name: self.bias_src_info
            }

        self.src_input_dic.update(src_infos)

        gate_weights, gate_rec_weights, gate_bias = self.convert_params_to_snpe(
            weights, rec_weights, bias, hidden_size
        )
        params_dict = {}

        # Transform input weights from iofc order to ifoc order and reshape
        gate_weights = np.take(gate_weights, (0, 2, 1, 3), axis=0)
        gate_weights = gate_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)
        if not graph.has_buffer(input_weights_name) or not self.is_const_postprocessed(graph.get_producer_op(input_weights_name).tensor, gate_weights):
            # Add/Update the constant in the graph
            params_dict[input_weights_name] = gate_weights

        # Transform recurrent weights from iofc order to ifoc order and reshape
        gate_rec_weights = np.take(gate_rec_weights, (0, 2, 1, 3), axis=0)
        gate_rec_weights = gate_rec_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)
        if not graph.has_buffer(rec_weights_name) or not self.is_const_postprocessed(graph.get_producer_op(rec_weights_name).tensor, gate_rec_weights):
            params_dict[rec_weights_name] = gate_rec_weights

        # Transform biases from iofc order to ifoc order and reshape
        gate_bias = np.take(gate_bias, (0, 2, 1, 3), axis=0)
        gate_bias = gate_bias.reshape(-1,)
        if not graph.has_buffer(bias_name) or not self.is_const_postprocessed(graph.get_producer_op(bias_name).tensor, gate_bias):
            params_dict[bias_name] = gate_bias

        # cell state weights is the only optional input that can be captured in Onnx lstm
        if cell_state_weights_name:
            params_dict[cell_state_weights_name] = self.cell_state_weights

        self.prepare_params_as_constants(graph, params_dict)

        return op_adapter.RolledLstmOp(op_name,
                                        direction=direction,
                                        time_major=True,
                                        c_0_input_name=c_0_input_name,
                                        h_0_input_name=h_0_input_name,
                                        # if c_0 and h_0 exist, reset_state_at_time_step_0 will be False
                                        reset_state_at_time_step_0=False if h_0_input_name and c_0_input_name else True,
                                        hidden_size=self.params.hidden_size)

    def create_unidirectional_lstm_qairt(self, graph, op_name, input_name_list, src_op, **kwargs):
        """
        Create a QAIRT multi-time-step LSTM op with gate-expanded parameter tensors.
        This reformulation keeps behavior identical while reducing duplication, clarifying gate
        handling (iofc -> ifoc), and centralizing constant updates.
        """
        # Resolve direction and initial states
        direction = kwargs['direction'] if kwargs else self.direction
        h_0_input_name = kwargs['h_0_input_name'] if kwargs else OPTIONAL_INPUTS.initial_h
        c_0_input_name = kwargs['c_0_input_name'] if kwargs else OPTIONAL_INPUTS.initial_c
        src_op_inputs = list(src_op.input) if src_op is not None else []

        # Resolve canon weights/bias set and hidden_size
        if kwargs:
            weights = kwargs['weights']
            rec_weights = kwargs['rec_weights']
            bias = kwargs['bias']
            hidden_size = kwargs['hidden_size']
        else:
            weights = self.weights
            rec_weights = self.rec_weights
            bias = self.bias
            hidden_size = self.params.hidden_size

        # Trace inputs for gate-expanded tensors
        trace_indices = [
            self.LSTM_INPUT_WEIGHT_TO_FORGET_GATE_IDX,
            self.LSTM_INPUT_WEIGHT_TO_CELL_GATE_IDX,
            self.LSTM_INPUT_WEIGHT_TO_OUTPUT_GATE_IDX,
            self.LSTM_INPUT_WEIGHT_TO_INPUT_GATE_IDX,
            self.LSTM_RECURRENT_WEIGHT_TO_FORGET_GATE_IDX,
            self.LSTM_RECURRENT_WEIGHT_TO_CELL_GATE_IDX,
            self.LSTM_RECURRENT_WEIGHT_TO_OUTPUT_GATE_IDX,
            self.LSTM_RECURRENT_WEIGHT_TO_INPUT_GATE_IDX,
            self.LSTM_BIAS_TO_FORGET_GATE_IDX,
            self.LSTM_BIAS_TO_CELL_GATE_IDX,
            self.LSTM_BIAS_TO_OUTPUT_GATE_IDX,
            self.LSTM_BIAS_TO_INPUT_GATE_IDX
        ]
        src_infos = {input_name_list[idx]: (op_name, TraceType.OP) for idx in trace_indices}
        self.src_input_dic.update(src_infos)

        # Determine input size from data input buffer
        input_data_name = self.input_names[self.LSTM_IR_DATA_INPUT_IDX]
        input_size = graph.get_buffer(input_data_name).shape[-1]

        # Convert to SNPE tensors (gate-major), then reorder gates from iofc -> ifoc
        gate_weights, gate_rec_weights, gate_bias = self.convert_params_to_snpe(weights, rec_weights, bias, hidden_size)

        def reorder_to_ifoc(gate_tensor):
            # ONNX iofc ordering -> backend ifoc ordering along gate axis (0)
            return np.take(gate_tensor, (0, 2, 1, 3), axis=0)

        # Input weights: reshape to 2D and split per gate
        in_w = reorder_to_ifoc(gate_weights).reshape(self.no_of_gates * hidden_size, -1)
        in_w_split = self.split_lstm_tensor_per_gate(in_w)

        # Recurrent weights: reshape to 2D and split per gate
        rec_w = reorder_to_ifoc(gate_rec_weights).reshape(self.no_of_gates * hidden_size, -1)
        rec_w_split = self.split_lstm_tensor_per_gate(rec_w)

        # Biases: flatten and split per gate
        b = reorder_to_ifoc(gate_bias).reshape(-1, )
        b_split = self.split_lstm_tensor_per_gate(b)

        params_dict = {}

        def add_param_if_needed(t_name, t_value):
            # Add/Update constant only if not present or outdated
            if not graph.has_buffer(t_name) or not self.is_const_postprocessed(graph.get_producer_op(t_name).tensor, t_value):
                params_dict[t_name] = t_value

        # Input weights mapped to gates
        add_param_if_needed(
            input_name_list[self.LSTM_INPUT_WEIGHT_TO_FORGET_GATE_IDX],
            self.reshape_split_tensor(in_w_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_INPUT_WEIGHT_TO_FORGET_GATE_IDX]],
                                      desired_shape=(hidden_size, input_size))
        )
        add_param_if_needed(
            input_name_list[self.LSTM_INPUT_WEIGHT_TO_CELL_GATE_IDX],
            self.reshape_split_tensor(in_w_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_INPUT_WEIGHT_TO_CELL_GATE_IDX]],
                                      desired_shape=(hidden_size, input_size))
        )
        add_param_if_needed(
            input_name_list[self.LSTM_INPUT_WEIGHT_TO_OUTPUT_GATE_IDX],
            self.reshape_split_tensor(in_w_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_INPUT_WEIGHT_TO_OUTPUT_GATE_IDX]],
                                      desired_shape=(hidden_size, input_size))
        )
        add_param_if_needed(
            input_name_list[self.LSTM_INPUT_WEIGHT_TO_INPUT_GATE_IDX],
            self.reshape_split_tensor(in_w_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_INPUT_WEIGHT_TO_INPUT_GATE_IDX]],
                                      desired_shape=(hidden_size, input_size))
        )

        # Recurrent weights mapped to gates
        add_param_if_needed(
            input_name_list[self.LSTM_RECURRENT_WEIGHT_TO_FORGET_GATE_IDX],
            self.reshape_split_tensor(rec_w_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_RECURRENT_WEIGHT_TO_FORGET_GATE_IDX]],
                                      desired_shape=(hidden_size, hidden_size))
        )
        add_param_if_needed(
            input_name_list[self.LSTM_RECURRENT_WEIGHT_TO_CELL_GATE_IDX],
            self.reshape_split_tensor(rec_w_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_RECURRENT_WEIGHT_TO_CELL_GATE_IDX]],
                                      desired_shape=(hidden_size, hidden_size))
        )
        add_param_if_needed(
            input_name_list[self.LSTM_RECURRENT_WEIGHT_TO_OUTPUT_GATE_IDX],
            self.reshape_split_tensor(rec_w_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_RECURRENT_WEIGHT_TO_OUTPUT_GATE_IDX]],
                                      desired_shape=(hidden_size, hidden_size))
        )
        add_param_if_needed(
            input_name_list[self.LSTM_RECURRENT_WEIGHT_TO_INPUT_GATE_IDX],
            self.reshape_split_tensor(rec_w_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_RECURRENT_WEIGHT_TO_INPUT_GATE_IDX]],
                                      desired_shape=(hidden_size, hidden_size))
        )

        # Biases mapped to gates
        add_param_if_needed(
            input_name_list[self.LSTM_BIAS_TO_FORGET_GATE_IDX],
            self.reshape_split_tensor(b_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_BIAS_TO_FORGET_GATE_IDX]],
                                      desired_shape=(hidden_size, ))
        )
        add_param_if_needed(
            input_name_list[self.LSTM_BIAS_TO_CELL_GATE_IDX],
            self.reshape_split_tensor(b_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_BIAS_TO_CELL_GATE_IDX]],
                                      desired_shape=(hidden_size, ))
        )
        add_param_if_needed(
            input_name_list[self.LSTM_BIAS_TO_OUTPUT_GATE_IDX],
            self.reshape_split_tensor(b_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_BIAS_TO_OUTPUT_GATE_IDX]],
                                      desired_shape=(hidden_size, ))
        )
        add_param_if_needed(
            input_name_list[self.LSTM_BIAS_TO_INPUT_GATE_IDX],
            self.reshape_split_tensor(b_split[self.LSTM_IR_TO_GATE_INDICES[self.LSTM_BIAS_TO_INPUT_GATE_IDX]],
                                      desired_shape=(hidden_size, ))
        )

        # Note:
        # - input_name_list[10] and [11] (initial h/c) are added in create_rnn/create_bidirectional_module
        # - input_name_list[12..15] are layer norm weights; stateful LSTM does not provide these and they remain empty
        # - input_name_list[18..20] are peephole weights; stateful LSTM does not provide these

        # Commit constants (weights/rec_weights/biases per gate)
        self.prepare_params_as_constants(graph, params_dict)

        if len(src_op_inputs) > 1 and src_op_inputs[1]:
            self._copy_encoding(src_op_inputs[1], [
                input_name_list[self.LSTM_INPUT_WEIGHT_TO_FORGET_GATE_IDX],
                input_name_list[self.LSTM_INPUT_WEIGHT_TO_CELL_GATE_IDX],
                input_name_list[self.LSTM_INPUT_WEIGHT_TO_OUTPUT_GATE_IDX],
                input_name_list[self.LSTM_INPUT_WEIGHT_TO_INPUT_GATE_IDX]
            ], graph)

        # Copy encodings for recurrent (hidden state) weights to each gate-specific tensor
        if len(src_op_inputs) > 2 and src_op_inputs[2]:
            self._copy_encoding(src_op_inputs[2], [
                input_name_list[self.LSTM_RECURRENT_WEIGHT_TO_FORGET_GATE_IDX],
                input_name_list[self.LSTM_RECURRENT_WEIGHT_TO_CELL_GATE_IDX],
                input_name_list[self.LSTM_RECURRENT_WEIGHT_TO_OUTPUT_GATE_IDX],
                input_name_list[self.LSTM_RECURRENT_WEIGHT_TO_INPUT_GATE_IDX]
            ], graph)

        # Copy encodings for gate biases to each gate-specific tensor
        if len(src_op_inputs) > 3 and src_op_inputs[3]:
            self._copy_encoding(src_op_inputs[3], [
                input_name_list[self.LSTM_BIAS_TO_FORGET_GATE_IDX],
                input_name_list[self.LSTM_BIAS_TO_CELL_GATE_IDX],
                input_name_list[self.LSTM_BIAS_TO_OUTPUT_GATE_IDX],
                input_name_list[self.LSTM_BIAS_TO_INPUT_GATE_IDX]
            ], graph)

        return op_adapter.LstmOp(name=op_name,
                                 direction=direction,
                                 time_major=True,
                                 c_0_input_name=c_0_input_name,
                                 h_0_input_name=h_0_input_name,
                                 # if c_0 and h_0 exist, reset_state_at_time_step_0 will be False
                                 reset_state_at_time_step_0=False if h_0_input_name and c_0_input_name else True,
                                 hidden_size=self.params.hidden_size)

    def create_bidirectional_lstm(self, src_op, converter_context):
        return self.create_bidirectional_module(src_op, converter_context, self.weights, self.rec_weights,
                                                self.bias,
                                                self.params, self.create_unidirectional_lstm, rnn_type = 'LSTM')

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'input_forget':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)
        elif attr_name == "activations":
            lstm_supported_activations = ["Sigmoid", "Tanh", "Tanh"]
            if len(attr_value) == 6:
                # bidirectional case has double the number of activations
                lstm_supported_activations = lstm_supported_activations*2
            log_assert(attr_value == lstm_supported_activations,
                       "Received activations list: {} differs from LSTM op {} supported activations list: {}",
                       attr_value, src_op.name, lstm_supported_activations)


OnnxTranslations.register_translation(OnnxLSTMTranslation(),
                                      converter_type('LSTM', 'onnx'),
                                      op_adapter.LstmOp.TRANSLATION_KEY)
