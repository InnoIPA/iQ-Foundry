# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.onnx.onnx_translations import (
    OnnxTranslationBase,
    OnnxTranslations,
)
from qti.aisw.converters.onnx.util import (
    OnnxAttrProtoUtil,
    converter_type,
    extract_onnx_type,
)
from qti.aisw.converters.onnx.op_schema import OpSchemaBase, OpSchemaDict
from qti.aisw.converters.common.converter_ir.op_adapter import BlockOpAdapterMap
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.onnx.rnn_translations import OnnxLSTMTranslation
from qti.aisw.converters.onnx.rnn_translations import OnnxGruTranslation


# ------------------------------------------------------------------------------
#   QNN Op
# ------------------------------------------------------------------------------
class OnnxBlockOpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.qnn_op = None

    def extract_input_names(self, src_op, converter_context):
        return [str(input) for input in src_op.input]

    def extract_output_names(self, src_op, converter_context):
        return [str(output) for output in src_op.output]

    def extract_parameters(self, src_op, converter_context, **kwargs):
        # extract the parameter of the source ops
        param_dict = {}
        for attr in src_op.attribute:
            code = OnnxAttrProtoUtil.enum_to_strCode[attr.type]
            attr_value = extract_onnx_type(code, attr)
            param_dict[attr.name] = attr_value

        neuron_types = ["ElementWiseNeuron"]  # relu_min_max

        # First check whether the optype is supported or not. If yes, call the
        # respective op constructor using the translation map.
        # Sub op types (like Neuron) needs to be handled separately
        # since qnn doesn't have these sub op concept.
        if (
            src_op.op_type in neuron_types
            and src_op.operation
            == converter_context.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX
        ):
            return op_adapter.ElementwiseNeuronOp(
                str(src_op.name),
                operation=converter_context.ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX,
                **param_dict,
            )
        elif src_op.op_type in BlockOpAdapterMap.translations.keys():
            return BlockOpAdapterMap.translations[src_op.op_type](
                str(src_op.name), **param_dict
            )
        else:
            raise ValueError(
                "Undefined QNN op type {} for node name {}".format(
                    src_op.op_type, src_op.name
                )
            )

    def add_op(self, src_op, context, **kwargs):
        def add_static_tensor_as_constant_op(node_name):
            if not graph.has_node(node_name):
                constant_op = self.fetch_constant_op(node_name, context)
                graph.add(op=constant_op, input_names=[], output_names=[node_name])

        def is_static_input(input_name):
            return context.weights.has(input_name)

        def add_static_inputs_to_graph(input_names):
            static_input_names = [name for name in input_names if is_static_input(name)]
            for static_input_name in static_input_names:
                add_static_tensor_as_constant_op(static_input_name)

        graph = context.ir_graph
        op = self.extract_parameters(src_op, context, **kwargs)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)
        add_static_inputs_to_graph(input_names)
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        return node


OnnxTranslations.register_translation(
    OnnxBlockOpTranslation(), converter_type("block_op_MaskedSoftmax", "onnx")
)
OnnxTranslations.register_translation(
    OnnxBlockOpTranslation(), converter_type("block_op_Buffer", "onnx")
)
OnnxTranslations.register_translation(
    OnnxBlockOpTranslation(), converter_type("block_op_CombinedNms", "onnx")
)



class OnnxBlockOpStatefulLstmTranslation(OnnxLSTMTranslation):
    def __init__(self):
        from qti.aisw.converters.common.converter_ir.op_properties.stateful_lstm import (
            NUM_INPUTS,
            NUM_OUTPUTS,
            IR_RESET_IDX,
            IR_NUM_INPUTS,
            IR_TO_ONNX_INDICES,
            LSTM_IR_NUM_INPUTS,
            LSTM_IR_RESET_IDX
        )

        OnnxLSTMTranslation.__init__(self)
        schema_dict = OpSchemaDict("StatefulLstm")
        schema = OpSchemaBase("StatefulLstm")
        schema.numInputs = NUM_INPUTS
        schema.numOutputs = NUM_OUTPUTS
        schema.version = ["1"]
        schema._attributes = {
            "activations": ["activations", "ls", ["Sigmoid", "Tanh", "Tanh"]],
            "direction": ["direction", "s", "forward"],
            "hidden_size": ["hidden_size", "i"],
            "input_forget": ["input_forget", "i", 0],
            "layout": ["layout", "i", 0],
        }
        schema_dict.add_schema(schema, 1)
        self._op_schemas.append(schema_dict)

        self.NUM_INPUTS = NUM_INPUTS
        self.NUM_OUTPUTS = NUM_OUTPUTS
        self.IR_RESET_IDX = IR_RESET_IDX
        self.IR_NUM_INPUTS = IR_NUM_INPUTS
        self.IR_TO_ONNX_INDICES = IR_TO_ONNX_INDICES
        self.LSTM_IR_NUM_INPUTS = LSTM_IR_NUM_INPUTS
        self.LSTM_IR_RESET_IDX = LSTM_IR_RESET_IDX
        self.ONNX_RESET_IDX = IR_TO_ONNX_INDICES[IR_RESET_IDX]

    def extract_input_names(self, src_op, converter_context):
        graph = converter_context.ir_graph
        base_lstm_names = super().extract_input_names(src_op, converter_context)
        if getattr(graph, 'qairt_converter', False) and OnnxTranslations.is_cpp_ir_pass('MULTI_TIME_STEPS_LSTM'):
            # Converted IR is LSTM
            self.IR_NUM_INPUTS = self.LSTM_IR_NUM_INPUTS
            self.IR_RESET_IDX = self.LSTM_IR_RESET_IDX

        if (
            len(self.input_names) > self.ONNX_RESET_IDX
            and self.input_names[self.ONNX_RESET_IDX]
        ):
            if len(base_lstm_names) == self.IR_NUM_INPUTS:
                base_lstm_names[self.IR_RESET_IDX] = self.input_names[
                    self.ONNX_RESET_IDX
                ]
            elif len(base_lstm_names) == self.IR_NUM_INPUTS - 1:
                base_lstm_names.append(self.input_names[self.ONNX_RESET_IDX])
            else:
                raise ValueError(
                    "Wrong number of inputs extracted from source node: "
                    f"Expected between {self.NUM_INPUTS - 1}-{self.NUM_INPUTS}, "
                    f"got {len(base_lstm_names)}."
                )
        return base_lstm_names

    def add_op(self, src_op, context, **kwargs):
        rv = super().add_op(src_op, context, **kwargs)
        return rv


OnnxTranslations.register_translation(
    OnnxBlockOpStatefulLstmTranslation(),
    converter_type("block_op_StatefulLstm", "onnx"),
)


class OnnxBlockOpStatefulGruTranslation(OnnxGruTranslation):
    def __init__(self):
        from qti.aisw.converters.common.converter_ir.op_properties.stateful_gru import (
            NUM_INPUTS,
            NUM_OUTPUTS,
            IR_RESET_IDX,
            IR_NUM_INPUTS,
            IR_TO_ONNX_INDICES,
        )

        OnnxGruTranslation.__init__(self)
        schema_dict = OpSchemaDict("StatefulGru")
        schema = OpSchemaBase("StatefulGru")
        schema.numInputs = NUM_INPUTS
        schema.numOutputs = NUM_OUTPUTS
        schema.version = ["1"]
        schema._attributes = {
            "activations": ["activations", "ls", ["Sigmoid", "Tanh", "Tanh"]],
            "direction": ["direction", "s", "forward"],
            "hidden_size": ["hidden_size", "i"],
            "linear_before_reset": ["linear_before_reset", "i", 0],
            "layout": ["layout", "i", 0],
        }
        schema_dict.add_schema(schema, 1)
        self._op_schemas.append(schema_dict)

        self.NUM_INPUTS = NUM_INPUTS
        self.NUM_OUTPUTS = NUM_OUTPUTS
        self.IR_RESET_IDX = IR_RESET_IDX
        self.IR_NUM_INPUTS = IR_NUM_INPUTS
        self.IR_TO_ONNX_INDICES = IR_TO_ONNX_INDICES
        self.ONNX_RESET_IDX = IR_TO_ONNX_INDICES[IR_RESET_IDX]

    def extract_input_names(self, src_op, converter_context):
        base_gru_names = super().extract_input_names(src_op, converter_context)
        if (
            len(self.input_names) > self.ONNX_RESET_IDX
            and self.input_names[self.ONNX_RESET_IDX]
        ):
            if len(base_gru_names) == self.IR_NUM_INPUTS:
                base_gru_names[self.IR_RESET_IDX] = self.input_names[
                    self.ONNX_RESET_IDX
                ]
            elif len(base_gru_names) == self.IR_NUM_INPUTS - 1:
                base_gru_names.append(self.input_names[self.ONNX_RESET_IDX])
            else:
                raise ValueError(
                    "Wrong number of inputs extracted from source node: "
                    f"Expected between {self.NUM_INPUTS - 1}-{self.NUM_INPUTS}, "
                    f"got {len(base_gru_names)}."
                )
        return base_gru_names


OnnxTranslations.register_translation(
    OnnxBlockOpStatefulGruTranslation(), converter_type("block_op_StatefulGru", "onnx")
)


class OnnxBlockOpElementWiseMuxTranslation(OnnxBlockOpTranslation):
    def __init__(self):
        OnnxBlockOpTranslation.__init__(self)

    def extract_parameters(self, src_op, converter_context, **kwargs):
        return op_adapter.ElementWiseMuxOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxBlockOpElementWiseMuxTranslation(),
                                      converter_type('block_op_elementwisemux', 'onnx'),
                                      op_adapter.ElementWiseMuxOp.TRANSLATION_KEY)