# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
from .onnx_translations import *
from .util import *

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir.op_graph import TraceType
from qti.aisw.converters.common.utils.converter_utils import (track_transform,
                                                              qnn_to_numpy_datatype,
                                                              add_new_user_quantization_overrides)
from qti.aisw.converters.common.utils.translation_utils import get_broadcasted_shape
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders

# ------------------------------------------------------------------------------
#   Cast
# ------------------------------------------------------------------------------
class OnnxCastTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Cast', [1, 6, 9, 13, 19, 21, 23, 24])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        log_warning("Only numerical type cast is supported. The cast op: {} will be interpreted at conversion time".
                    format(src_op.name))
        input_names = list(map(str, src_op.input))
        params = extract_attributes(src_op,
                                    attr_infos=[('to', 'i', 0)],
                                    schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)

        cast_dtype = onnx_to_np_dtype.get(params.to).name
        from_type = converter_context.tensor_to_np_dtype.get(input_names[0])
        if from_type:
            from_type = from_type.name
        # Raise error when cast type is not in list of supported types
        if cast_dtype is None:
            raise ValueError("Cannot convert cast op {}: Cast type {} is not supported".
                             format(src_op.name, cast_dtype.name))
        const_op = self.fetch_constant_op(input_names[0], converter_context, dtype=cast_dtype, prunable=False,
                                          fail_if_dynamic=False, fail_if_not_found=True)
        if const_op:
            log_debug1("Node {} with static input(s) is resolved as Constant Op and interpreted during conversion".format(str(src_op.name)))
            const_op.tensor = np.asarray(const_op.tensor, dtype=cast_dtype)
            track_transform(const_op.constant_tensor,
                            ir_graph.TransformType.TRANSFORM_CAST,
                            {"datatype": const_op.constant_tensor.dtype.name})
            was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in input_names])
            converter_context.insert_weights(str(src_op.output[0]), const_op.tensor, was_scalar, [src_op.name], input_names,
                                             const_op.constant_tensor.transform_manager)
            return None
        # if the input of cast op is scalar, output will also be a scalar
        # adding the output tensor name to the scalar tensors so that the
        # output shapes for the next node can be computed accordingly
        if str(src_op.input[0]) in converter_context.scalar_tensor:
            converter_context.scalar_tensor.add(str(src_op.output[0]))

        if not from_type:
            return op_adapter.CastOp(str(src_op.name), to_type=cast_dtype)
        return op_adapter.CastOp(str(src_op.name), from_type=from_type, to_type=cast_dtype)

    def extract_input_names(self, src_op, converter_context):
        if not converter_context.ir_graph.has_buffer(str(src_op.input[0])) and converter_context.weights.has(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxCastTranslation(),
                                      converter_type('Cast', 'onnx'))


# ------------------------------------------------------------------------------
#   ChannelShuffle
# ------------------------------------------------------------------------------
class OnnxChannelShuffleTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, converter_context):
        # Note: Schema is not used here since this is not a valid Onnx Op
        params = extract_attributes(src_op,
                                    ('groups', 'i'))
        return op_adapter.ChannelShuffleOp(src_op.name, num_groups=params.groups)


OnnxTranslations.register_translation(OnnxChannelShuffleTranslation(),
                                      converter_type('Channel_Shuffle', 'onnx'),
                                      op_adapter.ChannelShuffleOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Clip
# ------------------------------------------------------------------------------
class OnnxClipTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Clip', [1, 6, 11, 12])
        self.input_names = []

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        self.input_names = []
        op = self.extract_parameters(src_op, converter_context)
        if op is None:
            return
        output_names = self.extract_output_names(src_op, converter_context)

        # If any one of the inputs (Min,Max) are dynamic then
        # expand the clip Op to Min and Max Op i.e. Min(max, Max(input, min))
        # because we do not support dynamic values for Min, Max inputs.
        #
        #   input min  max                   input      min
        #    \    |    /                      \         /
        #     \   |   /                        \_______/
        #      \__|__/                         |  Max  |
        #      |clip |        ------>          |_______|
        #      |_____|                   max        /
        #         |                        \       /
        #     out |                         \_____/
        #         |                         | Min |
        #                                   |_____|
        #                                      |
        #                                  out |
        #                                      |

        if isinstance(op, list):
            max_op = op[0]
            min_op = op[1]
            input_names = src_op.input
            min_is_dynamic = graph.has_buffer(min_op.name) and graph.get_buffer(min_op.name).producer.op.type != op_adapter.ConstantOp.TRANSLATION_KEY
            max_is_dynamic = graph.has_buffer(max_op.name) and graph.get_buffer(max_op.name).producer.op.type != op_adapter.ConstantOp.TRANSLATION_KEY

            if not min_is_dynamic:
                min_constant_op = op_adapter.ConstantOp(input_names[1], tensor=converter_context.weights.fetch(input_names[1]))
                if not graph.has_buffer(min_constant_op.name):
                    min_node = graph.add(min_constant_op, [], [min_constant_op.name])
                    self.insert_constant_trace_info(src_op.name, min_node, converter_context)

            if not max_is_dynamic:
                max_constant_op = op_adapter.ConstantOp(input_names[2], tensor=converter_context.weights.fetch(input_names[2]))
                if not graph.has_buffer(max_constant_op.name):
                    max_node = graph.add(max_constant_op, [], [max_constant_op.name])
                    self.insert_constant_trace_info(src_op.name, max_node, converter_context)

            max_node = graph.add(max_op,[input_names[0],input_names[1]], [input_names[1] + str("_output_0")])
            self.add_src_op_info(max_node.op.name, src_op, graph)
            self.insert_default_trace_info(src_op, max_node, converter_context)
            min_node = graph.add(min_op,[max_node.output_names[0],input_names[2]], output_names)
            self.add_src_op_info(min_node.op.name, src_op, graph)
            self.insert_default_trace_info(src_op, min_node, converter_context)
            return min_node
        else:
            input_names = self.extract_input_names(src_op, converter_context)
            node = graph.add(op, input_names, output_names)
            self.add_src_op_info(node.op.name, src_op, graph)
            self.insert_default_trace_info(src_op, node, converter_context)
            return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())

        if src_op.input[0] not in graph.buffers and converter_context.weights.has(src_op.input[0]):
            input0_op = op_adapter.ConstantOp(src_op.input[0], tensor=converter_context.weights.fetch(src_op.input[0]))
            input0_node = graph.add(input0_op, [], [src_op.input[0]])
            self.insert_constant_trace_info(src_op.input[0], input0_node, converter_context)
            graph.add_src_op_info(input0_op.name, [], [src_op.input[0]])

        min_name = str(src_op.input[1]) if len(src_op.input) > 1 else ''
        max_name = str(src_op.input[2]) if len(src_op.input) > 2 else ''
        max_is_dynamic = graph.has_buffer(max_name) and graph.get_buffer(max_name).producer.op.type != op_adapter.ConstantOp.TRANSLATION_KEY
        min_is_dynamic = graph.has_buffer(min_name) and graph.get_buffer(min_name).producer.op.type != op_adapter.ConstantOp.TRANSLATION_KEY

        if min_is_dynamic or max_is_dynamic:
            min_op = op_adapter.ElementwiseBinaryOp(min_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM)
            max_op = op_adapter.ElementwiseBinaryOp(max_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM)
            self.input_names= list(src_op.input)
            return [max_op, min_op]

        self.input_names.append(src_op.input[0])
        min_op = self.fetch_constant_op(min_name, converter_context)
        if min_op is None:
            min_val = params.min if 'min' in params else np.finfo(np.float32).min
        else:
            min_val = min_op.tensor.item(0)

        max_op = self.fetch_constant_op(max_name, converter_context)
        if max_op is None:
            max_val = params.max if 'max' in params else np.finfo(np.float32).max
        else:
            max_val = max_op.tensor.item(0)

        input_names = list(map(str, src_op.input))
        const_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False,
                                          fail_if_dynamic=False, fail_if_not_found=True)
        if const_op:
            log_debug1("Node {} with static input(s) is resolved as Constant Op and interpreted during conversion".format(str(src_op.name)))
            data = const_op.tensor
            clip_data = np.clip(data, min_val, max_val)
            # remove empty string from input_names, since optional inputs may be empty string in Clip
            was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in input_names if input_name != ''])
            converter_context.insert_weights(str(src_op.output[0]), clip_data, was_scalar, [src_op.name], input_names)

            # add constant node for later ops
            # without the output buffer from this node, it will cause error in later op
            if src_op.output[0] not in graph.buffers:
                new_const_op = op_adapter.ConstantOp(src_op.output[0], tensor=clip_data)
                output_names = [src_op.output[i] for i in range(len(src_op.output))]
                node = graph.add(new_const_op, [], output_names)
                op_source_info = [(src_op.name, TraceType.OP), (src_op.output[0], TraceType.TENSOR)]
                for input_name in input_names:
                    op_source_info.append((input_name, TraceType.TENSOR))
                self.insert_trace_info([node, graph.get_buffer(src_op.output[0])], op_source_info, converter_context)
                graph.add_src_op_info(new_const_op.name, [], output_names)
            return None

        return op_adapter.ElementwiseNeuronOp(src_op.name,
                                              operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX,
                                              min_value=min_val,
                                              max_value=max_val)

    def extract_input_names(self, src_op, converter_context):
        return self.input_names


OnnxTranslations.register_translation(OnnxClipTranslation(), converter_type('Clip', 'onnx'))


# ------------------------------------------------------------------------------
#   Concat
# ------------------------------------------------------------------------------
class OnnxConcatTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Concat', [1, 4, 11, 13])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op = self.extract_parameters(src_op, converter_context)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        input_names_with_null_tensor = []
        if op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
            for input_name in input_names:
                if not graph.has_buffer(input_name) and converter_context.weights.has(input_name):
                    const_op = self.fetch_constant_op(input_name, converter_context, prunable=False)
                    shape = const_op.tensor.shape
                    # Some ONNX models (saved from pytorch) have empty tensor (one dimension is 0)
                    # Add checking here to remove empty tensor in concat becasue it is meaningless
                    if 0 in shape:
                        input_names_with_null_tensor.append(input_name)
                    else:
                        const_node = graph.add(const_op, [], input_name)
                        self.insert_constant_trace_info(input_name, const_node, converter_context)
                        graph.add_src_op_info(input_name, None, const_node.output_names[0])
                else:
                    # TODO
                    # Short term solution : use this else branch.
                    # Long term solution : use QNN 0-Dim feature, when it is fully ready need refine here.
                    input_buf = graph.get_buffer(input_name)
                    shape = input_buf.shape
                    # Check if there is 0-dim tensor of Concat Op's input.
                    if 0 in shape.dims:
                        input_names_with_null_tensor.append(input_name)

        for input_name in input_names_with_null_tensor:
            input_names.remove(input_name)

        if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            self.add_src_op_info(op.name, src_op, graph)
            node = graph.add(op, [], output_names)
            op_source_info = [(src_op.name, TraceType.OP), (output_names[0], TraceType.TENSOR)]
            for input_name in input_names:
                op_source_info.append((input_name, TraceType.TENSOR))
            self.insert_trace_info([node, graph.get_buffer(output_names[0])], op_source_info, converter_context)
            return node

        self.add_src_op_info(op.name, src_op, graph)
        node = graph.add(op, input_names, output_names)
        self.insert_default_trace_info(src_op, node, converter_context)
        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())

        # static concatenation used for reshaping shape tensors
        if converter_context.weights.has_all(src_op.input):
            data = [converter_context.weights.fetch(input_name) for input_name in src_op.input]
            concat_data = np.concatenate(data, params.axis)
            converter_context.insert_weights(str(src_op.output[0]), concat_data,
                                             src_op_names=[src_op.name], src_tensor_names=src_op.input)
            return None

        # handle single input concats
        if len(src_op.input) == 1:
            if converter_context.weights.has_all(src_op.input):
                converter_context.insert_weights(str(src_op.output[0]), converter_context.weights.fetch(src_op.input[0]),
                                                 src_op_names=[src_op.name], src_tensor_names=src_op.input)
                return None
            return op_adapter.IdentityOp(src_op.name)

        # handle all constant input to concat
        input_names = list(map(str, src_op.input))
        const_input_ops = []
        for input_name in input_names:
            const_input_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if const_input_op is not None:
                const_input_ops.append(const_input_op)
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            data = []
            for const_input_op in const_input_ops:
                data.append(const_input_op.tensor)
            concat_data = np.concatenate(data, params.axis)
            converter_context.insert_weights(str(src_op.output[0]), concat_data,
                                             src_op_names=[src_op.name], src_tensor_names=input_names)
            return op_adapter.ConstantOp(str(src_op.output[0]), concat_data)

        return op_adapter.ConcatOp(src_op.name, axis=params.axis)

    def extract_input_names(self, src_op, converter_context):
        # If this was translated to a static op don't return input names
        if converter_context.weights.has_all(src_op.input):
            return []
        else:
            return list(map(str, src_op.input))

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.has_all(src_op.input):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxConcatTranslation(),
                                      converter_type('Concat', 'onnx'),
                                      op_adapter.ConcatOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Constant
# ------------------------------------------------------------------------------
class OnnxConstantTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Constant', [1, 9, 11, 12, 13, 19, 21, 23, 24])

    def extract_parameters(self, src_op, converter_context):
        # attr_infos contain the attribute used for the output of the constant op.
        # for opset 13, it can be one of the following : value, value_float, value_floats, value_int,
        # value_ints, value_string, value_strings, sparse_tensor.
        # for opset 1 and 9, it can be only value.
        # sparse tensor is not currently supported by the converter.
        if src_op.attribute[0].type == 11:
            raise ValueError("SparseTensor attribute type for op {} of type {} is not supported".
                             format(src_op.name, src_op.op_type))
        attr_infos = [(src_op.attribute[0].name, OnnxAttrProtoUtil.enum_to_strCode[src_op.attribute[0].type], "")]
        params = extract_attributes(src_op, schema=self.op_schema(), attr_infos=attr_infos)
        # ONNX return np "array scalar" for ONNX scalar.
        # the problem is, "array scalar" has shape attribute as an empty tuple.
        # which may break backends.
        # So we reshape "array scalar" to exactly an array with shape (1, )
        was_scalar = False
        value = np.array(params.get(attr_infos[0][0]))
        if not value.shape:
            value = value.reshape(1)
            was_scalar = True

        converter_context.insert_weights(src_op.output[0], value, was_scalar, [src_op.name], src_op.input)
        # Constant op is a special case... the output name is the real name
        return op_adapter.ConstantOp(src_op.output[0], value)

    def infer_output_shapes(self, op, input_shapes):
        return [list(op.tensor.shape)]


OnnxTranslations.register_translation(OnnxConstantTranslation(),
                                      converter_type('Constant', 'onnx'),
                                      op_adapter.ConstantOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ConstantOfShape
# ------------------------------------------------------------------------------
class OnnxConstantOfShapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ConstantOfShape', [9, 20, 21, 23, 24])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        def extract_default_attributes(name):
            ret = NamedDict()
            ret[name] = np.array([0]).astype(np.float32)
            return ret
        params = extract_attributes(src_op, schema=self.op_schema(), default_attrs=extract_default_attributes("value"))

        input_names = list(map(str, src_op.input))
        was_scalar = False

        # Only support when input is static
        const_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_not_found=True)

        log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
        shape = const_op.tensor.astype(np.int32)
        tensor_dtype = downcast_dtype_64bit_to_32bit(src_op.name, params.value.dtype)
        data = np.full(shape, params.value[0], dtype=tensor_dtype)
        if not data.shape:
            data = data.reshape(1)
            was_scalar = True
        converter_context.insert_weights(src_op.output[0], data, was_scalar, [src_op.name], src_op.input)
        return op_adapter.ConstantOp(src_op.output[0], data)

    def extract_input_names(self, src_op, converter_context):
        return []


OnnxTranslations.register_translation(OnnxConstantOfShapeTranslation(),
                                      converter_type('ConstantOfShape', 'onnx'))

# ------------------------------------------------------------------------------
#   DequantizeLinear
# ------------------------------------------------------------------------------
class OnnxDequantizeLinearTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('DequantizeLinear', [10, 13, 19, 21, 23, 24])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op, enc = self.extract_parameters(src_op, converter_context)
        if op is None:
            return
        if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            input_names = []
        else:
            input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        node = graph.add(op, input_names, output_names)
        self.insert_default_trace_info(src_op, node, converter_context)

        graph.add_quantization_params(node.op.name,
                                      output_encodings=enc)

        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        graph = converter_context.ir_graph
        # Three inputs data, scale(s), and zero point(s)
        input_names = list(map(str,src_op.input))
        output_names = list(map(str,src_op.output))
        # default values
        axis = 1
        block_size = 0

        log_assert(len(input_names) >= 2,
                   "Invalid number of inputs to DequantizeLinear Op {}. Expected at least 2, but got {}",
                   src_op.name, len(input_names))

        # Retrieve the scales
        scale_op = self.fetch_constant_op(input_names[1], converter_context, prunable=False, fail_if_dynamic=False)
        if scale_op is not None:
            scale = np.array(scale_op.tensor).astype(np.float32)
        else:
            raise ValueError("No scale provided, only static scale or constant is supported for op: {} of type: {}"
                             .format(src_op.name, src_op.op_type))

        # Check if zero point provided, otherwise use default of 0
        # zp must match the scale shape
        zp = np.zeros(scale.shape).astype(np.uint8)
        if len(input_names) > 2:
            zp_op = self.fetch_constant_op(input_names[2], converter_context, prunable=False, fail_if_dynamic=False)
            if zp_op is not None:
                zp = zp_op.tensor
            else:
                raise ValueError("No zero point provided, only static scale or constant is supported for op: {} of type: {}"
                                 .format(src_op.name, src_op.op_type))

        input_const_op = self.fetch_constant_op(input_names[0], converter_context, fail_if_not_found=False, fail_if_dynamic=False)
        input_rank = None

        if input_const_op is not None:
            x_data = input_const_op.tensor
            input_rank = len(x_data.shape)
        else:
            input_buffer = graph.get_buffer(input_names[0])
            input_rank = len(input_buffer.shape)

        if 'axis' in params:
            axis = params.axis
            if axis < 0:
                axis += input_rank

        if 'block_size' in params:
            block_size = params.block_size

        if 'output_dtype' in params and params.output_dtype:
            # Currently CPU supports only float32 for Dequantize output datatype
            output_dtype = onnx_tensor_dtype_to_np_dtype(params.output_dtype).name
            if output_dtype != "float32":
                raise ValueError(f"'output_dtype: '{output_dtype} is not supported for {src_op.name}")

        output_name = str(output_names[0])
        enc, zp = get_encoding(output_name, scale, zp, axis, block_size)

        w_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        if w_op is not None:
            # It's quantized parameters, quantize and store
            w = w_op.tensor
            if isinstance(w, onnx.onnx_ml_pb2.TensorProto):
                w = decode_tensorproto_raw_data(w).reshape(w.dims)

            # Microscaling (MxFP) types cannot be dequantized inline via (w - zp) * scale
            # because they use block-level E8M0 scales with a different encoding scheme.
            # Pass through the decoded values as-is; the microscaling encoding metadata
            # (block scales, value encoding) is carried in `enc`. Downstream,
            # QuantizeOp -> DequantizeOp is squashed into a ConvertOp that the QNN
            # backend executes natively. The IrQuantizer's packMicroscaling() later
            # converts these float32 values into compact uint8 E2M1 codes.
            if is_microscaling_dtype(enc.get("dtype", "")):
                converter_context.insert_weights(output_name, w, src_op_names=[src_op.name],
                                                 src_tensor_names=src_op.input)
                return op_adapter.ConstantOp(output_name, w), enc

            # for per-channel broadcasting
            orig_shape = None
            if len(scale.shape) == 1 and scale.shape[0] != 1:
                new_shape = [1] * len(w.shape)
                new_shape[axis] = len(scale)
                scale = scale.reshape(new_shape)
                zp = zp.reshape(new_shape)
            elif len(scale.shape) >= 2 and w.shape[axis] != scale.shape[axis]:
                # Block quantization: infer block_size from shape mismatch
                inferred_block_size = block_size if block_size > 0 else w.shape[axis] // scale.shape[axis]
                orig_shape = list(w.shape)
                block_shape = list(orig_shape)
                block_shape[axis] = orig_shape[axis] // inferred_block_size
                block_shape.insert(axis + 1, inferred_block_size)
                w = w.reshape(block_shape)
                scale = np.expand_dims(scale, axis + 1)
                zp = np.expand_dims(zp, axis + 1)
            w = (w.astype(scale.dtype) - zp.astype(scale.dtype)) * scale
            if orig_shape is not None:
                w = w.reshape(orig_shape)
            converter_context.insert_weights(output_name, w, src_op_names=[src_op.name], src_tensor_names=src_op.input)
            return op_adapter.ConstantOp(output_name, w), enc

        stripped_enc = {k:enc[k] for k in enc if k != 'name'}
        if isinstance(stripped_enc.get('min', 0.0), np.ndarray) and stripped_enc['min'].size > 1:
            return op_adapter.DequantizeOp(src_op.name, bw=enc["bw"]), enc
        return op_adapter.DequantizeOp(src_op.name, **stripped_enc), enc

    def extract_input_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        return [input_shapes[0]]

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxDequantizeLinearTranslation(),
                                      converter_type('DequantizeLinear', 'onnx'),
                                      op_adapter.DequantizeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Expand
# ------------------------------------------------------------------------------
class OnnxExpandTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Expand', [8, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        src_input_names = list(map(str, src_op.input))

        static_input_name = str(src_input_names[0])
        input_constant_op = self.fetch_constant_op(src_input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        shape_constant_op = self.fetch_constant_op(src_input_names[1], converter_context, dtype=np.int32, fail_if_not_found=True)
        log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_input_names[1]))
        if shape_constant_op is None:
            raise ValueError("Expand Op {} only support static shape tensor: {}".format(src_op.name, src_input_names[1]))
        shape = shape_constant_op.tensor.tolist()

        backend_name = converter_context.backend_info_obj.backend_name() if converter_context.backend_info_obj else None
        if backend_name == "AIC" and input_constant_op:
            self.add_constant_src_op(static_input_name, input_constant_op, converter_context)
        elif input_constant_op:
            # The output dtype is stored in a dictionary named tensor_to_np_dtype which is filled by reading elem_type
            # field present in value_info of the ONNX operator. It is possible that value_info is not present for this
            # node in the original ONNX framework graph. Hence, we use dtype from the Constant folded input op.
            out_data = np.array((input_constant_op.tensor) * np.ones(shape), dtype=input_constant_op.tensor.dtype)
            was_scalar = True if not out_data.shape else False
            if was_scalar:
                out_data = out_data.reshape(1,)
            converter_context.insert_weights(str(src_op.output[0]), out_data, was_scalar, [src_op.name], src_input_names)
            return None

        # handle the weights(shape) source because it is generated from framework op/tensor
        converter_context.update_weights_trace_info_for_op(src_op.name, src_input_names[1])
        input_buf = graph.get_buffer(src_input_names[0])
        input_dtype = qnn_to_numpy_datatype(input_buf.dtype)

        op = op_adapter.ExpandOp(name=src_op.name,
                                 shape=shape,
                                 input_dtype=input_dtype)
        return op

    def extract_input_names(self, src_op, converter_context):
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxExpandTranslation(), converter_type('Expand', 'onnx'))


# ------------------------------------------------------------------------------
#   Flatten
# ------------------------------------------------------------------------------
class OnnxFlattenTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Flatten', [1, 9, 11, 13, 21, 23, 24])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        axis = params.axis

        # When input is static, ensure they are preprocessed statically.
        input_name = str(src_op.input[0])
        if converter_context.weights.has(input_name):
            # static flatten of weight parameters
            output_name = str(src_op.output[0])
            w = converter_context.weights.fetch(input_name)
            pre_axes = w.shape[:axis]
            post_axes = w.shape[axis:]
            output_shape = [product(pre_axes), product(post_axes)]
            w = np.reshape(w, output_shape)
            converter_context.insert_weights(output_name, w, src_op_names=[src_op.name], src_tensor_names=src_op.input)
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))
            return None

        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_shape = input_buf.shape

        pre_axes = input_shape[:axis]
        post_axes = input_shape[axis:]
        output_shape = [product(pre_axes), product(post_axes)]

        # Otherwise this is a dynamic flatten so add the flatten/reshape op
        return op_adapter.ReshapeOp(src_op.name, shape=output_shape)

    def extract_input_names(self, src_op, converter_context):
        return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxFlattenTranslation(), converter_type('Flatten', 'onnx'))


# ------------------------------------------------------------------------------
#   Gather
# ------------------------------------------------------------------------------
class OnnxGatherTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Gather', [1, 11, 13])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        const_input_op, translated_ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        # const_input_op should only be 1 (either data or indices) or None.
        # if const_input_op = None => either both data and indices are constant or both are dynamic
        if const_input_op and not graph.has_buffer(const_input_op.name):
            node = graph.add(const_input_op, [], const_input_op.name)
            self.insert_constant_trace_info(const_input_op.name, node, converter_context)
            graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops[0].type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # when gather op is represented as a constant op
            last_node = graph.add(translated_ops[0], [], output_names)
            op_source_info = [(src_op.name, TraceType.OP), (output_names[0], TraceType.TENSOR)]
            for input_name in input_names:
                op_source_info.append((input_name, TraceType.TENSOR))
            self.insert_trace_info([last_node, graph.get_buffer(output_names[0])], op_source_info, converter_context)
            self.add_src_op_info(last_node.op.name, src_op, graph)
        else:
            # when gather is represented as gather or gather + reshape
            if len(translated_ops) == 2:
                gather_output_names = [output_names[0] + '_pre_reshape']
                if graph.user_quantization_overrides:
                    encoding = graph.get_overridden_encoding(list(src_op.output)[0], False)
                    if encoding is not None:
                        graph.set_overridden_encoding(gather_output_names[0], encoding, False)
            else:
                gather_output_names = [output_names[0]]

            last_node = graph.add(translated_ops[0], input_names, gather_output_names)
            if len(translated_ops) == 2:
                self.insert_trace_info([last_node, graph.get_buffer(gather_output_names[0])], (src_op.name, TraceType.OP), converter_context)
            else:
                self.insert_default_trace_info(src_op, last_node, converter_context)
            graph.add_src_op_info(last_node.op.name, None, gather_output_names[0])

            if len(translated_ops) == 2:
                last_node = graph.add(translated_ops[1], gather_output_names, output_names)
                self.insert_trace_info(last_node, (src_op.name, TraceType.OP), converter_context)
                for output_name in output_names:
                    self.insert_trace_info(graph.get_buffer(output_name), (output_name, TraceType.TENSOR), converter_context)
                graph.add_src_op_info(last_node.op.name, None, last_node.output_names[0])

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])
        if graph.has_buffer(input_data_name):
            # get input dims from buffer
            input_buf = graph.get_buffer(input_data_name)
            input_buf_shape = input_buf.shape
            input_data_dims = input_buf_shape.dims
            input_data_rank = input_buf.rank()
        else:
            # get input dims from weight tensor
            input_data = converter_context.weights.fetch(input_data_name)
            # set input_buf_shape to None since there may be no buffer for the input currently
            input_buf_shape = None
            input_data_dims = input_data.shape
            input_data_rank = len(input_data_dims)
        # axis can be negtive in ONNX, but it must not less than 0 in QNN
        axis = params.axis if params.axis >= 0 else (input_data_rank + params.axis)
        if axis < 0 or axis >= input_data_rank:
            raise ValueError("Axis param for Gather op {} should be in [-{}, {}), but got {}".
                             format(src_op.name, input_data_rank, input_data_rank, params.axis))

        input_names = list(map(str, src_op.input))
        const_input_ops = []
        const_input_op = self.fetch_constant_op(input_data_name, converter_context, dtype=None, prunable=False, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        scalar_index = False
        const_index_op = self.fetch_constant_op(indices_name, converter_context, dtype=np.int32, prunable=False,
                                                fail_if_dynamic=False)
        if const_index_op is not None:
            const_input_ops.append(const_index_op)
            scalar_index = converter_context.weights.was_scalar(indices_name)
        else:
            # TODO: deprecate it after 0d tensor is fully supported
            input_indices_buf = graph.get_buffer(indices_name)
            indices_op = input_indices_buf.producer.op
            if input_indices_buf.rank() == 0:
                if indices_op.type == op_adapter.InputOp.TRANSLATION_KEY:
                    scalar_index = True
                    input_indices_buf.shape = [1]
                    indices_op.shape = [1]
                else:
                    raise ValueError("Cannot support Gather op {} for 0D input when the indices is dynamic tensor.".
                                     format(src_op.name))

            if indices_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                indices_op.quantizable = False
                indices_op.tensor = indices_op.tensor.astype(np.int32)
                track_transform(indices_op.constant_tensor,
                                ir_graph.TransformType.TRANSFORM_CAST,
                                {"datatype": indices_op.constant_tensor.dtype.name})

        const_op = const_input_ops[0] if len(const_input_ops) == 1 else None

        # If both input and indices are static then interpret gather and return const op
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor
            # TODO: deprecate the if condition after 0d tensor is fully supported
            # Constant op will output 1D tensor even if the output is a scalar in onnx,
            # so we need to retrieve the scalar value from tensor.
            was_scalar = converter_context.weights.was_scalar(indices_name)
            if was_scalar:
                indices = indices.item()
            gather_data = np.take(input_data, indices, axis=axis)
            was_result_scalar = False if gather_data.shape else True
            # reshape the output tensor to 1D if it's a scalar
            if was_result_scalar:
                gather_data = gather_data.reshape(1)
            converter_context.insert_weights(str(src_op.output[0]), gather_data, was_result_scalar, [src_op.name], src_op.input)
            translated_ops.append(op_adapter.ConstantOp(src_op.output[0], gather_data))
            return const_op, translated_ops

        translated_ops.append(op_adapter.GatherOp(src_op.name, axis=axis))

        # TODO: deprecate it after 0d tensor is fully supported
        if (indices_name in converter_context.scalar_tensor) or scalar_index:
            output_shape = input_data_dims[:axis] + input_data_dims[axis+1:]
            # if Gather output is scalar do not do a post reshape since there is no support
            # for scalar inputs/outputs in IR or backends
            if len(output_shape):
                if isinstance(input_buf_shape, op_adapter.BufferShape) and input_buf_shape.is_dynamic():
                    raise ValueError("Cannot support Gather op {} for dynamic input when output is a scalar.".
                                     format(src_op.name))
                reshape_op_name = src_op.name
                if src_op.name:
                    reshape_op_name = 'Reshape_post_' + src_op.name
                translated_ops.append(op_adapter.ReshapeOp(reshape_op_name,
                                                           shape=output_shape))
            else:
                # TODO: deprecate it after 0d tensor is fully supported
                converter_context.scalar_tensor.add(str(src_op.output[0]))

        return const_op, translated_ops


OnnxTranslations.register_translation(OnnxGatherTranslation(),
                                      converter_type('Gather', 'onnx'),
                                      op_adapter.GatherOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GatherElements
# ------------------------------------------------------------------------------
class OnnxGatherElementsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GatherElements', [11, 13])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_ops, translated_ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        # input_ops size should only be 1 or 2 or 0([]).
        # if input_ops = [] => both data and indices are constant or both are dynamic
        if input_ops:
            for input_op in input_ops:
                node = graph.add(input_op, [], input_op.name)
                self.insert_constant_trace_info(input_op.name, node, converter_context)
                graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops == []:
            last_node = None
        else:
            last_node = graph.add(translated_ops[0], input_names, output_names)
            self.insert_default_trace_info(src_op, last_node, converter_context)
            self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def _perform_static_gather_elements(self, src_op, input_data: np.ndarray, input_indices: np.ndarray, axis: np.int32):
        # validate the inputs
        input_dim = input_data.ndim
        indices_dim = input_indices.ndim
        if input_dim != indices_dim or input_dim < 1:
            raise ValueError("Two inputs for GatherElements op {} should have same rank and be not smaller than 1, "
                             "but the invalid ranks of input are {} and {}.".
                             format(src_op.name, input_dim, indices_dim))
        # internal helper functions
        def _get_element(tensor_data, index_list_format:list):
            for dim in index_list_format:
                tensor_data = tensor_data[dim]
            return np.copy(tensor_data)
        def _set_element(tensor_data, index_list_format:list, data_to_set):
            while len(index_list_format) != 0:
                target_ind = index_list_format.pop(-1)
                tensor_to_set = _get_element(tensor_data, index_list_format)
                tensor_to_set[target_ind] = data_to_set
                data_to_set = tensor_to_set
            return tensor_to_set
        def _get_all_index_in_list_format(curr_indices, tensor, all_index_in_list_format):
            while np.size(tensor[-1]) == 0:
                tensor.pop[-1]
            while len(tensor) > 0:
                last_data = tensor.pop(-1)
                next_indices = curr_indices.copy()
                next_indices.append(len(tensor))
                if type(last_data) == list:
                    _get_all_index_in_list_format(next_indices, last_data, all_index_in_list_format)
                else:
                    all_index_in_list_format.append(next_indices)
            return all_index_in_list_format
        # output has the same shape with input_indices, hence get the indexes of input_indices for output initialize
        all_index=_get_all_index_in_list_format([], np.copy(input_indices).tolist(), [])
        # init the static_gether_elements to input_indices and renew all data in it
        # reference https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements
        static_gether_elements = np.copy(input_indices).astype(input_data.dtype)
        for ind in all_index:
            data_to_set_index = ind.copy()
            data_to_set_index[axis] = _get_element(input_indices, ind)
            data_to_set = _get_element(input_data, data_to_set_index)
            static_gether_elements = _set_element(static_gether_elements, ind, data_to_set)

        return static_gether_elements

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        input_ops = []
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])
        axis = params.axis

        input_names = list(map(str, src_op.input))
        const_input_ops = []
        const_input_op = self.fetch_constant_op(input_data_name, converter_context, dtype=None, prunable=False, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)
        const_input_op = self.fetch_constant_op(indices_name, converter_context, dtype=np.int32, prunable=False,
                                                fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        # If both input and indices are static then interpret gather and return const op
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor
            gather_elements_data = self._perform_static_gather_elements(src_op, input_data, indices, axis=axis)
            converter_context.insert_weights(str(src_op.output[0]), gather_elements_data, [src_op.name], src_op.input)
            return input_ops, translated_ops

        # If only input is stored as weights then create a corresponding const op
        if not graph.has_buffer(input_data_name) and converter_context.weights.has(input_data_name):
            input_data = converter_context.weights.fetch(input_data_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(input_data_name, input_data))

        # If only indices is stored as weights then create a corresponding const op
        if not graph.has_buffer(indices_name) and converter_context.weights.has(indices_name):
            indices = converter_context.weights.fetch(indices_name, prunable=False).astype(np.int32)
            input_ops.append(op_adapter.ConstantOp(indices_name, indices, quantizable=False))
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices_op.tensor = indices_op.tensor.astype(np.int32)

        translated_ops.append(op_adapter.GatherElementsOp(src_op.name, axis=axis))

        return input_ops, translated_ops


OnnxTranslations.register_translation(OnnxGatherElementsTranslation(),
                                      converter_type('GatherElements', 'onnx'),
                                      op_adapter.GatherElementsOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GatherND
# ------------------------------------------------------------------------------
class OnnxGatherNDTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GatherND', [11, 12, 13])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_ops, translated_ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        # input op should only be 1 or 2 or None.
        # if input_op = None => all inputs are constant or all are dynamic
        # When input ops are None, GatherND is a constant op
        if input_ops:
            for input_op in input_ops:
                node = graph.add(input_op, [], input_op.name)
                self.insert_constant_trace_info(input_op.name, node, converter_context)
                graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops[0].type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # when gather_nd op is represented as a constant op i.e input ops is None
            last_node = graph.add(translated_ops[0], [], output_names)
            op_source_info = [(src_op.name, TraceType.OP), (output_names[0], TraceType.TENSOR)]
            for input_name in input_names:
                op_source_info.append((input_name, TraceType.TENSOR))
            self.insert_trace_info([last_node, graph.get_buffer(output_names[0])], op_source_info, converter_context)
            self.add_src_op_info(last_node.op.name, src_op, graph)
        else:
            # when gather_nd op has one or more dynamic inputs
            last_node = graph.add(translated_ops[0], input_names, output_names)
            self.insert_default_trace_info(src_op, last_node, converter_context)
            self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(),
                                    attr_infos=[('batch_dims', 'i', 0)])
        batch_dims = params["batch_dims"]

        input_ops = []
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])

        input_names = list(map(str, src_op.input))
        const_input_ops = []

        # Create prunable const ops for all inputs if set, inputs are data and indices tensors
        const_input_op = self.fetch_constant_op(input_data_name, converter_context, dtype=None, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        const_indices_op = self.fetch_constant_op(indices_name, converter_context, dtype=np.uint32,
                                                  quantizable=False,
                                                  fail_if_dynamic=False)
        if const_indices_op is not None:
            const_input_ops.append(const_indices_op)

        # If all inputs are static, then perform static Gather and return
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor

            gather_nd_data = self._perform_static_gather_nd(src_op, input_data, indices, batch_dims)
            converter_context.insert_weights(str(src_op.output[0]), gather_nd_data,
                                             src_op_names=[src_op.name], src_tensor_names=src_op.input)
            translated_ops.append(op_adapter.ConstantOp(src_op.output[0], gather_nd_data))
            return input_ops, translated_ops

        # If input is stored as weights then create a corresponding const op
        input_data, indices = None, None
        if not graph.has_buffer(input_data_name) and converter_context.weights.has(input_data_name):
            input_data = converter_context.weights.fetch(input_data_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(input_data_name, input_data))

        # If indices is stored as weights then create a corresponding const op
        if not graph.has_buffer(indices_name) and converter_context.weights.has(indices_name):
            indices = converter_context.weights.fetch(indices_name, prunable=False).astype(np.uint32)
            indices_op = op_adapter.ConstantOp(indices_name, indices, quantizable=False)
            input_ops.append(indices_op)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices = indices_op.tensor = indices_op.tensor.astype(np.uint32)

        # fetch attribute param batch_dims's real value and reset the init value of it.
        translated_ops.append(op_adapter.GatherNDOp(src_op.name, batch_dims=batch_dims))

        return input_ops, translated_ops

    def _perform_static_gather_nd(self, src_op, input_data: np.ndarray, indices: np.ndarray,
                                   batch_dims: np.uint32):
        if batch_dims < 0:
            raise TypeError("Cannot perform static GatherND op {}. Expected batch_dims should be 0 or positive integer.".
                            format(src_op.name))

        data = np.copy(input_data)

        # Note the data rank - will be reused multiple times later
        data_rank = len(data.shape)

        # Check input tensors' shape/rank condition
        assert indices.shape[-1] <= data_rank

        #The list of data/indice shape of batch_dims
        batch_dims_shape = []

        #The number of elements in the batch_dims for data/indice array
        batch_dims_size = 1

        # Check the shape of indice and data are identical for batch dims.
        for i in range(batch_dims):
            batch_dims_shape.append(indices.shape[i])
            batch_dims_size *= indices.shape[i]

        # Compute shape of output array
        if (indices.shape[-1] == data_rank - batch_dims):
            output_shape = batch_dims_shape + list(indices.shape)[batch_dims:-1]
        else:
            output_shape = batch_dims_shape + list(indices.shape)[batch_dims:-1] \
                + list(data.shape)[batch_dims + indices.shape[-1]:]

        # Placeholder for output data
        output_data_buffer = []

        # Flatten 'indices' to 2D array
        reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

        # Flatten 'data' to array of shape (batch_dim_size, data.shape[batch_dimes:])
        reshaped_data = data.reshape((batch_dims_size, ) + data.shape[batch_dims:])

        # gather each scalar value from 'data'
        for batch_dim in range(reshaped_indices.shape[0]):
            for outer_dim in range(reshaped_indices.shape[1]):
                gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
                output_data_buffer.append(reshaped_data[(batch_dim,) + gather_index])

        return np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)

    def extract_input_names(self, src_op, converter_context):
        if converter_context.weights.has_all(src_op.input):
            return []
        else:
            return super().extract_input_names(src_op, converter_context)

OnnxTranslations.register_translation(OnnxGatherNDTranslation(),
                                      converter_type('GatherND', 'onnx'),
                                      op_adapter.GatherNDOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   HardSwish
# ------------------------------------------------------------------------------
class OnnxHardSwishTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('HardSwish', [14, 22])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseNeuronOp(src_op.name,
                                              operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH,
                                              alpha=1/6,
                                              beta=0.5)

    def extract_input_names(self, src_op, converter_context):
        return [list(src_op.input)[0]]

OnnxTranslations.register_translation(OnnxHardSwishTranslation(), converter_type('HardSwish', 'onnx'))


# ------------------------------------------------------------------------------
#   IsInf
# ------------------------------------------------------------------------------
class OnnxIsInfTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('IsInf', [10, 20])

    def extract_parameters(self, src_op, converter_context, **kwargs):
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)
        detect_negative = params.get("detect_negative", 1)
        detect_positive = params.get("detect_positive", 1)
        return op_adapter.IsInfOp(src_op.name,
                                  detect_negative=bool(detect_negative),
                                  detect_positive=bool(detect_positive))

OnnxTranslations.register_translation(OnnxIsInfTranslation(), converter_type('IsInf', 'onnx'))


# ------------------------------------------------------------------------------
#   IsNan
# ------------------------------------------------------------------------------
class OnnxIsNanTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('IsNaN', [9, 13, 20])

    def extract_parameters(self, src_op, converter_context, **kwargs):
        return op_adapter.IsNanOp(src_op.name)

OnnxTranslations.register_translation(OnnxIsNanTranslation(), converter_type('IsNaN', 'onnx'))


# ------------------------------------------------------------------------------
#   NonZero
# ------------------------------------------------------------------------------
class OnnxNonZeroTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('NonZero', [9, 13])
        self.input_names = None
        self.output_names = None

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        op = self.extract_parameters(src_op, converter_context)
        if op is None:
            return
        log_warning("\n \n WARNING! The model contains a NonZero Op. This produces a " \
                     "dynamically shaped output tensor. It is common for this to cause"\
                      "compilation to fail e.g. This can lead to errors in operations like"\
                      "MatMul, which depend on the shapes of the inputs.. \n \n")
        intermediate_names = [name + '_intermediate' for name in self.output_names]
        graph.add_src_op_info(op.name, self.input_names, intermediate_names)
        node = graph.add(op, self.input_names, intermediate_names)
        trace_target_list = [node]
        for name in intermediate_names:
            trace_target_list.append(graph.get_buffer(name))
        self.insert_trace_info(trace_target_list, (src_op.name, TraceType.OP), converter_context)

        transpose_op = op_adapter.TransposeOp(name=src_op.name + '_transpose', perm=[1,0])
        graph.add_src_op_info(transpose_op.name, intermediate_names, self.output_names)
        transpose_node = graph.add(transpose_op, intermediate_names, self.output_names)
        self.insert_trace_info(transpose_node, (src_op.name, TraceType.OP), converter_context)
        for output_name in self.output_names:
            self.insert_trace_info(graph.get_buffer(output_name), (output_name, TraceType.TENSOR), converter_context)
        return transpose_node

    def extract_parameters(self, src_op, converter_context):
        self.input_names = self.extract_input_names(src_op, converter_context)
        self.output_names = self.extract_output_names(src_op, converter_context)

        # handle constant input to NonZero
        input_const_op = self.fetch_constant_op(self.input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        if input_const_op is not None:
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_const_data = input_const_op.tensor
            nonzero_output_data = np.array(np.nonzero(input_const_data))
            converter_context.insert_weights(self.output_names[0], nonzero_output_data,
                                             src_op_names=[src_op.name], src_tensor_names=self.input_names)
            return None

        return op_adapter.NonZeroOp(name=src_op.name)


OnnxTranslations.register_translation(OnnxNonZeroTranslation(),
                                      converter_type('NonZero', 'onnx'),
                                      op_adapter.NonZeroOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   OneHot
# ------------------------------------------------------------------------------
class OnnxOneHotTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('OneHot', [9, 11])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        ops = self.extract_parameters(src_op, converter_context)
        if ops is None:
          return
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        if len(ops) == 2:
            onehot_output_name = [output_names[0] + '_pre_reshape']
        else:
            onehot_output_name = [output_names[0]]

        last_node = graph.add(ops[0], input_names, onehot_output_name)
        if len(ops) == 2:
            self.insert_trace_info([last_node, graph.get_buffer(onehot_output_name[0])], (src_op.name, TraceType.OP), converter_context)
        else:
            self.insert_default_trace_info(src_op, last_node, converter_context)
        graph.add_src_op_info(last_node.op.name, input_names[0], onehot_output_name[0])

        if len(ops) == 2:
            last_node = graph.add(ops[1], onehot_output_name, output_names)
            self.insert_trace_info(last_node, (src_op.name, TraceType.OP), converter_context)
            for output_name in output_names:
                self.insert_trace_info(graph.get_buffer(output_name), (output_name, TraceType.TENSOR), converter_context)
            graph.add_src_op_info(last_node.op.name, onehot_output_name[0], last_node.output_names[0])

        return last_node

    def calculate_onehot(self, depth, values, indices, axis):
        if isinstance(depth, np.ndarray):
            depth = depth[0]
        output_shape = list(indices.shape)
        if axis < 0:
            axis = axis + len(output_shape) + 1
        output_shape.insert(axis, depth)
        onehot_vector = np.full(output_shape, values[0])
        for idx, value in np.ndenumerate(indices):
            if -depth <= value <= depth-1:
                index = list(idx)
                index.insert(axis, int(value % depth))
                onehot_vector[tuple(index)] = values[1]
        return onehot_vector

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(map(str, src_op.input))
        ops = []

        depth_const_op = self.fetch_constant_op(input_names[1], converter_context)
        depth = depth_const_op.tensor[0]
        if depth < 0:
            raise ValueError("Depth input for OneHot Op {} must be non-negative, but got {}.".
                             format(src_op.name, depth))

        values_const_op = self.fetch_constant_op(input_names[2], converter_context)
        values = values_const_op.tensor

        axis = params['axis']
        axis_min = -len(values.shape) - 1
        axis_max = len(values.shape)
        if axis > axis_max or axis < axis_min:
            raise ValueError("OneHot op {} axis parameter should be in range {} to {}".
                             format(src_op.name, axis_min, axis_max))

        if converter_context.weights.has_all(src_op.input):
            indices_op = self.fetch_constant_op(input_names[0], converter_context)
            indices = indices_op.tensor
            onehot_data = self.calculate_onehot(depth, values, indices, params['axis'])
            converter_context.weights.insert(str(src_op.output[0]),onehot_data)
            return None

        ops.append(op_adapter.OneHotOp(src_op.name, depth=depth, on_value=values[1], off_value=values[0], axis=params.axis))

        # if indices input was a scalar then reshape one_hot output
        if converter_context.weights.has(input_names[0]) and converter_context.weights.was_scalar(input_names[0]):
            output_shape = [depth]
            reshape_op_name = src_op.name
            if src_op.name:
                reshape_op_name = 'Reshape_post_' + src_op.name
            ops.append(op_adapter.ReshapeOp(reshape_op_name,
                                            shape=output_shape))

        return ops

    def extract_input_names(self, src_op, converter_context):
        # Filter depth and values from the input
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxOneHotTranslation(),
                                      converter_type('OneHot', 'onnx'))


# ------------------------------------------------------------------------------
#   Pad
# ------------------------------------------------------------------------------
class OnnxPadTranslation(OnnxTranslationBase):
    class OnnxPadMode:
        CONSTANT = 'constant'
        REFLECT = 'reflect'
        EDGE =  'edge'
    supported_modes = {OnnxPadMode.CONSTANT : ir_graph.QNN_OP_PAD_SCHEME_CONSTANT,
                       OnnxPadMode.REFLECT : ir_graph.QNN_OP_PAD_SCHEME_MIRROR_REFLECT,
                       OnnxPadMode.EDGE : ir_graph.QNN_OP_PAD_SCHEME_EDGE}

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Pad', [1, 2, 11, 13, 18, 19, 21, 23, 24])\
            .register_method(self.validate_attribute_values)
        self.pads = []
        self.negative_pads_present = False

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.negative_pads_present = False
        self.pads = []
        node_op = self.extract_parameters(src_op, converter_context)
        if node_op.type == 'CropAndResize':
            input_buf = graph.get_buffer(str(src_op.input[0]))
            rank = len(input_buf.shape)
            #Add axis tracking
            axis_format = input_buf.axis_format
            if axis_format == 'NCHW':
                y1 = self.pads[2]
                x1 = self.pads[3]
                y2 = self.pads[6]
                x2 = self.pads[7]
                height = input_buf.shape[2]
                width = input_buf.shape[3]
            elif axis_format == 'NHWC':
                y1 = self.pads[1]
                x1 = self.pads[2]
                y2 = self.pads[5]
                x2 = self.pads[6]
                height = input_buf.shape[1]
                width = input_buf.shape[2]
            else:
                log_error("Negative padding is currently supported for NCHW and NHWC images only. "
                          "Unsupported axis format {} for Pad op {}".format(axis_format, src_op.name))
                sys.exit(1)

            y1 = -y1 / (height -1)
            y2 = 1 + (y2 / (height -1))
            x1 = -x1 / (width - 1)
            x2 = 1 + (x2 / (width - 1))
            crop_boxes = np.array([y1,x1,y2,x2]).reshape(1,4)
            crop_boxes_op = op_adapter.ConstantOp(src_op.name+'_crop_boxes',tensor=crop_boxes)
            graph.add(crop_boxes_op,[],[src_op.name+'_crop_boxes'], axis_formats=['NONTRIVIAL'])
            batch_index = np.arange(input_buf.shape[0])
            batch_index_op = op_adapter.ConstantOp(src_op.name+'_batch_index',tensor=batch_index)
            graph.add(batch_index_op,[],[src_op.name+'_batch_index'], axis_formats=['ANY'])
            output_names = self.extract_output_names(src_op, converter_context)
            node = graph.add(node_op,[str(src_op.input[0]), crop_boxes_op.name, batch_index_op.name], output_names)
            self.add_src_op_info(node.op.name, src_op, graph)
            self.insert_default_trace_info(src_op, node, converter_context)
            return node
        else:
            if node_op is None:
                return
            input_names = self.extract_input_names(src_op, converter_context)
            output_names = self.extract_output_names(src_op, converter_context)
            node = graph.add(node_op, input_names, output_names)
            self.add_src_op_info(node.op.name, src_op, graph)
            self.insert_default_trace_info(src_op, node, converter_context)
            return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        pads_name = str(src_op.input[1]) if len(src_op.input) > 1 else ''
        const_name = str(src_op.input[2]) if len(src_op.input) > 2 else ''
        pads = None

        if pads_name:
            pads_op = self.fetch_constant_op(pads_name, converter_context, dtype=np.int32)
            if pads_op is not None:
                pads = pads_op.tensor
        elif 'pads' in params:
            pads = params.pads
        elif 'paddings' in params:
            pads = params.paddings

        if pads is None:
            raise ValueError("Failed to retrieve pads value on {} source Pad op {}".
                             format(src_op.op_type, src_op.name))

        input_buf = graph.get_buffer(str(src_op.input[0]))
        rank = len(input_buf.shape)

        #If axes is provided in the input,axes tensor will be fetched & if not provided, the axes tensor will be
        #[0,1,...,input_buf.rank()-1]
        if len(src_op.input) > 3:
            axes_input = str(src_op.input[3])
            if not graph.has_buffer(axes_input):
                axes_op = self.fetch_constant_op(axes_input, converter_context, dtype=np.int32)
                if axes_op is not None:
                    axes = axes_op.tensor.tolist()
                else:
                    raise ValueError("axes input for source Pad op {} is neither in converter_context.weights nor in graph.buffer".
                                     format(str(src_op.name)))
            else:
                axes = graph.get_buffer(axes_input).shape
            #Handling the negative axes scenario
            neg_axes = [idx for idx, ele in enumerate(axes) if ele<0]
            if neg_axes:
                for neg_idx in neg_axes:
                    axes[neg_idx] += input_buf.rank()
            axes = np.asarray(axes, dtype=np.dtype('int32'))
        else:
            axes = np.asarray([ele for ele in range(input_buf.rank())], dtype=np.dtype('int32'))
        #Checking the condition whether len(axes) == len(pads)/2
        log_assert(len(axes) == int(len(pads)/2),
                   "For Pad op {}, length of the axes: {} must equal (# pads/2): {}",
                   src_op.name, len(axes), int(len(pads) / 2))
        if len(axes) != rank:
            #Updating the pad tensor based on axes
            pads_zero = np.zeros(2*rank, dtype=int)
            for idx in range(len(axes)):
                 pads_zero[axes[idx]] = pads[idx]
                 pads_zero[axes[idx] + int(len(pads_zero)/2)] = pads[idx + int(len(pads)/2)]
            pads = pads_zero

        self.negative_pads_present = any([0 if val >= 0 else 1 for val in pads])
        self.pads = pads

        # Handling negative pads
        if self.negative_pads_present:
            if input_buf.axis_format == "NCHW":
                resize_dims = np.array([input_buf.shape[2]+pads[2]+pads[6],input_buf.shape[3]+pads[3]+pads[7]])
            elif input_buf.axis_format == "NHWC":
                resize_dims = np.array([input_buf.shape[1]+pads[1]+pads[5],input_buf.shape[2]+pads[2]+pads[6]])
            else:
                log_error("Negative padding is currently supported for NCHW and NHWC images only. "
                          "Invalid axis format {} for input buffer {}".format(input_buf.axis_format, input_buf))
                sys.exit(1)
            if params.mode != self.OnnxPadMode.CONSTANT:
                log_warning("Pad op {} with negative padding doesn't support padding mode {}. Ignoring padding "
                            "mode {} and using mode {} instead.".format(
                                src_op.name, params.mode, params.mode, self.OnnxPadMode.CONSTANT))
            constant_value = 0.0
            if const_name:
                const_op = self.fetch_constant_op(const_name, converter_context, dtype=np.int32)
                if const_op is not None:
                    constant_value = const_op.tensor[0]

            return op_adapter.CropAndResizeOp(src_op.name,
                                              resize_dims=resize_dims,
                                              extrapolation_value=constant_value)

        else:
            # Pads/paddings need to be translated from r1_begin, r2_begin...r1_end, r2_end, ...
            # to pairs (r1_begin, r1_end), (r2_begin, r2_end)...
            pad_pairs = []
            for index in range(rank):
                pad_pairs.append([pads[index], pads[index + rank]])
            pad_pairs = np.asarray(pad_pairs, dtype=np.dtype('int32'))

            constant_value = 0
            if const_name:
                const_op = self.fetch_constant_op(const_name, converter_context, dtype=np.int32)
                if const_op is not None:
                    constant_value = const_op.tensor[0]
            elif 'value' in params:
                constant_value = params.value

            return op_adapter.PadOp(src_op.name,
                                    scheme=self.supported_modes[params.mode],
                                    pad_amount=pad_pairs,
                                    pad_constant_value=constant_value)

    def extract_input_names(self, src_op, converter_context):
        # Filter if there are any parameters like 'pads' in inputs
        # For example, 'pads' are already handled in extract_parameters
        return [str(src_op.input[0])]

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'mode':
            src_op_mode = attr_value
            if src_op_mode not in OnnxPadTranslation.supported_modes:
                raise ValueError("Unsupported mode {} set in Pad op {}".
                                 format(src_op_mode, src_op.name))


OnnxTranslations.register_translation(OnnxPadTranslation(),
                                      converter_type('Pad', 'onnx'),
                                      op_adapter.PadOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   QuantizeLinear
# ------------------------------------------------------------------------------
class OnnxQuantizeLinearTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('QuantizeLinear', [10, 13, 19, 21, 23, 24])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op, enc = self.extract_parameters(src_op, converter_context)
        if op is None:
            return
        if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            input_names = []
        else:
            input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        node = graph.add(op, input_names, output_names)
        if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            trace_target_list = [node]
            for output_name in output_names:
                trace_target_list.append(graph.get_buffer(output_name))
            self.insert_trace_info(trace_target_list, (src_op.name, TraceType.OP), converter_context)
        else:
            self.insert_default_trace_info(src_op, node, converter_context)

        graph.add_quantization_params(node.op.name,
                                      output_encodings=enc)

        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())

        # Three inputs data, scale(s), and zero point(s)
        graph = converter_context.ir_graph
        input_names = list(map(str,src_op.input))
        output_names = list(map(str,src_op.output))
        axis = 1
        block_size = 0

        log_assert(len(input_names) >= 2,
                   "Invalid number of inputs to QuantizeLinear Op {}. Expected at least 2, but got {}",
                   src_op.name, len(input_names))

        # Retrieve the scales
        scale_op = self.fetch_constant_op(input_names[1], converter_context, prunable=False ,fail_if_dynamic=False)
        if scale_op is not None:
            scale = np.array(scale_op.tensor).astype(np.float32)
        else:
            raise ValueError("No scale provided, only static svalue is supported for op: {} of type: {}".
                             format(src_op.name, src_op.op_type))

        # Check if zero point provided, otherwise use default of 0
        # zp must match the scale shape
        zp = np.zeros(scale.shape).astype(np.uint8)
        if len(input_names) > 2:
            zp_op = self.fetch_constant_op(input_names[2], converter_context, prunable=False, fail_if_dynamic=False)
            if zp_op is not None:
                zp = zp_op.tensor
            else:
                raise ValueError("No zero point provided, only static value is supported for Zero point for op: {} of type: {}".
                                 format(src_op.name, src_op.op_type))

        input_const_op = self.fetch_constant_op(input_names[0], converter_context, fail_if_not_found=False, fail_if_dynamic=False)
        input_rank = None

        if input_const_op is not None:
            x_data = input_const_op.tensor
            input_rank = len(x_data.shape)
        else:
            input_buffer = graph.get_buffer(input_names[0])
            input_rank = len(input_buffer.shape)

        if 'axis' in params:
            axis = params.axis
            if axis < 0:
                axis += input_rank

        if 'block_size' in params:
            block_size = params.block_size

        if 'precision' in params and params.precision:
            raise ValueError(f"'precision' attribute is not currently supported for {src_op.name}")

        if 'saturate' in params and params.saturate > 1:
            raise ValueError(f"'saturate' attribute is not currently supported for {src_op.name}")

        output_name = str(output_names[0])
        enc, zp = get_encoding(output_name, scale, zp, axis, block_size)

        if 'output_dtype' in params and params.output_dtype:
            # Currently output_dtype value same as what we infer from get_encoding (zero point) is supported
            output_dtype = onnx_tensor_dtype_to_np_dtype(params.output_dtype).name
            enc_dtype = enc["dtype"]
            if output_dtype != enc_dtype:
                raise ValueError(f"output_dtype({output_dtype}) is different from zero point type ({enc_dtype}) for {src_op.name}")

        w_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        if w_op is not None:
            # Check if input has microscaling encoding — if so, don't fold, create QuantizeOp
            # so the downstream squash_quant_dequant pass can produce a Convert op
            input_op_name = input_names[0]
            if input_op_name in graph.quantization_params:
                input_enc = graph.quantization_params[input_op_name].get('output_encodings', [{}])
                if input_enc:
                    input_enc = input_enc[0]
                if is_microscaling_dtype(input_enc.get("dtype", "")):
                    return op_adapter.QuantizeOp(src_op.name, bw=enc["bw"]), enc

            # It's quantized parameters, quantize and store
            w = w_op.tensor
            if isinstance(w, onnx.onnx_ml_pb2.TensorProto):
                w = decode_tensorproto_raw_data(w).reshape(w.dims)
            # for per-channel broadcasting
            orig_shape = None
            if len(scale.shape) == 1 and scale.shape[0] != 1:
                new_shape = [1] * len(w.shape)
                new_shape[axis] = len(scale)
                scale = scale.reshape(new_shape)
                zp = zp.reshape(new_shape)
            elif len(scale.shape) >= 2 and w.shape[axis] != scale.shape[axis]:
                # Block quantization: infer block_size from shape mismatch
                inferred_block_size = block_size if block_size > 0 else w.shape[axis] // scale.shape[axis]
                orig_shape = list(w.shape)
                block_shape = list(orig_shape)
                block_shape[axis] = orig_shape[axis] // inferred_block_size
                block_shape.insert(axis + 1, inferred_block_size)
                w = w.reshape(block_shape)
                scale = np.expand_dims(scale, axis + 1)
                zp = np.expand_dims(zp, axis + 1)
            qmin, qmax = get_quantized_range(zp, enc["bw"], enc.get("dtype"))
            w = np.clip((np.rint(w/scale) + zp), qmin, qmax).astype(zp.dtype)
            if orig_shape is not None:
                w = w.reshape(orig_shape)
            converter_context.insert_weights(output_name, w, src_op_names=[src_op.name], src_tensor_names=src_op.input)
            return op_adapter.ConstantOp(output_name, w), enc

        stripped_enc = {k:enc[k] for k in enc if k != 'name'}
        return op_adapter.QuantizeOp(src_op.name, **stripped_enc), enc

    def extract_input_names(self, src_op, converter_context):
        # If this was translated to a static op don't return names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        return [input_shapes[0]]

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxQuantizeLinearTranslation(),
                                      converter_type('QuantizeLinear', 'onnx'),
                                      op_adapter.QuantizeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Range
# ------------------------------------------------------------------------------
class OnnxRangeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Range', [11])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = list(map(str, src_op.input))
        const_inputs = []

        # Only support when all inputs are static
        for input_name in input_names:
            const_op = self.fetch_constant_op(input_name, converter_context, prunable=False)
            if const_op is not None:
                const_inputs.append(const_op.tensor)

        log_assert(len(const_inputs) == 3,
                   "Invalid number of static inputs to Range Op {}. Expected 3, but got {}",
                   src_op.name, len(const_inputs))

        start = const_inputs[0].item(0)
        limit = const_inputs[1].item(0)
        delta = const_inputs[2].item(0)

        # range type is determined by inputs which are expected to be all of same type
        dtype = downcast_dtype_64bit_to_32bit(src_op.output[0],
                                              const_inputs[0].dtype)

        range_output = np.arange(start, limit, delta,dtype=dtype)
        converter_context.insert_weights(str(src_op.output[0]), range_output, 
                                         src_op_names=[src_op.name], src_tensor_names=src_op.input)
        return op_adapter.ConstantOp(src_op.output[0], range_output)

    def extract_input_names(self, src_op, converter_context):
        return []


OnnxTranslations.register_translation(OnnxRangeTranslation(),
                                      converter_type('Range', 'onnx'))


# ------------------------------------------------------------------------------
#   Reshape
# ------------------------------------------------------------------------------
class OnnxReshapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Reshape', [1, 5, 13, 14, 19, 21, 23, 24])

    def extract_parameters(self, src_op, converter_context):
        # There are two main versions of ONNX Reshape
        #    1. The old reshape, where shape is provided as an attribute
        #    2. The new reshape, where the shape is provided as a second input
        #
        # Backends and the converter support two versions of Reshape:
        #    1. Dynamic reshaping with a statically provided output shape
        #    2. Static reshaping, performed at conversion time
        #
        # Backends dont support the 2nd ONNX Reshape explicitly, however in the converter we can
        # pass static shape as suppl. attribute of our IR and still allow the network to be resizable for
        # limited cases. In addition, if a Shape' layer provided the shape it will have been saved
        # as static,
        # eg weight data, in the converter and all ops operating on that data will
        # become static ops and will be pruned during the final conversion.
        graph = converter_context.ir_graph
        shape = []
        params = extract_attributes(src_op, schema=self.op_schema())
        if len(src_op.input) > 1:
            shape_input = str(src_op.input[1])
            # only support constant for second input, if dynamic fetch will fail.
            shape = self.fetch_constant_op(shape_input, converter_context, fail_if_not_found=True,
                                           dtype=np.int32).tensor.tolist()
        else:
            if 'shape' in params:
                shape = params.shape

        # allowzero will make difference only if the shape has one of dimension equal to zero
        if 0 in shape and ('allowzero' in params and params.allowzero == 1):
            raise ValueError("Unsupported value of {} provided as allowzero attribute for Reshape Op {}. Only 0 (default) is supported.".format(1, src_op.name))

        input_name = str(src_op.input[0])
        const_input_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_input_op is not None:
            # static reshape of weight parameters
            output_name = str(src_op.output[0])
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, shape))

            const_input_tensor = const_input_op.tensor
            if len(shape) != 0:
                # Shape attribute of numpy Reshape and onnx Reshape is different, e.g., shape=[0,0,1,-1] is not supported in numpy reshape
                # Recalculating the shape as per numpy standards to avoid error
                shapeNew = ir_graph.ReshapeOp.calculate_shape(src_op.name,list(const_input_tensor.shape), shape)
                const_input_tensor = np.reshape(const_input_tensor, shapeNew)
                track_transform(const_input_op.constant_tensor,
                                ir_graph.TransformType.TRANSFORM_RESHAPE,
                                {"shape": shapeNew})
                converter_context.insert_weights(output_name, const_input_tensor, src_op_names=[src_op.name], src_tensor_names=src_op.input)
            else:
                # Shape (second input) could be an empty shape, which means reshape to a scalar
                # Set was_scalar = True in this case
                converter_context.insert_weights(output_name, const_input_tensor, True, [src_op.name], src_op.input)
            return None
        else:
            # dynamic reshape of activations
            input_buff = graph.get_buffer(input_name)
            log_assert(len(shape) != 0 or (input_buff.rank() == 1 and input_buff.shape[0] == 1),
                       "Reshape op {}: Unable to retrieve reshape shape", src_op.name)

            if input_buff.shape.is_dynamic():
                return self.translate_dynamic(src_op, input_buff, shape)

            # Handling when both input and output dims are equal.
            if input_buff.shape == shape:
                return op_adapter.IdentityOp(src_op.name)

            def is_transpose_equivalent(input_buf_shape, output_buf_shape):
                """
                Check if reshape can be replaced with Transpose
                :param input_buf_shape: shape of input buffer
                :param output_buf_shape: shape of output buffer
                :return: True if Reshape op can be replaced with Transpose
                """
                if (len(input_buf_shape) != len(output_buf_shape)):
                    return False
                modified_input_buf_shape = [ x for x in input_buf_shape if x != 1]
                modified_output_buf_shape = [ x for x in output_buf_shape if x != 1]
                if (modified_input_buf_shape != modified_output_buf_shape):
                    return False
                return True

            # Keep the output shape as [1] and set the output buff as a scalar tensor
            if len(shape) == 0:
                shape = input_buff.get_buf_dims()
                converter_context.scalar_tensor.add(str(src_op.output[0]))
                if len(src_op.input) > 1:
                    # handle the weights(shape) source if it is generated from framework op/tensor
                    converter_context.update_weights_trace_info_for_op(src_op.name, str(src_op.input[1]))
                return op_adapter.ReshapeOp(src_op.name, shape=shape)
            # Check if ReshapeOp is just permuting the shape of input buffer and hence working as a TransposeOp
            elif is_transpose_equivalent(input_buff.shape, shape):
                perm = []
                input_buf_shape = input_buff.shape
                if isinstance(input_buf_shape, op_adapter.BufferShape):
                    input_buf_shape = input_buf_shape.dims
                else:
                    input_buf_shape = input_buf_shape.copy()

                for value in shape:
                    index = input_buf_shape.index(value)
                    perm.append(index)
                    input_buf_shape[index] = None
                if len(src_op.input) > 1:
                    # handle the weights(shape) source if it is generated from framework op/tensor
                    converter_context.update_weights_trace_info_for_op(src_op.name, str(src_op.input[1]))
                return op_adapter.TransposeOp(src_op.name, perm=perm)
            else:
                if len(src_op.input) > 1:
                    # handle the weights(shape) source if it is generated from framework op/tensor
                    converter_context.update_weights_trace_info_for_op(src_op.name, str(src_op.input[1]))
                return op_adapter.ReshapeOp(src_op.name, shape=shape)


    def is_squeeze(self, input_shape, output_shape):
        def remove_ones(shape):
            new_shape = [dim for dim in shape if dim!=1]
            return new_shape

        # input_dims_remove_ones, output_dims_remove_ones are in order
        input_dims_remove_ones = remove_ones(input_shape)
        output_dims_remove_ones = remove_ones(output_shape)

        if input_dims_remove_ones == output_dims_remove_ones:
            if len(input_shape) > len(output_shape):
                return True
        return False

    def is_expand_dims(self, input_shape, output_shape):
        def remove_ones(shape):
            new_shape = [dim for dim in shape if dim!=1]
            return new_shape

        # input_dims_remove_ones, output_dims_remove_ones are in order
        input_dims_remove_ones = remove_ones(input_shape)
        output_dims_remove_ones = remove_ones(output_shape)

        if input_dims_remove_ones == output_dims_remove_ones:
            if len(input_shape) < len(output_shape):
                return True
        return False


    def is_identity(self, input_shape, output_shape):
        return input_shape == output_shape


    # Reshape with dynamic inputs can be supported only in certain scenarios
    # translating to Squeeze/ExpandDims/Identity being one of them
    # Further, Reshape translation to Squeeze/ExpandDims/Identity is currently supported only in below cases
    # 1. The input dim is dynamic in only one axes
    # 2. Squeeze or ExpandDims is operating only one axes
    # 3. The input dims/max-dims can be correlated with the output shape
    def translate_dynamic(self, src_op, input_buff, output_shape):
        input_shape = input_buff.shape

        if len(input_shape.dynamic_axes) != 1:
            raise ValueError("Cannot translate Reshape Op {} with dynamic input shape {} to output shape {}.".
                             format(src_op.name, input_buff.shape, output_shape))

        input_shape_dims = input_buff.shape.dims
        # canocalize output shape
        output_shape = ir_graph.ReshapeOp.calculate_shape(src_op.name, list(input_shape_dims), output_shape)

        # check if identity
        if self.is_identity(input_shape_dims, output_shape):
            return op_adapter.IdentityOp(src_op.name)

        # check if squeeze, expandDims
        rank_diff = np.abs(len(input_shape_dims) - len(output_shape))
        if rank_diff != 1:
            raise ValueError("Cannot translate Reshape Op {} with dynamic input shape {} to output shape {}.".
                             format(src_op.name, input_buff.shape, output_shape))

        if self.is_squeeze(input_shape_dims, output_shape):
            axes = []
            for i in range(len(output_shape)):
                if input_shape_dims[i] != output_shape[i]:
                    axes = [i]
                    break
            return op_adapter.SqueezeOp(src_op.name, axes=axes)
        elif self.is_expand_dims(input_shape_dims, output_shape):
            axes = []
            for i in range(len(input_shape_dims)):
                if input_shape_dims[i] != output_shape[i]:
                    axes = [i]
                    break
            return op_adapter.ExpandDimsOp(src_op.name, axes=axes)
        else:
            raise ValueError("Cannot translate Reshape Op {} with dynamic input shape {} to output shape {}.".
                             format(src_op.name, input_buff.shape, output_shape))


    def extract_input_names(self, src_op, converter_context):
        input_name = str(src_op.input[0])
        if converter_context.weights.consumed(input_name):
            return []
        else:
            return [input_name]

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxReshapeTranslation(),
                                      converter_type('Reshape', 'onnx'),
                                      op_adapter.ReshapeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Resize
# ------------------------------------------------------------------------------
class OnnxResizeTranslation(OnnxTranslationBase):
    SUPPORTED_RESIZE_MODES = ['nearest', 'linear', 'bilinear', 'cubic']
    SUPPORTED_COORD_TRANSFORM_MODES = ['asymmetric', 'align_corners', 'half_pixel', 'tf_half_pixel_for_nn',
                                       'pytorch_half_pixel']
    SUPPORTED_NEAREST_MODES = ['round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil']

    onnx_to_ir_transformation_mode = {
        "asymmetric": ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC,
        "align_corners": ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS,
        "half_pixel": ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL,
        "tf_half_pixel_for_nn": ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL,
        "pytorch_half_pixel": ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_PYTORCH_HALF_PIXEL,
    }

    onnx_to_ir_interpolation_mode = {
        "nearest": ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST,
        "linear": ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR,
        "bilinear": ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR,
        "cubic": ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC,
    }

    onnx_to_ir_nearest_mode = {
        "round_prefer_floor": ir_graph.QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR,
        "round_prefer_ceil": ir_graph.QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_CEIL,
        "floor": ir_graph.QNN_OP_RESIZE_NEAREST_MODE_FLOOR,
        "ceil": ir_graph.QNN_OP_RESIZE_NEAREST_MODE_CEIL,
    }

    default_transformation_mode = 'half_pixel'

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        schema_dict = self.register_op_schema('Resize', [10, 11, 13, 18, 19])
        schema_dict.replace_default_values(mode='nearest')
        schema_dict.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        resize_schema = self.op_schema()
        input_data_name = str(src_op.input[0])
        # If input is stored as weights, then fetch it and create a corresponding const op
        if not graph.has_buffer(input_data_name) and converter_context.weights.has(input_data_name):
            input_tensor = converter_context.weights.fetch(input_data_name)
            input_op = op_adapter.ConstantOp(input_data_name, input_tensor)
            input_node = graph.add(input_op, [], input_op.name)
            self.insert_constant_trace_info(input_op.name, input_node, converter_context)
            graph.add_src_op_info(input_node.op.name, None, input_node.output_names[0])
        elif graph.has_buffer(input_data_name):
            input_tensor = graph.get_buffer(input_data_name)
        input_shape = input_tensor.shape
        input_rank = len(input_shape)
        params = extract_attributes(src_op, attr_infos=[('mode', 's', 'nearest'),
                                                        ('coordinate_transformation_mode', 's', self.default_transformation_mode),
                                                        ('cubic_coeff_a', 'f', -0.75),
                                                        ('exclude_outside', 'i', 0),
                                                        ('nearest_mode', 's', 'round_prefer_floor'),
                                                        ('axes', 'li', list(range(input_rank))),
                                                        ('antialias', 'i', 0),
                                                        ('keep_aspect_ratio_policy', 's', 'stretch')],
                                    schema=resize_schema, validate=True)
        transformation_mode = params.coordinate_transformation_mode
        ir_cubic_coeff = params.cubic_coeff_a

        if input_rank not in [5, 4, 3]:
            raise ValueError("3D/4D/5D input to resize(scale) op {} is expected, but got shape: {}".
                             format(src_op.name, input_shape))

        ir_exclude_outside = False if params.exclude_outside == 0 else True
        param_antialias =  False if params.antialias == 0 else True
        ir_transformation_mode = self.onnx_to_ir_transformation_mode.get(transformation_mode)
        ir_interpolation_mode = self.onnx_to_ir_interpolation_mode.get(params.mode)
        ir_nearest_mode = self.onnx_to_ir_nearest_mode.get(params.nearest_mode)
        if not params.mode == "nearest" and transformation_mode == "tf_half_pixel_for_nn":
            raise ValueError("Resize op {} is not supported with mode {} and transformation_mode {}".
                             format(src_op.name, params.mode, transformation_mode))

        def get_output_dims_from_scales(scales):
            # Using numpy.multiply with data type np.float32
            # to align with the precision of C++ (onnx) float multiply when inferring shape.
            # e.g. 55*4.072727=223.99998 will be round up to 224 if using float in C++.
            output_dims = [np.floor(np.multiply(scale, shape, dtype=np.float32)) for scale, shape in zip(scales, input_shape)]
            return [output_dim.item() for output_dim in output_dims]

        def get_scales_from_sizes(sizes):
            # Opset 11 has 4th parameter as output sizes,
            # here we are calculating scales from output sizes
            # per onnx spec, scales is float.
            if len(sizes) == input_rank:
                scales = list(map(float, sizes))
                return [scale / shape for scale, shape in zip(scales, input_shape)]
            # From Opset 18, length of sizes can be same as the rank
            # of input or the length of axes. If derived from sizes,dimension
            # of scale needs to be expanded to match rank of input.
            if 'axes' not in params:
                raise ValueError("For Resize Op {} axes should be present with sizes if scale is empty".format(src_op.name))
            axes = params['axes']
            if not len(sizes) == len(axes):
                raise ValueError("For Resize Op {},"
                                 "Rank of sizes {} should match either rank of input {} or length of axes {}."
                                 .format(src_op.name, len(sizes), input_rank, len(axes)))
            scales = np.ones(input_rank)
            for size, axis in zip(sizes, axes):
                scales[axis] = size / input_shape[axis]
            return scales

        def get_scales_from_axes(scales_tensor, axes, input_rank):
            # From Opset 18, length of scales can be same as the rank
            # of input or the length of axes. If derived from axes,dimension
            # of scale needs to be expanded to match rank of input.
            if not len(scales_tensor) == len(axes):
                raise ValueError("For Resize Op {},"
                                 "Rank of scales {} should match either rank of input {} or length of axes {}."
                                 .format(src_op.name, len(scales_tensor), input_rank, len(axes)))
            expanded_scales = np.array(np.ones(input_rank))
            expanded_scales[axes] = scales_tensor
            return expanded_scales

        def get_static_tensor(tensor_name, type, graph, dtype=np.float32):
            """
            :param tensor_name:
            :param type: String for scales or sizes
            :param graph: IrOpGraph
            :param dtype: datatype in numpy type
            :return: numpy tensor or None. Can raise Error
            """
            tensor = None
            if converter_context.weights.has(tensor_name):
                tensor = converter_context.weights.fetch(tensor_name).astype(dtype)
            elif graph.has_buffer(tensor_name):
                if isinstance(graph.get_buffer(tensor_name).producer.op, op_adapter.ConstantOp):
                    tensor = graph.get_buffer(tensor_name).producer.op.tensor
                else:
                    raise TypeError("Resize Op {}: Dynamic {} input ({}) not supported".format(src_op.name,
                                                                                               type,
                                                                                               tensor_name))
            return tensor

        def is_tensor_empty(tensor):
            return tensor.shape == (0,)

        # roi input is currently unsupported in Resize Op.
        if transformation_mode == "tf_crop_and_resize" and len(src_op.input) > 2:
            roi_name = str(src_op.input[1])
            roi_tensor = get_static_tensor(roi_name, "roi", graph)
            if roi_tensor is not None:
                raise ValueError("Resize Op {}: roi ({}) is unsupported.".
                        format(src_op.name, roi_name))

        # Handle sizes input if src_op has 4 inputs and scales input is ''
        if len(src_op.input) == 4:
            sizes_name = str(src_op.input[-1])
            scales_name = str(src_op.input[-2])
            scales_tensor = get_static_tensor(scales_name, "scales", graph)

            if scales_tensor is None or is_tensor_empty(scales_tensor):
                # per onnx spec, size is int64. But int32 may be enough.
                sizes_tensor = get_static_tensor(sizes_name, "sizes", graph, dtype=np.int32)
                if sizes_tensor is None or not sizes_tensor.shape:
                    raise ValueError("Resize Op {}: One of scales ({}) or sizes ({}) needs to be provided".format(
                        src_op.name, scales_name, sizes_name
                    ))
                sizes = sizes_tensor.tolist()
                if 'keep_aspect_ratio_policy' in params:
                    if 'axes' in params:
                        axes = params['axes']
                        # Handling the negative axes scenario
                        neg_axes = [idx for idx, ele in enumerate(axes) if ele < 0]
                        if neg_axes:
                            for neg_idx in neg_axes:
                                axes[neg_idx] += input_rank
                    else:
                        axes = list(range(input_rank))
                    if params.keep_aspect_ratio_policy == "not_larger":
                        if len(input_shape) == len(sizes) and len(sizes) != len(axes):
                            scale_value = min([sizes[idx] / input_shape[idx] for idx in axes])
                            for size_ele in axes:
                                sizes[size_ele] = round(scale_value * input_shape[size_ele])
                        elif len(sizes) == len(axes):
                            scale_value = min([sizes[idx] / input_shape[axes[idx]] for idx in range(len(axes))])
                            input_cpy = list(np.copy(np.array(input_shape)))
                            sizes = input_cpy
                            for size_ele in axes:
                                sizes[size_ele] = round(scale_value * input_shape[size_ele])
                        else:
                            # Length of sizes should be same as the rank of input or the length of axes
                            raise ValueError("For source op {} of type {},"
                                             "length of sizes {} should match either rank of input {} or length of axes."
                                             .format(src_op.name, src_op.op_type, len(sizes), len(input_shape), len(axes)))

                    elif params.keep_aspect_ratio_policy == "not_smaller":
                        if len(input_shape) == len(sizes) and len(sizes) != len(axes):
                            scale_value = max([sizes[idx] / input_shape[idx] for idx in axes])
                            for size_ele in axes:
                                sizes[size_ele] = round(scale_value * input_shape[size_ele])
                        elif len(sizes) == len(axes):
                            scale_value = max([sizes[idx] / input_shape[axes[idx]] for idx in range(len(axes))])
                            input_cpy = list(np.copy(np.array(input_shape)))
                            sizes = input_cpy
                            for size_ele in axes:
                                sizes[size_ele] = round(scale_value * input_shape[size_ele])
                        else:
                            # Length of sizes should be same as the rank of input or the length of axes
                            raise ValueError("For source op {} of type {},"
                                             "length of sizes {} should match either rank of input {} or length of axes."
                                             .format(src_op.name, src_op.op_type, len(sizes), len(input_shape), len(axes)))

                scales = get_scales_from_sizes(sizes)
            else:
                # Opset 18: Handle the scales when parsed with axes
                if len(scales_tensor) < input_rank and 'axes' in params:
                    axes = params['axes']
                    scales_tensor = get_scales_from_axes(
                        scales_tensor, axes, input_rank)
                scales = scales_tensor.tolist()
                if scales[0] != 1.0:
                    raise ValueError("Resize Op {} does not support resize along Batch Dimension".format(src_op.name))
                if scales[1] != 1.0:
                    raise ValueError("Resize Op {} does not support resize along Channel Dimension".format(src_op.name))

                sizes = get_output_dims_from_scales(scales)
                # Calculate scales again to account for align_corners
                scales = get_scales_from_sizes(sizes)
        elif len(src_op.input) > 1:
            scales_name = str(src_op.input[-1])
            scales_tensor = get_static_tensor(scales_name, "scales", graph)
            if scales_tensor is None or not scales_tensor.shape:
                raise ValueError("Resize Op {}: scales ({}) tensor is invalid".format(
                    src_op.name, scales_name
                ))
            if len(scales_tensor) < input_rank and 'axes' in params:
                axes = params['axes']
                scales_tensor = get_scales_from_axes(
                    scales_tensor, axes, input_rank)
            scales = scales_tensor.tolist()
            if scales[0] != 1.0:
                raise ValueError("Resize Op {} does not support resize along Batch Dimension".format(src_op.name))
            if scales[1] != 1.0:
                raise ValueError("Resize Op {} does not support resize along Channel Dimension".format(src_op.name))

            sizes = get_output_dims_from_scales(scales)
            # Calculate scales again to account for align_corners
            scales = get_scales_from_sizes(sizes)
        else:
            # deprecated. Added for Upsample version 7 and below
            scales = extract_attributes(src_op, attr_infos=[('scales', 'lf')], schema=resize_schema, validate=True).scales

        return op_adapter.ResizeOp(src_op.name,
                                   exclude_outside=ir_exclude_outside,
                                   transformation_mode=ir_transformation_mode,
                                   interpolation_mode=ir_interpolation_mode,
                                   antialias=param_antialias,
                                   nearest_mode=ir_nearest_mode,
                                   cubic_coeff=ir_cubic_coeff,
                                   scale_depth=scales[-3],
                                   scale_height=scales[-2],
                                   scale_width=scales[-1])

    @classmethod
    def validate_attribute_values(cls, src_op, attr_name, attr_value):
        if attr_name == 'mode':
            src_op_mode = attr_value
            if src_op_mode not in cls.SUPPORTED_RESIZE_MODES:
                raise ValueError("{} mode for Resize op {} was not supported. Please choose from modes: {}".
                                 format(src_op_mode, src_op.name, cls.SUPPORTED_RESIZE_MODES))
        elif attr_name == 'scales':
            scales = attr_value
            if scales[0] != 1 or scales[1] != 1:
                log_warning("For Resize op {}, only resizing Height and Width is supported, ignoring other dimensions.".
                            format(src_op.name))
        elif attr_name == 'coordinate_transformation_mode':
            src_op_mode = attr_value
            if src_op_mode not in cls.SUPPORTED_COORD_TRANSFORM_MODES:
                raise ValueError("transformation mode {} was not supported for Resize op {}. "
                                 "Please choose from modes: {}".format(src_op_mode, src_op.name, cls.SUPPORTED_COORD_TRANSFORM_MODES))
        elif attr_name == 'nearest_mode':
            src_op_mode = attr_value
            if src_op_mode not in cls.SUPPORTED_NEAREST_MODES:
                raise ValueError(
                    "nearest mode {} for Resize op {} was not supported. Please choose from modes: {}"
                    .format(src_op_mode, src_op.name, cls.SUPPORTED_NEAREST_MODES))

    def extract_input_names(self, src_op, converter_context):
        if len(src_op.input) > 2:
            return [str(src_op.input[0])]
        else:
            return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]

    def infer_output_shapes(self, op, input_shapes):
        log_debug("Resize op {}: inferred output shape {}".format(op.name, op.output_shape))
        return [op.output_shape]


OnnxTranslations.register_translation(OnnxResizeTranslation(),
                                      converter_type('Resize', 'onnx'),
                                      op_adapter.ResizeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ScatterElements
# ------------------------------------------------------------------------------
class OnnxScatterElementsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ScatterElements', [11, 13, 16])
        self.register_op_schema('Scatter', [9, 11])
        self.reduction_types = {"none": ir_graph.QNN_OP_SCATTER_ELEMENTS_REDUCTION_NONE,
                                "add": ir_graph.QNN_OP_SCATTER_ELEMENTS_REDUCTION_ADD,
                                "mul": ir_graph.QNN_OP_SCATTER_ELEMENTS_REDUCTION_MUL,
                                "max": ir_graph.QNN_OP_SCATTER_ELEMENTS_REDUCTION_MAX}

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_ops, translated_ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        # the number of input op should be 1 or 2, or no input ops
        # if input_op = None => inputs are all constant or all dynamic
        if input_ops:
            for input_op in input_ops:
                node = graph.add(input_op, [], input_op.name)
                self.insert_constant_trace_info(input_op.name, node, converter_context)
                graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops[0].type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # when scatter_elements op is represented as a constant op, i.e input ops is None
            last_node = graph.add(translated_ops[-1], [], output_names)
            op_source_info = [(src_op.name, TraceType.OP), (output_names[0], TraceType.TENSOR)]
            for input_name in input_names:
                op_source_info.append((input_name, TraceType.TENSOR))
            self.insert_trace_info([last_node, graph.get_buffer(output_names[0])], op_source_info, converter_context)
        else:
            # when scatter_elements op has one or more dynamic inputs
            last_node = graph.add(translated_ops[-1], input_names, output_names)
            self.insert_default_trace_info(src_op, last_node, converter_context)
        self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def check_reduction(self, src_op, reduction: str = "none"):
        if reduction not in self.reduction_types:
            raise TypeError("Cannot perform ScatterElements op {}. Expected reduction type"
                    " to be one of: {}, instead got: {}".format(
                        src_op.name, list(self.reduction_types.keys()), reduction))

    def _perform_static_scatter_elements(self, input_data: np.ndarray, indices: np.ndarray,
                                   updates: np.ndarray, reduction: str = "none"):

        # Perform only reduction = none since that is supported in opset 11,13
        # No need to reject other reduction values since the attribute only exists in opset 16
        # TODO: Check for other reduction types once new version is added

        static_scatter_data = np.copy(input_data)
        for idx_tuple in np.ndindex(indices.shape):
            update_value = updates[idx_tuple]
            idx_list = list(idx_tuple)
            idx_list[self.axis] = indices[idx_tuple]
            idx_tuple = tuple(idx_list)
            if self.reduction == "add":
                static_scatter_data[idx_tuple] += update_value
            elif self.reduction == "mul":
                static_scatter_data[idx_tuple] *= update_value
            elif self.reduction == "max":
                static_scatter_data[idx_tuple] = max(static_scatter_data[idx_tuple], update_value)
            else:
                static_scatter_data[idx_tuple] = update_value

        return static_scatter_data

    def check_duplicate_indices(self, input_data, indices, op_name):
        # when indices and input_data are constant, we can check whether they have duplicate indices
        if indices is not None and input_data is not None:

            # check to ensure unique indices if reduction is none
            unique_indices = set()
            for idx_tuple in np.ndindex(indices.shape):
                idx_list = list(idx_tuple)
                idx_list[self.axis] = indices[idx_tuple]
                if idx_list[self.axis] < 0:
                    idx_list[self.axis] += input_data.shape[self.axis]
                idx_tuple = tuple(idx_list)
                if idx_tuple not in unique_indices:
                    unique_indices.add(idx_tuple)
                else:
                    log_warning("Duplicate scatter elements indices detected when reduction is set to None for Op {}. "
                                "This is not recommended and may result in inconsistent output and accuracy issues".
                                format(op_name))

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_ops = []
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])
        updates_name = str(src_op.input[2])
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type))
        self.axis = params.axis
        self.reduction = params.get('reduction', 'none')
        self.check_reduction(src_op, self.reduction)
        input_names = list(map(str, src_op.input))
        const_input_ops = []

        # Create prunable const ops for all inputs if set
        const_input_op = self.fetch_constant_op(input_data_name, converter_context, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        const_indices_op = self.fetch_constant_op(indices_name, converter_context,
                                                  quantizable=False,
                                                  fail_if_dynamic=False)
        if const_indices_op is not None:
            const_input_ops.append(const_indices_op)

        const_updates_op = self.fetch_constant_op(updates_name, converter_context, fail_if_dynamic=False)
        if const_updates_op is not None:
            const_input_ops.append(const_updates_op)

        # If all inputs are static, then perform static scatter elements and return constant
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor
            updates = const_input_ops[2].tensor
            self.check_duplicate_indices(input_data, indices, src_op.name)
            scatter_data = self._perform_static_scatter_elements(input_data, indices, updates)
            converter_context.insert_weights(str(src_op.output[0]), scatter_data,
                                             src_op_names=[src_op.name], src_tensor_names=src_op.input)
            translated_ops.append(op_adapter.ConstantOp(src_op.output[0], scatter_data))
            return input_ops, translated_ops

        # If input is stored as weights then create a corresponding const op
        input_data, indices = None, None
        if not graph.has_buffer(input_data_name) and converter_context.weights.has(input_data_name):
            input_data = converter_context.weights.fetch(input_data_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(input_data_name, input_data))

        # If indices is stored as weights then create a corresponding const op
        if not graph.has_buffer(indices_name) and converter_context.weights.has(indices_name):
            indices = converter_context.weights.fetch(indices_name, prunable=False).astype(np.int32)
            # canonicalize negative indice value
            if input_data is not None:
                input_shape = input_data.shape
            else:
                input_shape = graph.get_buffer(input_data_name).shape
            indices[indices < 0] += input_shape[self.axis]
            indices_op = op_adapter.ConstantOp(indices_name, indices, quantizable=False)
            input_ops.append(indices_op)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices = indices_op.tensor = indices_op.tensor.astype(np.int32)

        self.check_duplicate_indices(input_data, indices, src_op.name)

        # If updates input is stored as weights then create a corresponding const op
        if not graph.has_buffer(updates_name) and converter_context.weights.has(updates_name):
            updates = converter_context.weights.fetch(updates_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(updates_name, updates))

        translated_ops.append(op_adapter.ScatterElementsOp(src_op.name,
                                                           axis=params.axis,
                                                           reduction=self.reduction_types[self.reduction]))

        return input_ops, translated_ops

    def extract_input_names(self, src_op, converter_context):
        if converter_context.weights.has_all(src_op.input):
            return []
        else:
            return super().extract_input_names(src_op, converter_context)


OnnxTranslations.register_translation(OnnxScatterElementsTranslation(),
                                      converter_type('ScatterElements', 'onnx'),
                                      converter_type('Scatter', 'onnx'))


# ------------------------------------------------------------------------------
#   ScatterND
# ------------------------------------------------------------------------------
class OnnxScatterNDTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ScatterND', [11, 13, 16])
        self.reduction_types = {"none": ir_graph.QNN_OP_SCATTER_ND_REDUCTION_NONE,
                                "add": ir_graph.QNN_OP_SCATTER_ND_REDUCTION_ADD,
                                "mul": ir_graph.QNN_OP_SCATTER_ND_REDUCTION_MUL}

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_ops, translated_ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        # input op should only be 1 or 2 or None.
        # if input_op = None => all inputs are constant or all are dynamic
        # When input ops are None, scatter ND is a constant op
        if input_ops:
            for input_op in input_ops:
                node = graph.add(input_op, [], input_op.name)
                self.insert_constant_trace_info(input_op.name, node, converter_context)
                graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops[0].type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # when scatter_nd op is represented as a constant op i.e input ops is None
            last_node = graph.add(translated_ops[0], [], output_names)
            op_source_info = [(src_op.name, TraceType.OP), (output_names[0], TraceType.TENSOR)]
            for input_name in input_names:
                op_source_info.append((input_name, TraceType.TENSOR))
            self.insert_trace_info([last_node, graph.get_buffer(output_names[0])], op_source_info, converter_context)
            self.add_src_op_info(last_node.op.name, src_op, graph)
        else:
            # when scatter nd op has one or more dynamic inputs
            last_node = graph.add(translated_ops[0], input_names, output_names)
            self.insert_default_trace_info(src_op, last_node, converter_context)
            self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def check_reduction(self, src_op, reduction: str = "none"):
        if reduction not in self.reduction_types:
            raise TypeError("Cannot perform ScatterND op {}. Expected reduction type"
                            " to be one of: {}, instead got: {}".format(
                                src_op.name, list(self.reduction_types.keys()), reduction))

    def _perform_static_scatter_nd(self, input_data: np.ndarray, indices: np.ndarray,
                                   updates: np.ndarray, reduction: str = "none"):
        static_scatter_data = np.copy(input_data)
        update_idx = indices.shape[:-1]
        for idx in np.ndindex(update_idx):
            update_value = updates[idx]
            if reduction == "add":
                static_scatter_data[tuple(indices[idx])] += update_value
            elif reduction == "mul":
                static_scatter_data[tuple(indices[idx])] *= update_value
            else:
                static_scatter_data[tuple(indices[idx])] = update_value


        return static_scatter_data

    def extract_parameters(self, src_op, converter_context):
        # Note there are no attributes to extract for versions 11, 13

        graph = converter_context.ir_graph
        input_ops = []
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])
        updates_name = str(src_op.input[2])
        params = extract_attributes(src_op,
                                    attr_infos=[('reduction', 's', "none")],
                                    schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)
        self.reduction = params.reduction
        self.check_reduction(src_op, self.reduction)
        input_names = list(map(str, src_op.input))
        const_input_ops = []

        # Create prunable const ops for all inputs if set
        const_input_op = self.fetch_constant_op(input_data_name, converter_context, dtype=None, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        const_indices_op = self.fetch_constant_op(indices_name, converter_context,
                                                  quantizable=False,
                                                  fail_if_dynamic=False)
        if const_indices_op is not None:
            const_input_ops.append(const_indices_op)

        const_updates_op = self.fetch_constant_op(updates_name, converter_context, fail_if_dynamic=False)
        if const_updates_op is not None:
            const_input_ops.append(const_updates_op)

        # If all inputs are static, then perform static scatter and return
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor
            updates = const_input_ops[2].tensor
            scatter_data = self._perform_static_scatter_nd(input_data, indices, updates, self.reduction)
            converter_context.insert_weights(str(src_op.output[0]), scatter_data,
                                             src_op_names=[src_op.name], src_tensor_names=src_op.input)
            translated_ops.append(op_adapter.ConstantOp(src_op.output[0], scatter_data))
            return input_ops, translated_ops

        # If input is stored as weights then create a corresponding const op
        input_data, indices = None, None
        if not graph.has_buffer(input_data_name) and converter_context.weights.has(input_data_name):
            input_data = converter_context.weights.fetch(input_data_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(input_data_name, input_data))

        # If indices is stored as weights then create a corresponding const op
        if not graph.has_buffer(indices_name) and converter_context.weights.has(indices_name):
            indices = converter_context.weights.fetch(indices_name, prunable=False).astype(np.uint32)
            indices_op = op_adapter.ConstantOp(indices_name, indices, quantizable=False)
            input_ops.append(indices_op)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices = indices_op.tensor = indices_op.tensor.astype(np.uint32)

        if indices is not None:
            if np.any(indices < 0):
                if input_data is None:
                    raise ValueError("Cannot resolve constant negative indices for ScatterND op {} indices: "
                                     "{} if input data is not static".format(src_op.name, indices_name))
                else:
                    with np.nditer(indices, op_flags=['readwrite']) as it:
                        for index in it:
                            if index < 0:
                                index += len(input_data.shape)

            # check to ensure unique indices if reduction is none
            # TODO: Change when onnx version is updated as reduction is none for opset version < 16
            update_indices = indices.shape[:-1]
            unique_indices = set()
            for idx in np.ndindex(update_indices):
                # hash to place list value in unique_indices set
                idx_list = tuple(indices[idx].tolist())
                if idx_list not in unique_indices:
                    unique_indices.add(idx_list)
                else:
                    log_warning("Duplicate scatter indices detected when reduction is set to None for Op {}. "
                                "This is not recommended and can result in inconsistent output and accuracy issues".
                                format(src_op.name))

        # If updates is stored as weights then create a corresponding const op
        if not graph.has_buffer(updates_name) and converter_context.weights.has(updates_name):
            updates = converter_context.weights.fetch(updates_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(updates_name, updates))

        translated_ops.append(op_adapter.ScatterNDOp(src_op.name, reduction=self.reduction_types[self.reduction]))

        return input_ops, translated_ops

    def extract_input_names(self, src_op, converter_context):
        if converter_context.weights.has_all(src_op.input):
            return []
        else:
            return super().extract_input_names(src_op, converter_context)


OnnxTranslations.register_translation(OnnxScatterNDTranslation(),
                                      converter_type('ScatterND', 'onnx'),
                                      op_adapter.ScatterNDOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   SequenceConstruct
# ------------------------------------------------------------------------------
class OnnxSequenceConstructTranslation(OnnxTranslationBase):
    def __init__(self):
        super().__init__()
        self.register_op_schema('SequenceConstruct', [11])

    def add_op(self, src_op, context, **kwargs):
        log_error("Unsupported Op Type: SequenceConstruct. Please use "
                  "--perform_sequence_construct_optimizer to optimize the SequenceConstruct Op "
                  "or use --out_tensor_name to specify the output names.")
        sys.exit(1)


OnnxTranslations.register_translation(OnnxSequenceConstructTranslation(),
                                      converter_type('SequenceConstruct', 'onnx'))


# ------------------------------------------------------------------------------
#   Shape
# ------------------------------------------------------------------------------
class OnnxShapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Shape', [1, 13, 15])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
        input_name = str(src_op.input[0])

        constant_op = self.fetch_constant_op(input_name, converter_context, dtype=np.int32, fail_if_not_found=True,
                                             fail_if_dynamic=False)
        if constant_op:
            shape = constant_op.tensor.shape
        elif graph.has_buffer(input_name):
            shape = graph.get_buffer(input_name).shape

        start = 0
        end = len(shape)
        schema = self.op_schema(op_type=src_op.op_type)
        if schema.version[0] > 1:
            params = extract_attributes(src_op,
                                    attr_infos=[('start', 'i', 0), ('end', 'i', len(shape))],
                                    schema=schema, validate=True)
            start = params.start
            end = params.end

        output_name = str(src_op.output[0])
        shape = np.asarray(shape, dtype=np.int32)[start:end]
        converter_context.insert_weights(output_name, shape,
                                         src_op_names=[src_op.name],src_tensor_names=src_op.input)
        return None

    def extract_input_names(self, src_op, converter_context):
        return []

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxShapeTranslation(),
                                      converter_type('Shape', 'onnx'))


# ------------------------------------------------------------------------------
#   Size
# ------------------------------------------------------------------------------
class OnnxSizeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Size', [1, 13, 15, 19, 21, 23, 24])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
        input_name = str(src_op.input[0])

        constant_op = self.fetch_constant_op(input_name, converter_context, dtype=np.int32, fail_if_not_found=True,
                                             fail_if_dynamic=False)
        if constant_op:
            shape = constant_op.tensor.shape
        elif graph.has_buffer(input_name):
            shape = graph.get_buffer(input_name).shape.dims

        size = np.array([math.prod(shape)])
        output_name = str(src_op.output[0])
        converter_context.insert_weights(output_name, size,
                                         src_op_names=[src_op.name],src_tensor_names=src_op.input,
                                         was_scalar=True)
        return None

    def extract_input_names(self, src_op, converter_context):
        return []

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]

OnnxTranslations.register_translation(OnnxSizeTranslation(),
                                      converter_type('Size', 'onnx'))


# ------------------------------------------------------------------------------
#   Slice
# ------------------------------------------------------------------------------
class OnnxSliceTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Slice', [1, 10, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        is_null_tensor = False
        output_shape = []
        null_axes = []
        graph = converter_context.ir_graph
        input_names = [str(x) for x in src_op.input]
        params = extract_attributes(src_op, schema=self.op_schema())
        const_inputs_params = self._fetch_inputs_as_params(src_op, converter_context, params)
        params.update(const_inputs_params)

        log_assert(len(params.starts) == len(params.axes),
                   "Node {}: expected same number of starts as axes",
                   src_op.name)
        log_assert(len(params.ends) == len(params.axes),
                   "Node {}: expected same number of ends as axes",
                   src_op.name)
        log_assert(all(params.steps),
                   "Node {}: expected all steps != 0",
                   src_op.name)

        # Static slicing used for shape tensors
        if converter_context.weights.has(input_names[0]):
            data = converter_context.weights.fetch(input_names[0])
            for i in range(len(params.axes)):
                start, end = self.get_indices(params.starts[i],
                                              params.ends[i],
                                              params.steps[i],
                                              data.shape[params.axes[i]])
                data = data.take(indices=list(range(start, end, params.steps[i])), axis=params.axes[i])
            output_name = str(src_op.output[0])
            converter_context.insert_weights(output_name, data,
                                             src_op_names=[src_op.name], src_tensor_names=input_names)
            # If the input has overridden but the output doesn't,
            # Copy the input enc to output to avoid accuracy drop
            enc_in = graph.get_overridden_encoding(input_names[0])
            enc_out = graph.get_overridden_encoding(output_name)
            if enc_in is not None and enc_out is None:
                if 'version' in graph.user_quantization_overrides and graph.user_quantization_overrides['version'] == '2.0.0':
                    add_new_user_quantization_overrides(graph, output_name, "encodings", enc_in.copy())
                else:
                    add_new_user_quantization_overrides(graph, output_name, "param_encodings", enc_in.copy())
            return None

        input_buf = graph.get_buffer(input_names[0])
        input_buf_dy_axes = input_buf.shape.dynamic_axes
        rank = input_buf.rank()
        begin = [0] * rank
        end = [0] * rank
        strides = [0] * rank

        for index, axis in enumerate(params.axes):
            begin[axis], end[axis] = self.get_indices(params.starts[index],
                                                      params.ends[index],
                                                      params.steps[index],
                                                      input_buf.shape.dims[axis],
                                                      is_dynamic=axis in input_buf_dy_axes)
            strides[axis] = params.steps[index]

            # If the input data is dynamic and start is out of bounds with positive steps, i.e., start will be equal to dim
            # Or if start is equal to end, then
            # Output will be null tensor
            # Still need to add to the weights so that subsequent node tracks properly
            if begin[axis]==input_buf.shape.dims[axis] or begin[axis] == end[axis]:
                is_null_tensor = True
                null_axes.append(axis)
                continue

            # canonicalization won't happen if axis is in dy_axes
            if axis in input_buf_dy_axes:
                continue

            # add check to find if there is empty data case or out-of-range indices
            log_assert(begin[axis] < end[axis] if strides[axis] > 0 else begin[axis] > end[axis],
                       "Node {}: invalid stride for begin {} and end {} at axis {}",
                       src_op.name, begin[axis], end[axis], axis)
            log_assert(0 <= begin[axis] < input_buf.shape.dims[axis],
                       "Node {}: begin:{} at axis {} is out-of-range",
                       src_op.name, begin[axis])
            log_assert(-1 <= end[axis] <= input_buf.shape.dims[axis],
                       "Node {}: end:{} at axis {} is out-of-range",
                       src_op.name, end[axis], axis)

        for i, stride in enumerate(strides):
            if not stride:
                begin[i], end[i] = 0, input_buf.shape.dims[i]
                strides[i] = 1

        if is_null_tensor:
            data = []
            shape = self.calc_null_tensor_shape(input_buf.shape.dims, null_axes, begin, end, strides)
            data = np.array(data).reshape(shape)
            output_name = str(src_op.output[0])
            converter_context.insert_weights(output_name, data,
                                             src_op_names=[src_op.name], src_tensor_names=src_op.input)
            return None

        ranges = list(map(list, zip(begin, end, strides)))
        op = op_adapter.StridedSliceOp(name=src_op.name, ranges=ranges)

        # there is null tensor logic in onnx translation so we can not clean out
        # canonicalization logic
        # TODO: clean up canonicalization logic in translation
        return op

    def calc_null_tensor_shape(self, input_shape, null_axes, begin, end, strides):
        shape = input_shape
        for j in range(len(shape)):
            if j in null_axes:
                shape[j] = 0
            else:
                begin_index = begin[j] if begin[j] > 0 else begin[j] + shape[j]
                end_index = end[j] if end[j] > 0 else end[j] + shape[j]
                shape[j] = int(np.ceil((end[j] - begin[j])/strides[j]))
        return shape

    def _fetch_inputs_as_params(self, src_op, converter_context, params):
        # opset 10,11 need handle 5 inputs, fetch constant input and add it to params
        # NOTE: Runtime does not allow dynamic input for starts, ends, axes and steps
        # input indices: data: 0, starts: 1, ends: 2, axes: 3(optional), steps: 4(optional)
        graph = converter_context.ir_graph
        input_names = [str(x) for x in src_op.input]
        rank = 0
        if graph.has_buffer(input_names[0]):
            input_buf = graph.get_buffer(input_names[0])
            rank = input_buf.rank()
        elif converter_context.weights.has(input_names[0]):
            rank = len(converter_context.weights.fetch(input_names[0], prunable=False).shape)
        keys = ['data', 'starts', 'ends', 'axes', 'steps']
        if len(src_op.input) >= 3:
            for key, name in zip(keys[1:], input_names[1:]):
                # ONNX may use empty string as a placeholder
                # So add an and-condition to further check it.
                if name and converter_context.weights.has(name):
                    # handle INT_MAX and INT_MIN case in ONNX spec, require fetch int64 directly
                    # case: INT64_MAX -> cast to float and cast to int64 -> INT64_MIN
                    # case: INT64_MAX -> cast to int32 -> -1
                    params[key] = converter_context.weights.fetch(name, dtype=np.int64, prunable=False).tolist()
                    if key == 'axes':
                        for axis in params['axes']:
                            log_assert(-rank <= axis <= rank-1,
                                       "expected axis range from {} to {} for Slice op {}, but got {}",
                                       -rank, rank-1, src_op.name, axis)
                elif graph.has_buffer(name):
                    raise ValueError("Invalid inputs {} for Slice op {}, only constant inputs supported".
                                     format(name, src_op.name))

        if 'axes' not in params or len(params.axes) == 0:
            params['axes'] = list(range(len(params.starts)))
        if 'steps' not in params or len(params.steps) == 0:
            params['steps'] = list([1] * len(params.starts))

        return params

    def extract_input_names(self, src_op, converter_context):
        graph = converter_context.ir_graph
        # If this was translated to a static op don't return input names
        if converter_context.weights.has(str(src_op.input[0])):
            return []
        else:
            # Handle constant and initializer cases, do not add them to input_names to avoid prune error.
            actual_input_names = []
            for input_name in map(str, src_op.input):
                if input_name in graph.buffers and not converter_context.weights.has(input_name):
                    actual_input_names.append(input_name)
                else:
                    # handle the weights source if it is generated from framework op/tensor
                    converter_context.update_weights_trace_info_for_op(src_op.name, input_name)
            return actual_input_names

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.has(str(src_op.input[0])):
            return []
        else:
            return list(map(str, src_op.output))

    @staticmethod
    def get_indices(start, end, step, dim, is_dynamic=False):
        if is_dynamic:
            # If start/end is out of bounds and step is positive,
            # start/end will be dim for out of bounds case
            start = min(start, dim)
            end = min(end, dim)

            # Don't canonicalize negative values of start and end for dynamic case
            return start, end

        # Negative values mean wrap around, like in python
        if start < 0:
            start = int(start % dim)

        if step < 0:
            # higher than the size, however, means stop at the end - 1.
            start = min(start, dim-1)
            end = max(end, -(dim+1))
        else:
            # If start is out of bounds and step is positive,
            # start will be dim for out of bounds case
            start=min(start,dim)
            end = min(end, dim)

        if end < 0:
            end = end + dim

        return start, end


OnnxTranslations.register_translation(OnnxSliceTranslation(),
                                      converter_type('Slice', 'onnx'),
                                      op_adapter.StridedSliceOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Split
# ------------------------------------------------------------------------------
class OnnxSplitTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Split', [1, 2, 11, 13, 18])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()

        if schema.version[0] < 13:
            schema.replace_default_values(split=[])
            params = extract_attributes(src_op, schema=schema)
            split_index = ir_graph.SplitOp.convert_sizes_to_indices(params.split)
        else:
            # Start from onnx split-18, new attribute 'num_outputs' added. Either input 'split' or the
            # attribute 'num_outputs' should be specified, but not both. Here converter does not need
            # 'num_outputs' attribute, hence only keep 'axis' in schema attributes.
            new_attributes = schema.attributes(names=['axis'])
            params = extract_attributes(src_op, schema=schema, attr_infos=new_attributes)
            if len(src_op.input) > 1:
                # backend only support split_index as a parameter, so need split input as a constant
                const_split_op = self.fetch_constant_op(str(src_op.input[1]), converter_context,
                                                        fail_if_dynamic=True, fail_if_not_found=True)
                split = const_split_op.tensor
                split_index = ir_graph.SplitOp.convert_sizes_to_indices(split)
            else:
                split_index = []

        const_input_op = self.fetch_constant_op(str(src_op.input[0]), converter_context, fail_if_dynamic=False)
        if const_input_op is not None:
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            # empty split index for onnx represents equal splitting when the num of output > 1
            if len(split_index) == 0 and len(src_op.output) > 1:
                split_index = len(src_op.output)
            w = const_input_op.tensor
            w = np.array_split(w, split_index, params.axis)
            # To account for multiple consumers
            for i in range(len(w)):
                converter_context.insert_weights(src_op.output[i], w[i],
                                                 src_op_names=[src_op.name], src_tensor_names=src_op.input)
            return None

        return op_adapter.SplitOp(src_op.name,
                                  axis=params.axis,
                                  split_index=split_index)

    def extract_input_names(self, src_op, converter_context):
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxSplitTranslation(),
                                      converter_type('Split', 'onnx'),
                                      op_adapter.SplitOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Squeeze
# ------------------------------------------------------------------------------
class OnnxSqueezeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Squeeze', [1, 11, 13, 21, 23, 24])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_name = str(src_op.input[0])
        params = extract_attributes(src_op, schema=self.op_schema())

        axes = []
        if len(src_op.input) > 1:
            axes_input = str(src_op.input[1])
            # only support constant for second input, if dynamic fetch will fail.
            axes = self.fetch_constant_op(axes_input, converter_context, dtype=np.int32).tensor.tolist()
        elif 'axes' in params:
            axes = params.axes

        const_input_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_input_op is not None:
            # static squeeze of weight parameters
            output_name = str(src_op.output[0])
            w = converter_context.weights.fetch(input_name)
            if not len(axes):
                axes = [i for i, s in enumerate(w.shape) if s == 1]
            output_shape = self._get_squeezed_shape(w.shape, axes)

            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))
            w = np.reshape(w, output_shape)
            # The w here might be a np "array scalar" whose shape attribute is
            # an empty tuple, which may break backends.
            # So we reshape "array scalar" to exactly an array with shape (1, )
            was_scalar = False
            if not w.shape:
                was_scalar = True
                w = w.reshape(1)
            converter_context.insert_weights(output_name, w, was_scalar, [src_op.name], src_op.input)
            return None

        # input is not a static parameter
        input_buf = graph.get_buffer(input_name)
        input_shape = input_buf.shape[:]

        if not len(axes):
            axes = [i for i, s in enumerate(input_shape) if s == 1]

        if not all(x < len(input_shape) for x in axes):
            raise ValueError("Squeeze op {} dims {} greater than input rank {}".format(src_op.name, axes, len(input_shape)))
        if not all((input_shape[x] == 1) for x in axes):
            raise ValueError("input_shape[axes] all point to dims==1 for Squeeze op {}".format(src_op.name))

        output_shape = self._get_squeezed_shape(input_shape, axes)
        # if output tensor is 0d, add it to scalar tensor set
        if not output_shape: # Reshape 0D to 1D
          output_shape = [1]
          converter_context.scalar_tensor.add(str(src_op.output[0]))
        return op_adapter.ReshapeOp(src_op.name, shape=output_shape)

    def extract_input_names(self, src_op, converter_context):
        return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    @staticmethod
    def _get_squeezed_shape(input_shape, axes):
        positive_axes=[]
        for i in axes:
            if i < 0:
                # update axes if negative
                positive_axes.insert(i, len(input_shape) + i)
            else:
                positive_axes.insert(i, i)
        output_shape = [s for i, s in enumerate(input_shape) if i not in positive_axes]
        return output_shape

OnnxTranslations.register_translation(OnnxSqueezeTranslation(), converter_type('Squeeze', 'onnx'))


# ------------------------------------------------------------------------------
#   Tile
# ------------------------------------------------------------------------------
class OnnxTileTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Tile', [1, 6, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = list(map(str, src_op.input))

        input_constant_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False,
                                                   fail_if_dynamic=False)
        static_input_name = str(input_names[0])
        if input_constant_op:
            input_rank = len(input_constant_op.tensor.shape)
        else:
            input_rank = len(graph.get_buffer(src_op.input[0]).shape)

        if len(input_names) == 3:
            # Represents Tile-1
            tiles = converter_context.weights.fetch(src_op.input[1])
            axis = converter_context.weights.fetch(src_op.input[2])
            repeats = [1] * input_rank
            repeats[axis] = tiles
        elif len(input_names) == 2:
            # Represents Tile-6
            repeats = self.fetch_constant_op(src_op.input[1], converter_context).tensor
        else:
            raise ValueError("Only versions {} of {} node {} are supported".format(self.get_supported_version(),
                                                                                   src_op.op_type, src_op.name))

        backend_name = converter_context.backend_info_obj.backend_name() if converter_context.backend_info_obj else None
        if backend_name == "AIC" and input_constant_op:
            self.add_constant_src_op(static_input_name, input_constant_op, converter_context)
        elif input_constant_op:
            output_tensor = np.tile(input_constant_op.tensor, repeats)
            converter_context.insert_weights(src_op.output[0], output_tensor,
                                             src_op_names=[src_op.name], src_tensor_names=src_op.input)
            return None

        return op_adapter.TileOp(src_op.name, multiples=repeats)

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxTileTranslation(),
                                      converter_type('Tile', 'onnx'),
                                      op_adapter.TileOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Transpose
# ------------------------------------------------------------------------------
class OnnxTransposeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Transpose', [1, 13, 21, 23, 24])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        def extract_default_attributes(name, graph, input_name):
            ret = NamedDict()
            if graph.has_buffer(input_name):
                input_shape = graph.get_buffer(input_name).shape
            else:
                const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False, fail_if_not_found=True)
                input_shape = const_op.tensor.shape
            perm = list(range(len(input_shape))) # creates a list like [0, 1,.., n]
            perm.reverse() # default behaviour of Transpose should reverse the permute [n, n-1, .., 0]
            ret[name] = perm
            return ret
        input_name = str(src_op.input[0])
        params = extract_attributes(src_op, schema=self.op_schema(), default_attrs=extract_default_attributes("perm", graph, input_name))
        const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False, fail_if_not_found=True)
        if const_op is not None:
            # static permute of weight parameters
            output_name = str(src_op.output[0])
            w = const_op.tensor
            log_debug1('for Transpose op {}, static input: {} to: {}'.format(src_op.name, input_name, w.shape))
            log_debug1('transpose shape to : {}'.format(params.perm))
            w = np.ascontiguousarray(np.transpose(w, params.perm))
            track_transform(const_op.constant_tensor,
                            ir_graph.TransformType.TRANSFORM_TRANSPOSE,
                            {"perm": params.perm})
            converter_context.insert_weights(output_name, w, src_op_names=[src_op.name], src_tensor_names=src_op.input)
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, w.shape))

            # If there are overrides present for input to the TransposeOp, but no overrides for output, copy them to its output
            enc_in = graph.get_overridden_encoding(input_name)
            enc_out = graph.get_overridden_encoding(output_name)
            if enc_in is not None and enc_out is None:
                if 'version' in graph.user_quantization_overrides and graph.user_quantization_overrides['version'] == '2.0.0':
                    add_new_user_quantization_overrides(graph, output_name, "encodings", enc_in.copy())
                else:
                    add_new_user_quantization_overrides(graph, output_name, "param_encodings", enc_in.copy())

            return None

        log_debug1('input: {} to: {}'.format(input_name,graph.get_buffer(input_name).shape))
        log_debug1('transpose shape to : {}'.format(params.perm))
        return op_adapter.TransposeOp(src_op.name, params.perm)

    def extract_input_names(self, src_op, converter_context):
        return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        # return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxTransposeTranslation(),
                                      converter_type('Transpose', 'onnx'),
                                      op_adapter.TransposeOp.TRANSLATION_KEY)


# -----------------------------------------------------------------------------
#   Unsqueeze
# ------------------------------------------------------------------------------
class OnnxUnsqueezeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Unsqueeze', [1, 11, 13, 21, 23, 24])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        axes = []
        if len(src_op.input) > 1:
            axes_input = str(src_op.input[1])
            # only support constant for second input, if dynamic fetch will fail.
            axes = self.fetch_constant_op(axes_input, converter_context, dtype=np.int32).tensor.tolist()
        elif 'axes' in params:
            axes = params.axes

        if len(set(axes)) != len(axes):
            raise ValueError("Duplicate unsqueeze dims {} for Unsqueeze op {}".format(axes, src_op.name))

        input_name = str(src_op.input[0])

        const_input_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_input_op is not None:
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            w = const_input_op.tensor
            shape = [] if converter_context.weights.was_scalar(input_name) else w.shape
            output_shape = self._get_unsqueezed_shape(shape, axes)
            w = np.reshape(w, output_shape)
            output_name = str(src_op.output[0])
            converter_context.insert_weights(output_name, w, src_op_names=[src_op.name], src_tensor_names=src_op.input)
            if graph.has_quantization_param(const_input_op.name):
                graph.copy_quantization_param(const_input_op.name, output_name, input_name, output_name)
            elif graph.user_quantization_overrides:
                encoding = graph.get_overridden_encoding(const_input_op.name, False)
                if encoding is not None:
                    graph.set_overridden_encoding(output_name, encoding, False)
            return None

        # input is not a static parameter
        input_buf = graph.get_buffer(input_name)
        input_shape = input_buf.shape[:]

        new_rank = len(input_shape) + len(axes)
        if not all(x < new_rank for x in axes):
            raise ValueError("Unsqueeze op {} got dims {} greater than new rank {}".
                             format(src_op.name, axes, new_rank))
        if len(src_op.input) > 1:
            # handle the weights(axes) source because it is generated from framework op/tensor
            converter_context.update_weights_trace_info_for_op(src_op.name, str(src_op.input[1]))
        output_shape = self._get_unsqueezed_shape(input_shape, axes)
        # if input shape is 1d, but is scalar tensor, reduce rank of output_shape by 1
        if input_name in converter_context.scalar_tensor:
            output_shape = output_shape[:-1]
        # Otherwise this is a dynamic unsqueeze so add the unsqueeze/reshape op
        return op_adapter.ReshapeOp(src_op.name, shape=output_shape)

    def extract_input_names(self, src_op, converter_context):
        return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    @staticmethod
    def _get_unsqueezed_shape(org_shape, axes):
        output_rank =  len(org_shape) + len(axes)
        output_shape = [0] * output_rank

        # make axis values non negative
        axes = [a + output_rank if a < 0 else a for a in axes]

        for axis in axes:
            output_shape[axis] = 1

        # include input dims
        org_idx = 0
        for i, dim in enumerate(output_shape):
            if dim == 0:
                output_shape[i] = org_shape[org_idx]
                org_idx += 1

        return output_shape


OnnxTranslations.register_translation(OnnxUnsqueezeTranslation(), converter_type('Unsqueeze', 'onnx'))


# ------------------------------------------------------------------------------
#   Upsample
# ------------------------------------------------------------------------------
class OnnxUpsampleTranslation(OnnxResizeTranslation):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.default_transformation_mode = 'asymmetric'
        self.register_op_schema('Upsample', [1, 7, 9])\
            .register_method(self.validate_attribute_values)


OnnxTranslations.register_translation(OnnxUpsampleTranslation(),
                                      converter_type('Upsample', 'onnx'))


# ------------------------------------------------------------------------------
#   Where
# ------------------------------------------------------------------------------
class OnnxWhereTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Where', [9, 16])
        self.input_names = None
        self.is_expand_added = False

    def add_op(self, src_op, converter_context):
        self.is_expand_added = False
        graph = converter_context.ir_graph
        op = self.extract_parameters(src_op, converter_context)
        if op is None:
            return

        if self.is_expand_added:
            if src_op.input[1] in graph.buffers:
                input_names = [graph.get_buffer(src_op.input[1]).consumers[0].output_names[0]]
            else:
                input_names = [graph.get_buffer(src_op.input[2]).consumers[0].output_names[0]]
        else:
            input_names = self.extract_input_names(src_op, converter_context)

        output_names = self.extract_output_names(src_op, converter_context)

        if op.type == op_adapter.IdentityOp.TRANSLATION_KEY:
            self.add_src_op_info(op.name, src_op, graph)
            node = graph.add(op, input_names, output_names)
            self.insert_default_trace_info(src_op, node, converter_context)
            return node

        for input_name in self.input_names:
            const_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            # Add fetched constant op to graph, if it doesn't exist
            if const_op is not None:
                if not graph.has_buffer(input_name):
                    const_node = graph.add(const_op, [], input_name)
                    self.insert_constant_trace_info(input_name, const_node, converter_context)
                    graph.add_src_op_info(const_op.name, None, const_node.output_names[0])

        # Add elementwise src op info
        self.add_src_op_info(op.name, src_op, graph)
        last_node = graph.add(op, self.input_names, output_names)
        self.insert_default_trace_info(src_op, last_node, converter_context)
        return last_node

    def add_expand(self, graph, input_name, op_name, output_shape, converter_context):
        expand_op = op_adapter.ExpandOp(input_name + "expand", shape = output_shape)
        expand_node = graph.add(expand_op, [input_name], [op_name + "_expand_output"])
        self.is_expand_added = True
        self.insert_constant_trace_info(expand_op.name, expand_node, converter_context)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.input_names = list(map(str, src_op.input))

        condition_op = self.fetch_constant_op(self.input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        branch1_op = self.fetch_constant_op(self.input_names[1], converter_context, prunable=False, fail_if_dynamic=False)
        branch2_op = self.fetch_constant_op(self.input_names[2], converter_context, prunable=False, fail_if_dynamic=False)

        if condition_op:
            if branch1_op and branch2_op:
                data = np.where(condition_op.tensor, branch1_op.tensor, branch2_op.tensor)
                was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in self.input_names])
                converter_context.insert_weights(str(src_op.output[0]), data, was_scalar, [src_op.name], self.input_names)
                return None

            condition_tensor = condition_op.tensor.flatten()
            # Check Identity cases: Either all True yielding a pass-through of input1 or all False
            # yielding a pass-through of input2
            if all(condition for condition in condition_tensor):
                if not graph.has_buffer(self.input_names[1]):
                    branch1_node = graph.add(branch1_op, [], [self.input_names[1]])
                    self.insert_constant_trace_info(branch1_op.name, branch1_node, converter_context)

                input_shapes = []
                input_shapes.append(list(graph.get_buffer(src_op.input[1]).shape))

                if branch2_op is not None:
                    input_shapes.append(list(branch2_op.shape))
                else:
                    input_shapes.append(list(graph.buffers[self.input_names[2]].shape))

                input_shapes.append(list(condition_op.shape))

                output_shape = get_broadcasted_shape(input_shapes, axis_order=AxisOrders.ONNX, align_channels=False)

                if list(graph.get_buffer(src_op.input[1]).shape) != output_shape:
                    self.add_expand(graph, src_op.input[1], src_op.name , output_shape, converter_context)

                self.input_names = [self.input_names[1]]
                return op_adapter.IdentityOp(src_op.name)
            elif all(not condition for condition in condition_tensor):
                if not graph.has_buffer(self.input_names[2]):
                    branch2_node = graph.add(branch2_op, [], [self.input_names[2]])
                    self.insert_constant_trace_info(branch2_op.name, branch2_node, converter_context)

                input_shapes = []
                input_shapes.append(list(graph.get_buffer(src_op.input[2]).shape))
                if branch1_op is not None:
                    input_shapes.append(list(branch1_op.shape))
                else:
                    input_shapes.append(list(graph.buffers[self.input_names[1]].shape))

                input_shapes.append(list(condition_op.shape))
                output_shape = get_broadcasted_shape(input_shapes, axis_order=AxisOrders.ONNX, align_channels=False)

                if list(graph.get_buffer(src_op.input[2]).shape) != output_shape:
                    self.add_expand(graph, src_op.input[2], src_op.name , output_shape, converter_context)

                self.input_names = [self.input_names[2]]
                return op_adapter.IdentityOp(src_op.name)

        return op_adapter.ElementwiseTernaryOp(name=src_op.name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SELECT)

    def extract_input_names(self, src_op, converter_context):
        return self.input_names

OnnxTranslations.register_translation(OnnxWhereTranslation(),
                                      converter_type('Where', 'onnx'))