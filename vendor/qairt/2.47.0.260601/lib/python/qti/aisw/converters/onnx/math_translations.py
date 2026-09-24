# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


from .onnx_translations import *

import distutils
from distutils import version
import packaging
from enum import Enum
import numpy as np

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.converter_ir.op_graph import TraceType
from qti.aisw.converters.common.converter_ir.op_graph_utils import apply_scaling_to_constant
from qti.aisw.converters.common.converter_ir.op_graph_optimizations import IROptimizations
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker

# ------------------------------------------------------------------------------
#   Abs
# ------------------------------------------------------------------------------
class OnnxAbsTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Abs', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS
        self.numpy_op = np.abs


OnnxTranslations.register_translation(OnnxAbsTranslation(),
                                      converter_type('Abs', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS])


# ------------------------------------------------------------------------------
#   Add
# ------------------------------------------------------------------------------
class OnnxAddTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Add', [1, 6, 7, 13, 14])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD
        self.numpy_op = np.add


OnnxTranslations.register_translation(OnnxAddTranslation(),
                                      converter_type('Add', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD])


# ------------------------------------------------------------------------------
#   And
# ------------------------------------------------------------------------------
class OnnxAndTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('And', [1, 7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND
        self.numpy_op = np.logical_and


OnnxTranslations.register_translation(OnnxAndTranslation(),
                                      converter_type('And', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND])


# ------------------------------------------------------------------------------
#   ArgMax, ArgMin
# ------------------------------------------------------------------------------
class OnnxArgOpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ArgMax', [1, 11, 12, 13])
        self.register_op_schema('ArgMin', [1, 11, 12, 13])

    def extract_parameters(self, src_op, converter_context):
        schema = self.op_schema(op_type=src_op.op_type)
        # these parameters depends on src_op.op_type(ArgMax/ArgMin)
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type), validate=True)

        if "select_last_index" in params and params.select_last_index == 1:
            log_warning("select_last_index is set to 1 for the op - {}. Only 0(default value) is supported. "
                        "This may cause accuracy issues.".format(src_op.name))

        if str(src_op.op_type) == 'ArgMax':
            arg_type = ir_graph.QNN_OP_ARGMAX
        elif str(src_op.op_type) == 'ArgMin':
            arg_type = ir_graph.QNN_OP_ARGMIN

        return op_adapter.ArgOp(str(src_op.name),
                                arg_type = arg_type,
                                axis=params.axis,
                                keep_dims=params.keepdims)


OnnxTranslations.register_translation(OnnxArgOpTranslation(),
                                      converter_type('ArgMax', 'onnx'),
                                      converter_type('ArgMin', 'onnx'),
                                      op_adapter.ArgOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Asin
# ------------------------------------------------------------------------------
class OnnxAsinTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Asin', [7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN
        self.numpy_op = np.arcsin


OnnxTranslations.register_translation(OnnxAsinTranslation(),
                                      converter_type('Asin', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN])


# ------------------------------------------------------------------------------
#   Atan
# ------------------------------------------------------------------------------
class OnnxAtanTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Atan', [7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN
        self.numpy_op = np.arctan


OnnxTranslations.register_translation(OnnxAtanTranslation(),
                                      converter_type('Atan', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN])


# ------------------------------------------------------------------------------
#   Ceil
# ------------------------------------------------------------------------------
class OnnxCeilTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Ceil', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL
        self.numpy_op = np.ceil


OnnxTranslations.register_translation(OnnxCeilTranslation(),
                                      converter_type('Ceil', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL])


# ------------------------------------------------------------------------------
#   Cos
# ------------------------------------------------------------------------------
class OnnxCosTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Cos', [7, 22])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS
        self.numpy_op = np.cos


OnnxTranslations.register_translation(OnnxCosTranslation(),
                                      converter_type('Cos', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS])


# ------------------------------------------------------------------------------
#   CumSum
# ------------------------------------------------------------------------------
class OnnxCumSumTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('CumSum', [11, 14])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(src_op.input)
        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_rank = input_buf.rank()

        # extract axis param
        const_op = self.fetch_constant_op(input_names[1], converter_context, fail_if_dynamic=True)
        axis = const_op.tensor.astype(np.int32).item(0)

        # handle the weights(axis) source because it is generated from framework op/tensor
        converter_context.update_weights_trace_info_for_op(src_op.name, input_names[1], src_op_names=[], src_tensor_names=[input_names[1]])

        # check for axis to be in [-r,r-1]
        if axis not in range(-input_rank, input_rank):
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} CumSum op {}"
                             .format(axis, input_names[1], src_op.op_type, src_op.name))
        if axis < 0:
            axis += input_rank

        # extract reverse and exclusive params
        reverse = params.reverse if 'reverse' in params else 0
        if reverse not in (0, 1):
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} CumSum op {}"
                             .format(reverse, "reverse", src_op.op_type, src_op.name))
        exclusive = params.exclusive if 'exclusive' in params else 0
        if exclusive not in (0, 1):
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} CumSum op {}"
                             .format(exclusive, "exclusive", src_op.op_type, src_op.name))

        # axis received as input, but added as param in our IR graph
        return op_adapter.CumSumOp(str(src_op.name),
                                   axis=axis,
                                   reverse=bool(reverse),
                                   exclusive=bool(exclusive))

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]

OnnxTranslations.register_translation(OnnxCumSumTranslation(),
                                      converter_type('CumSum', 'onnx'),
                                      op_adapter.CumSumOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Div
# ------------------------------------------------------------------------------
class OnnxDivTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Div', [1, 6, 7, 13, 14])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE
        self.numpy_op = np.divide


OnnxTranslations.register_translation(OnnxDivTranslation(),
                                      converter_type('Div', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE])


# ------------------------------------------------------------------------------
#   Einsum
# ------------------------------------------------------------------------------
class OnnxEinsumTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Einsum', [12])
        self.transform_pattern = ""
        self.perm=[]

    class Pattern(Enum):
        MATMUL = 1
        MATMUL_TRANSPOSE_a = 2 # matmul(transpose(a), b).
        MATMUL_TRANSPOSE_b = 3 # matmul(a, transpose(b)).
        TRANSPOSE = 4

    def post_process_equation(self, src_op_name, equation):
        operand_str = []
        # check equation is explicit or implicit
        # only can support explicit equation, will enable implicit next step.
        if '->' not in equation:
            raise ValueError('Converter can not support implicit equation for Einsum op {}. Got {}.'
                             .format(src_op_name, equation))
        # explicit equation
        else:
            lhs_equ, rhs_equ = equation.split('->')
            # TODO: limitations will be enabled next step.
            # can not support "..." in equation.
            # can not support inputs more then 2 tensors.
            # can not support outputs more then 1 tensor.
            if lhs_equ.count("...") >= 1 or rhs_equ.count("...") >= 1:
                raise ValueError('Converter can not support "..." broadcast in equation for Einsum op {}. Got {}.'
                                 .format(src_op_name, equation))
            elif len(lhs_equ.split(",")) > 2:
                raise ValueError('Converter can not support more then two inputs for Einsum op {}. Got {}.'
                                 .format(src_op_name, equation))
            elif len(rhs_equ.split(",")) > 1:
                raise ValueError('Converter can not support more then one output for Einsum op {}. Got {}.'
                                 .format(src_op_name, equation))

        # Einsum may have two or only one input.
        # when there are two inputs just split lhs_equ with ',' otherwise keep lhs_equ as the only one input.
        # invalid check and produce operand_str.
        # numpy reference https://np.org/doc/stable/reference/generated/np.einsum.html
        for lhs_item in lhs_equ.split(","):
            lhs_item = lhs_item.strip()
            if not lhs_item.isalpha():
                raise ValueError("equation {} has invalid symbol that is not alphabet for Einsum op {}."
                                 .format(equation, src_op_name))
            operand_str.append(lhs_item)
        operand_str.append(rhs_equ)

        # operand_str include inputs and output in equation
        return operand_str

    def extract_transform_pattern(self, src_op_name, operand_str, origin_equation):
        # scenario : two inputs, one output
        if len(operand_str) == 3:
            lhs_equ_in_first = operand_str[0]
            lhs_equ_in_second = operand_str[1]
            rhs_equ = operand_str[2]

            # extract transform pattern
            # pattern: matmul(a,b)
            # eg: "hbwij, hbwjc->hbwic" to "matmul"
            if lhs_equ_in_first[-1] == lhs_equ_in_second[-2] \
                and rhs_equ[-1] == lhs_equ_in_second[-1] \
                and rhs_equ[-2] == lhs_equ_in_first[-2]:
                    self.transform_pattern = self.Pattern.MATMUL
            # pattern: matmul(transpose(a),b)
            # eg: "hbwpc,hbwpq->hbwcq" to "transopse(a) + matmul"
            elif lhs_equ_in_first[-2] == lhs_equ_in_second[-2] \
                and rhs_equ[-1] == lhs_equ_in_second[-1] \
                and rhs_equ[-2] == lhs_equ_in_first[-1]:
                    self.transform_pattern = self.Pattern.MATMUL_TRANSPOSE_a
            # pattern: matmul(a,transpose(b))
            # eg: "hbwpc,hbwqc->hbwpq" to "transopse(b) + matmul"
            elif lhs_equ_in_first[-1] == lhs_equ_in_second[-1] \
                and rhs_equ[-1] == lhs_equ_in_second[-2] \
                and rhs_equ[-2] == lhs_equ_in_first[-2]:
                    self.transform_pattern = self.Pattern.MATMUL_TRANSPOSE_b
            else:
                # TODO: More patterns will be supported next step.
                raise ValueError('Converter can not support equation {} for Einsum op {}.'.
                                 format(origin_equation, src_op_name))
        # scenario : one input, one output
        elif len(operand_str) == 2:
            lhs_equ_in_first = operand_str[0]
            rhs_equ = operand_str[1]
            # extract transform pattern
            # pattern: transpose(a)
            # eg: "mnkj->mknj" to "transpose" with perm (0,2,1,3)
            if len(rhs_equ) == len(lhs_equ_in_first) \
                and set(rhs_equ) == set(lhs_equ_in_first):
                    # calculate perm parameter and set transform pattern.
                    self.perm = [lhs_equ_in_first.index(s) for s in rhs_equ]
                    self.transform_pattern = self.Pattern.TRANSPOSE
            else:
                # TODO: More patterns will be supported next step.
                raise ValueError('Converter can not support equation {} for Einsum op {}.'
                                 .format(origin_equation, src_op_name))
        else:
            raise ValueError('Got invalid equation {} for Einsum op {}'
                             .format(origin_equation, src_op_name))

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = list(map(str, src_op.input))
        output_names = list(map(str, src_op.output))
        params = extract_attributes(src_op, schema=self.op_schema())

        # remove redundant spaces in equation.
        # eg: " hbwpc,  hbwqc ->  hbwpq " to "hbwpc,hbwqc->hbwpq"
        origin_equation = params.equation.replace(" ", "")

        # check inputs number match or not between operand str and Einsum Op.
        operand_str = self.post_process_equation(src_op.name, origin_equation)
        equ_input_num = len(operand_str) - 1
        op_input_num = len(input_names)
        if equ_input_num != op_input_num:
            raise ValueError("inputs number {} in equation not match inputs number {} of Einsum Op {}." \
                .format(equ_input_num, op_input_num, src_op.name))

        # extract transform pattern with inputs and output string in the valid equation.
        self.extract_transform_pattern(src_op.name, operand_str, origin_equation)

        # Transform Einsum to other Ops according to equation.
        # pattern 1: Transform Einsum to Matmul.
        if self.transform_pattern == self.Pattern.MATMUL:
            # If input A is static, add a Constant Op
            first_input_name = input_names[0]
            if not graph.has_buffer(first_input_name):
                input_constant_op = self.fetch_constant_op(first_input_name, converter_context, prunable=False, fail_if_dynamic=False)
                if input_constant_op:
                    input_constant_node = graph.add(input_constant_op, [], [first_input_name])
                    self.insert_constant_trace_info(first_input_name, input_constant_node, converter_context)
                    graph.add_src_op_info(input_constant_node.op.name, [], [first_input_name])

            # If input B is constant, add a Constant Op
            second_input_name = input_names[1]
            if not graph.has_buffer(second_input_name):
                weights_constant_op = self.fetch_constant_op(second_input_name, converter_context, prunable=False, fail_if_dynamic=False)
                if weights_constant_op:
                    weight_node = graph.add(weights_constant_op, [], [second_input_name])
                    self.insert_constant_trace_info(second_input_name, weight_node, converter_context)
                    graph.add_src_op_info(weight_node.op.name, [], [second_input_name])

            return op_adapter.MatMulOp(name=str(src_op.name), transpose_in0=False, transpose_in1=False)

        # pattern 2: Transform Einsum to Matmul(transpose(in0), in1)
        elif self.transform_pattern == self.Pattern.MATMUL_TRANSPOSE_a:
            matmul_op = op_adapter.MatMulOp(src_op.name, transpose_in0=True, transpose_in1=False)
            graph.add_src_op_info(src_op.name, input_names, output_names)
            node = graph.add(matmul_op, input_names, output_names)
            self.insert_default_trace_info(src_op, node, converter_context)

            return None

        # pattern 3: Transform Einsum to Matmul(in0, transpose(in1))
        elif self.transform_pattern == self.Pattern.MATMUL_TRANSPOSE_b:
            matmul_op = op_adapter.MatMulOp(src_op.name, transpose_in0=False, transpose_in1=True)
            graph.add_src_op_info(src_op.name, input_names, output_names)
            node = graph.add(matmul_op, input_names, output_names)
            self.insert_default_trace_info(src_op, node, converter_context)

            return None

        # pattern 4: Transform Einsum to Transpose.
        elif self.transform_pattern == self.Pattern.TRANSPOSE:
            const_op = self.fetch_constant_op(input_names[0], converter_context, fail_if_dynamic=False, fail_if_not_found=True)
            if const_op is not None:
                # static permute of weight parameters
                output_name = str(src_op.output[0])
                w = const_op.tensor
                log_debug1('Einsum op {}, static input: {} to: {}'.format(src_op.name, input_names[0], w.shape))
                log_debug1('transpose shape to : {}'.format(params.perm))
                w = np.transpose(w, params.perm)
                converter_context.insert_weights(output_name, w, src_op_names=[src_op.name], src_tensor_names=src_op.input)
                log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_names[0], output_name, w.shape))

                return None

            log_debug1('Einsum op {}, input: {} to: {}'.format(src_op.name, input_names[0], graph.get_buffer(input_names[0]).shape))
            log_debug1('transpose shape to : {}'.format(params.perm))

            return op_adapter.TransposeOp(src_op.name, params.perm)

        else:
            # TODO: More patterns will be supported next step.
            raise ValueError('No pattern matched, converter can not support equation {} of Einsum Op {}.'
                             .format(origin_equation, src_op.name))


OnnxTranslations.register_translation(OnnxEinsumTranslation(),
                                      converter_type('Einsum', 'onnx'))


# ------------------------------------------------------------------------------
#   Elu
# ------------------------------------------------------------------------------
class OnnxEluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Elu', [1, 6, 22])

    def extract_parameters(self, src_op, converter_context):
        # these parameters belong to Elu
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                              operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU,
                                              alpha=params.alpha)


OnnxTranslations.register_translation(OnnxEluTranslation(),
                                      converter_type('Elu', 'onnx'))


# ------------------------------------------------------------------------------
#   Equal
# ------------------------------------------------------------------------------
class OnnxEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Equal', [1, 7, 11, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL
        self.numpy_op = np.equal


OnnxTranslations.register_translation(OnnxEqualTranslation(),
                                      converter_type('Equal', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL])


# ------------------------------------------------------------------------------
#   Erf
# ------------------------------------------------------------------------------
class OnnxErfTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Erf', [9, 13])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ErfOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxErfTranslation(),
                                      converter_type('Erf', 'onnx'),
                                      op_adapter.ErfOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Exp
# ------------------------------------------------------------------------------
class OnnxExpTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Exp', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP
        self.numpy_op = np.exp


OnnxTranslations.register_translation(OnnxExpTranslation(),
                                      converter_type('Exp', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP])


# ------------------------------------------------------------------------------
#   Floor
# ------------------------------------------------------------------------------
class OnnxFloorTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Floor', [1, 6])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR
        self.numpy_op = np.floor


OnnxTranslations.register_translation(OnnxFloorTranslation(),
                                      converter_type('Floor', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR])


# ------------------------------------------------------------------------------
#   GEMM
# ------------------------------------------------------------------------------
class OnnxGemmTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Gemm', [1, 6, 7, 9, 11, 13])
        self.params = None

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph

        # Use MatMul for LPAI, otherwise use FC
        backend_name = converter_context.backend_info_obj.backend_name() if converter_context.backend_info_obj else None
        if backend_name == "LPAI":
            op = self.extract_parameters_matmul(src_op, converter_context)
        else:
            op = self.extract_parameters(src_op, converter_context)

        if op is None:
            return

        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        if len(input_names) >= 3:
            # Matmul/FC bias needs to be rank 1 for QNN, however in ONNX, it can be rank 2
            if graph.has_buffer(self.bias_name) and graph.get_buffer(self.bias_name).rank() == 2:
                fc_output_name = self.input_name + self.weights_name + '_fc'
                fc_node = graph.add(op, [self.input_name, self.weights_name], [fc_output_name])
                self.insert_trace_info([fc_node, graph.get_buffer(fc_output_name)], (src_op.name, TraceType.OP), converter_context)
                graph.add_src_op_info(fc_node.op.name, [self.input_name, self.weights_name], [fc_output_name])
                eleadd_op = op_adapter.ElementwiseBinaryOp(str(fc_output_name + self.bias_name + '_add'), operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
                eleadd_node = graph.add(eleadd_op, [fc_output_name , self.bias_name], output_names)
                self.insert_trace_info([eleadd_node, graph.get_buffer(output_names[0])], (src_op.name, TraceType.OP), converter_context)
                graph.add_src_op_info(eleadd_node.op.name, [fc_output_name , self.bias_name], output_names)
                return eleadd_node

        self.add_src_op_info(op.name, src_op, graph)
        node = graph.add(op, input_names, output_names)
        self.insert_default_trace_info(src_op, node, converter_context)
        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        log_warning("GEMM operation is not supported in the general case, attempting to interpret as FC")
        self.params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        input_names = list(map(str, src_op.input))

        if not graph.has_buffer(input_names[0]) and converter_context.weights.has(input_names[0]):
            input_constant_op = self.fetch_constant_op(input_names[0], converter_context,
                                                                     fail_if_dynamic=False)
            if input_constant_op:
                input_constant_node = graph.add(input_constant_op, [], [input_names[0]])
                self.insert_constant_trace_info(input_names[0], input_constant_node, converter_context)
                graph.add_src_op_info(input_constant_node.op.name, [], [input_names[0]])

        # In newer opset versions, bias is made an optional parameter
        # in the Gemm operator. Default value of bias in this case is 0
        # Get weights
        weights_constant_op = None  # Initialize to None
        if not graph.has_buffer(input_names[1]):
            weights_constant_op = self.fetch_constant_op(input_names[1], converter_context, fail_if_dynamic=False)
            # If the weight is a constant, add the constant op to graph
            if weights_constant_op:
                weights_constant_node = graph.add(weights_constant_op, [], [input_names[1]])
                self.insert_constant_trace_info(input_names[1], weights_constant_node, converter_context)
                graph.add_src_op_info(weights_constant_node.op.name, [], [input_names[1]])
            # If the weight is neither in weight nor in graph, raise an error
            else:
                raise ValueError("Gemm Op {}'s weight is neither in converter_context.weights nor in graph.buffer"
                                 .format(str(src_op.name)))

        # Add the multiplied op to graph for both dynamic and constant weights
        # Transpose input if transB is given, Weights = weights'
        if self.params.transB:
            trans_weight_name = input_names[1] + '_permute'
            if not graph.has_buffer(trans_weight_name):
                permute_op = op_adapter.TransposeOp(trans_weight_name, perm=[1, 0])
                permute_node = graph.add(permute_op, [input_names[1]], [trans_weight_name])
                permute_enc = graph.get_overridden_encoding(trans_weight_name)
                if permute_enc is not None:
                    if 'version' in graph.user_quantization_overrides and graph.user_quantization_overrides['version'] == '2.0.0':
                        add_new_user_quantization_overrides(graph, input_names[1], "encodings", permute_enc.copy())
                    else:
                        add_new_user_quantization_overrides(graph, input_names[1], "activation_encodings", permute_enc.copy())
                self.insert_trace_info([permute_node, graph.get_buffer(trans_weight_name)], (src_op.name, TraceType.OP), converter_context)
                graph.add_src_op_info(permute_op.name, [input_names[1]], [trans_weight_name])
        else:
            trans_weight_name = input_names[1]

        # Transpose input if transA is given
        if self.params.transA:
            trans_input_name = input_names[0] + '_permute'
            if not graph.has_buffer(trans_input_name):
                permute_op = op_adapter.TransposeOp(trans_input_name, perm=[1, 0])
                permute_node = graph.add(permute_op, [input_names[0]], [trans_input_name])
                permute_enc = graph.get_overridden_encoding(trans_input_name)
                if permute_enc is not None:
                    if 'version' in graph.user_quantization_overrides and graph.user_quantization_overrides['version'] == '2.0.0':
                        add_new_user_quantization_overrides(graph, input_names[0], "encodings", permute_enc.copy())
                    else:
                        add_new_user_quantization_overrides(graph, input_names[0], "activation_encodings", permute_enc.copy())
                self.insert_trace_info([permute_node, graph.get_buffer(trans_input_name)], (src_op.name, TraceType.OP), converter_context)
                graph.add_src_op_info(permute_op.name, [input_names[0]], [trans_input_name])
            self.input_name = trans_input_name
        else:
            self.input_name = input_names[0]

        # Weights = alpha * weights
        if self.params.alpha != 1:
            alpha = self.params.alpha
            if weights_constant_op:
                apply_scaling_to_constant(graph, weights_constant_op, alpha)
                self.weights_name = trans_weight_name
            else:
                # Use input[0]'s name in case the alpha is different
                alpha_constant_name = input_names[0] + "_alpha"
                if not graph.has_buffer(alpha_constant_name):
                    # If alpha is a scalar, reshape it into a 1D array.
                    if np.isscalar(alpha):
                        alpha = np.asarray([alpha], dtype = np.dtype('float32'))
                    alpha_constant_op = op_adapter.ConstantOp(alpha_constant_name, tensor=alpha)
                    alpha_node = graph.add(alpha_constant_op, [], [alpha_constant_name])
                    self.insert_trace_info([alpha_node, graph.get_buffer(alpha_constant_name)], (src_op.name, TraceType.OP), converter_context)
                    graph.add_src_op_info(alpha_constant_name, None, [alpha_constant_name])
                alpha_mul_name = alpha_constant_name + "_multiply"
                if not graph.has_buffer(alpha_mul_name):
                    alpha_mul_op = op_adapter.ElementwiseBinaryOp(alpha_mul_name,
                                                                operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
                    alpha_mul_node = graph.add(alpha_mul_op, [trans_weight_name, alpha_constant_name], [alpha_mul_name])
                    self.insert_trace_info([alpha_mul_node, graph.get_buffer(alpha_mul_name)], (src_op.name, TraceType.OP), converter_context)
                    graph.add_src_op_info(alpha_mul_name, [trans_weight_name, alpha_constant_name], [alpha_mul_name])
                self.weights_name = alpha_mul_name
        else:
            self.weights_name = trans_weight_name

        # Get bias
        if len(src_op.input) >= 3:
            bias_constant_op = None  # Initialize to None
            # If the bias is a constant, add the constant op to graph
            if not graph.has_buffer(input_names[2]):
                if converter_context.weights.has(input_names[2]):
                    # squeezing the bias to remove all single dimensional entries
                    bias = np.atleast_1d(np.squeeze(converter_context.weights.fetch(input_names[2])))
                    if bias.ndim == 1 and graph.get_buffer(trans_weight_name).shape[1] != len(bias):
                        bias = np.resize(bias,(graph.get_buffer(trans_weight_name).shape[1],))
                    bias_constant_op = op_adapter.ConstantOp(input_names[2], tensor=bias)
                    bias_node = graph.add(bias_constant_op, [], [input_names[2]])
                    self.insert_constant_trace_info(input_names[2], bias_node, converter_context)
                    graph.add_src_op_info(input_names[2], None, input_names[2])
                else:
                    raise ValueError("Gemm Op {}'s bias is neither in converter_context.weights nor in graph.buffer"
                                     .format(str(src_op.name)))
            # Add the multiplied op to graph for both dynamic and constant bias
            # Bias = beta * bias
            if self.params.beta != 1:
                beta = self.params.beta
                if bias_constant_op:
                    apply_scaling_to_constant(graph, bias_constant_op, beta)
                    self.bias_name = input_names[2]
                else:
                    # Use input[0]'s name in case the beta is different
                    beta_constant_name = input_names[0] + "_beta"
                    if not graph.has_buffer(beta_constant_name):
                        # If beta is a scalar, reshape it into a 1D array.
                        if np.isscalar(beta):
                            beta = np.asarray([beta], dtype = np.dtype('float32'))
                        beta_constant_op = op_adapter.ConstantOp(beta_constant_name, tensor=beta)
                        beta_node = graph.add(beta_constant_op, [], [beta_constant_name])
                        self.insert_trace_info([beta_node, graph.get_buffer(beta_constant_name)], (src_op.name, TraceType.OP), converter_context)
                    beta_mul_name = beta_constant_name + "_multiply"
                    if not graph.has_buffer(beta_mul_name):
                        beta_mul_op = op_adapter.ElementwiseBinaryOp(beta_mul_name,
                                                                    operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
                        beta_mul_node = graph.add(beta_mul_op, [input_names[2], beta_constant_name], [beta_mul_name])
                        self.insert_trace_info([beta_mul_node, graph.get_buffer(beta_mul_name)], (src_op.name, TraceType.OP), converter_context)
                        graph.add_src_op_info(beta_mul_name, [input_names[2], beta_constant_name], [beta_mul_name])
                    self.bias_name = beta_mul_name
            else:
                self.bias_name = input_names[2]

            return op_adapter.FullyConnectedOp(str(src_op.name),
                                               bias_op_name=self.bias_name)
        else:
            return op_adapter.FullyConnectedOp(str(src_op.name))

    def extract_parameters_matmul(self, src_op, converter_context):
        graph = converter_context.ir_graph
        log_warning(code_to_message.get_warning_message("WARNING_GEMM"))
        self.params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        self.input_names = list(map(str, src_op.input))

        # In newer opset versions, bias is made an optional parameter
        # in the Gemm operator. Default value of bias in this case is 0

        # If input A is static, add a Constant Op
        first_input_name = self.input_names[0]
        if not graph.has_buffer(first_input_name):
            static_input_op = self.fetch_constant_op(first_input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if static_input_op:
                static_input_node = graph.add(static_input_op, [], [first_input_name])
                self.insert_constant_trace_info(first_input_name, static_input_node, converter_context)
                graph.add_src_op_info(static_input_node.op.name, [], [first_input_name])

        first_input = graph.buffers[self.input_names[0]]
        if first_input.rank() > 2:
            # Add the Reshape Op in context of fully connected op
            # Common usecase: AveragePool [B,2048,1,1] -> Reshape [B,2048]
            # This issue captured here: https://github.com/microsoft/onnxruntime/issues/2045#issuecomment-539794539
            reshape_name = str(src_op.name) + "_reshape"
            reshape_output_name = first_input.name + ".reshape"
            new_input_shape = [first_input.shape[0], np.prod(first_input.shape[1:])]
            reshape_op = op_adapter.ReshapeOp(name=reshape_name, shape=new_input_shape)
            reshape_node = graph.add(reshape_op, [first_input.name], [reshape_output_name])
            self.insert_trace_info([reshape_node, graph.get_buffer(reshape_output_name)], (self.input_names[0], TraceType.TENSOR), converter_context)
            graph.add_src_op_info(reshape_node.op.name, [first_input.name], [reshape_output_name])
            self.input_name = reshape_output_name
        else:
            self.input_name = first_input_name

        # If input B is constant, add a Constant Op
        second_input_name = str(self.input_names[1])
        weights_constant_op = None  # Initialize to None
        if not graph.has_buffer(second_input_name):
            weights_constant_op = self.fetch_constant_op(second_input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if weights_constant_op:
                weight_node = graph.add(weights_constant_op, [], [second_input_name])
                self.insert_constant_trace_info(second_input_name, weight_node, converter_context)
                graph.add_src_op_info(weight_node.op.name, [], [second_input_name])

        second_input = graph.buffers[self.input_names[1]]
        log_assert(len(second_input.shape) == 2,
                   "Invalid shape for the second input to Gemm Op {}. Expected shape len is 2"
                   "but got {}".format(src_op.name, len(second_input.shape)))

        # weights = alpha * weights
        if self.params.alpha != 1:
            alpha = self.params.alpha
            if weights_constant_op:
                apply_scaling_to_constant(graph, weights_constant_op, alpha)
                self.weights_name = second_input_name
            else:
                alpha_constant_name = self.input_names[0] + "_alpha"
                if not graph.has_buffer(alpha_constant_name):
                    alpha_constant_op = op_adapter.ConstantOp(alpha_constant_name, tensor=alpha)
                    alpha_node = graph.add(alpha_constant_op, [], [alpha_constant_name])
                    self.insert_trace_info([alpha_node, graph.get_buffer(alpha_constant_name)], (src_op.name, TraceType.OP), converter_context)
                    graph.add_src_op_info(alpha_constant_name, None, [alpha_constant_name])

                alpha_mul_name = alpha_constant_name + "_multiply"
                if not graph.has_buffer(alpha_mul_name):
                    alpha_mul_op = op_adapter.ElementwiseBinaryOp(alpha_mul_name,
                                                                operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
                    alpha_mul_node = graph.add(alpha_mul_op, [second_input_name, alpha_constant_name], [alpha_mul_name])
                    self.insert_trace_info([alpha_mul_node, graph.get_buffer(alpha_mul_name)], (src_op.name, TraceType.OP), converter_context)
                    graph.add_src_op_info(alpha_mul_name, [second_input_name, alpha_constant_name], [alpha_mul_name])
                self.weights_name = alpha_mul_name
        else:
            self.weights_name = second_input_name

        # Get bias
        if len(src_op.input) >= 3:
            bias_constant_op = None  # Initialize to None
            bias_input_name = self.input_names[2]
            # If the bias is a constant, add the constant op to graph
            if not graph.has_buffer(bias_input_name):
                if converter_context.weights.has(bias_input_name):
                    # squeezing the bias to remove all single dimensional entries
                    bias = np.atleast_1d(np.squeeze(converter_context.weights.fetch(bias_input_name)))
                    bias_constant_op = op_adapter.ConstantOp(bias_input_name, tensor=bias)
                    bias_node = graph.add(bias_constant_op, [], [bias_input_name])
                    self.insert_constant_trace_info(bias_input_name, bias_node, converter_context)
                    graph.add_src_op_info(bias_input_name, None, bias_input_name)
                else:
                    raise ValueError("Gemm op {}'s bias is neither in converter_context.weights nor in graph.buffer".
                                     format(str(src_op.name)))

            # Add the multiplied op to graph for both dynamic and constant bias
            # Bias = beta * bias
            if self.params.beta != 1:
                beta = self.params.beta
                if bias_constant_op:
                    apply_scaling_to_constant(graph, bias_constant_op, beta)
                    self.bias_name = bias_input_name
                else:
                    # Use input[0]'s name in case the beta is different
                    beta_constant_name = self.input_names[0] + "_beta"
                    if not graph.has_buffer(beta_constant_name):
                        beta_constant_op = op_adapter.ConstantOp(beta_constant_name, tensor=beta)
                        beta_node = graph.add(beta_constant_op, [], [beta_constant_name])
                        self.insert_trace_info([beta_node, graph.get_buffer(beta_constant_name)], (src_op.name, TraceType.OP), converter_context)
                    beta_mul_name = beta_constant_name + "_multiply"
                    if not graph.has_buffer(beta_mul_name):
                        beta_mul_op = op_adapter.ElementwiseBinaryOp(beta_mul_name,
                                                                    operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
                        beta_mul_node = graph.add(beta_mul_op, [bias_input_name, beta_constant_name], [beta_mul_name])
                        self.insert_trace_info([beta_mul_node, graph.get_buffer(beta_mul_name)], (src_op.name, TraceType.OP), converter_context)
                        graph.add_src_op_info(beta_mul_name, [bias_input_name, beta_constant_name], [beta_mul_name])
                    self.bias_name = beta_mul_name
            else:
                self.bias_name = bias_input_name


        return op_adapter.MatMulOp(str(src_op.name),
                                transpose_in0=self.params.transA,
                                transpose_in1=self.params.transB)

    def extract_input_names(self, src_op, converter_context):
        if len(src_op.input) >= 3:
            if(self.params.transA):
                return [str(src_op.input[0]) + '_permute', self.weights_name, self.bias_name]
            else:
                return [str(src_op.input[0]), self.weights_name, self.bias_name]
        else:
            if(self.params.transA):
                return [str(src_op.input[0]) + '_permute', self.weights_name]
            else:
            	return [str(src_op.input[0]), self.weights_name]


OnnxTranslations.register_translation(OnnxGemmTranslation(), converter_type('Gemm', 'onnx'))


# ------------------------------------------------------------------------------
#   Greater
# ------------------------------------------------------------------------------
class OnnxGreaterTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Greater', [1, 7, 9, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER
        self.numpy_op = np.greater


OnnxTranslations.register_translation(OnnxGreaterTranslation(),
                                      converter_type('Greater', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER])


# ------------------------------------------------------------------------------
#   GreaterOrEqual
# ------------------------------------------------------------------------------
class OnnxGreaterOrEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GreaterOrEqual', [12])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL
        self.numpy_op = np.greater_equal


# GreaterOrEqual is announced in ONNX 1.7.0, add if statement to avoid warning
if packaging.version.Version(onnx.__version__) >= packaging.version.Version("1.7.0"):
    OnnxTranslations.register_translation(OnnxGreaterOrEqualTranslation(),
                                          converter_type('GreaterOrEqual', 'onnx'),
                                          op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL])


# ------------------------------------------------------------------------------
#   HardSigmoid
# ------------------------------------------------------------------------------
class OnnxHardSigmoidTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('HardSigmoid', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                   operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID,
                                   alpha=params.alpha,
                                   beta=params.beta)


OnnxTranslations.register_translation(OnnxHardSigmoidTranslation(),
                                      converter_type('HardSigmoid', 'onnx'))


# ------------------------------------------------------------------------------
#   Identity
# ------------------------------------------------------------------------------
class OnnxIdentityTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Identity', [1, 13, 14, 16, 19, 21, 23, 24])

    def extract_parameters(self, src_op, converter_context):
        # if the input buffer is not in the graph, that means
        # it is a const input. We replace all const inputs with a
        # const op. Otherwise the identity op is a no-op that
        # gets squashed later.
        graph = converter_context.ir_graph
        if not graph.has_buffer(src_op.input[0]):
            const_input = converter_context.weights.fetch(str(src_op.input[0]))
            converter_context.insert_weights(str(src_op.output[0]), const_input, 
                                             src_op_names=[src_op.name], src_tensor_names=src_op.input)

            # If there are overrides present for input to the IdentityOp, copy them to its output
            enc = graph.get_overridden_encoding(src_op.input[0])
            if enc is not None:
                if 'version' in graph.user_quantization_overrides and graph.user_quantization_overrides['version'] == '2.0.0':
                    add_new_user_quantization_overrides(graph, src_op.output[0], "encodings", enc.copy())
                else:
                    add_new_user_quantization_overrides(graph, src_op.output[0], "param_encodings", enc.copy())
            return op_adapter.ConstantOp(src_op.output[0], const_input)

        return op_adapter.IdentityOp(str(src_op.name))

    def extract_input_names(self, src_op, converter_context):
        # if the input buffer is not in the graph, that means
        # it is a const input. We replace all const inputs with a
        # const op which do not need an input name.
        if not converter_context.ir_graph.has_buffer(src_op.input[0]):
            return []
        return str(src_op.input[0])


OnnxTranslations.register_translation(OnnxIdentityTranslation(),
                                      converter_type('Identity', 'onnx'))


# ------------------------------------------------------------------------------
#   Less
# ------------------------------------------------------------------------------
class OnnxLessTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Less', [1, 7, 9, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS
        self.numpy_op = np.less


OnnxTranslations.register_translation(OnnxLessTranslation(),
                                      converter_type('Less', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS])


# ------------------------------------------------------------------------------
#   LessOrEqual
# ------------------------------------------------------------------------------
class OnnxLessOrEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LessOrEqual', [12])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL
        self.numpy_op = np.less_equal


# LessOrEqual is announced in ONNX 1.7.0, add if statement to avoid warning
if packaging.version.Version(onnx.__version__) >= packaging.version.Version("1.7.0"):
    OnnxTranslations.register_translation(OnnxLessOrEqualTranslation(),
                                          converter_type('LessOrEqual', 'onnx'),
                                          op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL])


# ------------------------------------------------------------------------------
#   LpNormalization
# ------------------------------------------------------------------------------
class OnnxLpNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LpNormalization', [1])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())

        if params.p != 2:
            raise ValueError("Only the L2-Norm is supported. "
                             "Found order of {} for LpNormalization op {}".format(params.p, src_op.name))

        # we use the default value of epsilon here
        return op_adapter.L2NormOp(src_op.name,
                                   axis=params.axis)


OnnxTranslations.register_translation(OnnxLpNormalizationTranslation(),
                                      converter_type('LpNormalization', 'onnx'),
                                      op_adapter.L2NormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Log
# ------------------------------------------------------------------------------
class OnnxLogTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Log', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG
        self.numpy_op = np.log


OnnxTranslations.register_translation(OnnxLogTranslation(),
                                      converter_type('Log', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG])


# ------------------------------------------------------------------------------
#   LogSoftmax
# ------------------------------------------------------------------------------
class OnnxLogSoftmaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LogSoftmax', [1, 11, 13])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        translated_ops = self.extract_parameters(src_op, converter_context)

        # for version < 13, requires flattened 2d tensor
        if len(translated_ops) == 3:
            # translated_ops = [reshape, logsoftmax, reshape]
            # add pre reshape
            pre_reshape_output_names = [input_names[0] + '_reshape']
            pre_reshape_node = graph.add(translated_ops[0], input_names, pre_reshape_output_names)
            self.insert_trace_info([pre_reshape_node, graph.get_buffer(pre_reshape_output_names[0])], (src_op.name, TraceType.OP), converter_context)
            graph.add_src_op_info(pre_reshape_node.op.name, input_names, pre_reshape_output_names)

            # add logsoftmax
            logsoftmax_output_names = [input_names[0] + '_logsoftmax']
            logsoftmax_node = graph.add(translated_ops[1], pre_reshape_node.output_names, logsoftmax_output_names)
            self.insert_trace_info([logsoftmax_node, graph.get_buffer(logsoftmax_output_names[0])], (src_op.name, TraceType.OP), converter_context)
            graph.add_src_op_info(logsoftmax_node.op.name, pre_reshape_node.output_names, logsoftmax_output_names)

            # add post reshape
            input_buf = graph.get_buffer(input_names[0])
            post_reshape_node = graph.add(translated_ops[2], logsoftmax_node.output_names, output_names,
                                          axis_formats=[input_buf.axis_format])
            self.insert_trace_info(post_reshape_node, (src_op.name, TraceType.OP), converter_context)
            for output_name in output_names:
                self.insert_trace_info(graph.get_buffer(output_name), (output_name, TraceType.TENSOR), converter_context)
            graph.add_src_op_info(post_reshape_node.op.name, logsoftmax_node.output_names, output_names)
            last_node = post_reshape_node
            # copy quantization_overrides info to each new op
            rename_user_quantization_overrides(graph, output_names[0], logsoftmax_output_names[0])

        else:
            # translated_ops = [logsoftmax]
            logsoftmax_node = graph.add(translated_ops[0], input_names, output_names)
            self.insert_default_trace_info(src_op, logsoftmax_node, converter_context)
            graph.add_src_op_info(logsoftmax_node.op.name, input_names, output_names)
            last_node = logsoftmax_node

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        params = extract_attributes(src_op, schema=schema)
        if schema.version[0] < 13:
            axis = getattr(params, "axis", 1)
        else:
            axis = getattr(params, "axis", -1)

        origin_shape = graph.get_buffer(src_op.input[0]).shape

        if origin_shape.is_dynamic():
            log_warning("Cannot insert Reshape as it does not support dynamic inputs.")

        # only insert reshape if input rank larger than 2 and input is not dynamic,
        # in such case adding reshape won't influence result
        if schema.version[0] < 13 and origin_shape.rank > 2 and not origin_shape.is_dynamic():
            # for logsoftmax with older version, it will flatten dimension after axis(include) and calculate logsoftmax on it
            shape = [*origin_shape[:axis], np.prod(origin_shape[axis:])]

            # logsoftmax on axis
            logsoftmax_op = op_adapter.LogSoftmaxOp(str(src_op.name), axis=axis)

            # if input_shape and output_shape of reshape is identical, return op only
            if origin_shape == shape:
                return [logsoftmax_op]

            # flatten dimension after axis (include)
            pre_reshape_node_name = str(src_op.input[0]) + '_flatten'
            pre_reshape_op = op_adapter.ReshapeOp(name=pre_reshape_node_name, shape=shape)

            # reshape to origin shape
            post_reshape_node_name = str(src_op.output[0]) + '_reshape'
            post_reshape_op = op_adapter.ReshapeOp(name=post_reshape_node_name, shape=origin_shape)

            translated_ops = [pre_reshape_op, logsoftmax_op, post_reshape_op]
        else:
            translated_ops = [op_adapter.LogSoftmaxOp(str(src_op.name), axis=axis)]
        return translated_ops

OnnxTranslations.register_translation(OnnxLogSoftmaxTranslation(),
                                      converter_type('LogSoftmax', 'onnx'),
                                      op_adapter.LogSoftmaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Matmul
# ------------------------------------------------------------------------------
class OnnxMatMulTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('MatMul', [1, 9, 13])
        self.input_names = []
        self.output_names = []

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        ops = self.extract_parameters(src_op, converter_context)
        if isinstance(ops, list):
            # we need to insert post reshape behind matmul
            # since we have expanded A or B in extract_parameters()
            matmul_op, reshape_op = ops
            matmul_output_names = [self.output_names[0] + '_pre_reshape']
            matmul_node = graph.add(matmul_op, self.input_names, matmul_output_names)
            graph.add_src_op_info(matmul_node.op.name, self.input_names, matmul_output_names)
            reshape_node = graph.add(reshape_op, matmul_output_names, self.output_names)
            graph.add_src_op_info(reshape_node.op.name, matmul_output_names, self.output_names)
            return reshape_node

        # normal flow for matmul op
        # No need to insert post reshape behind
        op = ops

        if not op:
            # Return last node
            return graph.list_nodes()[-1]

        node = graph.add(op, self.input_names, self.output_names)
        self.insert_default_trace_info(src_op, node, converter_context)
        self.add_src_op_info(node.op.name, src_op, graph)

        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.input_names = self.extract_input_names(src_op, converter_context)
        self.output_names = self.extract_output_names(src_op, converter_context)

        # If input A is static, add a Constant Op
        static_input_name = str(self.input_names[0])
        if not graph.has_buffer(static_input_name):
            input_constant_op = self.fetch_constant_op(static_input_name, converter_context, fail_if_dynamic=False)
            if input_constant_op:
                input_constant_node = graph.add(input_constant_op, [], [static_input_name])
                self.insert_constant_trace_info(static_input_name, input_constant_node, converter_context)
                graph.add_src_op_info(input_constant_node.op.name, [], [static_input_name])

        # If input A is an 1-D tensor [n], unsqueeze to [1, n]
        first_input = graph.buffers[self.input_names[0]]
        if len(first_input.shape) == 1:
            first_unsqueeze_name = str(src_op.name) + "_first_unsqueeze"
            first_unsqueeze_output_name = first_input.name + ".unsqueeze"
            first_unsqueeze_op = op_adapter.ReshapeOp(first_unsqueeze_name, shape=[1, first_input.shape[0]])
            first_unsqueeze_node = graph.add(first_unsqueeze_op, [first_input.name], [first_unsqueeze_output_name])
            self.insert_trace_info([first_unsqueeze_node, graph.get_buffer(first_unsqueeze_output_name)], (self.input_names[0], TraceType.TENSOR), converter_context)
            graph.add_src_op_info(first_unsqueeze_node.op.name, [first_input.name], [first_unsqueeze_output_name])
            self.input_names[0] = first_unsqueeze_output_name

        # If input B is constant, add a Constant Op
        weight_input_name = str(self.input_names[1])
        if not graph.has_buffer(weight_input_name):
            weight_constant_op = self.fetch_constant_op(weight_input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if weight_constant_op:
                weight_constant_node = graph.add(weight_constant_op, [], [weight_input_name])
                self.insert_constant_trace_info(weight_input_name, weight_constant_node, converter_context)
                graph.add_src_op_info(weight_constant_node.op.name, [], [weight_input_name])

        # If input B is an 1-D tensor [n], unsqueeze to [n, 1]
        second_input = graph.buffers[self.input_names[1]]
        if len(second_input.shape) == 1:
            second_unsqueeze_name = str(src_op.name) + "_second_unsqueeze"
            second_unsqueeze_output_name = second_input.name + ".unsqueeze"
            second_unsqueeze_op = op_adapter.ReshapeOp(second_unsqueeze_name, shape=[second_input.shape[0], 1])
            second_unsqueeze_node = graph.add(second_unsqueeze_op, [second_input.name], [second_unsqueeze_output_name])
            self.insert_trace_info([second_unsqueeze_node, graph.get_buffer(second_unsqueeze_output_name)], (self.input_names[1], TraceType.TENSOR), converter_context)
            graph.add_src_op_info(second_unsqueeze_node.op.name, [second_input.name], [second_unsqueeze_output_name])
            self.input_names[1] = second_unsqueeze_output_name

        if len(first_input.shape) == 1 and len(second_input.shape) == 1:
            # if src input A and src input B are both 1D tensor, matmul src output shape will be 0D tensor
            # which is not supported in current backend
            raise ValueError("ERROR: Unsupported 0d output tensor for {} op {}".format(src_op.op_type, src_op.name))
        elif len(first_input.shape) != 1 and len(second_input.shape) != 1:
            # No post reshape op is needed, return one matmul op
            # Since ONNX Matmul does not support matrix transpose,
            # both transpose_in0 and transpose_in1 are set False
            return op_adapter.MatMulOp(name=str(src_op.name),
                                       transpose_in0=False,
                                       transpose_in1=False)
        else:
            # Post reshape op is needed after matmul op because we have expanded 1D input tensor
            # Since ONNX Matmul does not support matrix transpose,
            # both transpose_in0 and transpose_in1 are set False
            matmul_op = op_adapter.MatMulOp(name=str(src_op.name),
                                            transpose_in0=False,
                                            transpose_in1=False)
            reshape_op_name = src_op.name
            if src_op.name:
                reshape_op_name = 'Reshape_post_' + src_op.name

            if len(first_input.shape) == 1 and len(second_input.shape) != 1:
                # We have prepended 1 for A input, so post reshape is needed after matmul
                # Example: A is [3], B is [1,4,3,7] --> output shape is [1,4,7]
                output_shape = second_input.shape[:-2] + second_input.shape[-1:]
            elif len(first_input.shape) != 1 and len(second_input.shape) == 1:
                # We have appended 1 for B input, so post reshape is needed after matmul
                # Example: A is [1,4,5,3], B is [3] --> output shape is [1,4,5]
                output_shape = first_input.shape[:-1]

            reshape_op = op_adapter.ReshapeOp(reshape_op_name,
                                              shape=output_shape)
            return [matmul_op, reshape_op]


OnnxTranslations.register_translation(OnnxMatMulTranslation(), converter_type('MatMul', 'onnx'))


# ------------------------------------------------------------------------------
#   MatMulNBits
# ------------------------------------------------------------------------------
class OnnxMatMulnBitsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []
        self.output_names = []

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op = self.extract_parameters(src_op, converter_context)

        if op is None:
            # return last node
            return graph.list_nodes()[-1]

        node = graph.add(op, self.input_names[:2], self.output_names)
        self.insert_default_trace_info(src_op, node, converter_context)
        self.add_src_op_info(node.op.name, src_op, graph)

        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.input_names = self.extract_input_names(src_op, converter_context)
        self.output_names = self.extract_output_names(src_op, converter_context)
        params = extract_attributes(src_op, attr_infos=[('K', 'i'),
                                                        ('N', 'i'),
                                                        ('bits', 'i'),
                                                        ('block_size', 'i')])
        # block-wise column-wise
        axis = 1

        log_assert(params.bits == 4,
                   "Invalid value for bits={}. Supported value is 4 for MatMulNBits op {}".
                   format(params.bits, src_op.name))

        # fetch input B
        log_assert(converter_context.weights.has(self.input_names[1]),
                   "Input B for MatMulNBits op {} needs to be a constant matrix".
                   format(src_op.name))
        input_B = converter_context.weights.fetch(self.input_names[1], prunable=False)
        # dtype_bits = np.dtype(B.dtype).itemsize * 8
        log_assert(input_B.dtype == "uint8",
                   "Input B for MatMulNBits op {} shall be stored as as uint8_t".
                   format(src_op.name))

        n_blocks_per_col = (params.K + params.block_size - 1) // params.block_size
        blob_size = (params.block_size * params.bits + 7) // 8
        log_assert(input_B.shape == (params.N, n_blocks_per_col, blob_size),
                   "Invalid shape for the second input B to MatMulNBits Op {}. Expected shape len is {}"
                   "but got {}".format(src_op.name, (params.N, n_blocks_per_col, blob_size), input_B.shape))

        scales = converter_context.weights.fetch(self.input_names[2], prunable=False)

        # Validate that K is divisible by block_size for vectorized implementation
        if params.K % params.block_size != 0:
          raise ValueError(
              "K ({}) must be divisible by block_size ({}) for vectorized dequantization. "
              "Expected K to be a multiple of block_size.".format(params.K, params.block_size)
          )

        if len(self.input_names) == 4:
            zero_points = converter_context.weights.fetch(self.input_names[3], prunable=False)
        else:
            zero_points_size = params.N * ((n_blocks_per_col * params.bits + 7) // 8)
            zero_points = np.zeros(zero_points_size, dtype=input_B.dtype)

        def dequantize_B(B_quant, scales, zero_points, K, N, block_size, bits):
            # Only works for 4 bits
            log_assert(bits == 4,
                   "Invalid value for bits={}. Supported value is 4 for MatMulNbits".
                   format(bits))

            n_blocks_per_col = K // block_size
            mask = (1 << bits) - 1
            # Precompute zero-points for all blocks
            total_blocks = N * n_blocks_per_col
            bit_positions = np.arange(total_blocks, dtype=np.int64) * bits
            byte_indices = (bit_positions // 8).astype(np.int64)
            bit_offsets = (bit_positions % 8).astype(np.int32)

            log_assert(byte_indices.max() < len(zero_points),
             "Zero-points array size mismatch for MatMulNBits op. "
             "Expected at least {} bytes but got {}.".format(byte_indices.max() + 1, len(zero_points)))
            # Extract zero-points in vectorized way
            zero_points_unpacked = ((zero_points[byte_indices] >> bit_offsets) & mask).astype(zero_points.dtype)

           # Unpack quantized values for all columns and blocks
            lower = B_quant[:, :, :block_size//2] & mask
            upper = (B_quant[:, :, :block_size//2] >> bits) & mask

            # Convert to float32 once before arithmetic
            lower_float = lower.astype(np.float32)
            upper_float = upper.astype(np.float32)

            # Reshape scales and zero_points for broadcasting
            scales_reshaped = scales.reshape(N, n_blocks_per_col, 1)
            zero_points_reshaped = zero_points_unpacked.reshape(N, n_blocks_per_col, 1).astype(np.float32)

            # Compute dequantized values
            # Compute dequantized values by interleaving directly
            lower_dequant = (lower_float - zero_points_reshaped) * scales_reshaped
            upper_dequant = (upper_float - zero_points_reshaped) * scales_reshaped

            # Stack and reshape in one operation
            B_dequant_blocks = np.stack([lower_dequant, upper_dequant], axis=-1).reshape(N, n_blocks_per_col, block_size)

            # Reshape to (K, N)
            B_dequant = B_dequant_blocks.transpose(1, 2, 0).reshape(K, N)
            return B_dequant, zero_points_unpacked

        B_dequant, zero_points_unpacked = dequantize_B(input_B, scales, zero_points,
                                                       params.K, params.N,
                                                       params.block_size, params.bits)
        if not graph.has_buffer(self.input_names[1]):
            B_op = op_adapter.ConstantOp(self.input_names[1], tensor=B_dequant)
            B_node = graph.add(B_op, [], [self.input_names[1]])
            self.insert_constant_trace_info(self.input_names[1], B_node, converter_context)
            graph.add_src_op_info(B_node.op.name, [], [self.input_names[1]])

        enc, _ = get_encoding(self.output_names[0], scales, zero_points_unpacked, axis, params.block_size)
        # align with BQ encoding format
        enc['enc_type'] = "PER_BLOCK"
        enc['is_sym'] = enc['is_symmetric']
        enc['bw'] = params.bits
        new_enc = IROptimizations.extract_encoding_dict(
            graph.get_buffer(self.input_names[1]).producer.op.name,
            enc=[enc],
            version="1.0.0"
        )
        graph.add_quantization_params(
            graph.get_buffer(self.input_names[1]).producer.op.name,
            output_encodings=new_enc
        )

        return op_adapter.MatMulOp(name=str(src_op.name),
                                   transpose_in0=False,
                                   transpose_in1=False)

OnnxTranslations.register_translation(OnnxMatMulnBitsTranslation(), converter_type('MatMulNBits', 'onnx'))


# ------------------------------------------------------------------------------
#   Max
# ------------------------------------------------------------------------------
class OnnxMaxTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Max', [1, 6, 8, 12, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM
        self.numpy_op = np.maximum


OnnxTranslations.register_translation(OnnxMaxTranslation(),
                                      converter_type('Max', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM])

# --------------------------------------------------------------------------------
#   Mean
# --------------------------------------------------------------------------------
class OnnxMeanTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Mean', versions=[1, 6, 8, 13])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.MeanOp(name=src_op.name)

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)

        node_op.name = graph.naming_policy.get_op_name(node_op)

        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, converter_context)
        return node

OnnxTranslations.register_translation(OnnxMeanTranslation(),
                                      converter_type('Mean', 'onnx'))


# ------------------------------------------------------------------------------
#   Min
# ------------------------------------------------------------------------------
class OnnxMinTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Min', [1, 6, 8, 12, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM
        self.numpy_op = np.minimum


OnnxTranslations.register_translation(OnnxMinTranslation(),
                                      converter_type('Min', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM])


# ------------------------------------------------------------------------------
#   Mod
# ------------------------------------------------------------------------------
class OnnxModTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Mod', [10, 13])
        self.numpy_op = np.mod
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MOD

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        fmod = getattr(params, "fmod", 0)
        if fmod:
            self.numpy_op = np.fmod
            self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FMOD

        # reuse super function to perform constant folding and return op
        return super().extract_parameters(src_op, converter_context)


OnnxTranslations.register_translation(OnnxModTranslation(),
                                      converter_type('Mod', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MOD],
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FMOD])


# ------------------------------------------------------------------------------
#   Mul
# ------------------------------------------------------------------------------
class OnnxMulTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Mul', [1, 6, 7, 13, 14])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY
        self.numpy_op = np.multiply


OnnxTranslations.register_translation(OnnxMulTranslation(),
                                      converter_type('Mul', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY])


# ------------------------------------------------------------------------------
#   Neg
# ------------------------------------------------------------------------------
class OnnxNegTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Neg', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG
        self.numpy_op = np.negative


OnnxTranslations.register_translation(OnnxNegTranslation(),
                                      converter_type('Neg', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG])


# ------------------------------------------------------------------------------
#   Not
# ------------------------------------------------------------------------------
class OnnxNotTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Not', [1])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT
        self.numpy_op = np.logical_not


OnnxTranslations.register_translation(OnnxNotTranslation(),
                                      converter_type('Not', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT])


# ------------------------------------------------------------------------------
#   Or
# ------------------------------------------------------------------------------
class OnnxOrTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Or', [1, 7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR
        self.numpy_op = np.logical_or


OnnxTranslations.register_translation(OnnxOrTranslation(),
                                      converter_type('Or', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR])


# ------------------------------------------------------------------------------
#   Pow
# ------------------------------------------------------------------------------
class OnnxPowTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Pow', [1, 7, 12, 13, 15])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER
        self.numpy_op = np.power
        self.const_input_op_2_origin_name = None

    def extract_parameters(self, src_op, converter_context):
        self.input_names = self.extract_input_names(src_op, converter_context)
        const_input_op_1 = self.fetch_constant_op(self.input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        const_input_op_2 = self.fetch_constant_op(self.input_names[1], converter_context, prunable=False, fail_if_dynamic=False)
        # Check if the first input to the Pow op is dynamic and the second input is static
        if const_input_op_1 is None and const_input_op_2 is not None:
            """
            Check if the constant input to the Pow op is 2. If true translate the op into a Mul op since
            Pow(x, 2) == Mul(x, x)
            """

            tensor = const_input_op_2.tensor
            # Check if tensor is a scalar or a single-element array and its value is equal to 2
            if ((np.isscalar(tensor) and tensor == 2)
                    or (isinstance(tensor, np.ndarray) and tensor.size == 1 and tensor.item() == 2)):
                self.const_input_op_2_origin_name = const_input_op_2.name
                self.input_names.remove(const_input_op_2.name)
                self.input_names = self.input_names * 2
                operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY
                return op_adapter.ElementwiseBinaryOp(str(src_op.name), operation=operation)
        return super().extract_parameters(src_op, converter_context)

    def add_op(self, src_op, converter_context):
        node = super().add_op(src_op, converter_context)
        graph = converter_context.ir_graph
        if self.const_input_op_2_origin_name != None and graph.enable_trace:
            # if the extrace_parameters optimize the op to remove the static input 1
            # framework tracing should still record this information from framework model
            pow_node = graph.get_node_by_name(src_op.name)
            pow_trace_info = graph.get_trace_info(pow_node)
            pow_trace_info.append((self.const_input_op_2_origin_name, TraceType.TENSOR))
            graph.set_trace_info(pow_node, pow_trace_info)
        return node

OnnxTranslations.register_translation(OnnxPowTranslation(),
                                      converter_type('Pow', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER])


# ------------------------------------------------------------------------------
#   QLinearMatmul
# ------------------------------------------------------------------------------
class OnnxQLinearMatMulTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('QLinearMatMul', [10, 21])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        ops, enc = self.extract_parameters(src_op, converter_context)
        if isinstance(ops, list):
            # we need to insert post reshape behind matmul
            matmul_op, reshape_op = ops
            matmul_output_names = [self.output_names[0] + '_pre_reshape']
            matmul_node = graph.add(matmul_op, self.input_names, matmul_output_names)
            graph.add_quantization_params(matmul_node.op.name, output_encodings = enc)
            graph.add_src_op_info(matmul_node.op.name, self.input_names, matmul_output_names)
            reshape_node = graph.add(reshape_op, matmul_output_names, self.output_names)
            graph.add_src_op_info(reshape_node.op.name, matmul_output_names, self.output_names)
            return reshape_node

        # normal flow for matmul op
        # No need to insert post reshape behind
        op = ops
        node = graph.add(op, self.input_names, self.output_names)
        self.insert_default_trace_info(src_op, node, converter_context)
        graph.add_quantization_params(node.op.name,
                                      output_encodings = enc)
        self.add_src_op_info(node.op.name, src_op, graph)

        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.input_names = self.extract_input_names(src_op, converter_context)
        self.output_names = self.extract_output_names(src_op, converter_context)
        inputs = src_op.input
        outputs = src_op.output

        input_buffer = graph.get_buffer(str(inputs[0]))

        # Raise error if len(inputs) != 8
        log_assert(len(inputs) == 8,
                   "Invalid number of inputs to QLinearMatMul Op {}. Expected number of inputs 8,"
                   "but got {}".format(src_op.name, len(inputs)))

        # Retrieve first input scale
        first_input_scale_op = self.fetch_constant_op(inputs[1], converter_context, prunable=False,
                                                                             fail_if_dynamic=False)
        if first_input_scale_op is not None:
            first_input_scale = np.array(first_input_scale_op.tensor).astype(np.float32)
        else:
            raise ValueError("Input scale for first input not provided for op: {} of type: {}"
                                                         .format(src_op.name, src_op.op_type))

        # Retrieve first input zero point
        first_input_zero_point = self.fetch_constant_op(inputs[2], converter_context,
                                                         prunable=False, fail_if_dynamic=False)
        if first_input_zero_point is not None:
            first_input_zp = np.array(first_input_zero_point.tensor)
        else:
            raise ValueError("Input zero point for first input not  provided for op: {} of type: {}"
                                                               .format(src_op.name, src_op.op_type))

        # If first input  is static, add a Constant Op
        if converter_context.weights.has(inputs[0]):
            static_input_name = inputs[0]
            tensor = converter_context.weights.fetch(static_input_name, prunable=False)
            # Dequantizing the static input
            tensor_split = np.array_split(tensor, len(first_input_scale))
            split_tensorlist =[]
            for idx in range(len(tensor_split)):
                dequant_tensorlist = first_input_scale[idx] * (tensor_split[idx] - first_input_zp[idx])
                split_tensorlist.append(dequant_tensorlist)
            dequant_array = np.concatenate(split_tensorlist, axis=0)
            tensor = dequant_array
            if not graph.has_buffer(static_input_name):
                static_input_op = op_adapter.ConstantOp(static_input_name, tensor=tensor)
                static_input_node = graph.add(static_input_op, [], [static_input_name])
                self.insert_constant_trace_info(static_input_name, static_input_node, converter_context)
                graph.add_src_op_info(static_input_node.op.name, [], [static_input_name])

        # Retrieve first input encoding and add it to graph as quantization params
        first_input_enc, _ = get_encoding(inputs[0], first_input_scale, first_input_zp)
        graph.add_quantization_params(graph.get_buffer(inputs[0]).producer.op.name,
                                      output_encodings = first_input_enc)

        # If first input is an 1-D tensor [n], unsqueeze to [1, n]
        if len(input_buffer.shape) == 1:
            first_unsqueeze_name = str(src_op.name) + "_first_unsqueeze"
            unsqueeze_output_name = first_unsqueeze_name
            first_unsqueeze_op = op_adapter.ReshapeOp(first_unsqueeze_name,
                                                      shape=[1, input_buffer.shape[0]])
            first_unsqueeze_node = graph.add(first_unsqueeze_op, [input_buffer.name],
                                                        [unsqueeze_output_name])
            self.insert_trace_info([first_unsqueeze_node,
                                    graph.get_buffer(unsqueeze_output_name)],
                                   (self.input_names[0], TraceType.TENSOR), converter_context)
            graph.add_src_op_info(first_unsqueeze_node.op.name, [input_buffer.name],
                                                      [unsqueeze_output_name])
            self.input_names[0] = unsqueeze_output_name

        # Retrieve second input scale
        second_input_scale_op = self.fetch_constant_op(inputs[4], converter_context, prunable=False,
                                                                             fail_if_dynamic=False)
        if second_input_scale_op is not None:
            second_input_scale = np.array(second_input_scale_op.tensor).astype(np.float32)
        else:
            raise ValueError("Input scale for second input not provided for op: {} of type: {}"
                                                         .format(src_op.name, src_op.op_type))

        # Retrieve second input zero point
        second_input_zero_point = self.fetch_constant_op(inputs[5], converter_context,
                                                         prunable=False, fail_if_dynamic=False)
        if second_input_zero_point is not None:
            second_input_zp = np.array(second_input_zero_point.tensor)
        else:
            raise ValueError("Input zero point for second input not provided for op: {} of type: {}"
                                                              .format(src_op.name, src_op.op_type))

        # If second input is constant, add a Constant Op
        if converter_context.weights.has(inputs[3]):
            weight_input_name = inputs[3]
            weights = converter_context.weights.fetch(weight_input_name, prunable=False)
            # Dequantizing the static input
            weight_split = np.array_split(weights, len(second_input_scale), axis=1)
            split_quantlist =[]
            for idx in range(len(weight_split)):
                dequant_list = second_input_scale[idx] * (weight_split[idx] - second_input_zp[idx])
                split_quantlist.append(dequant_list)
            dequant_array = np.concatenate(split_quantlist, axis=1)
            weights = dequant_array
            if not graph.has_buffer(weight_input_name):
                weights_constant_op = op_adapter.ConstantOp(weight_input_name, tensor=weights)
                weight_node = graph.add(weights_constant_op, [], [weight_input_name])
                self.insert_constant_trace_info(weight_input_name, weight_node, converter_context)
                graph.add_src_op_info(weight_node.op.name, [], [weight_input_name])

        second_input_buffer = graph.get_buffer(str(inputs[3]))

        # Retrieve second input encoding and add it to graph as quantization params
        second_input_enc, _ = get_encoding(inputs[3],  second_input_scale, second_input_zp)
        graph.add_quantization_params(graph.get_buffer(inputs[3]).producer.op.name,
                                      output_encodings = second_input_enc)

        # If second input is an 1-D tensor [n], unsqueeze to [n, 1]
        if len(second_input_buffer.shape) == 1:
            second_unsqueeze_name = str(src_op.name) + "_second_unsqueeze"
            unsqueeze_output_name = second_unsqueeze_name
            second_unsqueeze_op = op_adapter.ReshapeOp(second_unsqueeze_name,
                                                       shape=[second_input_buffer.shape[0], 1])
            second_unsqueeze_node = graph.add(second_unsqueeze_op, [second_input_buffer.name],
                                                       [unsqueeze_output_name])
            self.insert_trace_info([second_unsqueeze_node,
                                    graph.get_buffer(unsqueeze_output_name)],
                                    (self.input_names[1], TraceType.TENSOR), converter_context)
            graph.add_src_op_info(second_unsqueeze_node.op.name, [second_input_buffer.name],
                                                       [unsqueeze_output_name])
            self.input_names[1] = unsqueeze_output_name

        # Retrieve output scale
        output_scale_op = self.fetch_constant_op(inputs[6], converter_context, prunable=False ,
                                                                       fail_if_dynamic=False)
        if output_scale_op is not None:
            output_scale = np.array(output_scale_op.tensor).astype(np.float32)
        else:
            raise ValueError("No output scale provided for op: {} of type: {}"
                                         .format(src_op.name, src_op.op_type))

        # Retrieve output zero point
        output_zero_point = self.fetch_constant_op(inputs[7], converter_context, prunable=False ,
                                                                           fail_if_dynamic=False)
        if output_zero_point is not None:
            output_zp = np.array(output_zero_point.tensor)
        else:
            raise ValueError("No output zero point provided for op: {} of type: {}"
                                            .format(src_op.name, src_op.op_type))

        # Retrieve the output encodings
        output_name = str(outputs[0])
        output_enc, _ = get_encoding(output_name, output_scale, output_zp)

        if len(input_buffer.shape) == 1 and len(second_input_buffer.shape) == 1:
            # if src input A and src input B are both 1D tensor, matmul src output shape will be 0D tensor
            # which is not supported in current backend
            raise ValueError("ERROR: Unsupported 0d output tensor for {} op {}"
                                         .format(src_op.op_type, src_op.name))

        elif len(input_buffer.shape) != 1 and len(second_input_buffer.shape) != 1:
            # No post reshape op is needed, return one matmul op
            # Since ONNX Matmul does not support matrix transpose,
            # both transpose_in0 and transpose_in1 are set False
            return op_adapter.MatMulOp(name=str(src_op.name),
                                       transpose_in0=False,
                                       transpose_in1=False), output_enc
        else:
            # Post reshape op is needed after matmul op because we have expanded 1D input tensor
            # Since ONNX Matmul does not support matrix transpose,
            # both transpose_in0 and transpose_in1 are set False
            matmul_op = op_adapter.MatMulOp(name=str(unsqueeze_output_name),
                                            transpose_in0=False,
                                            transpose_in1=False)
            reshape_op_name = src_op.name
            if src_op.name:
                reshape_op_name = src_op.name + '_post_reshape'

            if len(input_buffer.shape) == 1 and len(second_input_buffer.shape) != 1:
                # We have prepended 1 for A input, so post reshape is needed after matmul
                # Example: A is [3], B is [1,4,3,7] --> output shape is [1,4,7]
                output_shape = second_input_buffer.shape[:-2] + second_input_buffer.shape[-1:]
            elif len(input_buffer.shape) != 1 and len(second_input_buffer.shape) == 1:
                # We have appended 1 for B input, so post reshape is needed after matmul
                # Example: A is [1,4,5,3], B is [3] --> output shape is [1,4,5]
                output_shape = input_buffer.shape[:-1]

            reshape_op = op_adapter.ReshapeOp(reshape_op_name,
                                              shape = output_shape)
            return [matmul_op, reshape_op], output_enc

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0], src_op.input[3]]

OnnxTranslations.register_translation(OnnxQLinearMatMulTranslation(),
                                     converter_type('QLinearMatMul', 'onnx'))


# ------------------------------------------------------------------------------
#   RandomUniformLike
# ------------------------------------------------------------------------------
class OnnxRandomUniformLikeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('RandomUniformLike', [1, 22])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type), validate=True)

        input_name = str(src_op.input[0])
        input_dtype = converter_context.tensor_to_np_dtype.get(input_name)

        output_dtype = params.get('dtype')
        if output_dtype is None:
            # if dtype is not specified we check the input dtype
            if input_dtype and issubclass(input_dtype.type, np.floating):
                output_dtype = onnx_to_np_dtype.get(input_dtype)
        else:
            # If dtype is not float then we force it float32, as per the ONNX op definition it should be floating point.
            output_dtype = np.float32

        # If input is initializer add Constant Op to the graph
        if not graph.has_buffer(input_name):
            const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False, fail_if_not_found=True)
            const_node = graph.add(const_op, [], input_name)
            graph.add_src_op_info(input_name, None, input_name)
            self.insert_constant_trace_info(input_name, const_node, converter_context)

        # Determine the low and high values
        low = np.float32(params.get('low', 0.0))
        high = np.float32(params.get('high', 1.0))
        # Handle the seed attribute
        seed = params.get('seed')
        if seed:
            seed = np.float32(seed)

        return op_adapter.RandomUniformLikeOp(src_op.name, low=low, high=high,
                                              seed=seed, dtype=output_dtype)

    def extract_input_names(self, src_op, converter_context):
        return [str(src_op.input[0])]

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxRandomUniformLikeTranslation(),
                                      converter_type('RandomUniformLike', 'onnx'))

# ------------------------------------------------------------------------------
#   RandomNormalLike
# ------------------------------------------------------------------------------
class OnnxRandomNormalLikeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('RandomNormalLike', [1, 22])
        self.input_names = []

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type), validate=True)

        # Determine the low and high values
        mean = np.float32(params.get('mean', 0.0))
        scale = np.float32(params.get('scale', 1.0))

        input_name = str(src_op.input[0])
        input_dtype = converter_context.tensor_to_np_dtype.get(input_name)

        onnx_dtype = params.get('dtype')
        dtype = None
        if onnx_dtype and issubclass(onnx_to_np_dtype[onnx_dtype].type, np.floating):
            dtype = onnx_to_np_dtype[onnx_dtype].name
        elif input_dtype and issubclass(input_dtype.type, np.floating):
            # if dtype is not specified we use input dtype if it is a floating type
            dtype = input_dtype.name
        else:
            # we force it float32 if none of the above, as per the ONNX op definition it should be floating point.
            dtype = 'float32'

        # If input is initializer add Constant Op to the graph
        if graph.has_buffer(input_name):
            input_buf = graph.get_buffer(input_name)
            shape = input_buf.shape.dims
        else:
            const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False, fail_if_not_found=True)
            shape = const_op.tensor.shape

        new_input = np.array(shape, dtype=np.uint32)
        new_input_name = src_op.name + "_input"
        new_input_constant_op = op_adapter.ConstantOp(new_input_name, tensor=new_input, quantizable=False)
        new_input_constant_op.quantizable = False
        self.input_names.append(new_input_name)
        new_input_node = graph.add(new_input_constant_op, [], [new_input_name])
        graph.add_src_op_info(new_input_name, None, new_input_node.output_names[0])
        self.insert_constant_trace_info(new_input_name, new_input_node, converter_context)

        # Handle the seed attribute
        seed = params.get('seed')
        if seed:
            seed_input = np.asarray([seed], dtype=np.float32)
            seed_input_name = src_op.name + "_seed"
            seed_input_constant_op = op_adapter.ConstantOp(seed_input_name, tensor=seed_input, quantizable=False)
            seed_input_constant_op.quantizable = False
            self.input_names.append(seed_input_name)
            seed_input_node = graph.add(seed_input_constant_op, [], [seed_input_name])
            graph.add_src_op_info(seed_input_name, None, seed_input_node.output_names[0])
            self.insert_constant_trace_info(seed_input_name, seed_input_node, converter_context)

        return op_adapter.RandomNormalLikeOp(src_op.name, shape, mean=mean,
                                        scale=scale, dtype=dtype)

    def extract_input_names(self, src_op, converter_context):
        # return input names per the QNN op def
        # where the inputs are shape of the source input graph and seed if present
        return self.input_names

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxRandomNormalLikeTranslation(),
                                      converter_type('RandomNormalLike', 'onnx'))


# ------------------------------------------------------------------------------
#   Reciprocal
# ------------------------------------------------------------------------------
class OnnxReciprocalTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Reciprocal', [1, 6])
        self.input_names = []
        self.numpy_op = np.reciprocal

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.input_names = self.extract_input_names(src_op, converter_context)

        const_input_data = []
        for input_name in self.input_names:
            const_input_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if const_input_op is not None:
                const_input_data.append(const_input_op.tensor)
        if len(const_input_data) == len(self.input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            out_dtype = converter_context.tensor_to_np_dtype.get(str(src_op.output[0]))
            data = self.numpy_op(*const_input_data).astype(out_dtype)
            was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in self.input_names])
            converter_context.insert_weights(str(src_op.output[0]), data, was_scalar, [src_op.name], src_op.input)
            return None
        else:
            # if input is not const, map reciprocal to divide
            input_buf = graph.get_buffer(self.input_names[0])
            const_op_name = src_op.name + "_coeff"
            const_op = op_adapter.ConstantOp(const_op_name, np.ones((1,), np.float32))
            const_node = graph.add(const_op, [], [const_op_name])
            self.insert_trace_info([const_node, graph.get_buffer(const_op_name)], (src_op.name, TraceType.OP), converter_context)
            graph.add_src_op_info(const_node.op.name, [], [const_op_name])
            self.input_names.insert(0, const_op_name)
            return op_adapter.ElementwiseBinaryOp(str(src_op.name), operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE)

OnnxTranslations.register_translation(OnnxReciprocalTranslation(),
                                      converter_type('Reciprocal', 'onnx'))


# ------------------------------------------------------------------------------
#   ReduceBase
# ------------------------------------------------------------------------------
class OnnxReduceBaseTranslation(OnnxTranslationBase):
    def __init__(self, reduce_type):
        OnnxTranslationBase.__init__(self)
        self.reduce_type = reduce_type

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        input_name = src_op.input[0]
        const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_op is None:
            input_buf = graph.get_buffer(input_name)
            rank = input_buf.rank()
        else:
            rank = len(const_op.tensor.shape)

        # TODO: reducesum in opset13 is different with opset11
        # and other reduce types have a same attributes and inputs operation when in opset18.
        if (schema.version[0] < 13 and self.reduce_type =="ReduceSum") or \
            (schema.version[0] < 18 and self.reduce_type !="ReduceSum"):
            schema.replace_default_values(axes=range(rank))
            params = extract_attributes(src_op, schema=schema)
            axes = params.axes
        else:
            schema.replace_default_values(noop_with_empty_axes=0)
            params = extract_attributes(src_op, schema=schema)
            # if noop_with_empty_axes is true and axes is empty, input tensor will not be reduced.
            if params.noop_with_empty_axes and len(src_op.input) == 1:
                return op_adapter.IdentityOp(str(src_op.name))
            else:
                if len(src_op.input) != 1:
                    axes_op = self.fetch_constant_op(str(src_op.input[1]), converter_context, fail_if_dynamic=False)
                    if axes_op is not None:
                        axes = axes_op.tensor
                    else:
                        raise ValueError("Dynamic axes provided, only static axes is supported for op: {} of type: {}"
                                         .format(src_op.name, src_op.op_type))
                else:
                    # when axes is None,noop_with_empty_axes with 'false' is to reduce all axes.
                    axes = list(range(rank))

        # handle static input case with numpy logic
        if const_op:
            data = const_op.tensor
            if self.reduce_type == "ReduceL2":
                reduce_data = np.sqrt(np.sum(a=np.square(data), axis=tuple(axes), keepdims=bool(params.keepdims)))
            elif self.reduce_type == "ReduceMax":
                reduce_data = np.maximum.reduce(data, axis=tuple(axes), keepdims=bool(params.keepdims))
            elif self.reduce_type == "ReduceMean":
                reduce_data = np.mean(data, axis=tuple(axes), keepdims=bool(params.keepdims))
            elif self.reduce_type == "ReduceMin":
                reduce_data = np.minimum.reduce(data, axis=tuple(axes), keepdims=bool(params.keepdims))
            elif self.reduce_type == "ReduceProd":
                reduce_data = np.prod(data, axis=tuple(axes), keepdims=bool(params.keepdims))
            elif self.reduce_type == "ReduceSum":
                reduce_data = np.sum(data, axis=tuple(axes), keepdims=bool(params.keepdims))
            was_scalar = True if len(np.array(reduce_data).shape) == 0 else False
            converter_context.insert_weights(str(src_op.output[0]), np.array(reduce_data), was_scalar, [src_op.name], src_op.input)
            return None
        # if reduce output tensor is 0d, add it to scalar tensor set
        if params.keepdims==0 and len(axes)==input_buf.rank():
            converter_context.scalar_tensor.add(str(src_op.output[0]))
        return op_adapter.ReduceOp(str(src_op.name),
                                   reduce_type=self.reduce_type,
                                   axes=axes,
                                   keep_dims=params.keepdims)

    def extract_input_names(self, src_op, converter_context):
        return [str(src_op.input[0])]


# ------------------------------------------------------------------------------
#   ReduceL2
# ------------------------------------------------------------------------------
class OnnxReduceL2Translation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.IR_OP_REDUCE_L2)
        self.register_op_schema('ReduceL2', [1, 11, 13, 18])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceL2Translation(),
                                      converter_type('ReduceL2', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.IR_OP_REDUCE_L2])


# ------------------------------------------------------------------------------
#   ReduceMax
# ------------------------------------------------------------------------------
class OnnxReduceMaxTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_MAX)
        self.register_op_schema('ReduceMax', [1, 11, 12, 13, 18, 20])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceMaxTranslation(),
                                      converter_type('ReduceMax', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MAX])


# ------------------------------------------------------------------------------
#   ReduceMean
# ------------------------------------------------------------------------------
class OnnxReduceMeanTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_MEAN)
        self.register_op_schema('ReduceMean', [1, 11, 13, 18])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceMeanTranslation(),
                                      converter_type('ReduceMean', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MEAN])


# ------------------------------------------------------------------------------
#   ReduceMin
# ------------------------------------------------------------------------------
class OnnxReduceMinTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_MIN)
        self.register_op_schema('ReduceMin', [1, 11, 12, 13, 18, 20])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceMinTranslation(),
                                      converter_type('ReduceMin', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MIN])


# ------------------------------------------------------------------------------
#   ReduceProd
# ------------------------------------------------------------------------------
class OnnxReduceProdTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_PROD)
        self.register_op_schema('ReduceProd', [1, 11, 13, 18])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceProdTranslation(),
                                      converter_type('ReduceProd', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_PROD])


# ------------------------------------------------------------------------------
#   ReduceSum
# ------------------------------------------------------------------------------
class OnnxReduceSumTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_SUM)
        self.register_op_schema('ReduceSum', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceSumTranslation(),
                                      converter_type('ReduceSum', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_SUM])


# ------------------------------------------------------------------------------
# ReduceLogSumExp
# ------------------------------------------------------------------------------
# ReduceLogSumExp(x, axes, keepdims) = Log(ReduceSum(Exp(x), axes, keepdims))
# ------------------------------------------------------------------------------
# However, Exp(x) can easily overflow. Since we are doing exp(x) first, directly
# evaluating this expression may not be the optimal approach.
#
# To avoid overflow, this expression is evaluated as:
#
# Log(ReduceSum(Exp(x-max(x)), axes, keepdims)) + max(x)
# ------------------------------------------------------------------------------

class OnnxReduceLogSumExpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ReduceLogSumExp', [1, 11, 13, 18])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        translated_ops = self.extract_parameters(src_op, converter_context)

        # Constant folding case: return
        if translated_ops is None:
            return
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        # noop_with_empty_dims case
        if len(translated_ops) == 1:
            node = graph.add(translated_ops[0], [input_names[0]], output_names)
            self.add_src_op_info(node.op.name, src_op, graph)
            self.insert_default_trace_info(src_op, node, converter_context)
            return node

        # Translated graph:
        #        X
        #        |
        #        +-------+
        #        |       |
        #   +---------+  |
        #   |ReduceMax|  +
        #   +---------+ /
        #        |     /
        # +------+    /
        # |      |   /
        # |   +-----+
        # |   | Sub |
        # |   +-----+
        # |      |
        # |   +-----+
        # |   | Exp |
        # |   +-----+
        # |      |
        # | +---------+
        # | |ReduceSum|
        # | +---------+
        # |      |
        # |   +-----+
        # |   | Log |
        # |   +-----+
        # +----+ |
        #      | |
        #     +-----+
        #     | Add |
        #     +-----+
        #        |
        #        Y

        # Reduce max node:
        # Inputs: [X]
        # Outputs: [Reduce max op name]
        reduce_max_node = graph.add(translated_ops[0],
                                    [input_names[0]],
                                    [translated_ops[0].name])
        self.add_src_op_info(reduce_max_node.op.name, src_op, graph)
        self.insert_default_trace_info(src_op, reduce_max_node, converter_context)

        # Elementwise subtract node:
        # Inputs: [X, Output of Reduce max node]
        # Outputs: [Elementwise subtract node name]
        sub_node = graph.add(translated_ops[1],
                             [input_names[0], reduce_max_node.op.name],
                             [translated_ops[1].name])
        self.add_src_op_info(sub_node.op.name, src_op, graph)
        self.insert_default_trace_info(src_op, sub_node, converter_context)

        # Elementwise exp node:
        # Inputs: [Output of Elementwise subtract node]
        # Outputs: [Elementwise exp node name]
        exp_node = graph.add(translated_ops[2],
                             [sub_node.op.name],
                             [translated_ops[2].name])
        self.add_src_op_info(exp_node.op.name, src_op, graph)
        self.insert_default_trace_info(src_op, exp_node, converter_context)

        # Reduce sum node:
        # Inputs: [Output of Elementwise exp node]
        # Outputs: [Reduce sum node name]
        reduce_sum_node = graph.add(translated_ops[3],
                                    [exp_node.op.name],
                                    [translated_ops[3].name])
        self.add_src_op_info(reduce_sum_node.op.name, src_op, graph)
        self.insert_default_trace_info(src_op, reduce_sum_node, converter_context)

        # Elementwise log node:
        # Inputs: [Output of reduce sum node]
        # Outputs: [Elementwise log node name]
        log_node = graph.add(translated_ops[4],
                             [reduce_sum_node.op.name],
                             [translated_ops[4].name])
        self.add_src_op_info(log_node.op.name, src_op, graph)
        self.insert_default_trace_info(src_op, log_node, converter_context)

        # Elementwise add node:
        # Inputs: [Output of elementwise log node,
        #          Output of reduce max node]
        # Outputs: [Y]
        sum_node = graph.add(translated_ops[5],
                             [log_node.op.name, reduce_max_node.op.name],
                             output_names)
        self.add_src_op_info(sum_node.op.name, src_op, graph)
        self.insert_default_trace_info(src_op, sum_node, converter_context)
        return sum_node

    def reduce_log_sum_exp(self, data, axes, keepdims):
        # Function to calculate ReduceLogSumExp for constant folding case
        C = data.max()
        res = C + np.log(np.sum(np.exp(data - C), axis=tuple(axes), keepdims=keepdims))
        res = np.array(res)
        return res

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        input_name = str(src_op.input[0])
        const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)

        # Extract input tensor rank
        if const_op is None:
            input_buf = graph.get_buffer(input_name)
            rank = input_buf.rank()
        else:
            rank = len(const_op.tensor.shape)

        # Opset version < 18 has 'axes' as an attribute, while opset version >= 18
        # has 'axes' as an optional input, which defaults to range(input_rank)
        if schema.version[0] < 18:
            schema.replace_default_values(axes=range(rank))
            params = extract_attributes(src_op, schema=schema)
            axes = params.axes
        else:
            schema.replace_default_values(noop_with_empty_axes=0)
            params = extract_attributes(src_op, schema=schema)
            # if noop_with_empty_axes is true and axes is empty, input tensor will not be reduced.
            if params.noop_with_empty_axes and len(src_op.input) == 1:
                if const_op is not None:
                    converter_context.insert_weights(str(src_op.output[0]), const_op.tensor, 
                                                     src_op_names=[src_op.name], src_tensor_names=src_op.input)
                    return None
                else:
                    return [op_adapter.IdentityOp(str(src_op.name))]
            else:
                if len(src_op.input) != 1:
                    axes_op = self.fetch_constant_op(str(src_op.input[1]), converter_context, fail_if_dynamic=False)
                    if axes_op is not None:
                        axes = axes_op.tensor
                    else:
                        raise ValueError("Dynamic axes provided, only static axes is supported for op: {} of type: {}".
                                         format(src_op.name, src_op.op_type))
                else:
                    # when axes is None,noop_with_empty_axes with 'false' is to reduce all axes.
                    axes = list(range(rank))

        # Handle constant folding case
        if const_op is not None:
            result = self.reduce_log_sum_exp(data=const_op.tensor, axes=axes, keepdims=params.keepdims)
            was_scalar = True if not result.shape else False
            converter_context.insert_weights(str(src_op.output[0]), result, was_scalar, [src_op.name], src_op.input)
            return None

        max_op = op_adapter.ReduceOp(src_op.name + '_max',
                                     reduce_type="ReduceMax",
                                     axes=range(rank),
                                     keep_dims=False)
        sub_op = op_adapter.ElementwiseBinaryOp(src_op.name + '_normalized',
                                                operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT)
        exp_op = op_adapter.ElementwiseUnaryOp(src_op.name + '_normalized_exp',
                                               operation=ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP)
        reduce_op = op_adapter.ReduceOp(src_op.name + "_normalized_reduced_sum",
                                        reduce_type="ReduceSum",
                                        axes=axes,
                                        keep_dims=bool(params.keepdims))
        log_op = op_adapter.ElementwiseUnaryOp(src_op.name + "_normalized_reduced_log_sum",
                                               operation=ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG)
        sum_op = op_adapter.ElementwiseBinaryOp(src_op.name + "_reduced_log_sum",
                                                operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
        return [max_op, sub_op, exp_op, reduce_op, log_op, sum_op]

OnnxTranslations.register_translation(OnnxReduceLogSumExpTranslation(),
                                      converter_type('ReduceLogSumExp', 'onnx'))


# ------------------------------------------------------------------------------
#   ReduceSum
# ------------------------------------------------------------------------------
class OnnxReduceSumSquareTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ReduceSumSquare', [1, 11, 13, 18])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        const_op, power_op, reduce_sum_op = self.extract_parameters(src_op, converter_context)

        const_output_name = input_names[0] + '_exponents'
        const_node = graph.add(const_op, [], [const_output_name])
        graph.add_src_op_info(const_node.op.name, [], [const_output_name])

        power_output_name = input_names[0] + '_power'
        power_input_names = [input_names[0], const_output_name]
        power_node = graph.add(power_op, power_input_names, [power_output_name])
        graph.add_src_op_info(power_node.op.name, power_input_names, [power_output_name])

        reduce_sum_node = graph.add(reduce_sum_op, [power_output_name], output_names)
        graph.add_src_op_info(reduce_sum_node.op.name, [power_output_name], output_names)

        return reduce_sum_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        input_name = src_op.input[0]
        const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_op is None:
            input_buf = graph.get_buffer(input_name)
            rank = input_buf.rank()
        else:
            rank = len(const_op.tensor.shape)

        if schema.version[0] < 18:
            schema.replace_default_values(axes=range(rank))
            params = extract_attributes(src_op, schema=schema)
            axes = params.axes
        else:
            schema.replace_default_values(noop_with_empty_axes=0)
            params = extract_attributes(src_op, schema=schema)
            # if noop_with_empty_axes is true and axes is empty, input tensor will not be reduced.
            if params.noop_with_empty_axes and len(src_op.input) == 1:
                return op_adapter.IdentityOp(str(src_op.name))
            else:
                if len(src_op.input) != 1:
                    axes_op = self.fetch_constant_op(str(src_op.input[1]), converter_context, fail_if_dynamic=False)
                    if axes_op is not None:
                        axes = axes_op.tensor
                    else:
                        raise ValueError("Dynamic axes provided, only static axes is supported for op: {} of type: {}"
                                         .format(src_op.name, src_op.op_type))
                else:
                    # when axes is None,noop_with_empty_axes with 'false' is to reduce all axes.
                    axes = list(range(rank))

        elementwise_power = op_adapter.ElementwiseBinaryOp(str(src_op.name) + "_power", 
                                                           operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER)
        reduce_sum = op_adapter.ReduceOp(str(src_op.name) + "_reduced_sum",
                                                    reduce_type="ReduceSum",
                                                    axes=axes,
                                                    keep_dims=params.keepdims)

        input_shape = np.array(graph.get_buffer(src_op.input[0]).shape)
        reduce_dimensions = [input_shape[axis] if axis in axes else 1 for axis in range(len(input_shape))]
        exponent_tensor = np.zeros(reduce_dimensions) + 2
        constant_op = op_adapter.ConstantOp(str(src_op.name) + "_exponent",
                                            tensor=exponent_tensor)
        converter_context.insert_weights(constant_op.name, exponent_tensor, src_op_names=[src_op.name], src_tensor_names=src_op.input)

        return [constant_op, elementwise_power, reduce_sum]

OnnxTranslations.register_translation(OnnxReduceSumSquareTranslation(),
                                      converter_type('ReduceSumSquare', 'onnx'))


# ------------------------------------------------------------------------------
#   Relu
# ------------------------------------------------------------------------------
class OnnxReluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Relu', [1, 6, 13, 14])

    def extract_parameters(self, src_op, converter_context):
        input_names = self.extract_input_names(src_op, converter_context)
        constant_input_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_dynamic=False)

        if constant_input_op is not None:
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            # ReLU computation
            data = numpy.maximum(constant_input_op.tensor, 0)
            was_scalar = converter_context.weights.was_scalar(input_names[0])
            converter_context.insert_weights(str(src_op.output[0]), data, was_scalar, [src_op.name], src_op.input)
            return None

        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                 operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU)


OnnxTranslations.register_translation(OnnxReluTranslation(),
                                      converter_type('Relu', 'onnx'),
                                      op_adapter.ElementwiseNeuronOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Round
# ------------------------------------------------------------------------------
class OnnxRoundTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Round', [11, 22])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND
        self.numpy_op = np.round


OnnxTranslations.register_translation(OnnxRoundTranslation(),
                                      converter_type('Round', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND])


# ------------------------------------------------------------------------------
#   Selu
# ------------------------------------------------------------------------------
class OnnxSeluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Selu', [1, 6])

    def add_op(self, src_op, context, **kwargs):
        graph = context.ir_graph
        in_out_names, ops = self.extract_parameters(src_op, context)
        for in_out_name, op in zip(in_out_names, ops):
            node = graph.add(op, in_out_name[0], in_out_name[1])
            graph.add_src_op_info(node.op.name, in_out_name[0], in_out_name[1])
            self.insert_default_trace_info(src_op, node, context)
        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        src_op_name = str(src_op.name) if str(src_op.name) else 'selu'
        # Selu can be replaced by "gamma * Elu"
        ops = []
        # in_out_names records input names and output names of each op
        # The format is [[input names, output names], ...]
        in_out_names = []
        src_input_name, src_output_name = str(src_op.input[0]), str(src_op.output[0])
        elu_name = src_op_name + '_elu'
        if not graph.has_buffer(elu_name):
            elu_op =  op_adapter.ElementwiseNeuronOp(elu_name,
                                                     operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU,
                                                     alpha=params.alpha)
            ops.append(elu_op)
            in_out_names.append([[src_input_name], [elu_name]])

        gamma = params.gamma
        gamme_name = src_op_name + '_gamma'
        if not graph.has_buffer(gamme_name):
            gamma_op = op_adapter.ConstantOp(gamme_name, np.atleast_1d(gamma))
            gamma_node = graph.add(gamma_op, [], gamme_name)
            self.insert_constant_trace_info(gamme_name, gamma_node, converter_context)
            graph.add_src_op_info(gamme_name, None, gamma_node.output_names[0])

        mul_name = src_op_name + "_multiply"
        if not graph.has_buffer(src_output_name):
            mul_op = op_adapter.ElementwiseBinaryOp(mul_name,
                                                    operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
            ops.append(mul_op)
            in_out_names.append([[elu_name, gamme_name], [src_output_name]])

        return in_out_names, ops


OnnxTranslations.register_translation(OnnxSeluTranslation(),
                                      converter_type('Selu', 'onnx'))


# ------------------------------------------------------------------------------
#   Sigmoid
# ------------------------------------------------------------------------------
class OnnxSigmoidTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sigmoid', [1, 6, 13])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                   operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID, alpha=1.0)


OnnxTranslations.register_translation(OnnxSigmoidTranslation(), converter_type('Sigmoid', 'onnx'))


# ------------------------------------------------------------------------------
#   Sign
# ------------------------------------------------------------------------------
class OnnxSignTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sign', [13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN
        self.numpy_op = np.sign


OnnxTranslations.register_translation(OnnxSignTranslation(),
                                      converter_type('Sign', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN])


# ------------------------------------------------------------------------------
#   Sin
# ------------------------------------------------------------------------------
class OnnxSinTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sin', [7, 22])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN
        self.numpy_op = np.sin


OnnxTranslations.register_translation(OnnxSinTranslation(),
                                      converter_type('Sin', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN])


# ------------------------------------------------------------------------------
#   Softmax
# ------------------------------------------------------------------------------
class OnnxSoftmaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Softmax', [1, 11, 13])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        translated_ops = self.extract_parameters(src_op, converter_context)
        if len(translated_ops) == 3:
            # translate_ops = [reshape, softmax, reshape]
            # add pre reshape
            pre_reshape_output_names = [input_names[0] + '_reshape']
            pre_reshape_node = graph.add(translated_ops[0], input_names, pre_reshape_output_names)
            self.insert_trace_info([pre_reshape_node, graph.get_buffer(pre_reshape_output_names[0])], (src_op.name, TraceType.OP), converter_context)
            graph.add_src_op_info(pre_reshape_node.op.name, input_names, pre_reshape_output_names)

            # add softmax
            softmax_output_names = [input_names[0] + '_softmax']
            softmax_node = graph.add(translated_ops[1], pre_reshape_node.output_names, softmax_output_names)
            self.insert_trace_info([softmax_node, graph.get_buffer(softmax_output_names[0])], (src_op.name, TraceType.OP), converter_context)
            graph.add_src_op_info(softmax_node.op.name, pre_reshape_node.output_names, softmax_output_names)

            # add post reshape
            input_buf = graph.get_buffer(input_names[0])
            post_reshape_node = graph.add(translated_ops[2], softmax_node.output_names, output_names,
                                          axis_formats=[input_buf.axis_format])
            self.insert_trace_info(post_reshape_node, (src_op.name, TraceType.OP), converter_context)
            for output_name in output_names:
                self.insert_trace_info(graph.get_buffer(output_name), (output_name, TraceType.TENSOR), converter_context)
            graph.add_src_op_info(post_reshape_node.op.name, softmax_node.output_names, output_names)
            last_node = post_reshape_node
            # copy quantization_overrides info to each new op
            rename_user_quantization_overrides(graph, output_names[0], softmax_output_names[0])

        else:
            # translate_ops = [softmax]
            softmax_node = graph.add(translated_ops[0], input_names, output_names)
            self.insert_default_trace_info(src_op, softmax_node, converter_context)
            graph.add_src_op_info(softmax_node.op.name, input_names, output_names)
            last_node = softmax_node

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        params = extract_attributes(src_op, schema=schema)
        axis = getattr(params, "axis", 1)

        origin_shape = graph.get_buffer(src_op.input[0]).shape

        if origin_shape.is_dynamic():
            log_warning("Cannot insert Reshape as it does not support dynamic inputs.")

        # only insert reshape if input rank larger than 2 and input is not dynamic,
        # the reason is that in such case adding reshape won't influence result
        if schema.version[0] < 13 and origin_shape.rank > 2 and not origin_shape.is_dynamic():
            # for softmax with older version, it will flatten dimension after axis(include) and calculate softmax on it
            shape = [*origin_shape[:axis], np.prod(origin_shape[axis:])]

            # flatten dimension after axis (include)
            pre_reshape_node_name = str(src_op.input[0]) + '_flatten'
            pre_reshape_op = op_adapter.ReshapeOp(name=pre_reshape_node_name, shape=shape)

            # softmax on axis
            softmax_op = op_adapter.SoftmaxOp(str(src_op.name), axis=axis)

            # reshape to origin shape
            post_reshape_node_name = str(src_op.output[0]) + '_reshape'
            post_reshape_op = op_adapter.ReshapeOp(name=post_reshape_node_name, shape=origin_shape)

            translated_ops = [pre_reshape_op, softmax_op, post_reshape_op]
        else:
            translated_ops = [op_adapter.SoftmaxOp(str(src_op.name), axis=axis)]
        return translated_ops


OnnxTranslations.register_translation(OnnxSoftmaxTranslation(),
                                      converter_type('Softmax', 'onnx'),
                                      op_adapter.SoftmaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Softplus
# ------------------------------------------------------------------------------
class OnnxSoftplusTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Softplus', [1, 22])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseNeuronOp(str(src_op.name), operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS)


OnnxTranslations.register_translation(OnnxSoftplusTranslation(),
                                      converter_type('Softplus', 'onnx'))

# ------------------------------------------------------------------------------
#   STFT
# ------------------------------------------------------------------------------
class OnnxSTFTTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('STFT', [17])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = list(src_op.input)

        # Extract frame_step attribute
        frame_step_op = self.fetch_constant_op(input_names[1], converter_context, fail_if_dynamic=False)
        if frame_step_op is not None:
            frame_step = frame_step_op.tensor.astype(np.int32).item(0)
        else:
            raise ValueError("ERROR: frame_step input must be a constant tensor for STFT op {}"
                             .format(src_op.name))

        # Initialize window and frame_length as None
        window = None
        frame_length = None
        self.window_exists = False
        # Check if there are more than 2 inputs
        if len(input_names) > 2:
            # Extract window or frame_length from input_names[2]
            window_or_frame_length_op = self.fetch_constant_op(input_names[2], converter_context, fail_if_dynamic=False)
            if window_or_frame_length_op is not None:
                if window_or_frame_length_op.tensor.dtype in [np.float16, np.float32]:
                    self.window_exists = True
                    window = window_or_frame_length_op.tensor
                else:
                    frame_length = window_or_frame_length_op.tensor.astype(np.int32).item(0)

        # Add constant inputs to the graph
        if self.window_exists:
            constant_inputs = [input_names[0], input_names[2]]
        else:
            constant_inputs = [input_names[0]]

        for constant_input in constant_inputs:
            if not graph.has_buffer(constant_input) and converter_context.weights.has(constant_input):
                input_constant_op = self.fetch_constant_op(constant_input, converter_context, fail_if_dynamic=False)
                if input_constant_op:
                    input_constant_node = graph.add(input_constant_op, [], [constant_input])
                    self.insert_constant_trace_info(constant_input, input_constant_node, converter_context)
                    graph.add_src_op_info(input_constant_node.op.name, [], [constant_input])

        # Check if there are more than 3 inputs
        if len(input_names) > 3:
            # Extract frame_length from input_names[3]
            frame_length_op = self.fetch_constant_op(input_names[3], converter_context, fail_if_dynamic=False)
            if frame_length_op is not None:
                frame_length = frame_length_op.tensor.astype(np.int32).item(0)
            else:
                raise ValueError("ERROR: frame_length input must be a constant tensor for STFT op {}"
                                 .format(src_op.name))

        if frame_length == None:
            if self.window_exists:
                # If frame_length is not provided, set frame_length to window size
                frame_length = window.size
            else:
                # If neither frame_length nor window is provided, set frame_length to signal_length
                frame_length = graph.nodes_in_order[0].op.shape[1]

        # Extract onesided attribute
        params = extract_attributes(src_op, schema=self.op_schema())
        onesided = params.onesided if 'onesided' in params else 1
        if onesided not in (0, 1):
            raise ValueError("ERROR: Invalid value {} for onesided attribute for STFT op {}"
                             .format(onesided, src_op.name))

        return op_adapter.STFTOp(str(src_op.name),
                                 frame_length=frame_length,
                                 frame_step=frame_step,
                                 onesided=bool(onesided))

    def extract_input_names(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = list(src_op.input)
        input_buf = graph.get_buffer(input_names[0])
        input_rank = input_buf.rank()

        if input_rank == 2:
            # Insert ExpandDims since dims is always (batch_size, signal_length, 1)
            expand_dims_name = f"{input_names[0]}_expand_dims"

            if not graph.has_buffer(expand_dims_name):
                # Create ExpandDims op only if it doesn't already exist
                expand_dims_op = op_adapter.ExpandDimsOp(name=expand_dims_name, axes=[2])
                expand_dims_node = graph.add(expand_dims_op, [input_names[0]], [expand_dims_name])
                self.insert_trace_info([expand_dims_node, graph.get_buffer(expand_dims_name)], (input_names[0], TraceType.TENSOR), converter_context)
                graph.add_src_op_info(expand_dims_node.op.name, [input_names[0]], [expand_dims_name])

            input_names[0] = expand_dims_name

        if self.window_exists:
            return [input_names[0], input_names[2]]
        else:
            return [input_names[0]]

OnnxTranslations.register_translation(OnnxSTFTTranslation(),
                                      converter_type('Stft', 'onnx'),
                                      op_adapter.STFTOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Sub
# ------------------------------------------------------------------------------
class OnnxSubTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sub', [1, 6, 7, 13, 14])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT
        self.numpy_op = np.subtract


OnnxTranslations.register_translation(OnnxSubTranslation(),
                                      converter_type('Sub', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT])


# ------------------------------------------------------------------------------
#   Sum
# ------------------------------------------------------------------------------
class OnnxSumTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sum', [1, 6, 8, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD
        self.numpy_op = np.add


OnnxTranslations.register_translation(OnnxSumTranslation(), converter_type('Sum', 'onnx'))


# ------------------------------------------------------------------------------
#   Sqrt
# ------------------------------------------------------------------------------
class OnnxSqrtTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sqrt', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT
        self.numpy_op = np.sqrt


OnnxTranslations.register_translation(OnnxSqrtTranslation(),
                                      converter_type('Sqrt', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT])


# ------------------------------------------------------------------------------
#   Tanh
# ------------------------------------------------------------------------------
class OnnxTanhTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Tanh', [1, 6, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        input_quant_enc = graph.get_overridden_encoding(input_names[0])
        input_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_dynamic=False)

        if input_op is not None:
            # Add Constant Op if the input has external overrides
            if input_quant_enc:
                input_node = graph.add(input_op, [], [input_names[0]])
                self.insert_constant_trace_info(input_names[0], input_node, converter_context)
                graph.add_src_op_info(input_node.op.name, [], [input_names[0]])
            else:
                # Constant fold the Tanh Op
                log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
                # Tanh computation
                data = numpy.tanh(input_op.tensor)
                was_scalar = converter_context.weights.was_scalar(input_names[0])
                converter_context.insert_weights(str(src_op.output[0]), data, was_scalar, [src_op.name], src_op.input)
                return None

        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                   operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH,
                                   alpha=1.0,
                                   beta=1.0)


OnnxTranslations.register_translation(OnnxTanhTranslation(),
                                      converter_type('Tanh', 'onnx'))


# ------------------------------------------------------------------------------
#   ScaledTanh
# ------------------------------------------------------------------------------
class OnnxScaledTanhTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ScaledTanh', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        # these parameters belong to ScaledTanh
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                   operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH,
                                   alpha=params.alpha,
                                   beta=params.beta)


# scaledtanh is removed in ONNX release v1.5.0, add if statement to avoid warning
if packaging.version.Version(onnx.__version__) < packaging.version.Version("1.5.0"):
    OnnxTranslations.register_translation(OnnxScaledTanhTranslation(),
                                          converter_type('ScaledTanh', 'onnx'))


# ------------------------------------------------------------------------------
#   TopK
# ------------------------------------------------------------------------------
class OnnxTopKTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('TopK', [1, 10, 11, 24])

    def add_op(self, src_op, context, **kwargs):
        graph = context.ir_graph
        op, post_permute_ops = self.extract_parameters(src_op, context)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)

        if len(post_permute_ops) != 0:
            src_op.output[0] = src_op.output[0] + '_permute'
            src_op.output[1] = src_op.output[1] + '_permute'
            output_names = self.extract_output_names(src_op, context)
            node = graph.add(op, input_names, output_names)
            self.add_src_op_info(node.op.name, src_op, graph)
            post_permute_node_0 = graph.add(post_permute_ops[0], [output_names[0]], [output_names[0].replace('_permute', '')])
            graph.add_src_op_info(post_permute_ops[0].name, [output_names[0]], [output_names[0].replace('_permute', '')])
            post_permute_node_1 = graph.add(post_permute_ops[1], [output_names[1]], [output_names[1].replace('_permute', '')])
            graph.add_src_op_info(post_permute_ops[1].name, [output_names[1]], [output_names[1].replace('_permute', '')])
            return post_permute_node_0

        output_names = self.extract_output_names(src_op, context)
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(src_op.input)
        output_names = list(src_op.output)
        post_permute_ops = list()

        # If input[0] is a constant, add a Constant Op to the graph
        first_input_name = str(input_names[0])
        if not graph.has_buffer(first_input_name):
            input_constant_op = self.fetch_constant_op(first_input_name,
                                                       converter_context,
                                                       prunable=False,
                                                       fail_if_dynamic=False,
                                                       fail_if_not_found=True)
            if input_constant_op:
                input_constant_node = graph.add(input_constant_op, [], [first_input_name])
                self.insert_constant_trace_info(first_input_name,
                                                input_constant_node,
                                                converter_context)
                graph.add_src_op_info(input_constant_node.op.name, [], [first_input_name])

        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_rank = input_buf.rank()
        input_dims = input_buf.get_buf_dims()

        # extract K as input in versions 10, 11 and as parameter in version 1
        if len(input_names) == 2:
            const_op = self.fetch_constant_op(input_names[1], converter_context)
            log_assert(const_op is not None,
                       "Input tensor {} of TopK op {} could not be extracted.", input_names[1], src_op.name)
            k = const_op.tensor.astype(np.int64).item(0)
        else:
            k = params.k

        largest = params.largest if 'largest' in params else 1
        sorted = params.sorted if 'sorted' in params else 1

        if sorted == 0:
            log_warning("TopK op {}, attribute sorted={} not supported", src_op.name, sorted)

        axis = params.axis

        if axis < 0:
            axis += input_rank

        log_assert(input_rank >= 1,
                   "TopK Op {}, Rank of input tensor {} must be greater than or equal to 1", src_op.name, input_rank)
        if axis != input_rank - 1:
            permutation = [i for i in range(input_rank)]
            permutation[axis], permutation[len(permutation)-1] = permutation[len(permutation)-1], permutation[axis]
            permute_op = op_adapter.TransposeOp(input_names[0] + '_permute', perm=permutation)
            graph.add(permute_op, [input_names[0]], [input_names[0] + '_permute'])
            graph.add_src_op_info(permute_op.name, [input_names[0]], [input_names[0] + '_permute'])
            src_op.input[0] = input_names[0] + '_permute'
            post_permute_ops.append(op_adapter.TransposeOp(output_names[0] + '_permute', perm=permutation))
            post_permute_ops.append(op_adapter.TransposeOp(output_names[1] + '_permute', perm=permutation))
            axis = input_rank - 1
            input_dims = [input_dims[i] for i in permutation]

        if k < 0 or input_dims[axis] < k:
            raise ValueError("Input K value of TopK Op {} must be in the range [1, {}]".
                             format(src_op.name, input_dims[axis]))

        return op_adapter.TopKOp(src_op.name, k=k, largest=bool(largest)), post_permute_ops

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxTopKTranslation(),
                                      converter_type('TopK', 'onnx'),
                                      op_adapter.TopKOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Xor
# ------------------------------------------------------------------------------
class OnnxXorTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Xor', [1, 7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_XOR
        self.numpy_op = numpy.logical_xor


OnnxTranslations.register_translation(OnnxXorTranslation(),
                                      converter_type('Xor', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_XOR])