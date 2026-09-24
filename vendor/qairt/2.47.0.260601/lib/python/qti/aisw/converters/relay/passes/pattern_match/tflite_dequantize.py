# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
import tvm
from tvm.relay.frontend.common import set_span

DataTypeBitwidths = {
    "float16": 16,
    "float32": 32,
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
    "uint8": 8,
    "uint16": 16,
    "uint32": 32,
    None: 8,
}


def _sanitize_node_name(name):
    """Sanitize the given node name."""
    name = name.lower()
    name = name.replace("::", "_")
    name = name.replace(".", "_")
    return name


class DequantizeExpr(tvm.relay.ExprMutator):
    def __init__(self, dtype_dict):
        super().__init__()
        self.dtype_dict = dtype_dict

    def visit_var(self, var):
        # Dequantize var expr to fp32 with name hint, shape, and span
        if var.type_annotation.dtype in ["int8", "uint8", "int16", "uint16"]:
            var = tvm.relay.var(
                var.name_hint,
                shape=var.type_annotation.shape,
                dtype="float32",
                span=var.span,
            )
            self.dtype_dict[var.name_hint] = "float32"
        return var

    def visit_call(self, call):
        # Visit args first and replace qnn ops with nn ops
        new_args = [self.visit(arg) for arg in call.args]
        try:
            handler = getattr(self, _sanitize_node_name(call.op.name))
        except AttributeError:
            new_fn = self.visit(call.op)
            return tvm.relay.Call(
                new_fn, new_args, call.attrs, call.type_args, call.span
            )
        else:
            return handler(call, new_args)

    def visit_function(self, fn):
        # enter point
        new_body = self.visit(fn.body)

        # get free_vars after visiting body because new relay.Var may be added
        new_params = tvm.relay.analysis.free_vars(new_body)

        return tvm.relay.Function(
            list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs
        )

    def qnn_add(self, add_expr, args):
        new_add_expr = tvm.relay.add(args[0], args[1])
        new_add_expr = set_span(new_add_expr, add_expr.span)
        return new_add_expr

    def qnn_subtract(self, sub_expr, args):
        new_sub_expr = tvm.relay.subtract(args[0], args[1])
        new_sub_expr = set_span(new_sub_expr, sub_expr.span)
        return new_sub_expr

    def qnn_mul(self, mul_expr, args):
        new_mul_expr = tvm.relay.multiply(args[0], args[1])
        new_mul_expr = set_span(new_mul_expr, mul_expr.span)
        return new_mul_expr

    def qnn_requantize(self, requant_expr, args):
        if len(requant_expr.span.output_names):
            copy_expr = tvm.relay.copy(args[0])
            copy_expr = set_span(copy_expr, requant_expr.span)
            return copy_expr
        return args[0]

    def qnn_dequantize(self, dequant_expr, args):
        if len(dequant_expr.span.output_names):
            copy_expr = tvm.relay.copy(args[0])
            copy_expr = set_span(copy_expr, dequant_expr.span)
            return copy_expr
        return args[0]

    def qnn_quantize(self, quant_expr, args):
        if len(quant_expr.span.output_names):
            copy_expr = tvm.relay.copy(args[0])
            copy_expr = set_span(copy_expr, quant_expr.span)
            return copy_expr
        return args[0]

    def qnn_batch_matmul(self, batch_matmul_expr, args):
        new_input_expr = args[0]
        new_weight_expr = args[1]
        new_batch_matmul_expr = tvm.relay.nn.batch_matmul(
            new_input_expr, new_weight_expr
        )
        new_batch_matmul_expr = set_span(new_batch_matmul_expr, batch_matmul_expr.span)
        return new_batch_matmul_expr

    def qnn_concatenate(self, concatenate_expr, args):
        input_exprs = args[0]
        new_concatenate_expr = tvm.relay.concatenate(
            input_exprs, **concatenate_expr.attrs
        )
        new_concatenate_expr = set_span(new_concatenate_expr, concatenate_expr.span)
        return new_concatenate_expr

    def qnn_conv2d(self, conv2d_expr, args):
        new_data_expr = args[0]
        new_kernel_expr = args[1]
        conv2d_attrs = dict(conv2d_expr.attrs)
        conv2d_attrs["out_dtype"] = "float32"
        new_conv2d_expr = tvm.relay.nn.conv2d(
            new_data_expr, new_kernel_expr, **conv2d_attrs
        )
        new_conv2d_expr = set_span(new_conv2d_expr, conv2d_expr.span)
        return new_conv2d_expr

    def qnn_conv2d_transpose(self, conv2d_transpose_expr, args):
        new_data_expr = args[0]
        new_kernel_expr = args[1]
        conv2d_transpose_attrs = dict(conv2d_transpose_expr.attrs)
        conv2d_transpose_attrs["out_dtype"] = "float32"
        new_conv2d_transpose_expr = tvm.relay.op.nn.conv2d_transpose(
            new_data_expr, new_kernel_expr, **conv2d_transpose_attrs
        )
        new_conv2d_transpose_expr = set_span(
            new_conv2d_transpose_expr, conv2d_transpose_expr.span
        )
        return new_conv2d_transpose_expr

    def qnn_dense(self, dense_expr, args):
        new_data_expr = args[0]
        new_weight_expr = args[1]
        dense_attrs = dict(dense_expr.attrs)
        dense_attrs["out_dtype"] = "float32"
        new_dense_expr = tvm.relay.nn.dense(
            new_data_expr, new_weight_expr, **dense_attrs
        )
        new_dense_expr = set_span(new_dense_expr, dense_expr.span)
        return new_dense_expr

    def cast(self, cast_expr, args):
        source_op_type = cast_expr.span.op_type
        if source_op_type not in [
            "CAST",
            "DEQUANTIZE",
            "GATHER",
        ]:
            # Remove cast if it is from quantized ops
            if len(cast_expr.span.output_names):
                copy_expr = tvm.relay.copy(args[0])
                copy_expr = set_span(copy_expr, cast_expr.span)
                return copy_expr
            return args[0]
        else:
            # Need to create a new expr, otherwise duplicate buffer name issue could be encountered.
            new_fn = self.visit(cast_expr.op)
            return tvm.relay.Call(
                new_fn, args, cast_expr.attrs, cast_expr.type_args, cast_expr.span
            )


class PopulateQuantInfo(tvm.relay.ExprMutator):
    def __init__(self, span_to_encodings, use_quantize_v2):
        super().__init__()
        self.span_to_encodings = span_to_encodings
        self.use_quantize_v2 = use_quantize_v2

    def get_bitwidth_from_expr_and_qparams(self, expr, qparam):
        dtype = qparam.dtype
        bw = DataTypeBitwidths[dtype]

        # Check if the bitwidth can be downcast from 64 to 32 for Const expr
        # Remove this if QNN support 64-bit quantization
        if bw == 64 and isinstance(expr, tvm.relay.Constant):
            data = expr.data.numpy()
            scale = qparam.scale
            zero_point = qparam.zero_point

            # Quantize the float32 data to check if overflow
            data_quantized = np.round((data.astype(scale.dtype) / scale) + zero_point.astype(scale.dtype))
            data_quantized_int_32 = data_quantized.astype(np.int32)
            if np.allclose(data_quantized, data_quantized_int_32):
                bw = 32
        return bw

    def populate_quantization_info(self, new_expr):
        """
        this function should be overrided for those op having multiple output, e.g. relay.split
        since they need to populate each encoding for each output
        """

        span = new_expr.span
        if isinstance(span, tvm.relay.SequentialSpan):
            span = (
                span.spans[0]
                if isinstance(new_expr, tvm.relay.Constant)
                else span.spans[-1]
            )
        if span is None:
            return
        output_qparam_dict = span.output_qparams
        q_infos = []
        for output_name, output_qparam in output_qparam_dict.items():
            scales = output_qparam.scale
            zero_points = output_qparam.zero_point
            dtype = output_qparam.dtype

            # get bitwidth from expr and qparam
            bw = self.get_bitwidth_from_expr_and_qparams(new_expr, output_qparam)

            # tflite: scale*(q-zero_point)
            # QNN:    scale*(q+offset)
            # offset need to be negative of zero_point here
            if scales.size > 1:
                # per channel quantization
                q_info = []
                for zero_point, scale in zip(zero_points, scales):
                    q_info.append(
                        {
                            "bitwidth": bw,
                            "offset": -zero_point,
                            "scale": scale,
                            "is_symmetric": "True",  # symmetric is True for per channel quantization
                            "dtype": dtype if self.use_quantize_v2 else "int",
                        }
                    )
            else:
                # case1: tensor is symmetric with signed int, zero_point must be zero
                # case2: tensor is not symmetric with unsigned int dtype
                # case3: tensor is not symmetric with signed int dtype
                is_symmetric = (
                    dtype and dtype.startswith("int") and np.allclose(zero_points, 0)
                )

                if dtype and dtype.startswith("int") and not is_symmetric and not self.use_quantize_v2:
                    # activation is quantized to unsigned integer in SNPE/QNN v1,
                    # so we need to shift offset for signed integer
                    zero_points = zero_points + 2 ** (bw - 1)

                q_info = {
                    "bitwidth": bw,
                    "offset": -zero_points,
                    "scale": scales,
                    "is_symmetric": str(is_symmetric),
                    "dtype": dtype if self.use_quantize_v2 else "int",
                }

            # the length of encodings should align length of output_names here
            # so for op with multiple output, they should override this function to create q_info
            # for each output
            # e.g,
            # span_to_encodings.encodings = [[encodings1], [encodings2], ...]
            # span_to_encodings.output_names = [output_name1, output_name2, ...]
            q_infos.append(q_info)
        self.span_to_encodings[new_expr.span] = q_infos

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        new_call = tvm.relay.Call(
            new_fn, new_args, call.attrs, call.type_args, call.span
        )
        self.populate_quantization_info(call)

        return new_call

    def visit_var(self, var):
        self.populate_quantization_info(var)
        return var

    def visit_constant(self, const):
        self.populate_quantization_info(const)
        return const


@tvm.ir.transform.module_pass(opt_level=3)
class DequantizePass:
    def __init__(self, dtype_dict, span_to_encodings, use_quantize_v2):
        self.dtype_dict = dtype_dict
        self.span_to_encodings = span_to_encodings
        self.use_quantize_v2 = use_quantize_v2

    # This function can define a pass.
    def transform_module(self, mod, ctx):
        mod.update_func(
            mod.get_global_var("main"),
            DequantizeExpr(self.dtype_dict).visit(mod["main"]),
        )
        mod.update_func(
            mod.get_global_var("main"),
            PopulateQuantInfo(self.span_to_encodings, self.use_quantize_v2).visit(mod["main"]),
        )
        return mod
