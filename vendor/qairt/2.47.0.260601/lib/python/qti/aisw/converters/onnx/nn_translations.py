# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.utils import translation_utils
from qti.aisw.converters.common.converter_ir.op_graph import QuantParams, TraceType
from qti.aisw.converters.onnx import onnx_translations
import qti.aisw.converters.onnx.composable_custom_op_utils as ComposableCustomOp

# ------------------------------------------------------------------------------
#   AveragePool, MaxPool, LpPool
# ------------------------------------------------------------------------------
class OnnxPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('AveragePool', [1, 7, 10, 11, 19, 22])
        self.register_op_schema('MaxPool', [1, 8, 10, 11, 12, 22])
        self.register_op_schema('LpPool', [1, 2, 11, 18, 22])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        input_buffer = graph.get_buffer(input_names[0])

        if input_buffer.rank() == 5:
            default_dilations = [1, 1, 1]
            default_strides = [1, 1, 1]
            default_pads = [0, 0, 0, 0, 0, 0]
        else:
            default_dilations = [1, 1]
            default_strides = [1, 1]
            default_pads = [0, 0, 0, 0]

        params = extract_attributes(src_op, attr_infos=
        [('ceil_mode', 'i', 0),  # new attribute since MaxPool-10 and AveragePool-10
         ('storage_order', 'i', 0),  # new attribute since MaxPool-8
         ('dilations', 'li', default_dilations),  # new attribute since MaxPool-10
         ('count_include_pad', 'i', 0),  # new attribute since AveragePool-7
         ('strides', 'li', default_strides),
         # stride default to 1 if not present since MaxPool-11 and AveragePool-11
         ('kernel_shape', 'li'),
         ('pads', 'li', default_pads),
         ('auto_pad', 's', 'NOTSET'),
         ('p', 'i', 2)], # attribute for LpPool
                                    schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)

        if params.pads != default_pads and params.auto_pad != 'NOTSET':
            log_warning("auto_pad and pads attribute can not be used simultaneously."
                        "Ignoring auto_pad attribute for op {}".format(src_op.name))
            params.auto_pad = 'NOTSET'

        padding_size_strategy = extract_padding_mode(params.auto_pad, src_op.name, params.ceil_mode,
                                                     translation_utils.pads_righthanded(params.pads))
        if str(src_op.op_type) == 'AveragePool':
            if input_buffer.rank() == 5:
                pool_type = ir_graph.QNN_OP_POOL_AVG_3D
            else:
                pool_type = ir_graph.QNN_OP_POOL_AVG_2D
        elif str(src_op.op_type) == 'MaxPool':
            if input_buffer.rank() == 5:
                pool_type = ir_graph.QNN_OP_POOL_MAX_3D
            else:
                pool_type = ir_graph.QNN_OP_POOL_MAX_2D

            log_assert(len(src_op.output) == 1, "Optional output Indices in MaxPool op {} is not supported.", src_op.name)

        elif str(src_op.op_type) == 'LpPool':
            pool_type = ir_graph.QNN_OP_L2_POOL_2D

        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis
        if 'dilations' in params:
            for axis_dilation in params.dilations:
                if (axis_dilation != 1):
                    raise ValueError("Optype: {0}, Opname: {1} Unsupported value {2} in param: dilation. Current {0} "
                                     "translation supports only dilation value 1.".format(src_op.op_type, src_op.name, axis_dilation))
        if input_buffer.rank() == 3 or len(params.kernel_shape) == 1 :
            if pool_type == ir_graph.QNN_OP_L2_POOL_2D:
                return op_adapter.Pool1dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_x=params.kernel_shape[0],
                                           stride_x=params.strides[0],
                                           padx_before=params.pads[0],
                                           padx_after=params.pads[num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy],
                                           p=params.p)
            elif pool_type == ir_graph.QNN_OP_POOL_MAX_2D:
                return op_adapter.Pool1dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_x=params.kernel_shape[0],
                                           stride_x=params.strides[0],
                                           dilation_x=params.dilations[0],
                                           padx_before=params.pads[0],
                                           padx_after=params.pads[num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy])
            elif pool_type == ir_graph.QNN_OP_POOL_AVG_2D:
                return op_adapter.Pool1dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_x=params.kernel_shape[0],
                                           stride_x=params.strides[0],
                                           dilation_x=params.dilations[0],
                                           padx_before=params.pads[0],
                                           padx_after=params.pads[num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy],
                                           count_pad_for_edges=params.count_include_pad)
        elif input_buffer.rank() == 5:
            return op_adapter.Pool3dOp(src_op.name,
                                       pool_type=pool_type,
                                       size_z=params.kernel_shape[0],
                                       size_y=params.kernel_shape[1],
                                       size_x=params.kernel_shape[2],
                                       stride_z=params.strides[0],
                                       stride_y=params.strides[1],
                                       stride_x=params.strides[2],
                                       dilation_z=params.dilations[0],
                                       dilation_y=params.dilations[1],
                                       dilation_x=params.dilations[2],
                                       padz_before=params.pads[0],
                                       padz_after=params.pads[num_input_pads],
                                       pady_before=params.pads[1],
                                       pady_after=params.pads[1 + num_input_pads],
                                       padx_before=params.pads[2],
                                       padx_after=params.pads[2+num_input_pads],
                                       padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy],
                                       count_pad_for_edges=params.count_include_pad)
        else:
            log_assert(num_input_pads == 2,
                       "Maxpool op {}: Converter only support pads for 2D images for spatial values Height and Width. Got {}.",
                       src_op.name, num_input_pads)

            # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims
            if pool_type == ir_graph.QNN_OP_L2_POOL_2D:
                return op_adapter.Pool2dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_y=params.kernel_shape[0],
                                           size_x=params.kernel_shape[1],
                                           stride_y=params.strides[0],
                                           stride_x=params.strides[1],
                                           pady_before=params.pads[0],
                                           pady_after=params.pads[num_input_pads],
                                           padx_before=params.pads[1],
                                           padx_after=params.pads[1 + num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy],
                                           p=params.p)
            elif pool_type == ir_graph.QNN_OP_POOL_MAX_2D:
                return op_adapter.Pool2dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_y=params.kernel_shape[0],
                                           size_x=params.kernel_shape[1],
                                           stride_y=params.strides[0],
                                           stride_x=params.strides[1],
                                           dilation_y=params.dilations[0],
                                           dilation_x=params.dilations[1],
                                           pady_before=params.pads[0],
                                           pady_after=params.pads[num_input_pads],
                                           padx_before=params.pads[1],
                                           padx_after=params.pads[1 + num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy])
            elif pool_type == ir_graph.QNN_OP_POOL_AVG_2D:
                return op_adapter.Pool2dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_y=params.kernel_shape[0],
                                           size_x=params.kernel_shape[1],
                                           stride_y=params.strides[0],
                                           stride_x=params.strides[1],
                                           dilation_y=params.dilations[0],
                                           dilation_x=params.dilations[1],
                                           pady_before=params.pads[0],
                                           pady_after=params.pads[num_input_pads],
                                           padx_before=params.pads[1],
                                           padx_after=params.pads[1 + num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy],
                                           count_pad_for_edges=params.count_include_pad)


OnnxTranslations.register_translation(OnnxPoolTranslation(),
                                      converter_type('AveragePool', 'onnx'),
                                      converter_type('MaxPool', 'onnx'),
                                      converter_type('LpPool', 'onnx'),
                                      op_adapter.Pool1dOp.TRANSLATION_KEY,
                                      op_adapter.Pool2dOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   BatchNormalization
# ------------------------------------------------------------------------------
class OnnxBatchNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('BatchNormalization', [1, 6, 7, 9, 14, 15]) \
            .register_method(self.validate_attribute_values)
        self.weight = None
        self.bias = None
        self.weight_src = []
        self.bias_src = []
        self.ignore_param_encodings = False

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        # In case src_op name is absent, extract parameter is updating quantization_params
        # information with empty '' key. This will throw error later in op_graph_optimization
        # when the bn_params are accessed with python graph node name(generated with naming policy).
        # Thus updating src_op name with same name generated with naming policy.
        bn_op_name = graph.naming_policy.get_op_name_by_type(ir_graph.QNN_OP_BATCHNORM, legacy_translation_key='batchnorm')
        if (len(src_op.name) == 0):
            src_op.name = bn_op_name
        else:
            # If framework level name present, use that instead
            bn_op_name = src_op.name
        node_op = self.extract_parameters(src_op, converter_context)

        # Constant folding case
        if node_op is None:
            return

        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        node_op.name = bn_op_name
        weights_name = bn_op_name + "_bn_w"
        bias_name = bn_op_name + "_bn_b"

        (gamma_quant_enc, beta_quant_enc, _) = self.get_quant_param(src_op, converter_context)
        if self.ignore_param_encodings:
            if gamma_quant_enc:
                log_warning("Batchnorm op: {}, ignoring the Gamma encodings as input_mean is not all zero's" \
                "(OR) input_var is not all one's", src_op.name)
            if beta_quant_enc:
                log_warning("Batchnorm op: {}, ignoring the Beta encodings as input_mean is not all zero's" \
                "(OR) input_var is not all one's", src_op.name)

        else:
            # Propagate gamma/beta overrides to weights/bias if they are compatible
            if gamma_quant_enc:
                graph.set_overridden_encoding(weights_name, gamma_quant_enc)
            if beta_quant_enc:
                graph.set_overridden_encoding(bias_name, beta_quant_enc)

        weight_node = graph.add_constant_op(weights_name, [weights_name], self.weights, axis_formats=[AxisTracker.AxisFormat.ANY])
        self.insert_trace_info([weight_node, graph.get_output_buffers(weight_node)[0]], self.weight_src, converter_context)
        bias_node = graph.add_constant_op(bias_name, [bias_name], self.bias, axis_formats=[AxisTracker.AxisFormat.ANY])
        self.insert_trace_info([bias_node, graph.get_output_buffers(bias_node)[0]], self.bias_src, converter_context)

        input_names.append(weights_name)
        input_names.append(bias_name)
        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, converter_context)
        return node

    def get_quant_param(self, src_op, converter_context):
        graph = converter_context.ir_graph
        gamma_quant_enc = beta_quant_enc = input_quant_enc = None
        #Scenario 1: Quantization params are present in QDQ model
        if graph.quantization_params.keys() is not None:
            if graph.has_quantization_param(list(src_op.input)[0]):
                input_quant_enc = graph.get_layer_quantization_param(list(src_op.input)[0])
            if graph.has_quantization_param(list(src_op.input)[1]):
                gamma_quant_enc = graph.get_layer_quantization_param(list(src_op.input)[1])["output_encodings"]
                # Normalize encoding format for compatibility with extract_encoding_dict function in graph optimization
                gamma_quant_enc[0].pop("name", None)
                if "bw" in gamma_quant_enc[0]:
                    gamma_quant_enc[0]["bitwidth"] = gamma_quant_enc[0].pop("bw")
                if "is_symmetric" in gamma_quant_enc[0] and isinstance(gamma_quant_enc[0]["is_symmetric"], bool):
                    gamma_quant_enc[0]["is_symmetric"] = str(gamma_quant_enc[0].pop("is_symmetric"))
            if graph.has_quantization_param(list(src_op.input)[2]):
                beta_quant_enc = graph.get_layer_quantization_param(list(src_op.input)[2])["output_encodings"]
                beta_quant_enc[0].pop("name", None)
                if "bw" in beta_quant_enc[0]:
                    beta_quant_enc[0]["bitwidth"] = beta_quant_enc[0].pop("bw")
                if "is_symmetric" in beta_quant_enc[0] and isinstance(beta_quant_enc[0]["is_symmetric"], bool):
                    beta_quant_enc[0]["is_symmetric"] = str(beta_quant_enc[0].pop("is_symmetric"))
        #Scenario 2: Quantization params are present by overriding encoding file
        if graph.has_user_quantization_overrides() and gamma_quant_enc is None:
            input_quant_enc = graph.get_overridden_encoding(list(src_op.input)[0])
            gamma_quant_enc = graph.get_overridden_encoding(list(src_op.input)[1])
            beta_quant_enc = graph.get_overridden_encoding(list(src_op.input)[2])

        return gamma_quant_enc, beta_quant_enc, input_quant_enc

    def quant_override_exist(self, src_op, converter_context):
        (gamma_quant_enc, beta_quant_enc, input_quant_enc) = self.get_quant_param(src_op, converter_context)
        if input_quant_enc or gamma_quant_enc or beta_quant_enc:
            return True
        return False

    def extract_parameters(self, src_op, converter_context, do_const_fold=True):
        graph = converter_context.ir_graph
        schema = self.op_schema(op_type=src_op.op_type)

        params = extract_attributes(src_op, schema=schema, validate=True)

        if "training_mode" in params and params.training_mode == 1:
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} op {}".format(params.training_mode,
                                                                                            "training_mode",
                                                                                            src_op.op_type,
                                                                                            src_op.name))

        # spatial removed starting opset 9
        spatial = bool(params.spatial if 'spatial' in params else 1)
        if not spatial:
            raise ValueError("Batchnorm only supports across spatial. Got spatial = False for op {}".format(src_op.name))

        input_names = list(src_op.input)
        gamma, beta, mu, var = converter_context.weights.fetch(*input_names[1:])

        static_input_name = str(input_names[0])
        input_constant_op = self.fetch_constant_op(static_input_name, converter_context, fail_if_dynamic=False)

        # If batchnorm's input is static
        if input_constant_op is not None:
            # Check if any quantization override exists
            if self.quant_override_exist(src_op, converter_context) or not do_const_fold:
                # If a quantization override exists for any constants, do not perform constant folding.
                # Instead, add a Constant Op to the graph.
                input_constant_node = graph.add(input_constant_op, [], [static_input_name])
                self.insert_constant_trace_info(static_input_name, input_constant_node, converter_context)
                graph.add_src_op_info(input_constant_node.op.name, [], [static_input_name])
            else:
                # If no quantization override exists, perform constant folding.
                x = input_constant_op.tensor
                dim_ones = (1,) * (len(x.shape) - 2)
                gamma = gamma.reshape(-1, *dim_ones)
                beta = beta.reshape(-1, *dim_ones)
                mu = mu.reshape(-1, *dim_ones)
                var = var.reshape(-1, *dim_ones)

                # Apply the BatchNorm formula
                out_data = gamma * (x - mu) / np.sqrt(var + params.epsilon) + beta

                was_scalar = converter_context.weights.was_scalar(str(input_names[0]))
                # Insert the computed output data into the weights context
                converter_context.weights.insert(str(src_op.output[0]), out_data, was_scalar=was_scalar)
                return None

        # ONNX Op takes 5 inputs: (Input, Gamma, Beta, input_mean, input_var)
        # QNN Op takes 3 inputs: (Input, Weights, Bias)
        #     where Weights and Bias are calculated with below formulas:
        #         Weight = gamma/sqrt(var+epsilon)
        #         Bias   = -mu*gamma/sqrt(var+epsilon) + beta = -mu*weights + beta
        #
        # Param encodings propagation logic:
        #   if input_mean is all Zero's AND input_var is all One's:
        #     - (Weight, Bias) of QNN op = (Gamma, Beta) of ONNX Op
        #     - Propagate the encodings of Gamma, Beta of ONNX Op to Weight, Bias of QNN op.
        #   Otherwise,
        #     - A workaround until Epic: AISW-126516 is ready:
        #         Ignore the encodings of Gamma, Beta as the data range of Weights and Bias differs
        #         from that of Gamma and Beta.
        if not (mu.all() == 0 and var.all() == 1):
            self.ignore_param_encodings = True

        if graph.has_buffer(input_names[1]):
            converter_context.weights.weight_map[input_names[1]].consumed = False
        if graph.has_buffer(input_names[2]):
            converter_context.weights.weight_map[input_names[2]].consumed = False
        # y = gamma*( (x-mu)/sqrt(var+epsilon) ) + beta
        # weights = gamma/sqrt(var+epsilon)
        self.weights = gamma / np.sqrt(var + params.epsilon)
        # bias = -mu*gamma/sqrt(var+epsilon) + beta = -mu*weights + beta
        self.bias = -mu * self.weights + beta

        self.weight_src = [(input_names[1], TraceType.TENSOR), (input_names[4], TraceType.TENSOR)]
        self.bias_src = [(input_names[1], TraceType.TENSOR), (input_names[2], TraceType.TENSOR), \
                         (input_names[3], TraceType.TENSOR), (input_names[4], TraceType.TENSOR)]

        # cache gamma, beta bn parameters for HBA quant optimization algorithm
        graph.add_quantization_params(src_op.name, bn_params={"gamma": gamma, "beta": beta})

        return op_adapter.BatchnormOp(src_op.name)

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]

    @staticmethod
    def validate_attribute_values(self, attr_name, attr_value):
        # is_test is only supported in test mode, which is_test = 1
        if attr_name == 'is_test':
            log_assert(attr_value, "Only test mode for BatchNormalization Ops is supported.")

OnnxTranslations.register_translation(OnnxBatchNormalizationTranslation(),
                                      converter_type('BatchNormalization', 'onnx'),
                                      op_adapter.BatchnormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Conv
# ------------------------------------------------------------------------------
class OnnxConvTranslation(OnnxTranslationBase):

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Conv', [1, 11, 22])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        if not graph.has_buffer(input_names[0]) and converter_context.weights.has(input_names[0]):
            input_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
            if input_op and not graph.has_buffer(input_names[0]):
                input_node = graph.add(input_op, [], [input_names[0]], axis_formats=[AxisTracker.AxisFormat.NCS])
                self.insert_constant_trace_info(input_names[0], input_node, converter_context)

        input_buffer = graph.get_buffer(input_names[0])

        if input_buffer.rank() == 5:
            default_dilations = [1, 1, 1]
            default_strides = [1, 1, 1]
            default_pads = [0, 0, 0, 0, 0, 0]
        else:
            default_dilations = [1, 1]
            default_strides = [1, 1]
            default_pads = [0, 0, 0, 0]

        params = extract_attributes(src_op, attr_infos=
                                    [('dilations', 'li', default_dilations),
                                     ('strides', 'li', default_strides),
                                     ('kernel_shape', 'li'),
                                     ('pads', 'li', default_pads),
                                     ('auto_pad', 's', 'NOTSET'),
                                     ('group', 'i', 1)], schema=self.op_schema(), validate=True)

        # Extract weights and biases and add them as ConstantOp inputs to the Conv2dOp
        weights_constant_op = self.fetch_constant_op(input_names[1], converter_context, prunable=False, fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(input_names[1]):
            if params.kernel_shape:
                log_assert(tuple(params.kernel_shape) == weights_constant_op.tensor.shape[2:],
                           "kernel_shape parameter differs from weights shape for Conv op {}.",
                           src_op.name)

            log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(src_op.name,
                                                                                     weights_constant_op.tensor.shape))
            if len(weights_constant_op.tensor.shape) == 3:
                weights_node = graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            elif len(weights_constant_op.tensor.shape) == 4:
                weights_node = graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.OIHW])
            elif len(weights_constant_op.tensor.shape) == 5:
                weights_node = graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.OIDHW])
            else:
                raise NotImplementedError(
                        "Convolutional Tensor {} doesn't support {}-D translations for conv op {}."
                        .format(weights_constant_op.name, len(weights_constant_op.tensor.shape), src_op.name))

            self.insert_constant_trace_info(input_names[1], weights_node, converter_context)

        elif graph.has_buffer(input_names[1]):
            weight_buffer = graph.get_buffer(input_names[1])
            weight_rank = weight_buffer.rank()
            # Properly set the axis format if the buffer already exists in the graph
            if weight_rank == 3:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif weight_rank == 4:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.OIHW
            elif weight_rank == 5:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.OIDHW
            else:
                raise NotImplementedError(
                        "Convolutional Tensor {} doesn't support {}-D translations for conv op {}."
                        .format(weights_constant_op.name, len(weights_constant_op.tensor.shape), src_op.name))

        bias_op_name = None
        if len(input_names) > 2:
            bias_op_name = input_names[2]
            bias_constant_op = self.fetch_constant_op(input_names[2], converter_context, prunable=False, fail_if_dynamic=False)
            if bias_constant_op and not graph.has_buffer(input_names[2]):
                log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(src_op.name,
                                                                                      bias_constant_op.tensor.shape))
                bias_node = graph.add(bias_constant_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])
                self.insert_constant_trace_info(input_names[2], bias_node, converter_context)

        # Extract the remaining attributes and calculate the padding size
        padding_size_strategy = IRPaddingStrategies.py_to_c[extract_padding_mode(params.auto_pad, src_op.name)]

        input_buf = graph.get_buffer(input_names[0])
        weights_buf = graph.get_buffer(input_names[1])
        if input_buf.shape.is_dynamic() or weights_buf.shape.is_dynamic():
            if padding_size_strategy not in [ir_graph.PADDING_SIZE_EXPLICIT, ir_graph.PADDING_SIZE_EXPLICIT_FLOOR]:
                raise ValueError("For ConvOp {} with dynamic dimensions, padding strategy must be PADDING_SIZE_EXPLICIT, but got {}".
                                 format(src_op.name, padding_size_strategy))
        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis

        # set the input padding size
        input_shape = input_buf.shape.dims
        weights_shape = weights_buf.shape.dims

        if input_buffer.rank() == 3:
            pad_x = ir_graph.ConvOp.calc_conv_padding_size(input_shape[2],
                                                           weights_shape[2],
                                                           params.dilations[0],
                                                           params.strides[0],
                                                           padding_size_strategy,
                                                           params.pads)

            # Handle marking this Convolution as a DepthwiseConvolution
            num_input_channels = graph.src_axis_order.extract_1d_spatial_dims(input_shape)[-1]
            num_output_channels = graph.src_axis_order.extract_conv1d_weights_dims(weights_shape)[-1]
            convolution_class = op_adapter.Conv1dOp
            if params.group == num_input_channels and num_input_channels == num_output_channels:
                convolution_class = op_adapter.DepthwiseConv1dOp

            return convolution_class(src_op.name,
                                     bias_op_name=bias_op_name,
                                     padx_before=pad_x[0],
                                     padx_after=pad_x[1],
                                     padding_size_strategy=padding_size_strategy,
                                     stridex=params.strides[0],
                                     dilationx=params.dilations[0],
                                     groups=params.group,
                                     data_layout=AxisTracker.AxisFormat.NCF)
        elif input_buffer.rank() == 5:
            if num_input_pads != 3:
                raise ValueError("Conv3d op {}, the number of pads number shall be 3, instead of {}"
                                 .format(src_op.name, num_input_pads))

            pad_z = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[2],
                                                             weights_shape[2],
                                                             params.dilations[0],
                                                             params.strides[0],
                                                             padding_size_strategy,
                                                             [params.pads[0], params.pads[num_input_pads]])

            pad_y = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[3],
                                                             weights_shape[3],
                                                             params.dilations[1],
                                                             params.strides[1],
                                                             padding_size_strategy,
                                                             [params.pads[1], params.pads[1 + num_input_pads]])

            pad_x = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[4],
                                                             weights_shape[4],
                                                             params.dilations[2],
                                                             params.strides[2],
                                                             padding_size_strategy,
                                                             [params.pads[2], params.pads[2 + num_input_pads]])

            # Handle marking this Convolution as a DepthwiseConvolution
            num_input_channels = graph.src_axis_order.extract_3d_spatial_dims(input_shape)[-1]
            num_output_channels = graph.src_axis_order.extract_conv3d_weights_dims(weights_shape)[-1]
            convolution_class = op_adapter.Conv3dOp
            if params.group == num_input_channels and num_input_channels == num_output_channels and params.group > 1:
                raise ValueError("Depthwise 3D convolution operation {} is not yet implemented".format(src_op.name))

            return convolution_class(src_op.name,
                                     bias_op_name=bias_op_name,
                                     pady_before=pad_y[0],
                                     pady_after=pad_y[1],
                                     padx_before=pad_x[0],
                                     padx_after=pad_x[1],
                                     padz_before=pad_z[0],
                                     padz_after=pad_z[1],
                                     padding_size_strategy=padding_size_strategy,
                                     stridex=params.strides[2],
                                     stridey=params.strides[1],
                                     stridez=params.strides[0],
                                     dilationx=params.dilations[2],
                                     dilationy=params.dilations[1],
                                     dilationz=params.dilations[0],
                                     groups=params.group,
                                     data_layout=AxisTracker.AxisFormat.NCDHW)
        else:
            log_assert(num_input_pads == 2,
                       "Node {}: Converter only support pads for 2D images for spatial values Height and Width. Got {}.",
                       src_op.name, num_input_pads)

            # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims
            pad_y = ir_graph.ConvOp.calc_conv_padding_size(input_shape[2],
                                                           weights_shape[2],
                                                           params.dilations[0],
                                                           params.strides[0],
                                                           padding_size_strategy,
                                                           [params.pads[0], params.pads[num_input_pads]])

            pad_x = ir_graph.ConvOp.calc_conv_padding_size(input_shape[3],
                                                           weights_shape[3],
                                                           params.dilations[1],
                                                           params.strides[1],
                                                           padding_size_strategy,
                                                           [params.pads[1], params.pads[1 + num_input_pads]])
            # Handle marking this Convolution as a DepthwiseConvolution
            num_input_channels = graph.src_axis_order.extract_2d_spatial_dims(input_shape)[-1]
            num_output_channels = graph.src_axis_order.extract_conv2d_weights_dims(weights_shape)[-1]
            convolution_class = op_adapter.Conv2dOp
            if params.group == num_input_channels and num_input_channels == num_output_channels:
                convolution_class = op_adapter.DepthwiseConv2dOp

            return convolution_class(src_op.name,
                                     bias_op_name=bias_op_name,
                                     pady_before=pad_y[0],
                                     pady_after=pad_y[1],
                                     padx_before=pad_x[0],
                                     padx_after=pad_x[1],
                                     padding_size_strategy=padding_size_strategy,
                                     stridex=params.strides[1],
                                     stridey=params.strides[0],
                                     dilationx=params.dilations[1],
                                     dilationy=params.dilations[0],
                                     groups=params.group,
                                     data_layout=AxisTracker.AxisFormat.NCS)


OnnxTranslations.register_translation(OnnxConvTranslation(),
                                      converter_type('Conv', 'onnx'),
                                      op_adapter.Conv2dOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ConvTranspose
# ------------------------------------------------------------------------------
class OnnxConvTransposeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        schema_dict = self.register_op_schema('ConvTranspose', [1, 11, 22])
        schema_dict.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        input_buffer = graph.get_buffer(input_names[0])

        if input_buffer.rank() in [3, 4, 5]:
            spatial_rank = input_buffer.rank() - 2
            default_dilations = [1] * spatial_rank
            default_strides = [1] * spatial_rank
            default_pads = [0, 0] * spatial_rank
            default_output_shape = [0] * spatial_rank
            default_output_padding = [0] * spatial_rank
        else:
            raise NotImplementedError(
                "ConvTranspose op {} doesn't support {}-D translations.".format(src_op.name, input_buffer.rank()))

        params = extract_attributes(src_op,
                                    attr_infos=[('dilations', 'li', default_dilations),
                                                ('strides', 'li', default_strides),
                                                ('kernel_shape', 'li'),
                                                ('pads', 'li', default_pads),
                                                ('auto_pad', 's', 'NOTSET'),
                                                ('group', 'i', 1),
                                                ('output_shape', 'li', default_output_shape),
                                                ('output_padding', 'li', default_output_padding),],
                                    schema=self.op_schema(),
                                    validate=True)

        # Extract weights and biases and add them as ConstantOp inputs to the TransposeConv2dOp
        weights_constant_op = self.fetch_constant_op(input_names[1], converter_context, prunable=False, fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(input_names[1]):
            if params.kernel_shape:
                log_assert(tuple(params.kernel_shape) == weights_constant_op.tensor.shape[2:],
                           "kernel_shape parameter differs from weights shape for ConvTranspose op {}.",
                           src_op.name)

            params.kernel_shape = weights_constant_op.tensor.shape[2:]
            log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(src_op.name,
                                                                                     weights_constant_op.tensor.shape))
            if input_buffer.rank() == 3:
                weights_node = graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            elif input_buffer.rank() == 4:
                weights_node = graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.IOHW])
            elif input_buffer.rank() == 5:
                weights_node = graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.IODHW])
            else:
                raise NotImplementedError(
                    "ConvTranspose op {} doesn't support {}-D translations.".format(src_op.name, input_buffer.rank()))
            self.insert_constant_trace_info(input_names[1], weights_node, converter_context)

        elif graph.has_buffer(input_names[1]):
            # Properly set the axis format if the buffer already exists in the graph
            if input_buffer.rank() == 3:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif input_buffer.rank() == 4:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.IOHW
            elif input_buffer.rank() == 5:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.IODHW
            else:
                raise NotImplementedError(
                    "ConvTranspose op {} doesn't support {}-D translations.".format(src_op.name, input_buffer.rank()))

        bias_op_name = None
        if len(input_names) > 2:
            bias_op_name = input_names[2]
            bias_constant_op = self.fetch_constant_op(input_names[2], converter_context, prunable=False, fail_if_dynamic=False)
            if bias_constant_op and not graph.has_buffer(input_names[2]):
                log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(src_op.name,
                                                                                      bias_constant_op.tensor.shape))
                bias_node = graph.add(bias_constant_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])
                self.insert_constant_trace_info(input_names[2], bias_node, converter_context)

        # Extract the remaining attributes and calculate the padding size
        padding_mode = IRPaddingStrategies.py_to_c[extract_padding_mode(params.auto_pad, src_op.name)]
        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis

        # Extract and verify the output padding values
        output_padding = params.output_padding
        if any(output_padding):
            log_assert(output_padding < params.strides,
                       "Output padding for ConvTranspose op {} must be less than stride. Expected < {}, instead got {}.",
                       src_op.name, params.strides, output_padding)

        # Align padding strategy with QNN if output shape is explicitly specified
        if any(params.output_shape):
            # padding calculation differ in opset 1 and 11
            # check https://onnx.ai/onnx/operators/text_diff_ConvTranspose_1_11.html
            # we need to change padding mode accordingly to align padding calculation
            schema = self.op_schema()
            if params.auto_pad in ['SAME_UPPER', 'SAME_LOWER']:
                if schema.version[0] == 1:
                    auto_pad_mapping = {
                        'SAME_UPPER': 'SAME_LOWER',
                        'SAME_LOWER': 'SAME_UPPER'
                    }
                    log_warning(f'change auto_pad from {params.auto_pad} to {auto_pad_mapping[params.auto_pad]} to align with QNN padding calculation')
                    padding_mode = IRPaddingStrategies.py_to_c[extract_padding_mode(auto_pad_mapping[params.auto_pad], src_op.name)]
            elif params.auto_pad in ['NOTSET', 'VALID']:
                spatial_shape = input_buffer.get_buf_dims()[2:]
                total_paddings = []
                for idx in range(num_input_pads):
                    output_shape = params.strides[idx] * (spatial_shape[idx] - 1) + \
                                (params.kernel_shape[idx] - 1) * params.dilations[idx] + 1 + \
                                    params.output_padding[idx]
                    total_padding = output_shape - params.output_shape[idx]
                    total_paddings.append(total_padding)
                total_paddings = [max(0, total_padding) for total_padding in total_paddings]
                for idx, total_padding in enumerate(total_paddings):
                    # if pad_before + pad_afer not equal to total_padding, then pads is not save in source model, need to recalculate
                    pads = list(params.pads)
                    if params.pads[idx] + params.pads[idx + num_input_pads] != total_padding:
                        if schema.version[0] == 1:
                            pads[idx] = total_padding // 2
                            pads[idx + num_input_pads] = total_padding - (total_padding // 2)
                        else:
                            pads[idx] = total_padding - (total_padding // 2)
                            pads[idx + num_input_pads] = total_padding // 2
                    params.pads = pads
        weights_shape = graph.get_buffer(input_names[1]).shape
        if input_buffer.rank() == 3:
            pad_x = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][0],
                                                               params.dilations[0],
                                                               params.strides[0],
                                                               padding_mode,
                                                               [params.pads[0], params.pads[num_input_pads]],
                                                               output_padding[0])
            return op_adapter.TransposeConv1dOp(src_op.name,
                                                bias_op_name=bias_op_name,
                                                stridex=params.strides[0],
                                                padx_before=pad_x[0],
                                                padx_after=pad_x[1],
                                                output_paddingx=params.output_padding[0],
                                                padding_size_strategy=padding_mode,
                                                output_width=params.output_shape[0],
                                                groups=params.group)
        elif input_buffer.rank() == 4:
            pad_x = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][1],
                                                               params.dilations[1],
                                                               params.strides[1],
                                                               padding_mode,
                                                               [params.pads[1], params.pads[1+num_input_pads]],
                                                               output_padding[1])
            pad_y = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][0],
                                                               params.dilations[0],
                                                               params.strides[0],
                                                               padding_mode,
                                                               [params.pads[0], params.pads[num_input_pads]],
                                                               output_padding[0])
            # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims
            return op_adapter.TransposeConv2dOp(src_op.name,
                                                bias_op_name=bias_op_name,
                                                stridex=params.strides[1],
                                                stridey=params.strides[0],
                                                pady_before=pad_y[0],
                                                pady_after=pad_y[1],
                                                padx_before=pad_x[0],
                                                padx_after=pad_x[1],
                                                output_paddingx=params.output_padding[1],
                                                output_paddingy=params.output_padding[0],
                                                padding_size_strategy=padding_mode,
                                                output_height=params.output_shape[0],
                                                output_width=params.output_shape[1],
                                                groups=params.group)
        elif input_buffer.rank() == 5:
            pad_z, pad_y, pad_x = map(list, zip(params.pads[:3], params.pads[3:]))
            pad_x = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][2],
                                                               params.dilations[2],
                                                               params.strides[2],
                                                               padding_mode,
                                                               pad_x,
                                                               output_padding[2])
            pad_y = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][1],
                                                               params.dilations[1],
                                                               params.strides[1],
                                                               padding_mode,
                                                               pad_y,
                                                               output_padding[1])
            pad_z = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][0],
                                                               params.dilations[0],
                                                               params.strides[0],
                                                               padding_mode,
                                                               pad_z,
                                                               output_padding[0])
            # Note: For pads assumes 3D input where dimensions are NCDHW and DHW are the only spatial dims
            return op_adapter.TransposeConv3dOp(src_op.name,
                                                bias_op_name=bias_op_name,
                                                stridex=params.strides[2],
                                                stridey=params.strides[1],
                                                stridez=params.strides[0],
                                                padz_before=pad_z[0],
                                                padz_after=pad_z[1],
                                                pady_before=pad_y[0],
                                                pady_after=pad_y[1],
                                                padx_before=pad_x[0],
                                                padx_after=pad_x[1],
                                                dilationx=params.dilations[2],
                                                dilationy=params.dilations[1],
                                                dilationz=params.dilations[0],
                                                output_paddingx=params.output_padding[2],
                                                output_paddingy=params.output_padding[1],
                                                output_paddingz=params.output_padding[0],
                                                padding_size_strategy=padding_mode,
                                                output_depth=params.output_shape[0],
                                                output_height=params.output_shape[1],
                                                output_width=params.output_shape[2],
                                                groups=params.group)
        else:
            raise ValueError("Converter only supports 1D, 2D & 3D data for ConvTranspose op {}, but received input rank = {}".
                             format(src_op.name, input_buffer.rank()))

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'output_padding':
            log_assert(len(attr_value) <= 3,
                       "Output padding size of ConvTranspose op {} expected to be <= 3. Instead got {}.",
                       src_op.name, len(attr_value))


OnnxTranslations.register_translation(OnnxConvTransposeTranslation(),
                                      converter_type('ConvTranspose', 'onnx'),
                                      op_adapter.TransposeConv2dOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   FC
# ------------------------------------------------------------------------------
class OnnxFCTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)

        fc_op_name = graph.naming_policy.get_op_name(node_op)
        node_op.name = fc_op_name
        weights_name = input_names[1]
        weights_constant_op = op_adapter.ConstantOp(weights_name, tensor=self.weights)
        weight_node = graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        graph.add_src_op_info(weights_name, None, weight_node.output_names[0])

        if len(input_names) > 2:
            bias_name = input_names[2]
            bias_constant_op = op_adapter.ConstantOp(bias_name, tensor=self.bias)
            bias_node = graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add_src_op_info(bias_name, None, bias_node.output_names[0])

        self.insert_constant_trace_info(weights_name, weight_node, converter_context)
        if bias_name:
            self.insert_constant_trace_info(bias_name, bias_node, converter_context)

        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, converter_context)

        # Update weight/bias quantization info if it exists
        src_input_names = list(map(str, src_op.input))
        if graph.has_buffer(src_input_names[1]):
            graph.merge_quantization_params(graph.get_buffer(src_input_names[1]).producer.op.name,
                                            src_op.name, src_input_names[1], 'weights',
                                            encoding_type=QuantParams.PARAM_ENCODINGS)

        if len(src_input_names) > 2 and graph.has_buffer(src_input_names[2]):
                graph.merge_quantization_params(graph.get_buffer(src_input_names[2]).producer.op.name,
                                                src_op.name, src_input_names[2], 'bias',
                                                encoding_type=QuantParams.PARAM_ENCODINGS)

        return node

    def extract_parameters(self, src_op, converter_context):
        # Note: Schema is not used here since this op is not part of the Onnx spec.
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, attr_infos=
        [('axis', 'i', 1),
         ('axis_w', 'i', 1)])
        log_assert(params.axis == 1,
                   "Only an axis value of 1 is supported for FC layers {}.", src_op.name)
        log_assert(params.axis_w == 1,
                   "Only an axis_w value of 1 is supported for FC layers {}.", src_op.name)

        input_names = graph.naming_policy.get_input_names(src_op, src_op.input)
        if len(input_names) == 2:
            weights = converter_context.weights.fetch(input_names[1])
            self.bias_op_name = None
        else:
            weights, bias = converter_context.weights.fetch(*input_names[1:3])
            self.bias_op_name = input_names[2]
            self.bias = bias
        self.weights = weights
        self.weight_name = input_names[1]
        return op_adapter.FullyConnectedOp(src_op.name, bias_op_name=self.bias_op_name)

    def extract_input_names(self, src_op, converter_context):
        input_names = [str(_input) for _input in src_op.input]
        return input_names


OnnxTranslations.register_translation(OnnxFCTranslation(),
                                      converter_type('FC', 'onnx'),
                                      op_adapter.FullyConnectedOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#  Gelu  (need opset>=20, onnx>=1.15)
# ------------------------------------------------------------------------------
class OnnxGeluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Gelu', [20])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        if params.approximate not in ["none", "tanh"]:
            log_error("invalid value for params.approximate in Gelu Op {}".format(src_op.name))

        if params.approximate == "tanh":
            log_warning("Ignoring ONNX Gelu Op {} attribute 'approximate'='tanh'.  We don't support explicit "
                        "control of approximation. The inference behaviour of Gelu depends on QNN backend implementation"
                        .format(src_op.name))

        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        input_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        if input_op and not graph.has_buffer(input_names[0]):
            input_node = graph.add(input_op, [], [input_names[0]])
            self.insert_constant_trace_info(input_names[0], input_node, converter_context)
        return op_adapter.ElementwiseNeuronOp(name=src_op.name, operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU)


OnnxTranslations.register_translation(OnnxGeluTranslation(),
                                      converter_type('Gelu', 'onnx'))


# ------------------------------------------------------------------------------
#   GlobalAveragePool, GlobalMaxPool
# ------------------------------------------------------------------------------
class OnnxGlobalPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GlobalAveragePool', [1, 22])
        self.register_op_schema('GlobalMaxPool', [1, 22])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_buf = graph.get_buffer(str(src_op.input[0]))

        if str(src_op.op_type) == 'GlobalAveragePool':
            if input_buf.rank() == 5:
                pool_type = ir_graph.QNN_OP_POOL_AVG_3D
            else:
                # Note: pool1d currently shares the same pool type as pool2d,
                # because there is no "QNN_OP_POOL_AVG_1D" in QnnOpDef.h
                pool_type = ir_graph.QNN_OP_POOL_AVG_2D
        else:
            if input_buf.rank() == 5:
                pool_type = ir_graph.QNN_OP_POOL_MAX_3D
            else:
                # Note: pool1d currently shares the same pool type as pool2d,
                # because there is no "QNN_OP_POOL_MAX_1D" in QnnOpDef.h
                pool_type = ir_graph.QNN_OP_POOL_MAX_2D

        if input_buf.rank() == 3:
            return op_adapter.Pool1dOp(src_op.name,
                                       pool_type=pool_type,
                                       size_x=input_buf.shape[2],
                                       stride_x=input_buf.shape[2])
        elif input_buf.rank() == 4:
            return op_adapter.Pool2dOp(src_op.name,
                                       pool_type=pool_type,
                                       size_x=input_buf.shape[3],
                                       size_y=input_buf.shape[2],
                                       stride_x=input_buf.shape[3],
                                       stride_y=input_buf.shape[2])
        elif input_buf.rank() == 5:
            return op_adapter.Pool3dOp(src_op.name,
                                       pool_type=pool_type,
                                       size_x=input_buf.shape[4],
                                       size_y=input_buf.shape[3],
                                       size_z=input_buf.shape[2],
                                       stride_x=input_buf.shape[4],
                                       stride_y=input_buf.shape[3],
                                       stride_z=input_buf.shape[2])
        else:
            raise ValueError("Global pool op {} only supports 3D, 4D and 5D input, "
                             "but received input rank = {}".format(src_op.name, input_buf.rank()))


OnnxTranslations.register_translation(OnnxGlobalPoolTranslation(),
                                      converter_type('GlobalAveragePool', 'onnx'),
                                      converter_type('GlobalMaxPool', 'onnx'))


# ------------------------------------------------------------------------------
#   GridSample
# ------------------------------------------------------------------------------
class OnnxGridSampleTranslation(OnnxTranslationBase):
    SUPPORTED_MODES = ['linear', 'bilinear', 'nearest']
    SUPPORTED_PADDING_MODES = ['zeros', 'border', 'reflection']

    onnx_to_ir_mode = {
        "linear": ir_graph.QNN_OP_GRID_SAMPLE_MODE_BILINEAR,
        "bilinear": ir_graph.QNN_OP_GRID_SAMPLE_MODE_BILINEAR,
        "nearest": ir_graph.QNN_OP_GRID_SAMPLE_MODE_NEAREST,
    }

    onnx_to_ir_padding_mode = {
        "zeros": ir_graph.QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS,
        "border": ir_graph.QNN_OP_GRID_SAMPLE_PADDING_MODE_BORDER,
        "reflection": ir_graph.QNN_OP_GRID_SAMPLE_PADDING_MODE_REFLECTION,
    }

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GridSample', [16, 20, 22])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        grid_name = input_names[1]

        # handle constant grid to GridSample
        grid_const_op = self.fetch_constant_op(grid_name, converter_context, prunable=False, fail_if_dynamic=False)
        if grid_const_op and not graph.has_buffer(grid_name):
            grid_node = graph.add(grid_const_op, [], [grid_name], axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            graph.add_src_op_info(grid_name, None, grid_node.output_names[0])
            self.insert_constant_trace_info(grid_name, grid_node, converter_context)

        input_buf = graph.get_buffer(input_names[0])
        log_assert(len(input_buf.shape) in [4, 5],
                   "For GridSample op {}, only 4D and 5D inputs are supported. Got {}D input instead."
                   .format(op.name, len(input_buf.shape)))

        node = graph.add(op, input_names, output_names)
        graph.add_src_op_info(op.name, input_names, output_names)
        self.insert_default_trace_info(src_op, node, converter_context)

        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        log_assert(params.mode in self.SUPPORTED_MODES,
                   "Unsupported mode {} set in GridSampleOp {}.",
                   params.mode, src_op.name)

        log_assert(params.padding_mode in self.SUPPORTED_PADDING_MODES,
                   "Unsupported padding mode {} set in GridSampleOp {}.",
                   params.padding_mode, src_op.name)

        return op_adapter.GridSampleOp(src_op.name,
                                       align_corners=params.align_corners,
                                       mode=self.onnx_to_ir_mode[params.mode],
                                       padding_mode=self.onnx_to_ir_padding_mode[params.padding_mode])


OnnxTranslations.register_translation(OnnxGridSampleTranslation(),
                                      converter_type('GridSample', 'onnx'),
                                      op_adapter.GridSampleOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GroupNormalization
# ------------------------------------------------------------------------------
class OnnxGroupNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        super(OnnxGroupNormalizationTranslation, self).__init__()
        self.register_op_schema('GroupNormalization', [18, 21])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        _, weights_name, bias_name = input_names
        weights_constant_op = self.fetch_constant_op(weights_name, converter_context, prunable=False, fail_if_dynamic=False)
        input_buffer = graph.get_buffer(input_names[0])
        num_channels = input_buffer.get_buf_dims()[1]
        num_groups = node_op.group

        # for Groupnormalization opset-18, scale(or weight) and bias shape = [num_groups]
        # for Groupnormalization opset-21, scale(or weight) and bias shape = [num_channel]
        # QNN expects scale and bias shape = [num_channel]. Resize the shape of scale and bias if the
        # shape of the scale and bias is not equal to num of channels.
        if weights_constant_op and not graph.has_buffer(weights_name):
            weight_shape = weights_constant_op.tensor.shape[0]
            if weight_shape != num_channels:
                # Raise error if the shape of weight is not equal to number of groups
                log_assert(weight_shape == num_groups,
                           "Incorrect shape: {} of scale input {} for Groupnormalization "
                           "op {}. Expected: {} or {}",
                           weight_shape, weights_name, src_op.name, num_channels, num_groups)
                # update the weight tensor to per channel instead of per group.
                # For example, if the weight = [3, 7], num_group = 2 and num_channel = 4, then the new
                # weights is [3, 3, 7, 7]
                new_weights = []
                for w in weights_constant_op.tensor:
                    new_weights = new_weights + [w]*(num_channels//num_groups)
                new_weights = np.array(new_weights)
                weights_name = src_op.name + "_weight"
                new_weights_constant_op = op_adapter.ConstantOp(weights_name, tensor=new_weights)
                # update weight name
                input_names[1] = weights_name
                weight_node = graph.add(new_weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            else:
                weight_node = graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
                graph.add_src_op_info(weights_name, None, weight_node.output_names[0])
                self.insert_constant_trace_info(weights_name, weight_node, converter_context)
        else:
            weight_shape = graph.get_buffer(input_names[1]).shape[0]

            if weight_shape != num_channels:
                # Raise error if the shape of weight is not equal to number of groups
                log_assert(weight_shape == num_groups,
                           "Incorrect shape: {} of scale input {} for Groupnormalization "
                           "op {}. Expected: {} or {}",
                           weight_shape, weights_name, src_op.name, num_channels, num_groups)

                prefix = src_op.name + "_weight"
                # for dynamic weight, resize the weight by adding Reshape -> Tile -> Reshape
                pre_reshape_name = prefix + "_pre_reshape"
                graph.add(op_adapter.ReshapeOp(pre_reshape_name, shape=[weight_shape, 1]),
                          [input_names[1]], [pre_reshape_name])
                tile_name = prefix + "_tile"
                graph.add(op_adapter.TileOp(tile_name, multiples=[1, num_channels//num_groups]),
                          [pre_reshape_name], [tile_name])
                post_reshape_name = prefix + "_post_reshape"
                graph.add(op_adapter.ReshapeOp(post_reshape_name, shape=[num_channels]),
                          [tile_name], [post_reshape_name])

                # update weight name
                input_names[1] = prefix + "_post_reshape"

        bias_constant_op = self.fetch_constant_op(bias_name, converter_context, prunable=False, fail_if_dynamic=False)
        if bias_constant_op and not graph.has_buffer(bias_name):
            bias_shape = weights_constant_op.tensor.shape[0]
            if bias_shape != num_channels:
                # Raise error if the shape of bias is not equal to number of groups
                log_assert(bias_shape == num_groups,
                           "Incorrect shape: {} of bias input {} for Groupnormalization op {}. Expected: {} or {}",
                           bias_shape, bias_name, src_op.name, num_channels, num_groups)

                # update the bias tensor to per channel instead of per group similar to weight tensor.
                new_bias = []
                for w in bias_constant_op.tensor:
                    new_bias = new_bias + [w]*(num_channels//num_groups)
                new_bias = np.array(new_bias)
                bias_name = src_op.name + "_bias"
                new_bias_constant_op = op_adapter.ConstantOp(bias_name, tensor=new_bias)
                # update bias name
                input_names[2] = bias_name
                bias_node = graph.add(new_bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            else:
                bias_node = graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
                graph.add_src_op_info(bias_name, None, bias_node.output_names[0])
                self.insert_constant_trace_info(bias_name, bias_node, converter_context)
        else:
            bias_shape = graph.get_buffer(input_names[2]).shape[0]
            if bias_shape != num_channels:
                # Raise error if the shape of bias is not equal to number of groups
                log_assert(bias_shape == num_groups,
                           "Incorrect shape: {} of bias input {} for Groupnormalization op {}. Expected: {} or {}",
                           bias_shape, bias_name, src_op.name, num_channels, num_groups)

                prefix = src_op.name + "_bias"

                # for dynamic weight, resize the weight by adding Reshape -> Tile -> Reshape
                pre_reshape_name = prefix + "_pre_reshape"
                graph.add(op_adapter.ReshapeOp(pre_reshape_name, shape=[weight_shape, 1]),
                          [input_names[2]], [pre_reshape_name])
                tile_name = prefix + "_tile"
                graph.add(op_adapter.TileOp(tile_name, multiples=[1, num_channels//num_groups]),
                          [pre_reshape_name], [tile_name])
                post_reshape_name = prefix + "_post_reshape"
                graph.add(op_adapter.ReshapeOp(post_reshape_name, shape=[num_channels]),
                          [tile_name], [post_reshape_name])

                # update bias name
                input_names[2] = prefix + "_post_reshape"

        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, converter_context)
        return node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        return op_adapter.GroupNormOp(src_op.name,
                                      epsilon=params.epsilon,
                                      group=params.num_groups)


OnnxTranslations.register_translation(OnnxGroupNormalizationTranslation(),
                                      converter_type('GroupNormalization', 'onnx'),
                                      'groupnorm',
                                      op_adapter.GroupNormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   If
# ------------------------------------------------------------------------------
# This implements ONNX_IF op support with some limitations.
# ------------------------------------------------------------------------------
# Implementation:
#
# Fold constant If condition to either THEN or ELSE branch
# Fold THEN and/or ELSE branch if possible
# Decompose Unfoldable If op to eltwise_select(cond, then_output, else_output)
# ------------------------------------------------------------------------------
# Limitations:
#
# Doesn't support THEN and ELSE producing different output shapes
# ------------------------------------------------------------------------------
class OnnxIfTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('If', [1, 11, 13, 16, 19, 21, 23, 24])
        self.input_names = None
        self.then_output_is_modified = False
        self.is_ternary = False
        self.else_output_is_modified = False

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.is_ternary = False
        self.then_output_is_modified = False
        self.else_output_is_modified = False
        self.selected_branch_output_is_modified = False
        translated_ops = self.extract_parameters(src_op, converter_context)
        last_node = None
        if translated_ops is None:
            return

        # For If op with dynamic condition, and N outputs per branch,
        # we will have (2*N + 1) entries in self.input_names.
        # [cond_input, then_out_0, then_out_1, ..., then_out_(N-1), else_out_0, ..., else_out_(N-1)]
        # then_out_i is the i^th output of the then branch, and else_out_i is the i^th output of
        # the else branch.
        # We need to form N ElementwiseTernary ops with inputs as: [cond, then_out_0, else_out_0],
        # [cond, then_out_1, else_out_1], ... [cond, then_out_(N-1), else_out_(N-1)]

        # For If op with static condition, and N outputs per branch,
        # we will have (N) entries in self.input_names.
        # [then_out_0, then_out_2, ..., then_out_(N-1)] or [else_out_0, else_out_2, ..., else_out_(N-1)]
        # xxx_out_i is the i^th output of the selected branch
        # We need to form N Identity ops with input as: [xxx_out_0], [xxx_out_1], ... [xxx_out_(N-1)]
        # depending on the branch selected.

        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        no_of_branch_outputs = 0
        if self.is_ternary:
            no_of_branch_outputs = int((len(input_names) - 1) / 2)
        else:
            no_of_branch_outputs = len(input_names)
        for i in range(no_of_branch_outputs):
            op = translated_ops.get(i)
            if op is None:
                continue
            else:
                if op.type == op_adapter.IdentityOp.TRANSLATION_KEY:
                    self.add_src_op_info(op.name, src_op, graph)
                    op_input_names = [input_names[i]]
                    op_output_names = [output_names[i]]
                    node = graph.add(op, op_input_names, op_output_names)
                    self.insert_default_trace_info(src_op, node, converter_context)
                    last_node = node
                    continue
                else:
                    for input_name in self.input_names:
                        const_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
                        # Add fetched constant op to graph, if it doesn't exist
                        if const_op is not None:
                            if not graph.has_buffer(input_name):
                                const_node = graph.add(const_op, [], input_name)
                                self.insert_constant_trace_info(input_name, const_node, converter_context)
                                graph.add_src_op_info(const_op.name, None, const_node.output_names[0])

                    # Add elementwise src op info
                    condition_input = input_names[0]
                    self.add_src_op_info(op.name, src_op, graph)
                    shape = graph.get_buffer(condition_input).shape
                    rank = len(shape)

                    # We need the following assertions before adding the ElementwiseTernaryOp
                    # 1. Condition input must be a scalar or a 1D tensor with a single element.
                    #    If it is a tensor, then based on which entries are True and False,
                    #    we will be extracting data partially from the 'then' and 'else' branches.
                    #    Ideally, we want to select the entire data from 'else' branch output or
                    #    'then' branch output.
                    # 2. The 'then' and 'else' branch outputs must have the same shape. ElementwiseTernary
                    #    op allows their shapes to be broadcastable, but we don't want them to be broadcastable here.

                    if rank > 1:
                        raise ValueError(f"Input {input_names[0]} of op {src_op.name} must have rank 1, but got rank {rank}")
                    elif shape[0] != 1:
                        raise ValueError("Input {} of op {} must be either a scalar or a 1D tensor with single element"
                                         .format(input_names[0],src_op.name))

                    then_br_output_name = input_names[i+1]
                    else_br_output_name = input_names[i+1+no_of_branch_outputs]
                    then_br_output_shape = graph.get_buffer(then_br_output_name).shape
                    else_br_output_shape = graph.get_buffer(else_br_output_name).shape
                    if then_br_output_shape != else_br_output_shape:
                        raise ValueError("Inputs {} and {} of op {} must have same shape, but received {} and {}."
                                         .format(input_names[1], input_names[2], src_op.name, then_br_output_shape, else_br_output_shape))

                    if self.then_output_is_modified:
                        then_br_output_name = then_br_output_name[:-len('_subgraph_then')]
                    if self.else_output_is_modified:
                        else_br_output_name = else_br_output_name[:-len('_subgraph_else')]
                    then_dtype = converter_context.tensor_to_np_dtype[then_br_output_name]
                    else_dtype = converter_context.tensor_to_np_dtype[else_br_output_name]
                    if then_dtype != else_dtype:
                        raise RuntimeError("For op {}, subgraph outputs {} and {} must be "
                                           "of the same data type. Received {} and {}"
                                           .format(src_op.name,
                                                   then_br_output_name,
                                                   else_br_output_name,
                                                   then_dtype,
                                                   else_dtype))
                    # Now, we can add the ElementwiseTernaryOp to the graph
                    # self.input_names = [cond_input, then_out_0, then_out_1, ..., then_out_(N-1), else_out_0, ..., else_out_(N-1)]
                    # select_node_0_input_names = [cond_input, then_out_0, else_out_0]
                    # select_node_1_input_names = [cond_input, then_out_1, else_out_1], etc.
                    # no_of_branch_outputs = N
                    # cond_input_index = 0, then_out_0_idx = 1, else_out_0_idx = N+2
                    #                       then_out_1_idx = 2, else_out_1_idx = N+3 etc.
                    # In general, then_out_i_idx = i+1, else_out_i_idx = i+N+1. The indices of the Ternary op
                    # are extracted in this manner below.
                    select_node_input_names = [self.input_names[0], self.input_names[i+1], self.input_names[i+1+no_of_branch_outputs]]
                    select_node_output_names = [output_names[i]]
                    elementwise_select_node = graph.add(op, select_node_input_names, select_node_output_names)
                    self.insert_default_trace_info(src_op, elementwise_select_node, converter_context)
                    last_node = elementwise_select_node
        return last_node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        condition_input = list(map(str, src_op.input))
        condition_op = self.fetch_constant_op(condition_input[0], converter_context, prunable=False, fail_if_dynamic=False)
        if condition_op is not None:
            # In case the condition is a static tensor:
            #    1. Select the subgraph of 'then' or 'else' branch based on condition.
            #    2. Add the appropriate subgraph and return IdentityOp
            condition_tensor = condition_op.tensor.flatten()
            if len(condition_tensor) > 1:
                raise ValueError(f"In If Op {src_op.name}, input {self.input_names[0]} must be 0D or 1D with a single element.")

            if condition_tensor[0]:
                branch = params['then_branch']
            else:
                branch = params['else_branch']
            model = onnx.helper.make_model(branch)

            # Add the constants of the subgraph
            self._add_graph_initializers(branch, converter_context)

            # Add the nodes of the selected subgraph to the model
            for node in branch.node:
                # If the subgraph output is the same as the main graph If op output,
                # we need to modify the output name
                for j in range(len(node.output)):
                    if node.output[j] in src_op.output:
                        node.output[j] += '_subgraph_then'
                self._add_onnx_op(node, converter_context, model)

            # Store the outputs of the selected subgraph
            output_names = []
            for i in range(len(branch.output)):
                output_name = branch.output[i].name
                if output_name in src_op.output:
                    # If the subgraph output is the same as the main graph If op output,
                    # we need to modify the output name
                    output_name += '_subgraph_then'
                output_names.append(output_name)
            self.input_names = output_names

            # If the selected branch is statically foldable, then just add the output of the subgraph to the converter context.
            # For If op with static condition, and N outputs per branch,
            # we will have (N) entries in self.input_names.
            # [then_out_0, then_out_2, ..., then_out_(N-1)] or [else_out_0, else_out_2, ..., else_out_(N-1)]
            # xxx_out_i is the i^th output of the selected branch
            # We need to form N Identity ops with input as: [xxx_out_0], [xxx_out_1], ... [xxx_out_(N-1)]
            # depending on the branch selected.

            translated_ops = {}
            output_idx = 0
            for input_name in self.input_names:
                branch_output = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
                if branch_output is not None:
                    was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in self.input_names])
                    converter_context.insert_weights(str(src_op.output[output_idx]), branch_output.tensor, was_scalar, [src_op.name], src_op.input)
                else:
                    translated_ops[output_idx] = op_adapter.IdentityOp(src_op.name + '_' + str(output_idx))
                output_idx += 1
            if not translated_ops:
                return None
            else:
                return translated_ops
        else:
            # In case the condition is a dynamic tensor:
            #    1. Add both the subgraphs to the IR graphs
            #    2. Select the appropriate subgraph output based on the condition value using ElementwiseTernaryOp

            # Add the constants of the subgraphs
            self.is_ternary = True
            self._add_graph_initializers(params['then_branch'], converter_context)
            self._add_graph_initializers(params['else_branch'], converter_context)
            # Add then branch to model
            model = onnx.helper.make_model(params['then_branch'])
            for then_node in params['then_branch'].node:
                # If the subgraph output is the same as the main graph If op output,
                # we need to modify the output name
                for j in range(len(then_node.output)):
                    if then_node.output[j] in src_op.output:
                        then_node.output[j] += '_subgraph_then'
                self._add_onnx_op(then_node, converter_context, model)

            # Store output of then branch
            then_output_names = []
            for i in range(len(params['then_branch'].output)):
                then_output_name = params['then_branch'].output[i].name
                if then_output_name in src_op.output:
                    # If the subgraph output is the same as the main graph If op output,
                    # we need to modify the output name
                    then_output_name += '_subgraph_then'
                    self.then_output_is_modified = True
                then_output_names.append(then_output_name)
            then_branch_output = then_output_names

            # Add else branch to model
            model = onnx.helper.make_model(params['else_branch'])
            for else_node in params['else_branch'].node:
                # If the subgraph output is the same as the main graph If op output,
                # we need to modify the output name
                for j in range(len(else_node.output)):
                    if else_node.output[j] in src_op.output:
                        else_node.output[j] += '_subgraph_else'
                self._add_onnx_op(else_node, converter_context, model)

            # Store output of else branch
            else_output_names = []
            for i in range(len(params['else_branch'].output)):
                else_output_name = params['else_branch'].output[i].name
                if else_output_name in src_op.output:
                    # If the subgraph output is the same as the main graph If op output,
                    # we need to modify the output name
                    else_output_name += '_subgraph_else'
                    self.else_output_is_modified = True
                else_output_names.append(else_output_name)
            else_branch_output = else_output_names

            # The previous 2 outputs and condition input together form the inputs for the translated ElementwiseSelect op
            condition_input = list(map(str, src_op.input))
            if len(then_branch_output) != len(else_branch_output):
                raise ValueError("For If op {}, then and else branch must have same number of outputs."
                                 "Received {} for then branch and {} for else branch"
                                 .format(src_op.name, len(then_branch_output), len(else_branch_output)))

            # For If op with dynamic condition, and N outputs per branch,
            # we will have (2*N + 1) entries in self.input_names.
            # [cond_input, then_out_0, then_out_1, ..., then_out_(N-1), else_out_0, ..., else_out_(N-1)]
            # then_out_i is the i^th output of the then branch, and else_out_i is the i^th output of
            # the else branch.
            # We need to form N ElementwiseTernary ops with inputs as: [cond, then_out_0, else_out_0],
            # [cond, then_out_1, else_out_1], ... [cond, then_out_(N-1), else_out_(N-1)]
            self.input_names = condition_input + then_branch_output + else_branch_output
            translated_ops = {}
            for i in range(len(then_branch_output)):
                translated_ops[i] = op_adapter.ElementwiseTernaryOp(name=src_op.name + '_' + str(i),
                                                                    eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SELECT)
            return translated_ops

    def extract_input_names(self, src_op, converter_context):
        return self.input_names

OnnxTranslations.register_translation(OnnxIfTranslation(),
                                      converter_type('If', 'onnx'))


# ------------------------------------------------------------------------------
#   InstanceNormalization
# ------------------------------------------------------------------------------
class OnnxInstanceNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('InstanceNormalization', [1, 6, 22])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        bn_op_name = graph.naming_policy.get_op_name(node_op)
        node_op.name = bn_op_name

        src_input_names = list(map(str, src_op.input))
        weights_name = src_input_names[1]
        bias_name = src_input_names[2]
        weights_constant_op = self.fetch_constant_op(weights_name, converter_context, prunable=False, fail_if_dynamic=False)
        bias_constant_op = self.fetch_constant_op(bias_name, converter_context, prunable=False, fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(weights_name):
            weight_node = graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add_src_op_info(weights_name, None, weight_node.output_names[0])
            self.insert_constant_trace_info(weights_name, weight_node, converter_context)
        if bias_constant_op and not graph.has_buffer(bias_name):
            bias_node = graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add_src_op_info(bias_name, None, bias_node.output_names[0])
            self.insert_constant_trace_info(bias_name, bias_node, converter_context)
        input_names.append(weights_name)
        input_names.append(bias_name)

        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, converter_context)
        return node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        return op_adapter.InstanceNormOp(src_op.name,
                                         epsilon=params.epsilon,
                                         mode=ir_graph.QNN_OP_INSTANCE_NORM_MODE_MU_SIGMA,
                                         region=ir_graph.QNN_OP_INSTANCE_NORM_REGION_ACROSS_SPATIAL)

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxInstanceNormalizationTranslation(),
                                      converter_type('InstanceNormalization', 'onnx'),
                                      'instancenorm',
                                      op_adapter.InstanceNormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Inverse
# ------------------------------------------------------------------------------
#    - Finds inverse of a matrix
#    - Matrices of size 2x2 are supported
#    - Input dimensions:
#      # X - [..., 2, 2]
#    - Output dimensions:
#      # Y - [..., 2, 2]
#    - Computes inverse of each of the 2x2 matrices
# ------------------------------------------------------------------------------
class OnnxInverseTranslation(OnnxTranslationBase):
    def __init__(self):
        super(OnnxInverseTranslation, self).__init__()
        self.register_op_schema('Inverse', [1])

    def add_op(self, src_op, context):
        graph = context.ir_graph
        op = self.extract_parameters(src_op, context)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, context)
        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        input_buffer = graph.get_buffer(input_names[0])
        buf_shape = list(input_buffer.shape)
        if len(buf_shape) < 2:
            raise ValueError("For op {} the input rank must be at least 2, got rank {}"
                             .format(src_op.name, len(buf_shape)))
        if buf_shape[-1] != buf_shape[-2]:
            raise ValueError("For op {} the input must be a square matrix or a batch of square matrices,"
                             "got {}x{} matrix instead".format(src_op.name, buf_shape[-2], buf_shape[-1]))
        if buf_shape[-1] != 2:
            raise RuntimeError("For inverse op {}, only 2x2 matrices are supported, but got {}x{} matrix"
                               .format(src_op.name, buf_shape[-1], buf_shape[-1]))
        return op_adapter.InverseOp(name=src_op.name)

    def extract_input_names(self, src_op, converter_context):
        return list(map(str, src_op.input))

    def extract_output_names(self, src_op, converter_context):
        return list(map(str, src_op.output))

OnnxTranslations.register_translation(OnnxInverseTranslation(),
                                      converter_type('Inverse', 'onnx'),
                                      'inverse',
                                      op_adapter.InverseOp.TRANSLATION_KEY)

# ------------------------------------------------------------------------------
#   LayerNormalization
# ------------------------------------------------------------------------------
class OnnxLayerNormTranslation(OnnxTranslationBase):
    def __init__(self):
        super(OnnxLayerNormTranslation, self).__init__()
        self.register_op_schema('LayerNormalization', [17])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        data_constant_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        if data_constant_op:
            const_node = graph.add(data_constant_op, [], [input_names[0]], axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            graph.add_src_op_info(const_node.op.name, None, const_node.output_names[0])
            self.insert_constant_trace_info(input_names[0], const_node, converter_context)

        node_op = self.extract_parameters(src_op, converter_context)
        input_buf = graph.get_buffer(input_names[0])

        # Bias name is optional so extract bias name if 3 inputs given
        if len(input_names)>2:
            _, weights_name, bias_name = input_names
        else:
            _, weights_name = input_names
            bias_name = None

        node_op_axes = node_op.axes
        gamma_beta_shape = [input_buf.shape[axis] for axis in node_op_axes]

        weights_constant_op = self.fetch_constant_op(weights_name, converter_context, prunable=False, fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(weights_name):
            if tuple(gamma_beta_shape) != weights_constant_op.tensor.shape:
                if self.validate_gamma_beta_shape(weights_constant_op, node_op_axes, input_buf):
                    weights_constant_op.tensor = np.full(gamma_beta_shape, weights_constant_op.tensor)
                else:
                    raise ValueError("Invalid shape {} for weight tensor: Expected weight shape to be broadcastable to shape {}"
                                     .format(weights_constant_op.tensor.shape, tuple(gamma_beta_shape)))
            weight_node = graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add_src_op_info(weights_name, None, weight_node.output_names[0])
            self.insert_constant_trace_info(weights_name, weight_node, converter_context)

        if bias_name:
            bias_constant_op = self.fetch_constant_op(bias_name, converter_context, prunable=False, fail_if_dynamic=False)
            if bias_constant_op and not graph.has_buffer(bias_name):
                if tuple(gamma_beta_shape) != bias_constant_op.tensor.shape:
                    if self.validate_gamma_beta_shape(bias_constant_op, node_op_axes, input_buf):
                        bias_constant_op.tensor = np.full(gamma_beta_shape,  bias_constant_op.tensor)
                    else:
                        raise ValueError("Invalid shape {} for bias tensor: Expected bias shape to be broadcastable to shape {}"
                                         .format( bias_constant_op.tensor.shape, tuple(gamma_beta_shape)))
                bias_node = graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
                graph.add_src_op_info(bias_name, None, bias_node.output_names[0])
                self.insert_constant_trace_info(bias_name, bias_node, converter_context)
        if(len(output_names) > 1):
            log_warning("QNN Layernorm Op: {} can produce one output. The other 2 optional outputs {}, {} of"
            " Layernorm op are discarded. Conversion may fail if {} or {} have consumers.", src_op.name, output_names[1],
            output_names[2], output_names[1], output_names[2])
        node = graph.add(node_op, input_names, output_names[0])
        self.add_src_op_info(node_op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, converter_context)
        return node

    def validate_gamma_beta_shape(self, op, axes, input_buf):
        gamma_beta_rank = len(op.tensor.shape)
        layernorm_input_shape = input_buf.shape
        layernorm_input_rank = input_buf.rank()
        if gamma_beta_rank > layernorm_input_rank:
            return False
        one_extended_buffer_shape = [1] * (layernorm_input_rank -  gamma_beta_rank)  + list(op.tensor.shape)
        if any(dim != 1 for i, dim in enumerate(one_extended_buffer_shape) if i not in axes):
            # shape should be 1 for non-axes dimensions
            return False
        else:
            # make sure that squeezed_shape is broadcastable to input_shape on axes
            squeezed_shape = [dim for i, dim in enumerate(one_extended_buffer_shape) if i in axes]
            final_shape = [dim for i, dim in enumerate(layernorm_input_shape) if i in axes]
            return translation_utils.unidirectional_broadcastable(final_shape, squeezed_shape)

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        input_buf = graph.get_buffer(input_names[0])
        rank = input_buf.rank()
        axis = rank - 1
        if 'axis' in params:
            # Make the negative axis to a positive value.
            # axis=-1 with input[0].rank()=4, then axis will be 3
            axis = params.axis + rank if params.axis < 0 else params.axis
        # ONNX: axis means the first dim to normalize, QNN: axes means the normalization dims.
        # axis = [2] + input[0].rank = 4 -> axes = [2, 3]
        return op_adapter.LayerNormOp(src_op.name,
                                      epsilon=params.epsilon,
                                      axes=[i for i in range(axis, rank)])


OnnxTranslations.register_translation(OnnxLayerNormTranslation(),
                                      converter_type('LayerNormalization', 'onnx'),
                                      'layernorm',
                                      op_adapter.LayerNormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   MaxRoiPool
# ------------------------------------------------------------------------------
class OnnxMaxRoiPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('MaxRoiPool', [1, 22])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        if len(ops) > 1:
            roi_node = graph.add(ops[0], [], input_names[1])
            self.insert_constant_trace_info(input_names[1], roi_node, converter_context)
        last_node = graph.add(ops[-1], input_names, output_names)
        self.insert_default_trace_info(src_op, last_node, converter_context)

        # add src op info for roi pool operation
        self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        input_names = list(map(str, src_op.input))
        input_buf = graph.get_buffer(input_names[0])
        ops = []
        roi_name = input_names[1]

        if not graph.has_buffer(roi_name):
            roi_values = converter_context.weights.fetch(roi_name, prunable=False)
            roi_tensor = roi_values.astype(np.float32)
            roi_shape = roi_tensor.shape
            ops.append(op_adapter.ConstantOp(roi_name, roi_tensor))
        else:
            roi_shape = graph.get_buffer(roi_name).shape

        output_shape = [roi_shape[0],
                        input_buf.shape[1],
                        params.pooled_shape[0],
                        params.pooled_shape[1]]

        ops.append(op_adapter.RoiPoolingOp(src_op.name,
                                           output_shape,
                                           pooled_size_h=params.pooled_shape[0],
                                           pooled_size_w=params.pooled_shape[1],
                                           spatial_scale=params.spatial_scale))
        return ops

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]


OnnxTranslations.register_translation(OnnxMaxRoiPoolTranslation(),
                                      converter_type('MaxRoiPool', 'onnx'),
                                      op_adapter.RoiPoolingOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Prelu, LeakyRelu
# ------------------------------------------------------------------------------
# Also handles LeakyRelu as a bonus.
class OnnxPreluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('PRelu', [1, 6, 7, 9])
        self.register_op_schema('LeakyRelu', [1, 6])
        self.coeff = None
        self.reshape_inserted = False

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.reshape_inserted = False
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        if self.reshape_inserted:
            input_names[1] = list(graph.get_buffer(input_names[1]).consumers)[0].output_names[0]
        prelu_op_name = graph.naming_policy.get_op_name(node_op)
        node_op.name = prelu_op_name
        if str(src_op.op_type) == 'LeakyRelu':
            coeff_name = prelu_op_name + "_coeff"
            input_names.append(coeff_name)
        else:
            coeff_name = input_names[1]

        if not graph.has_buffer(coeff_name):
            coeff_constant_op = op_adapter.ConstantOp(coeff_name, tensor=self.coeff)
            coeff_node = graph.add(coeff_constant_op, [], [coeff_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            self.insert_trace_info([coeff_node, graph.get_buffer(coeff_name)], (prelu_op_name, TraceType.OP), converter_context)
            graph.add_src_op_info(coeff_name, None, coeff_node.output_names[0])

        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, converter_context)
        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = list(map(str, src_op.input))
        input_buf = graph.get_buffer(input_names[0])
        if str(src_op.op_type) == 'LeakyRelu':
            params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type),
                                        validate=True)
            slope = np.array([params.alpha], dtype=np.float32)
            self.coeff = slope
        else:
            rank_diff = self.calc_rank_diff(converter_context, input_names)
            self.coeff = None

            # Check for dynamic alpha input
            if graph.has_buffer(input_names[1]) and \
               not isinstance(graph.get_buffer(input_names[1]).producer.op,op_adapter.ConstantOp):
                slope = graph.get_buffer(input_names[1])
                if len(slope.shape) > 1 and rank_diff < 0:
                    # Prepending 1's to slope shape and then broadcasting to match input rank
                    pre_reshape_shape = [1] * abs(rank_diff) + list(slope.shape)
                    self.add_pre_reshape_op(src_op, converter_context, input_names[1], pre_reshape_shape)
                    self.reshape_inserted = True
            else: # Static alpha input
                slope = self.calc_static_slope(converter_context, input_names, rank_diff)
                # If coef buffer is not in graph, update the tensor shape in-place to match input rank
                if not graph.has_buffer(input_names[1]):
                    slope_op = op_adapter.ConstantOp(name=input_names[1], tensor=slope)
                    # Slope rank can be 1 or equal to input rank after calc_static_slope
                    # Assign slope axis format as ANY if slope rank is 1, otherwise as input axis format
                    slope_axis_format = input_buf.axis_format if input_buf.rank() == len(slope.shape) else AxisTracker.AxisFormat.ANY
                    slope_node = graph.add(slope_op, [], [input_names[1]], axis_formats=[slope_axis_format])
                    self.insert_constant_trace_info(slope_op.name, slope_node, converter_context)
                # Add ReshapeOp if coef buf is present in graph and its rank is neither 1 nor equal to input rank
                # ReshapeOp to reshape from coef_buf.shape to slope.shape
                else:
                    coef_buf = graph.get_buffer(input_names[1])
                    if len(coef_buf.shape.dims) > 1 and rank_diff < 0:
                        self.add_pre_reshape_op(src_op, converter_context, input_names[1], slope.shape)
                        self.reshape_inserted = True
                self.coeff = slope

        return op_adapter.PreluOp(src_op.name)

    def calc_rank_diff(self, converter_context, input_names):
        # Utility function to calculate rank diff between input and alpha
        rank_diff = 0
        graph = converter_context.ir_graph
        input_buf = graph.get_buffer(input_names[0])
        coef_name = input_names[1]
        slope_op = self.fetch_constant_op(coef_name, converter_context,
                                          fail_if_dynamic=False,
                                          fail_if_not_found=True)
        if slope_op is not None:
            rank_diff = len(slope_op.tensor.shape) - len(input_buf.shape)
        else:
            coef_buf = graph.get_buffer(coef_name)
            rank_diff = len(coef_buf.shape) - len(input_buf.shape)
        return rank_diff

    def calc_static_slope(self, converter_context, input_names, rank_diff):
        # Utility function to calculate broadcasted slope, as slope rank
        # must be either 1 or equal to input rank.
        graph = converter_context.ir_graph
        input_buf = graph.get_buffer(input_names[0])
        slope_op = self.fetch_constant_op(input_names[1], converter_context,
                                          fail_if_dynamic=False,
                                          fail_if_not_found=True)
        slope = slope_op.tensor
        if slope.shape == (1,):
            # only the slope has rank==1 and has only 1 value then we can do this broadcast
            slope = np.ones(input_buf.shape[1], dtype=np.float32) * slope[0]
        else:
            if rank_diff < 0 and len(slope.shape) > 1:
                # Prepending 1's to slope shape and then broadcasting to match input rank
                slope_shape = [1] * abs(rank_diff) + list(slope.shape)
                slope = np.broadcast_to(slope, slope_shape)
            slope = np.require(slope, dtype=np.float32)
        return slope

    def add_pre_reshape_op(self, src_op, converter_context, slope_name, slope_shape):
        graph = converter_context.ir_graph

        reshape_op = op_adapter.ReshapeOp(src_op.name+'_alpha_reshape',
                                          shape=slope_shape)
        reshape_op_input = slope_name
        reshape_op_output = reshape_op.name
        graph.add(reshape_op, [reshape_op_input], [reshape_op_output],
                  axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])


OnnxTranslations.register_translation(OnnxPreluTranslation(),
                                      converter_type('Prelu', 'onnx'),
                                      converter_type('LeakyRelu', 'onnx'),
                                      op_adapter.PreluOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Lrn
# ------------------------------------------------------------------------------
class OnnxLrnTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LRN', [1, 13])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.LrnOp(src_op.name,
                                alpha=params.alpha / params.size,
                                beta=params.beta,
                                bias=params.bias,
                                radius=int((params.size-1)/2),
                                region=ir_graph.QNN_OP_LRN_REGION_ACROSS_CHANNEL)


OnnxTranslations.register_translation(OnnxLrnTranslation(),
                                      converter_type('LRN', 'onnx'),
                                      op_adapter.LrnOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   NonMaxSuppression
# ------------------------------------------------------------------------------
class OnnxNonMaxSuppressionTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('NonMaxSuppression', [10, 11])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        if params.center_point_box:
            raise ValueError("NonMaxSuppression op {}: Only support minmax representation (y1,x1,y2,x2)".format(src_op.name))

        # Require static parameters
        input_names = list(map(str, src_op.input))
        if (len(input_names) > 2 and
            any([graph.has_buffer(buf) and not self.fetch_constant_op(buf, converter_context)
                 for buf
                 in input_names[2:]
                ]
               )
           ):
            raise ValueError(
                ('NonMaxSuppression op {}: Only supports static parameters for '
                'max_output_boxes_per_class, iou_threshold, score_threshold'.format(src_op.name))
            )
        max_output_boxes_per_class, iou_threshold, score_threshold = self.fetch_const_input(src_op, converter_context)
        # If max_output_boxes_per_class is negative, it will be set to input number of boxes
        if max_output_boxes_per_class < 0:
            max_output_boxes_per_class = graph.get_buffer(input_names[0]).shape[1]
        return op_adapter.NonMaxSuppressionOp(
            name=src_op.name,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_boxes_selected=max_output_boxes_per_class,
        )

    def fetch_const_input(self, src_op, converter_context):
        # per onnx spec
        max_output_boxes_per_class = 0
        iou_threshold = 0
        score_threshold = 0

        # Handle optional parameters
        input_names = list(map(str, src_op.input))
        if len(input_names) > 2 and input_names[2]:
            max_output_boxes_per_class = self.fetch_constant_op(input_names[2], converter_context).tensor
            max_output_boxes_per_class = int(max_output_boxes_per_class.item(0))
        if len(input_names) > 3 and input_names[3]:
            iou_threshold = self.fetch_constant_op(input_names[3], converter_context).tensor
            iou_threshold = iou_threshold.item(0)
        if len(input_names) > 4 and input_names[4]:
            score_threshold = self.fetch_constant_op(input_names[4], converter_context).tensor
            score_threshold = score_threshold.item(0)

        return max_output_boxes_per_class, iou_threshold, score_threshold

    def extract_input_names(self, src_op, converter_context):
        actual_input_name = []
        for inp in map(str, src_op.input):
            if not (inp and converter_context.weights.has(inp)):
                actual_input_name.append(inp)
        return actual_input_name

    def extract_output_names(self, src_op, converter_context):
        # align with QNN's second output to specify the valid number of selected indices
        output_name = str(src_op.output[0])
        return [output_name, output_name + '_valid_num_selected_indices']


OnnxTranslations.register_translation(OnnxNonMaxSuppressionTranslation(),
                                      converter_type('NonMaxSuppression', 'onnx'),
                                      op_adapter.NonMaxSuppressionOp.TRANSLATION_KEY)

# ------------------------------------------------------------------------------
#   RMS Normalization
# ------------------------------------------------------------------------------
class OnnxRMSNormTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('RMSNormalization', [23])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type), validate=True)
        inputs = src_op.input
        input_names = self.extract_input_names(src_op, converter_context)
        input_buf = graph.get_buffer(input_names[0])
        rank = input_buf.rank()

        # axis parameter (default:-1)
        axis = params.get('axis', -1)
        if 'axis' in params:
        # Make the negative axis to a positive value.
        # axis=-1 with input[0].rank()=4, then axis will be 3
            axis = params.axis + rank if params.axis < 0 else params.axis

        # The epsilon value to use to avoid division by zero. default is 1e-06
        epsilon = params.get('epsilon', 1e-06)

        # stash_type. default is 1
        stash_type  = params.get('stash_type', 1)
        if stash_type not in [1]:
            log_warning(f"RMSNormalization op {src_op.name}: Unsupported stash_type value {stash_type}. "
                        f"Expected 1 (float). Using default value 1.")
            stash_type = 1

        #fetch scale tensor
        scale_op = self.fetch_constant_op(inputs[1], converter_context, prunable=False,
                                          fail_if_dynamic=False)

        weights_name = input_names[1]
        if scale_op and not graph.has_buffer(input_names[1]):
            weight_node = graph.add(scale_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add_src_op_info(weights_name, None, weight_node.output_names[0])
            self.insert_constant_trace_info(weights_name, weight_node, converter_context)

        return op_adapter.RMSNormOp(src_op.name,
                                            axes=[i for i in range(axis, rank)],
                                            epsilon=epsilon,
                                            stash_type=stash_type)

    def extract_input_names(self, src_op, converter_context):

        input_names = [str(input_name) for input_name in src_op.input]

        return input_names

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxRMSNormTranslation(),
                                      converter_type('RMSNormalization', 'onnx'))

# ------------------------------------------------------------------------------
#   ROI Align
# ------------------------------------------------------------------------------
class OnnxRoiAlignTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('RoiAlign', [10, 16, 22])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        if len(ops) > 1:
            batch_indices_node = graph.add(ops[0], [], input_names[2])
            # add src_op info for added constant op
            graph.add_src_op_info(batch_indices_node.op.name, None,
                                  batch_indices_node.output_names[0])
            self.insert_constant_trace_info(input_names[2], batch_indices_node, converter_context)

        last_node = graph.add(ops[-1], input_names, output_names)
        # add src op info for roi align operation
        self.add_src_op_info(last_node.op.name, src_op, graph)
        self.insert_default_trace_info(src_op, last_node, converter_context)

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        params = extract_attributes(src_op, schema=schema)
        # coordinate_transformation_mode is a new attribute added in version 16.
        # It's default value is half_pixel but we only provide backward compatibility for now.
        if 'coordinate_transformation_mode' in params and params.coordinate_transformation_mode != "output_half_pixel":
            raise ValueError("ERROR: Unsupported value {} for coordinate_transformation_mode attribute for {} op {}".format(
                             params.coordinate_transformation_mode, src_op.op_type, src_op.name))
        ops = []
        indices_name = str(src_op.input[2])
        # If the input is stored as weights we need to create a const node
        if not graph.has_buffer(indices_name):
            indices = self.fetch_constant_op(indices_name, converter_context,
                                             prunable=False, dtype=np.int32)
            ops.append(indices)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op

            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices_op.tensor = indices_op.tensor.astype(np.int32)
        log_assert(params.spatial_scale != 0, "Unsupported value of 0 for spatial_scale attribute for {} op.", src_op.name)
        ops.append(op_adapter.RoiAlignOp(src_op.name,
                                         pooled_size_h=params.output_height,
                                         pooled_size_w=params.output_width,
                                         spatial_scale=(1.0/params.spatial_scale),
                                         sampling_ratio=params.sampling_ratio,
                                         mode=params.mode))
        return ops


OnnxTranslations.register_translation(OnnxRoiAlignTranslation(),
                                      converter_type('RoiAlign', 'onnx'),
                                      op_adapter.RoiAlignOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   SpaceToDepth
# ------------------------------------------------------------------------------
class OnnxSpaceToDepthTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('SpaceToDepth', [1, 13])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        return op_adapter.SpaceToDepthOp(
            name=src_op.name,
            block_size=[params.blocksize] * 2
        )

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxSpaceToDepthTranslation(),
                                      converter_type('SpaceToDepth', 'onnx'),
                                      op_adapter.SpaceToDepthOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   DepthToSpace
# ------------------------------------------------------------------------------
class OnnxDepthToSpaceTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('DepthToSpace', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        SUPPORTED_DEPTHTOSPACE_MODES = {'DCR': ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR,
                                        'CRD': ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_CRD}
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        if not 'mode' in params:
            mode = 'DCR'
        elif params.mode not in SUPPORTED_DEPTHTOSPACE_MODES:
            raise ValueError("Unsupported depthtospace mode {} of {} op {}".format(params.mode,
                                                                                   src_op.op_type,
                                                                                   src_op.name))
        else:
            mode = params.mode
        return op_adapter.DepthToSpaceOp(name=src_op.name,
                                         block_size=[params.blocksize] * 2,
                                         mode=SUPPORTED_DEPTHTOSPACE_MODES[mode])

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxDepthToSpaceTranslation(),
                                      converter_type('DepthToSpace', 'onnx'),
                                      op_adapter.DepthToSpaceOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   QLinearConv Op
# ------------------------------------------------------------------------------
class OnnxQLinearConvTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('QLinearConv', [10])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op, enc = self.extract_parameters(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        node = graph.add(op, self.input_names, output_names)
        self.insert_default_trace_info(src_op, node, converter_context)
        graph.add_quantization_params(node.op.name,
                                      param_encodings=enc["params_enc"],
                                      output_encodings=enc["outputs"])
        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        inputs = src_op.input
        outputs = src_op.output
        self.input_names = self.extract_input_names(src_op, converter_context)

        #Raise error if len(inputs) not greater than or equal to  8
        log_assert(len(inputs) >= 8,
                   "Invalid number of inputs to QLinearConv Op {}. Expected at least 8,"
                   "but got {}".format(src_op.name, len(inputs)))

        if not graph.has_buffer(inputs[0]) and converter_context.weights.has(inputs[0]):
            input_op = self.fetch_constant_op(inputs[0], converter_context, prunable=False,
                                                                    fail_if_dynamic=False)
            if input_op:
                input_node = graph.add(input_op, [], [inputs[0]],
                                               axis_formats=[AxisTracker.AxisFormat.NCS])
                self.insert_constant_trace_info(inputs[0], input_node, converter_context)

        input_buffer = graph.get_buffer(str(inputs[0]))

        if input_buffer.rank() == 5:
            default_dilations = [1, 1, 1]
            default_strides = [1, 1, 1]
            default_pads = [0, 0, 0, 0, 0, 0]
        else:
            default_dilations = [1, 1]
            default_strides = [1, 1]
            default_pads = [0, 0, 0, 0]

        params = extract_attributes(src_op, attr_infos=
                         [('dilations', 'li', default_dilations),
                          ('strides', 'li', default_strides),
                          ('kernel_shape', 'li'),
                          ('pads', 'li', default_pads),
                          ('auto_pad', 's', 'NOTSET'),
                          ('group', 'i', 1)], schema=self.op_schema(), validate=True)

        #Retrieve input scale
        input_scale_op = self.fetch_constant_op(inputs[1], converter_context, prunable=False,
                                                                        fail_if_dynamic=False)
        if input_scale_op is not None:
            input_scale = np.array(input_scale_op.tensor).astype(np.float32)
        else:
            raise ValueError("No input scale provided for op: {} of type: {}"
                             .format(src_op.name, src_op.op_type))

        #Retrieve input zero point
        input_zero_point = self.fetch_constant_op(inputs[2], converter_context, prunable=False,
                                                                         fail_if_dynamic=False)
        if input_zero_point is not None:
            input_zp = np.array(input_zero_point.tensor)
        else:
            raise ValueError("No input zero point provided for op: {} of type: {}"
                             .format(src_op.name, src_op.op_type))

        #Retrieve input encoding and add it to graph as quantization params
        input_enc, _ = get_encoding(inputs[0], input_scale, input_zp)
        graph.add_quantization_params(graph.get_buffer(inputs[0]).producer.op.name,
                                      output_encodings = input_enc)

        #Retrieve weights scale
        weights_scale_op = self.fetch_constant_op(inputs[4], converter_context, prunable=False,
                                                                         fail_if_dynamic=False)
        if weights_scale_op is not None:
            weights_scale = np.array(weights_scale_op.tensor).astype(np.float32)
        else:
            raise ValueError("No weights scale provided for op: {} of type: {}"
                             .format(src_op.name, src_op.op_type))

        #Retrieve weights zero point
        weights_zero_point = self.fetch_constant_op(inputs[5], converter_context, prunable=False,
                                                                           fail_if_dynamic=False)
        if weights_zero_point is not None:
            weights_zp = np.array(weights_zero_point.tensor)
        else:
            raise ValueError("No weights zero point provided for op: {} of type: {}"
                             .format(src_op.name, src_op.op_type))

        weights_enc, _ = get_encoding(inputs[3],  weights_scale, weights_zp)
        param_list = []
        param_list.append(weights_enc)

        # Extract weights and add them as ConstantOp inputs to the Conv2dOp
        weights_constant_op = self.fetch_constant_op(inputs[3], converter_context, prunable=False,
                                                                            fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(inputs[3]):
            #Dequantizing the weights
            w_split = np.array_split(weights_constant_op.tensor, len(weights_scale))
            split_qunatlist =[]
            for idx in range(len(w_split)):
                dequnat_list = weights_scale[idx] * (w_split[idx] - weights_zp[idx])
                split_qunatlist.append(dequnat_list)
            dequant_array = np.concatenate(split_qunatlist, axis=0)
            weights_constant_op.tensor =  dequant_array

            if params.kernel_shape:
                log_assert(tuple(params.kernel_shape) == weights_constant_op.tensor.shape[2:],
                     code_to_message.get_error_message("ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS"))
            log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(src_op.name,
                                                                weights_constant_op.tensor.shape))
            if len(weights_constant_op.tensor.shape) == 3:
                weights_node = graph.add(weights_constant_op, [], [inputs[3]],
                                                  axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            elif len(weights_constant_op.tensor.shape) == 4:
                weights_node = graph.add(weights_constant_op, [], [inputs[3]],
                                                  axis_formats=[AxisTracker.AxisFormat.OIHW])
            elif len(weights_constant_op.tensor.shape) == 5:
                weights_node = graph.add(weights_constant_op, [], [inputs[3]],
                                                  axis_formats=[AxisTracker.AxisFormat.OIDHW])
            else:
                raise NotImplementedError(
                    "Convolutional Tensor doesn't support {}-D translations for conv op {}."
                     .format(len(weights_constant_op.tensor.shape), src_op.name))

            self.insert_constant_trace_info(inputs[3], weights_node, converter_context)
            weights_enc['name'] = 'weights'

        elif graph.has_buffer(inputs[3]):
            weight_buffer = graph.get_buffer(inputs[3])
            # Properly set the axis format if the buffer already exists in the graph
            if len(weight_buffer.shape) == 3:
                graph.get_buffer(inputs[3]).axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif len(weight_buffer.shape) == 4:
                graph.get_buffer(inputs[3]).axis_format = AxisTracker.AxisFormat.OIHW
            elif len(weight_buffer.shape) == 5:
                graph.get_buffer(inputs[3]).axis_format = AxisTracker.AxisFormat.OIDHW
            else:
                raise NotImplementedError(
                    "Convolutional Tensor doesn't support {}-D translations for conv op {}."
                    .format(len(weights_constant_op.tensor.shape), src_op.name))

            #Handle static weight shared by multiple other ops
            if isinstance(weight_buffer.producer.op, op_adapter.ConstantOp):
                producer_op =  weight_buffer.producer.op
                weight_tensor = producer_op.tensor
                weight_tensor_copy = np.copy(weight_tensor)
                weight_op_name =  producer_op.name + "_kernel_weight"
                weight_const_op = op_adapter.ConstantOp(weight_op_name,
                                                        tensor=weight_tensor_copy)
                weight_const_node = graph.add(weight_const_op, [], [weight_op_name],
                                        axis_formats=[graph.get_buffer(inputs[3]).axis_format])
                self.insert_constant_trace_info(weight_op_name, weight_const_node, converter_context)
                weights_enc['name'] =  weight_op_name
                graph.set_overridden_encoding(weight_op_name, weights_enc)
                self.input_names[1] = weight_op_name

        #Retrieve output scale
        output_scale_op = self.fetch_constant_op(inputs[6], converter_context, prunable=False,
                                                                        fail_if_dynamic=False)
        if output_scale_op is not None:
            output_scale = np.array(output_scale_op.tensor).astype(np.float32)
        else:
            raise ValueError("No output scale provided for op: {} of type: {}"
                             .format(src_op.name, src_op.op_type))

        #Retrieve output zero point
        output_zero_point = self.fetch_constant_op(inputs[7], converter_context, prunable=False,
                                                                          fail_if_dynamic=False)
        if output_zero_point is not None:
            output_zp = np.array(output_zero_point.tensor)
        else:
            raise ValueError("No output zero point provided for op: {} of type: {}"
                             .format(src_op.name, src_op.op_type))

        #Retrieve the output encodings
        output_name = str(outputs[0])
        output_enc, _ = get_encoding(output_name, output_scale, output_zp)

        bias_op_name = None
        if len(inputs) > 8:
            bias_op_name = str(inputs[8])
            bias_constant_op = self.fetch_constant_op(inputs[8], converter_context, prunable=False,
                                                                            fail_if_dynamic=False)
            #Retrieve zero point and fetch the encodings
            #bias scale = input_scale * weights_scale
            #bias zero point will be zero and same shape as that of bias scale
            bias_scale = np.multiply(input_scale, weights_scale)
            bias_zp = np.zeros(bias_scale.shape).astype(np.int32)
            bias_enc, _ = get_encoding(inputs[8], bias_scale, bias_zp)
            param_list.append(bias_enc)

            if bias_constant_op and not graph.has_buffer(inputs[8]):
                #Dequantizing the bias
                bias_dequant = bias_scale * (bias_constant_op.tensor - bias_zp)
                bias_constant_op.tensor = np.array(bias_dequant).astype(np.float32)
                log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(src_op.name,
                                                                  bias_constant_op.tensor.shape))
                bias_node = graph.add(bias_constant_op, [], [inputs[8]],
                                                       axis_formats=[AxisTracker.AxisFormat.ANY])
                self.insert_constant_trace_info(inputs[8], bias_node, converter_context)
                bias_enc['name'] = 'bias'

            elif graph.has_buffer(inputs[8]):
                #Handle constant bias shared by multiple other ops
                bias_producer_op =  graph.get_buffer(inputs[8]).producer.op
                bias_tensor = bias_producer_op.tensor
                bias_tensor_copy = np.copy(bias_tensor)
                bias_op_name =  bias_producer_op.name + "_kernel_bias"
                bias_const_op = op_adapter.ConstantOp(bias_op_name, tensor=bias_tensor_copy)
                bias_const_node = graph.add(bias_const_op, [], [bias_op_name],
                                            axis_formats=[AxisTracker.AxisFormat.ANY])
                self.insert_constant_trace_info(bias_op_name, bias_const_node, converter_context)
                bias_enc['name'] =  bias_op_name
                graph.set_overridden_encoding(bias_op_name, bias_enc)
                self.input_names[2] = bias_op_name

        op_enc = {"params_enc": param_list, "outputs": output_enc}

        # Extract the remaining attributes and calculate the padding size
        padding_size_strategy = IRPaddingStrategies.py_to_c[extract_padding_mode(params.auto_pad,
                                                                                   src_op.name)]
        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis

        # set the input padding size
        input_shape = graph.get_buffer(inputs[0]).shape
        weights_shape = graph.get_buffer(inputs[3]).shape

        if input_buffer.rank() == 3:
            pad_x = ir_graph.ConvOp.calc_conv_padding_size(input_shape[2],
                                                           weights_shape[2],
                                                           params.dilations[0],
                                                           params.strides[0],
                                                           padding_size_strategy,
                                                           params.pads)

            # Handle marking this Convolution as a DepthwiseConvolution
            num_input_channels = graph.src_axis_order.extract_1d_spatial_dims(input_shape)[-1]
            num_output_channels = graph.src_axis_order.extract_conv1d_weights_dims(weights_shape)[-1]
            convolution_class = op_adapter.Conv1dOp
            if params.group == num_input_channels and num_input_channels == num_output_channels:
                convolution_class = op_adapter.DepthwiseConv1dOp
            self.convolution_class = convolution_class
            return convolution_class(src_op.name,
                                     bias_op_name=bias_op_name,
                                     padx_before=pad_x[0],
                                     padx_after=pad_x[1],
                                     padding_size_strategy=padding_size_strategy,
                                     stridex=params.strides[0],
                                     dilationx=params.dilations[0],
                                     groups=params.group,
                                     data_layout=AxisTracker.AxisFormat.NCF), op_enc

        elif input_buffer.rank() == 5:
            if num_input_pads != 3:
                raise ValueError("For Conv3d op {} the number of pads number shall be 3, instead of {}"
                                 .format(src_op.name, num_input_pads))

            pad_z = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[2],
                                                             weights_shape[2],
                                                             params.dilations[0],
                                                             params.strides[0],
                                                             padding_size_strategy,
                                                             [params.pads[0], params.pads[num_input_pads]])

            pad_y = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[3],
                                                             weights_shape[3],
                                                             params.dilations[1],
                                                             params.strides[1],
                                                             padding_size_strategy,
                                                             [params.pads[1], params.pads[1 + num_input_pads]])

            pad_x = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[4],
                                                             weights_shape[4],
                                                             params.dilations[2],
                                                             params.strides[2],
                                                             padding_size_strategy,
                                                             [params.pads[2], params.pads[2 + num_input_pads]])

            # Handle marking this Convolution as a DepthwiseConvolution
            num_input_channels = graph.src_axis_order.extract_3d_spatial_dims(input_shape)[-1]
            num_output_channels = graph.src_axis_order.extract_conv3d_weights_dims(weights_shape)[-1]
            convolution_class = op_adapter.Conv3dOp
            if params.group == num_input_channels and num_input_channels == num_output_channels and params.group > 1:
                raise ValueError("Depthwise 3D convolution operation is not yet implemented")

            return convolution_class(src_op.name,
                                     bias_op_name=bias_op_name,
                                     pady_before=pad_y[0],
                                     pady_after=pad_y[1],
                                     padx_before=pad_x[0],
                                     padx_after=pad_x[1],
                                     padz_before=pad_z[0],
                                     padz_after=pad_z[1],
                                     padding_size_strategy=padding_size_strategy,
                                     stridex=params.strides[2],
                                     stridey=params.strides[1],
                                     stridez=params.strides[0],
                                     dilationx=params.dilations[2],
                                     dilationy=params.dilations[1],
                                     dilationz=params.dilations[0],
                                     groups=params.group,
                                     data_layout=AxisTracker.AxisFormat.NCDHW), op_enc

        else:
            log_assert(num_input_pads == 2,
                       code_to_message.get_error_message("ERROR_NUMBER_OF_PADS_UNSUPPORTED")
                       (src_op.name, num_input_pads))
            # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims
            pad_y = ir_graph.ConvOp.calc_conv_padding_size(input_shape[2],
                                                           weights_shape[2],
                                                           params.dilations[0],
                                                           params.strides[0],
                                                           padding_size_strategy,
                                                           [params.pads[0], params.pads[num_input_pads]])

            pad_x = ir_graph.ConvOp.calc_conv_padding_size(input_shape[3],
                                                           weights_shape[3],
                                                           params.dilations[1],
                                                           params.strides[1],
                                                           padding_size_strategy,
                                                           [params.pads[1], params.pads[1 + num_input_pads]])
            # Handle marking this Convolution as a DepthwiseConvolution
            num_input_channels = graph.src_axis_order.extract_2d_spatial_dims(input_shape)[-1]
            num_output_channels = graph.src_axis_order.extract_conv2d_weights_dims(weights_shape)[-1]
            convolution_class = op_adapter.Conv2dOp
            if params.group == num_input_channels and num_input_channels == num_output_channels:
                convolution_class = op_adapter.DepthwiseConv2dOp

            return convolution_class(src_op.name,
                                     bias_op_name=bias_op_name,
                                     pady_before=pad_y[0],
                                     pady_after=pad_y[1],
                                     padx_before=pad_x[0],
                                     padx_after=pad_x[1],
                                     padding_size_strategy=padding_size_strategy,
                                     stridex=params.strides[1],
                                     stridey=params.strides[0],
                                     dilationx=params.dilations[1],
                                     dilationy=params.dilations[0],
                                     groups=params.group,
                                     data_layout=AxisTracker.AxisFormat.NCS), op_enc

    def extract_input_names(self, src_op, converter_context):
        if len(src_op.input) > 8:
            return [src_op.input[0], src_op.input[3], src_op.input[8]]
        return [src_op.input[0], src_op.input[3]]

OnnxTranslations.register_translation(OnnxQLinearConvTranslation(),
                                      converter_type('QLinearConv', 'onnx'))


# ------------------------------------------------------------------------------
#   ThresholdedRelu
#   TODO: Add unit test for this op
# ------------------------------------------------------------------------------
class OnnxThresholdedReluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ThresholdedRelu', [10, 22])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)

        node_op.name = graph.naming_policy.get_op_name(node_op)

        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, converter_context)
        return node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)
        return op_adapter.ThresholdedReluOp(src_op.name, alpha=params.alpha)

OnnxTranslations.register_translation(OnnxThresholdedReluTranslation(),
                                      converter_type('ThresholdedRelu', 'onnx'),
                                      op_adapter.ThresholdedReluOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   RotaryEmbedding
# ------------------------------------------------------------------------------
class OnnxRotaryEmbeddingTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('RotaryEmbedding', [23])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph

        # I/O names from the source node
        input_names  = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        # If input0 is rank-3 [B, S, H_total], reshape to 4D [B, N, S, H]
        x_name = input_names[0]
        try:
            x_buf = graph.get_buffer(x_name)
        except KeyError:
            x_buf = None

        if x_buf is not None and x_buf.rank() == 3:
            # Need num_heads to split H_total -> [N, H]
            params_for_rank = extract_attributes(
                src_op,
                attr_infos=[('num_heads', 'i', 0)],
                validate=False
            )
            num_heads = int(getattr(params_for_rank, 'num_heads', 0) or 0)
            if num_heads <= 0:
                raise ValueError(
                    f"RotaryEmbedding op {src_op.name}: 'num_heads' attribute is required when "
                    f"input is 3D [batch, sequence_length, hidden_size]"
                )

            b, s, h_total = list(x_buf.shape)
            if h_total % num_heads != 0:
                raise ValueError(
                    f"RotaryEmbedding op {src_op.name}: hidden_size ({h_total}) must be divisible "
                    f"by num_heads ({num_heads}) for 3D input"
                )
            h = h_total // num_heads
            # No strict evenness requirement; caches validate with //2

            # Insert reshape: [B, S, H_total] -> [B, N, S, H]
            reshape_out = x_name + "_reshaped_4d"
            reshape_op = op_adapter.ReshapeOp(
                name=src_op.name + "_reshape_3d_to_4d",
                shape=[b, num_heads, s, h]   # [B, N, S, H]
            )
            graph.add(reshape_op, [x_name], [reshape_out])
            input_names[0] = reshape_out  # feed reshaped tensor into RoPE

        # Build parameters; validation will now see 4D input
        node_op = self.extract_parameters(src_op, converter_context)

        # Add the RotaryEmbedding operation
        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, converter_context)
        return node

    def extract_parameters(self, src_op, converter_context):
        """
        Supports 3D input by relying on add_op() to reshape to 4D first.
        Valid final input is 4D: [B, N, S, H].
        Cache validation:
          - Without position_ids: expect caches shaped [B, S, rotary_dim // 2]
          - With position_ids: only validate last dimension = rotary_dim // 2
        """
        graph = converter_context.ir_graph

        params = extract_attributes(
            src_op,
            attr_infos=[('interleaved', 'i', 0),
                        ('rotary_embedding_dim', 'i', 0)],
            validate=False
        )

        interleaved = bool(params.interleaved)
        rotary_embedding_dim_attr = params.rotary_embedding_dim

        input_names = self.extract_input_names(src_op, converter_context)
        if len(input_names) < 3 or len(input_names) > 4:
            raise ValueError(
                f"RotaryEmbedding op {src_op.name} expects 3-4 inputs "
                f"(input, cos_cache, sin_cache, [position_ids]), got {len(input_names)}"
            )

        x_name   = input_names[0]
        cos_name = input_names[1]
        sin_name = input_names[2]
        pos_name = input_names[3] if len(input_names) == 4 else None

        # Prefer the reshaped alias if it exists
        alias = x_name + "_reshaped_4d"
        if graph.has_buffer(alias):
            x_name = alias

        input_buffer = graph.get_buffer(x_name)
        if input_buffer.rank() != 4:
            raise ValueError(
                f"RotaryEmbedding op {src_op.name} expects 4D input "
                f"[batch, num_heads, sequence_length, head_size], got {input_buffer.rank()}D"
            )

        # [B, N, S, H]
        batch_size = input_buffer.shape[0]
        num_heads  = input_buffer.shape[1]
        seq_len    = input_buffer.shape[2]
        head_size  = input_buffer.shape[3]

        # Decide whether rotary_embedding_dim was explicitly provided
        rotary_dim_explicit = bool(rotary_embedding_dim_attr) and int(rotary_embedding_dim_attr) > 0
        if rotary_dim_explicit:
            rotary_embedding_dim = int(rotary_embedding_dim_attr)
            if rotary_embedding_dim > head_size:
                raise ValueError(
                    f"RotaryEmbedding op {src_op.name}: rotary_embedding_dim ({rotary_embedding_dim}) "
                    f"cannot be larger than head_size ({head_size})"
                )
        else:
            # Use full head_size logically, but DO NOT set the attribute on the op
            rotary_embedding_dim = head_size

        # Cache validation
        expected_last_dim = rotary_embedding_dim // 2

        def _validate_cache_shape(cache_name, cache_kind):
            if not graph.has_buffer(cache_name):
                raise ValueError(
                    f"RotaryEmbedding op {src_op.name}: {cache_kind} buffer '{cache_name}' is missing. "
                    f"Both cos_cache and sin_cache are mandatory inputs for this operation."
                )
            cache_buf = graph.get_buffer(cache_name)
            cache_shape = list(cache_buf.shape)
            if pos_name is None:
                expected_cache_shape = [batch_size, seq_len, expected_last_dim]
                if cache_shape != expected_cache_shape:
                    raise ValueError(
                        f"RotaryEmbedding op {src_op.name}: {cache_kind} shape mismatch. "
                        f"Expected {expected_cache_shape}, got {cache_shape}"
                    )
            else:
                if len(cache_shape) == 0:
                    raise ValueError(
                        f"RotaryEmbedding op {src_op.name}: {cache_kind} must have at least 1 dimension"
                    )
                if cache_shape[-1] != expected_last_dim:
                    raise ValueError(
                        f"RotaryEmbedding op {src_op.name}: {cache_kind} last dimension mismatch. "
                        f"Expected {expected_last_dim}, got {cache_shape[-1]}"
                    )

        _validate_cache_shape(cos_name, "cos_cache")
        _validate_cache_shape(sin_name, "sin_cache")

        # Build the IR op; only attach rotary_embedding_dim when explicitly provided
        kwargs = {'interleaved': interleaved}
        if rotary_dim_explicit:
            kwargs['rotary_embedding_dim'] = rotary_embedding_dim

        return op_adapter.RotaryEmbeddingOp(src_op.name, **kwargs)

    def extract_input_names(self, src_op, converter_context):
        return [str(i) for i in src_op.input]

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]

OnnxTranslations.register_translation(
    OnnxRotaryEmbeddingTranslation(),
    converter_type('RotaryEmbedding', 'onnx'))
