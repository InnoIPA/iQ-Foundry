# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy

from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.converter_ir.op_adapter import ConstantOp, TransposeConv2dOp, TransposeConv3dOp, TransposeOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.tensorflow.layers.convolution import ConvolutionLayerBuilder


class DeconvolutionLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_DILATIONS = 'dilations'
    TF_ATTRIBUTE_STRIDES = 'strides'
    TF_ATTRIBUTE_PADDING = 'padding'
    TF_ATTRIBUTE_EXPLICIT_PADDING = 'explicit_paddings'

    def get_spatial_padding(self, deconv_op):
        try:
            paddings = deconv_op.get_attr(self.TF_ATTRIBUTE_EXPLICIT_PADDING)
        except ValueError:
            return [[0, 0] for _ in range(len(deconv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)))]

        spatial_padding = []
        # for NHWC, get HW dimensions, pads format:[[pady_before, pady_after], [padx_before, padx_after]]
        if [] != paddings:
            for i in range(1, len(paddings) // 2 - 1):
                spatial_padding.append([paddings[j] for j in range(2 * i, 2 * (i + 1))])

        return spatial_padding

    class Descriptor(LayerDescriptor):

        def __init__(self, name, nodes, deconv_op, bias_op, strides, padding_size_strategy, explicit_pads, input_tensor,
                     weights_tensor=None, bias_tensor=None):
            super(DeconvolutionLayerResolver.Descriptor, self).__init__('Deconvolution', name, nodes)
            self.deconv_op = deconv_op
            self.bias_op = bias_op
            self.strides = strides
            self.explicit_pads = explicit_pads
            self.padding_size_strategy = padding_size_strategy
            self.input_ops = [deconv_op] if bias_op is None else [deconv_op, bias_op]
            self.input_tensor = input_tensor
            self.weights_tensor = weights_tensor
            self.bias_tensor = bias_tensor

        def is_input_op(self, op):
            return op in self.input_ops

        def is_input_tensor(self, op, tensor):
            if (op == self.deconv_op and tensor == self.deconv_op.inputs[0]) or (tensor == self.deconv_op.outputs[0]):
                return False
            return True

        @property
        def output_names(self):
            if self.bias_op:
                output_name = str(self.bias_op.outputs[0].name)
            else:
                output_name = str(self.deconv_op.outputs[0].name)
            return [output_name]

    def __init__(self):
        super(DeconvolutionLayerResolver, self).__init__()
        self.graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('input_sizes', ['?']),
            NonConsumableConverterSequenceNode('weights_source', ['?']),
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('root', ['Conv2DBackpropInput', 'Conv3DBackpropInputV2']),
            NonConsumableConverterSequenceNode('bias_source', ['?']),
            ConverterSequenceNode('bias', ['Add', 'BiasAdd'])
        ])
        self.graph_sequence.set_inputs('root', ['input_sizes', 'weights_source', 'input'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = []

        # Basic deconvolution sequence
        self.graph_sequence.set_outputs(['root'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # Basic deconvolution sequence with bias
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('bias', ['root', 'bias_source'])
        self.graph_sequence.set_outputs(['bias'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        descriptors = []
        for match in matches:
            deconv_op = match['root']
            input_tensor = match['input'].outputs[0]
            consumed_nodes = list(match.consumed_nodes)

            weights_source_op = match['weights_source']
            weights_tensor = None
            if graph_helper.check_tensor_const_origin(weights_source_op.outputs[0])[0]:
                weights_tensor = graph_helper.evaluate_tensor_output(weights_source_op.outputs[0])

            bias_op, bias_tensor = None, None
            if 'bias' in match:
                bias_op = match['bias']
                bias_source_op = match['bias_source']
                if graph_helper.check_tensor_const_origin(bias_source_op.outputs[0])[0]:
                    bias_tensor = graph_helper.evaluate_tensor_output(bias_source_op.outputs[0])
                else:
                    continue

            # Extract attributes
            strides = deconv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding_size_strategy = deconv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            pads = self.get_spatial_padding(deconv_op)


            try:
                dilations = deconv_op.get_attr(self.TF_ATTRIBUTE_DILATIONS)
            except:
                dilations = [1] * len(strides)

            descriptor = DeconvolutionLayerResolver.Descriptor(str(deconv_op.name),
                                                               consumed_nodes,
                                                               deconv_op,
                                                               bias_op,
                                                               strides,
                                                               padding_size_strategy,pads,
                                                               input_tensor,
                                                               weights_tensor=weights_tensor,
                                                               bias_tensor=bias_tensor)
            if len(strides) >= 5:
                descriptor.dilationZ = dilations[1]
                descriptor.dilationY = dilations[2]
                descriptor.dilationX = dilations[3]
            else:
                descriptor.dilationY = dilations[1]
                descriptor.dilationX = dilations[2]

            descriptors.append(descriptor)
        return descriptors


class DeconvolutionLayerBuilder(LayerBuilder, object):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: DeconvolutionLayerResolver.Descriptor
        :rtype: int
        """
        input_dims = converter_context.graph_helper.get_op_output_shape(descriptor.deconv_op.inputs[2])
        filter_dims = converter_context.graph_helper.get_op_output_shape(descriptor.deconv_op.inputs[1])

        if descriptor.bias_op:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.bias_op)
        else:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.deconv_op)

        # actual input names ordering: [filter, activation, bias]
        orig_input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        # IR expects [activation, filter, bias] for ordering of inputs
        input_names = [self.get_input_names(converter_context, descriptor, input_descriptors)[1]]
        # Static weights
        if descriptor.weights_tensor is not None:
            weights_tensor = []
            if descriptor.deconv_op.type == 'Conv2DBackpropInput':
                weights_tensor = numpy.transpose(descriptor.weights_tensor, AxisTracker.AxisFormat.HWOI_TO_HWIO).copy()
            else:
                weights_tensor = numpy.transpose(descriptor.weights_tensor, AxisTracker.AxisFormat.DHWOI_TO_DHWIO).copy()
            if len(orig_input_names) > 1:
                if not ir_graph.has_buffer(orig_input_names[0]):
                    weights_const_op = ConstantOp(name=orig_input_names[0], tensor=weights_tensor)
                    if descriptor.deconv_op.type == 'Conv2DBackpropInput':
                        ir_graph.add(weights_const_op, [], orig_input_names[0], axis_formats=[AxisTracker.AxisFormat.HWIO])
                    else:
                        ir_graph.add(weights_const_op, [], orig_input_names[0], axis_formats=[AxisTracker.AxisFormat.DHWIO])
                else:
                    # if const op already exists then replace with a new const op with reshaped weights
                    orig_weight_const_op = ir_graph.get_producer_op(orig_input_names[0])
                    reshaped_weights_const_op = ConstantOp(name=orig_input_names[0], tensor=weights_tensor)
                    ir_graph.replace(orig_weight_const_op, reshaped_weights_const_op)
                input_names.append(orig_input_names[0])
            else:
                input_names.append(descriptor.layer_name + "_weight")
                weights_const_op = ConstantOp(name=input_names[-1], tensor=weights_tensor)
                if descriptor.deconv_op.type == 'Conv2DBackpropInput':
                    ir_graph.add(weights_const_op, [], input_names[-1], axis_formats=[AxisTracker.AxisFormat.HWIO])
                else:
                    ir_graph.add(weights_const_op, [], input_names[-1], axis_formats=[AxisTracker.AxisFormat.DHWIO])

            weights_buffer = ir_graph.get_buffer(input_names[-1])
            weights_buffer.shape = list(weights_tensor.shape)
        # Dynamic weights
        else:
            if len(orig_input_names) > 1:
                if not ir_graph.has_buffer(orig_input_names[0]):
                    raise ValueError("Dynamic weights on {} node {} are not added in ir_graph.".format(descriptor.layer_name, descriptor.layer_type))
                # Add transpose Op to change the buffer from HWOI/DHWOI to HWIO/DHWIO
                permute_op_name = orig_input_names[0] + "_permute"
                permute_order = AxisTracker.AxisFormat.HWOI_TO_HWIO if descriptor.deconv_op.type == 'Conv2DBackpropInput' \
                                                                    else AxisTracker.AxisFormat.DHWOI_TO_DHWIO
                permute_op = TransposeOp(permute_op_name, perm=permute_order)
                axis_format = AxisTracker.AxisFormat.HWIO if descriptor.deconv_op.type == 'Conv2DBackpropInput' \
                                                          else AxisTracker.AxisFormat.DHWIO
                ir_graph.add(permute_op, [orig_input_names[0]], [permute_op_name], axis_formats=[axis_format])

                input_names.append(permute_op_name)
            else:
                raise ValueError("Dynamic weights on {} node {} are not given.".format(descriptor.layer_name, descriptor.layer_type))

        # Handle biases, depending on source operation and already extracted tensor
        if descriptor.bias_tensor is not None:
            if len(orig_input_names) > 2:
                input_names.append(orig_input_names[2])
            else:
                input_names.append(descriptor.layer_name + "_bias")
            if not ir_graph.has_buffer(input_names[2]):
                bias_const_op = ConstantOp(name=input_names[-1], tensor=descriptor.bias_tensor)
                ir_graph.add(bias_const_op, [], input_names[-1], axis_formats=[AxisTracker.AxisFormat.ANY])
        elif descriptor.bias_op is not None and descriptor.bias_tensor is None:
            raise ValueError("Dynamic bias on {} node of type {} are unsupported.".format(
                descriptor.layer_name, descriptor.layer_type))

        if descriptor.deconv_op.type == "Conv3DBackpropInputV2":
            pads, ir_padding_strategy = ConvolutionLayerBuilder.calculate_padding_size(input_size=output_dims[-4:-1],
                                                                                       output_size=input_dims[-4:-1],
                                                                                       strides=descriptor.strides[1:4],
                                                                                       padding_size_strategy=descriptor.padding_size_strategy,
                                                                                       explicit_pads=descriptor.explicit_pads,
                                                                                       filter_dims=filter_dims,
                                                                                       dilation=[descriptor.dilationZ,
                                                                                                 descriptor.dilationY,
                                                                                                 descriptor.dilationX])

            return ir_graph.add(TransposeConv3dOp(name=descriptor.layer_name,
                                                  bias_op_name=descriptor.bias_op.name if descriptor.bias_op else None,
                                                  padz_before=pads[0][0],
                                                  padz_after=pads[0][1],
                                                  pady_before=pads[1][0],
                                                  pady_after=pads[1][1],
                                                  padx_before=pads[2][0],
                                                  padx_after=pads[2][1],
                                                  padding_size_strategy=ir_padding_strategy,
                                                  stridex=int(descriptor.strides[3]),
                                                  stridey=int(descriptor.strides[2]),
                                                  stridez=int(descriptor.strides[1]),
                                                  dilationx=descriptor.dilationX,
                                                  dilationy=descriptor.dilationY,
                                                  dilationz=descriptor.dilationZ,
                                                  output_width=output_dims[-2],
                                                  output_height=output_dims[-3],
                                                  output_depth=output_dims[-4],
                                                  groups=1),
                                input_names,
                                descriptor.output_names[0])
        else:
            pads, ir_padding_strategy = ConvolutionLayerBuilder.calculate_padding_size(input_size=output_dims[-3:-1],
                                                                                       output_size=input_dims[-3:-1],
                                                                                       strides=descriptor.strides[1:3],
                                                                                       padding_size_strategy=descriptor.padding_size_strategy,
                                                                                       explicit_pads=descriptor.explicit_pads,
                                                                                       filter_dims=filter_dims,
                                                                                       dilation=[descriptor.dilationY,
                                                                                                 descriptor.dilationX])

            return ir_graph.add(TransposeConv2dOp(name=descriptor.layer_name,
                                                  bias_op_name=descriptor.bias_op.name if descriptor.bias_op else None,
                                                  pady_before=pads[0][0],
                                                  pady_after=pads[0][1],
                                                  padx_before=pads[1][0],
                                                  padx_after=pads[1][1],
                                                  padding_size_strategy=ir_padding_strategy,
                                                  stridex=int(descriptor.strides[2]),
                                                  stridey=int(descriptor.strides[1]),
                                                  dilationx=descriptor.dilationX,
                                                  dilationy=descriptor.dilationY,
                                                  output_width=output_dims[-2],
                                                  output_height=output_dims[-3],
                                                  groups=1),
                                input_names,
                                descriptor.output_names[0])


