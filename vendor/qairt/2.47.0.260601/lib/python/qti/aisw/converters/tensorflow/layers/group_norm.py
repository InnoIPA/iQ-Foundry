# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from collections import OrderedDict

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import GroupNormOp, ConstantOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class GroupNormLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, group, epsilon, input_tensors):
            super(GroupNormLayerResolver.Descriptor, self).__init__('GroupNorm', name, operations)
            self.group = group
            self.epsilon = epsilon
            self.input_tensors = input_tensors

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):
        self.sequence = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('reshape', ['Reshape']),
            ConverterSequenceNode('mean', ['Mean']),
            ConverterSequenceNode('StopGradient', ['StopGradient']),
            ConverterSequenceNode('SquaredDifference', ['SquaredDifference']),
            ConverterSequenceNode('variance', ['Mean']),
            ConverterSequenceNode('epsilon', ['Identity', 'Const']),
            ConverterSequenceNode('add_0', ['Add', 'AddV2']),
            ConverterSequenceNode('Rsqrt', ['Rsqrt']),
            ConverterSequenceNode('gamma', ['Identity', 'Const']),
            ConverterSequenceNode('mul_0', ['Mul']),
            ConverterSequenceNode('mul_1', ['Mul']),
            ConverterSequenceNode('mul_2', ['Mul']),
            ConverterSequenceNode('beta', ['Identity', 'Const']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('add_1', ['Add', 'AddV2']),
            ConverterSequenceNode('reshape_1', ['Reshape']),
            NonConsumableConverterSequenceNode('shape', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('mean_reduction_indices', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('variance_reduction_indices', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('shape_1', ['Identity', 'Const']),
        ])
        self.sequence.set_inputs('reshape', ['input', 'shape'])
        self.sequence.set_inputs('mean', ['reshape', 'mean_reduction_indices'])
        self.sequence.set_inputs('StopGradient', ['mean'])
        self.sequence.set_inputs('SquaredDifference', ['reshape','StopGradient'])
        self.sequence.set_inputs('variance', ['SquaredDifference','variance_reduction_indices'])
        self.sequence.set_inputs('add_0', ['variance','epsilon'])
        self.sequence.set_inputs('Rsqrt', ['add_0'])
        self.sequence.set_inputs('mul_0', ['Rsqrt','gamma'])
        self.sequence.set_inputs('mul_1', ['reshape','mul_0'])
        self.sequence.set_inputs('mul_2', ['mean','mul_0'])
        self.sequence.set_inputs('sub', ['beta','mul_2'])
        self.sequence.set_inputs('add_1', ['mul_1','sub'])
        self.sequence.set_inputs('reshape_1', ['add_1', 'shape_1'])
        self.sequence.set_outputs(['reshape_1'])

        self.sequences = [self.sequence]

    def is_final_resolution(self):
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                input_op = match['reshape']
                shape = graph_helper.get_op_output_shape(input_op)
                rank = len(shape)

                beta_op = match['beta']
                beta = graph_helper.evaluate_tensor_output(beta_op.outputs[0])
                beta_shape = list(beta.shape)

                gamma_op = match['gamma']
                gamma = graph_helper.evaluate_tensor_output(gamma_op.outputs[0])
                gamma_shape = list(gamma.shape)

                epsilon_op = match['epsilon']
                epsilon = graph_helper.evaluate_tensor_output(epsilon_op.outputs[0])

                axes_op = match['mean_reduction_indices']
                axes = graph_helper.evaluate_tensor_output(axes_op.outputs[0])
                axes = [axes] if np.isscalar(axes) else axes.tolist()
                for i in range(len(axes)):
                    axes[i] = int(axes[i])
                    if axes[i] < 0:
                        axes[i] += rank

                axes_op_1 = match['variance_reduction_indices']
                axes_1 = graph_helper.evaluate_tensor_output(axes_op_1.outputs[0])
                axes_1 = [axes_1] if np.isscalar(axes_1) else axes_1.tolist()
                for i in range(len(axes_1)):
                    axes_1[i] = int(axes_1[i])
                    if axes_1[i] < 0:
                        axes_1[i] += rank

                # Add group norm constraints
                group = shape[-2]
                # Reshape op: N,H,W,C [1, 2, 2, 32] -> N,H,W,G,C/G [1, 2, 2, 16, 2], axes should be: [1,2,4]
                expected_axes = list(range(1, rank - 2)) + [rank -1]
                if axes != axes_1 or axes != expected_axes:
                    continue
                if beta_shape != gamma_shape or beta_shape != shape[-2:]:
                    continue

                consumed_nodes = match.consumed_nodes
                input_tensors = OrderedDict()
                input_tensors["gamma"] = gamma.reshape(-1)
                input_tensors["beta"] = beta.reshape(-1)

                potential_descriptors.append(GroupNormLayerResolver.Descriptor(str(input_op.name),
                                                                               consumed_nodes,
                                                                               group=group,
                                                                               epsilon=epsilon,
                                                                               input_tensors=input_tensors))
        return potential_descriptors


class GroupNormLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: GroupNormLayerResolver.Descriptor
        :rtype: int
        """
        if len(input_descriptors) > 1:
            non_constant_input_descriptors = []
            for input_descriptor in input_descriptors:
                if input_descriptor.layer_type != 'Constant':
                    non_constant_input_descriptors.append(input_descriptor)

            if len(non_constant_input_descriptors) == 1:
                input_name = self.get_input_name(converter_context, descriptor, non_constant_input_descriptors)
            else:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_LAYER_INPUT_COUNT_ERROR')
                                     (descriptor.layer_type, 1, len(input_descriptors)))

        else:
            input_name = self.get_input_name(converter_context, descriptor, input_descriptors)

        input_names = [input_name]

        # Add all the constant inputs
        for k,v in descriptor.input_tensors.items():
            if not isinstance(v, np.ndarray):
                v = np.atleast_1d(v)

            input_names.append(str(descriptor.layer_name)+"_"+k)
            ir_graph.add(ConstantOp(input_names[-1],
                                    v,
                                    quantizable=True),
                         [],
                         input_names[-1])

        # Add the group norm op
        return ir_graph.add(GroupNormOp(descriptor.layer_name,
                                        group = descriptor.group,
                                        epsilon = descriptor.epsilon),
                            input_names=input_names,
                            output_names=descriptor.output_names[0])
