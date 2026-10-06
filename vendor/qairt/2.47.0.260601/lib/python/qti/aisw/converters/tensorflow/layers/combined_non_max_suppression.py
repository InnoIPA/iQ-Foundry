# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import CombinedNmsOp
from qti.aisw.converters.common.utils.translation_utils import compare_values
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import ConverterError, get_const_op_value


class CombinedNonMaxSuppressionLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, max_output_size_per_class, max_total_size, iou_threshold, score_threshold, pad_per_class, clip_boxes, cnms_op, input_boxes_op, input_scores_op, output_names=None):
            super(CombinedNonMaxSuppressionLayerResolver.Descriptor, self).__init__('CombinedNonMaxSuppression', name, nodes, output_names=output_names)
            self.max_output_size_per_class = max_output_size_per_class
            self.max_total_size = max_total_size
            self.iou_threshold = iou_threshold
            self.score_threshold = score_threshold
            self.pad_per_class = pad_per_class
            self.clip_boxes = clip_boxes

            self.cnms_op = cnms_op

            # Input
            self.input_boxes_op = input_boxes_op
            self.input_scores_op = input_scores_op

            # Output
            self.output_boxes_op = None
            self.output_scores_op = None
            self.output_classes_op = None
            self.output_detections_op = None

        def is_input_tensor(self, op, tensor):
            if tensor.op.type == "Const" and \
                    any([compare_values(get_const_op_value(tensor.op), t) for t in [float(self.score_threshold),
                                                                                    int(self.max_total_size),
                                                                                    int(self.max_output_size_per_class),
                                                                                    float(self.iou_threshold),
                                                                                    bool(self.pad_per_class),
                                                                                    bool(self.clip_boxes)]]):
                return False
            return True

        def is_output_op(self, op):
            if op == self.output_boxes_op or op == self.output_scores_op or op == self.output_classes_op or op == self.output_detections_op:
                return True
            else:
                return False

    def __init__(self):

        sequence = GraphSequence([
            ConverterSequenceNode('root', ['CombinedNonMaxSuppression']),
            # Scores
            NonConsumableConverterSequenceNode('scores', ['?']),

            # Boxes
            NonConsumableConverterSequenceNode('boxes', ['?']),
            NonConsumableConverterSequenceNode('max_output_size_per_class', ['Const']),
            NonConsumableConverterSequenceNode('max_total_size', ['Const']),
            NonConsumableConverterSequenceNode('iou_threshold', ['Const']),
            NonConsumableConverterSequenceNode('score_threshold', ['?']),
        ])
        sequence.set_inputs('root', ['boxes', 'scores', 'max_output_size_per_class', 'max_total_size', 'iou_threshold', 'score_threshold'])#, 'pad_per_class', 'clip_boxes'])
        sequence.set_outputs(['root'])

        self.sequences = [sequence]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                # resolve layer for cnms operation
                cnms_op = match['root']
                input_boxes_op = match['boxes']
                input_scores_op = match['scores']

                max_output_size_per_class = graph_helper.evaluate_tensor_output(match['max_output_size_per_class'].outputs[0])
                max_total_size = graph_helper.evaluate_tensor_output(match['max_total_size'].outputs[0])
                iou_threshold = graph_helper.evaluate_tensor_output(match['iou_threshold'].outputs[0])
                score_threshold = graph_helper.evaluate_tensor_output(match['score_threshold'].outputs[0]) if 'score_threshold' in match else float(0)
                pad_per_class = cnms_op.get_attr('pad_per_class')
                clip_boxes = cnms_op.get_attr('clip_boxes')

                consumed_nodes = match.consumed_nodes
                cnms_descriptor = CombinedNonMaxSuppressionLayerResolver.Descriptor(
                    str(cnms_op.name), consumed_nodes, max_output_size_per_class, max_total_size, iou_threshold, score_threshold, pad_per_class,
                    clip_boxes, cnms_op, input_boxes_op, input_scores_op, output_names=[str(out.name) for out in cnms_op.outputs])

                descriptors.extend([cnms_descriptor])

        return descriptors


class CombinedNonMaxSuppressionLayerBuilder(LayerBuilder):

    @staticmethod
    def _compare_op_shapes(converter_context, ops):
        """
        Compares the shape of all ops in the list
        :param ops: list of ops
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :return: True if all are equal or empty list, False otherwise
        """
        if len(ops):
            shape = converter_context.graph_helper.get_op_output_shape(ops[0])  # get shape for first op
            for op in ops:
                if shape != converter_context.graph_helper.get_op_output_shape(op):
                    return False
        else:
            print("WARNING: empty list provided to compare combined nms ops shapes")
        return True

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: NonMaxSuppressionLayerResolver.Descriptor
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :rtype: int
        """
        names = {}
        for input_descriptor in input_descriptors:
            if input_descriptor.is_output_op(descriptor.input_boxes_op):
                names[descriptor.input_boxes_op] = input_descriptor.output_names[0]
            elif input_descriptor.is_output_op(descriptor.input_scores_op):
                names[descriptor.input_scores_op] = input_descriptor.output_names[0]

        if len(names) != 2:
            raise ConverterError("Failed to detect inputs for combined nms op.")

        input_names = [names[descriptor.input_boxes_op], names[descriptor.input_scores_op]]
        input_names.extend(list(set(self.get_input_names(converter_context, descriptor, input_descriptors)) - set(input_names)))

        return ir_graph.add(CombinedNmsOp(name=descriptor.layer_name,
                                   max_boxes_per_class = descriptor.max_output_size_per_class,
                                   max_total_boxes = descriptor.max_total_size,
                                   iou_threshold=descriptor.iou_threshold,
                                   score_threshold=descriptor.score_threshold,
                                   pad_per_class=descriptor.pad_per_class,
                                   clip_boxes=descriptor.clip_boxes),
                     input_names=input_names,
                     output_names=descriptor.output_names)