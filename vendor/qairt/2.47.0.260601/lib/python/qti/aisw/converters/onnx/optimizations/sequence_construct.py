# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
from qti.aisw.converters.common.framework_optimizer import FrameworkOptimizer
from qti.aisw.converters.common.utils.converter_utils import log_debug1, log_error
from qti.aisw.converters.onnx.util import (
    get_opset_version,
)


class ONNXSequenceConstructPatternOptimizer(FrameworkOptimizer):
    """
    ONNXSequenceConstructPatternOptimizer class is responsible for:
    Removing SequenceConstruct operators.
    """

    def __init__(self, loader):
        """
        Constructor of ONNXSequenceConstructPatternOptimizer
        """
        self.loader = loader
        self.model_opset = get_opset_version(self.loader.model)

    def __str__(self) -> str:
        """
        Function to get the string representation of ONNXSequenceConstructPatternOptimizer
        class.
        :returns str: String representation of class.
        """
        return "ONNX - SequenceConstruct PatternOptimizer"

    def update_names(self, base_names, origin_name, new_names):
        """
        Update the original name of the base name with the new name.
        """
        idx = list(base_names).index(origin_name)
        base_names.remove(origin_name)
        base_names[idx:idx] = new_names

    def optimize(self, sequence_construct_nodes, **kwargs):
        """
        Function to apply SequenceConstruct optimization logic to remove SequenceConstruct node
        since there is no sequence support in converter.
        :returns ONNXLoader instance.
        :raises:
            e: Exception raised in case of checker failure.
        """
        if len(sequence_construct_nodes) == 0:
            log_debug1(
                "The model doesn't contain any SequenceConstruct operators. "
                "Skipping SequenceConstruct Optimization."
            )
            return self.loader

        skip_shape_inference = kwargs.get("skip_shape_inference", False)
        for node in sequence_construct_nodes:
            graph_output_names = [g_tensor.name for g_tensor in self.loader.model.graph.output]
            # According to op def, SequenceConstruct has only one output
            node_output = node.output[0]
            if node_output in graph_output_names:
                # Update graph output names
                #   - remove SequenceConstruct op
                self.update_names(graph_output_names, node_output, list(node.input))
                self.loader.utils.update_output_names(graph_output_names)
            else:
                # There are nodes after the SequenceConstruct node.
                # The only case is when these nodes are sequence-related operations.
                # TODO: simplify Sequence series ops
                log_error("Currently, the converter does not support Sequence series nodes "
                          "after the Sequence Construct node.")
                sys.exit(-1)

            self.loader.utils.remove_node(node)
            # Cleanup model and apply topological sort.
            self.loader.utils.clean_model().topological_sort()

        try:
            self.loader.native_checker()
        except Exception as e:
            log_error(f"The Onnx native checker failed with Exception : {e} ")
            raise e

        if not skip_shape_inference:
            self.loader.utils.native_shape_inference()

        return self.loader
