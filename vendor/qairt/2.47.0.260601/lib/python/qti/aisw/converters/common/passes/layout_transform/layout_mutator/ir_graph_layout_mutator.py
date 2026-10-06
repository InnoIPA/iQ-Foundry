# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define layout mutator for IrGraph."""

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.passes.layout_transform import util
from qti.aisw.converters.common.passes.layout_transform.layout_mutator import (
    layout_mutator_base,
)


class IrGraphLayoutMutator(layout_mutator_base.LayoutMutatorBase):
    """Layout mutator for C++ IrGraph.

    Attributes:
        new_graph: A GraphWrapperApi instance to be mutated.

    Methods:
        get_src_inout_shapes: Get source input/output shapes.
        preserve_graph_output_layout: Preserve graph output layout if necessary.
    """

    def __init__(self, graph):
        self.new_graph = graph

    def _add_reshape_if_not_exist(
        self, reshape_name, input_buffer_name, output_buffer_name, shape, src_info=None
    ):
        """Add Reshape into new graph if not exist."""
        if self.new_graph.has_tensor(output_buffer_name):
            return False

        self.new_graph.add_reshape(
            reshape_name, [input_buffer_name], [output_buffer_name], shape
        )
        return True

    def _add_transpose(
        self,
        transpose_name,
        input_buffer_name,
        output_buffer_name,
        current_perm_seq,
        target_perm_seq,
        src_info=None,
    ):
        """Add Transpose to graph."""
        self.new_graph.add_transpose(
            transpose_name,
            [input_buffer_name],
            [output_buffer_name],
            util.calculate_perm_seq(current_perm_seq, target_perm_seq),
        )

    def _get_default_perm_seq(self, target_name, layout_recorder):
        """Get default perm seq for given name.

        Note "default" perm seq here means the one returned by layout inferers, comparing to those
        transposed by consumers, and thus the source and new buffer names in layout recorder are
        expected to be identical for extraction.
        """
        layout_map = {
            name: perm_seq
            for perm_seq, name in layout_recorder.get_layout_map(target_name).items()
        }
        return layout_map[target_name]

    def _get_shape_by_name(self, graph, name):
        """Get shape from graph by given name."""
        return graph.get_tensor_shape(name)

    def get_src_inout_shapes(self, op_type, input_buffer_names, attrs, layout_recorder):
        """Get source input/output shapes.

        Since there is only new graph for querying shapes which are already in new layout, source
        input shapes can only be acquired through fallback according to the perm seqs in layout
        memo. Source output shapes are then calculated by input shapes and necessary attributes
        through per-op shape inference APIs. Note that the fallback is inapplicable to output shapes
        as corresponding op is not yet added to the new graph at this point.

        Args:
            op_type: A str specifying target op type.
            input_buffer_names: A list of strs specifying inputs to be acquired shapes.
            attrs: A dict containing op attributes for inferring output shapes.
            layout_recorder: A LayoutRecorder instance for acquiring layouts.

        Returns:
            input_shapes: A list of list of ints representing source input shapes.
            output_shapes: A list of list of ints representing source output shapes.
        """

        def fallback_input_shape(input_name):
            """Fallback shape of given input."""
            new_input_shape = self._get_shape_by_name(self.new_graph, input_name)
            cur_perm_seq = self._get_default_perm_seq(input_name, layout_recorder)
            src_perm_seq = util.get_src_perm_seq(len(new_input_shape))
            fallback_perm_seq = util.calculate_perm_seq(cur_perm_seq, src_perm_seq)
            return [new_input_shape[axis] for axis in fallback_perm_seq]

        src_input_shapes = list(map(fallback_input_shape, input_buffer_names))
        src_output_shapes = getattr(
            self.new_graph.graph_helper, f"infer_{op_type.lower()}_output_shapes"
        )(src_input_shapes, attrs, axis_order=ir_graph.SpatialLastAxisOrder())
        return src_input_shapes, src_output_shapes

    def preserve_graph_output_layout(self, output_buffer_name, layout_recorder):
        """Preserve graph output layout if necessary.

        If the given output name is not in src perm, an additional Transpose will be added to graph
        to preserve layout, and the graph output will be replaced by the returned name.

        Args:
            output_buffer_name: A str specifying the graph output with layout to be preserved.
            layout_recorder: A LayoutRecorder instance.

        Returns:
            output_name/new_output_name: A str specifying possibly updated graph output name.
        """
        cur_perm_seq = self._get_default_perm_seq(output_buffer_name, layout_recorder)
        src_perm_seq = util.get_src_perm_seq(len(cur_perm_seq))
        if cur_perm_seq == src_perm_seq:
            return output_buffer_name

        new_output_name = util.generate_new_buffer_name(
            output_buffer_name, src_perm_seq
        )
        self._add_transpose(
            new_output_name,
            output_buffer_name,
            new_output_name,
            cur_perm_seq,
            src_perm_seq,
        )
        layout_recorder.update_perm_seq(new_output_name, new_output_name, src_perm_seq)

        return new_output_name
