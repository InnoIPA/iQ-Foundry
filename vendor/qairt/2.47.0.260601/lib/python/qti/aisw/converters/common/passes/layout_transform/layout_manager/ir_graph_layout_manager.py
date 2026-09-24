# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define layout manager for IrGraph."""

from qti.aisw.converters.common.passes.layout_transform import layout_defs, layout_mutator
from qti.aisw.converters.common.passes.layout_transform.layout_manager import layout_manager_base


class IrGraphLayoutManager(layout_manager_base.LayoutManagerBase):
    """Layout manager for transforming layouts of IrGraph.

    This manager dedicates to transform spatial-last IrGraph layout into spatial-first one and is
    expected to be applied during frontend translation. In other words, layout transform happens
    side-by-side with translation, and all ops are transformed to spatial-first layouts before
    adding into the graph.

    Note that layout inferers require input/output shapes before layout transform (i.e., "source"
    input/output shapes) instead of the ones afterwards. Since shapes acquired from graph are
    already in spatial-first layout, trivially querying input/output shapes is thus inapplicable.
    Refer to `IrGraphLayoutMutator.get_src_inout_shapes` for details of acquiring source shapes.

    In addition, since the graph is constructed simultaneously, layout manager could not be aware
    of graph output during the per-op transformation, resulting in graph output layout unable to be
    preserved internally. Therefore, method `preserve_graph_output_layout` is provided to fallback
    layout for graph output explicitly.

    Attributes:
        Refer to LayoutManagerBase for detailed attributes.
        SRC_LAYOUT_TABLE: A layout table in spatial-last format, specifying the source layout for
            heavily layout-sensitive ops.
        DESIRED_LAYOUT_TABLE: A layout table in spatial-first format, specifying the desired layout
            for heavily layout-sensitive ops.
        PREFERRED_PERM_SEQ_TABLE: A dict specifying preferred permuate sequences for various ranks,
            specifying the preferred permute sequence for layout-agnostic ops.

    Methods:
        apply_transform: Transform layout for given op type.
        preserve_graph_output_layout: Preserve layout of graph's output.
    """

    SRC_LAYOUT_TABLE = layout_defs.SPATIAL_LAST_LAYOUT_TABLE
    DESIRED_LAYOUT_TABLE = layout_defs.SPATIAL_FIRST_LAYOUT_TABLE
    # TODO: Support 0D cases.
    PREFERRED_PERM_SEQ_TABLE = {
        1: [(0,)],
        2: [(0, 1)],
        3: [(0, 2, 1), (1, 0, 2), (0, 1, 2)],
        4: [(0, 2, 3, 1), (0, 1, 2, 3)],
        5: [(0, 2, 3, 4, 1), (0, 1, 2, 3, 4)]
    }

    def __init__(self, graph):
        """Initialize manager.

        Args:
            graph: A GraphWrapperApi instance where layout transform applies.
        """
        super().__init__(
            self.SRC_LAYOUT_TABLE,
            self.DESIRED_LAYOUT_TABLE,
            self.PREFERRED_PERM_SEQ_TABLE,
            layout_mutator.IrGraphLayoutMutator(graph)
        )

    def apply_transform(self, op_type, input_names, output_names, attrs):
        """Apply transform for given op type.

        The overall process of transforming layout is divided into following steps:
            1. Get source input/output shapes to infer layout.
            2. Infer target layouts and attributes.
            3. Transform inputs according to target layouts.
            4. Update op attributes.
            5. Update layout memo.

        Args:
            op_type: A str specifying target op type.
            input_names: A list of strs specifying op input names.
            output_names: A list of strs specifying op output names.
            attrs: A dict containing op attributes.

        Returns:
            new_input_names: A list of strs specifying op input names after layout transform.
            new_attrs: A dict containing op attributes after layout transform.
        """
        inferer = self.get_layout_inferer(op_type)

        # Step 1.
        src_input_shapes, src_output_shapes = self.layout_mutator.get_src_inout_shapes(
            op_type, input_names, attrs, self.layout_recorder
        )

        # Step 2.
        target_input_perm_seqs, output_perm_seqs, new_attrs = (
            inferer.infer_target_layouts_and_attrs(
                input_names, src_input_shapes, src_output_shapes, attrs, self.layout_recorder
            )
        )

        # Step 3.
        new_input_names = self.layout_mutator.transform_input_buffers(
            input_names, target_input_perm_seqs, self.layout_recorder
        )

        # Step 4.
        for attr_name, attr_val in attrs.items():
            new_attrs.setdefault(attr_name, attr_val)

        # Step 5.
        self.layout_recorder.update_perm_seqs(output_names, output_names, output_perm_seqs)

        return new_input_names, new_attrs

    def preserve_graph_output_layout(self, output_name):
        """Preserve layout of graph output.

        Note that given buffer name is assumed to be graph output, and further checking is
        unavailable since the timing of setting graph output may happen afterwards.

        Args:
            output_name: A str specifying the output tensor name to be preserved layout.

        Returns:
            A str specifying the new output tensor name with layout preserved.
        """
        return self.layout_mutator.preserve_graph_output_layout(output_name, self.layout_recorder)
