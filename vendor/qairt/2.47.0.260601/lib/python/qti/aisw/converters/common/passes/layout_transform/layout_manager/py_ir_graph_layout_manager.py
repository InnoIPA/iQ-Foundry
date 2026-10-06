# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define layout mananger for PyIrGraph."""

from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.passes.layout_transform.layout_mutator import (
    PyIrGraphLayoutMutator,
)
from qti.aisw.converters.common.passes.layout_transform import layout_defs
from qti.aisw.converters.common.passes.layout_transform.layout_manager.layout_manager_base import (
    LayoutManagerBase,
)


class PyIrGraphLayoutManager(LayoutManagerBase):
    """Layout manager for transforming layouts of PyIrGraph.

    This manager dedicates to transform spatial-last layouts of PyIrGraph into spatial-first.

    Attributes:
        Refer to LayoutManagerBase for detailed attributes.
        SRC_LAYOUT_TABLE: A layout table in spatial-last format, specifying the source layout for
            heavily layout-sensitive ops.
        DESIRED_LAYOUT_TABLE: A layout table in spatial-first format, specifying the desired layout
            for heavily layout-sensitive ops.
        PREFERRED_PERM_SEQ_TABLE: A dict specifying preferred permuate sequences for various ranks,
            specifying the preferred permute sequence for layout-agnostic ops.

    Methods:
        apply_transform: Transform layout for given node.
    """

    SRC_LAYOUT_TABLE = layout_defs.SPATIAL_LAST_LAYOUT_TABLE
    DESIRED_LAYOUT_TABLE = layout_defs.SPATIAL_FIRST_LAYOUT_TABLE

    # TODO: Support 0D tensor cases.
    # TODO: Create testcases [Conv1D -> LSTM] to test the algorithm.
    PREFERRED_PERM_SEQ_TABLE = {
        1: [(0,)],
        2: [(0, 1)],
        3: [(0, 2, 1), (1, 0, 2), (0, 1, 2)],
        4: [(0, 2, 3, 1), (0, 1, 2, 3)],
        5: [(0, 2, 3, 4, 1), (0, 1, 2, 3, 4)],
    }

    def __init__(self, src_graph):
        """Initialize manager.

        Args:
            src_graph: An instance of IrOpGraph before layout transform.
        """
        layout_mutator = PyIrGraphLayoutMutator(src_graph)
        super().__init__(
            self.SRC_LAYOUT_TABLE,
            self.DESIRED_LAYOUT_TABLE,
            self.PREFERRED_PERM_SEQ_TABLE,
            layout_mutator,
            user_custom_io=src_graph.user_custom_io,
        )

    def apply_transform(self, src_node):
        """Transform layout for given node.

        The overall process of transforming layout is divided into following steps:
            1. Get required arguments to infer layout.
            2. Infer target layouts and attributes.
            3. Transform inputs according to target layouts.
            4. Add node into new graph.
            5. Fallback to source layout if necessary.

        Args:
            src_node: An instance of OpNode on source graph.
        """
        op_type = src_node.op.TRANSLATION_KEY
        inferer = self.get_layout_inferer(op_type)

        # Step 1.
        input_buffer_names = self.layout_mutator.get_src_input_buffer_names(src_node)
        output_buffer_names = self.layout_mutator.get_src_output_buffer_names(src_node)

        input_shapes = self.layout_mutator.get_src_buffer_shapes(input_buffer_names)
        output_shapes = self.layout_mutator.get_src_buffer_shapes(output_buffer_names)

        if isinstance(src_node.op, op_adapter.ConstantOp):
            # TODO
            # This is a workaround to avoid triggering loading data from disk for those defer-
            # weight-loading cases.
            src_attrs = {}
        else:
            src_attrs = src_node.op.list_params()

        # Step 2.
        target_input_perm_seqs, output_perm_seqs, new_attrs = (
            inferer.infer_target_layouts_and_attrs(
                input_buffer_names,
                input_shapes,
                output_shapes,
                src_attrs,
                self.layout_recorder,
            )
        )

        # Step 3.
        new_input_buffer_names = self.layout_mutator.transform_input_buffers(
            input_buffer_names, target_input_perm_seqs, self.layout_recorder, src_node.op.name
        )

        # Step 4.
        new_node = self.layout_mutator.add_op_to_new_graph(
            src_node,
            new_attrs,
            new_input_buffer_names,
            output_perm_seqs,
            self.layout_recorder,
        )

        # Step 5.
        self.layout_mutator.transform_output_buffers(
            output_buffer_names, output_perm_seqs, new_node, self.layout_recorder
        )
