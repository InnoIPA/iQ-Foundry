# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define mutators for mutating IRs during layout transform."""

import abc

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir import op_adapter, op_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker, AxisOrder
from qti.aisw.converters.common.passes.layout_transform import util
from qti.aisw.converters.common.utils import converter_utils


class LayoutMutatorBase(abc.ABC):
    """Base class for mutating IR-specific graph.

    As the mutator is operated on IR-specific graph, this base class only defines IR-agnostic
    mutations and leaves IR-specific ones for inherited classes to implement.

    Methods:
        preserve_graph_output_layout: Preserve graph output layout if necessary.
        transform_input_buffers: Transform input buffers by target input permute sequences.
    """

    @abc.abstractmethod
    def _add_reshape_if_not_exist(
        self,
        reshape_name,
        input_buffer_name,
        output_buffer_name,
        shape,
        src_info=None
    ):
        """Add Reshape into new graph if not exist."""
        raise NotImplementedError

    @abc.abstractmethod
    def _add_transpose(
        self,
        transpose_name,
        input_buffer_name,
        output_buffer_name,
        current_perm_seq,
        target_perm_seq,
        src_info=None
    ):
        """Add Transpose into new graph."""
        raise NotImplementedError

    def _align_input_rank(self, input_name, target_rank, layout_recorder, src_info=[]):
        """Align input rank with target one.

        This method aims to unsqueeze given input to target rank if necessary.
        """
        # In new graph, shape extracted from buffer with src name must have same rank as the
        # shape in src_graph, since that buffer is directly added by its producer
        # use that shape to infer rank of current input buffer
        input_shape = self._get_shape_by_name(self.new_graph, input_name)
        layout_map = layout_recorder.get_layout_map(input_name)
        rank = len(input_shape)
        if rank == target_rank:
            return

        # TODO: Support unsqueezing buffers not in src format.
        src_perm_seq = util.get_src_perm_seq(rank)
        converter_utils.log_assert(
            src_perm_seq in layout_map,
            f"Unsupported broadcast for buffer {input_name} as not in source format."
        )

        # Adopt input in src format.
        src_input_name = layout_map[src_perm_seq]
        src_input_shape = self._get_shape_by_name(self.new_graph, src_input_name)

        unsqueezed_shape = util.align_rank(src_input_shape, target_rank)
        unsqueezed_name = f"{src_input_name}_to_" + "_".join(map(str, unsqueezed_shape))

        # Unsequeeze if not already done.
        if self._add_reshape_if_not_exist(
            unsqueezed_name,
            src_input_name,
            unsqueezed_name,
            unsqueezed_shape,
            [(src_input_name, op_graph.TraceType.TENSOR), *src_info]
        ):
            # Update unsqueezed buffer into memo.
            unsqueezed_src_perm_seq = util.get_src_perm_seq(target_rank)
            layout_recorder.update_perm_seq(input_name, unsqueezed_name, unsqueezed_src_perm_seq)

    @abc.abstractmethod
    def _get_shape_by_name(self, graph, name):
        """Get shape from graph by given name."""

    def transform_input_buffers(
        self,
        input_buffer_names,
        target_perm_seqs,
        layout_recorder,
        src_op_name=None
    ):
        """Transform input buffers by target input permute sequences.

        This functions aims to transform inputs to match the given target permute sequences by
        inserting Transpose op ahead. Note that for inputs with ranks mismatched with target
        permute sequences, they will be unsqueezed by Reshape op to match the ranks.

        For example of input with shape [1,16,48] and target permute sequence (0,2,3,1):
            1. Unsqueeze input to [1,1,16,48] with rank 4.
            2. Transpose input to [1,16,48,1].

        Args:
            input_buffer_names: A list of strs specifying input buffer names.
            target_perm_seqs: A list of permute sequences specifying target layouts.
            layout_recorder: An instance of LayoutRecorder.
            src_op_name: A str specifying source op name.

        Returns:
            new_input_buffer_names: A list of strs specifying input buffer names for new node.
        """
        new_input_buffer_names = []
        for input_buffer_name, target_perm_seq in zip(
            input_buffer_names, target_perm_seqs
        ):
            # In Lstm/Gru cases, some inputs are optional, so its name may be empty string
            # For this case, just ignore it.
            if input_buffer_name == "":
                new_input_buffer_names.append(input_buffer_name)
                continue

            target_input_rank = len(target_perm_seq)
            self._align_input_rank(
                input_buffer_name,
                target_input_rank,
                layout_recorder,
                [(src_op_name, op_graph.TraceType.OP)]
            )

            existing_perm_seqs = layout_recorder.get_perm_seqs(
                input_buffer_name, target_input_rank
            )

            # Transpose buffer to target format if necessary.
            if target_perm_seq in existing_perm_seqs:
                buffer_name = layout_recorder.get_buffer_name_on_new_graph(
                    input_buffer_name, target_perm_seq
                )
                new_input_buffer_names.append(buffer_name)
            else:
                # Select which buffer to be transposed. Current policy is to select buffer in src
                # format if exists.
                src_perm_seq = util.get_src_perm_seq(target_input_rank)
                if src_perm_seq in existing_perm_seqs:
                    selected_input_perm_seq = src_perm_seq
                else:
                    selected_input_perm_seq = existing_perm_seqs[0]

                selected_input_buffer_name = (
                    layout_recorder.get_buffer_name_on_new_graph(
                        input_buffer_name, selected_input_perm_seq
                    )
                )

                # Add Transpose to match target perm seq.
                new_input_buffer_name = util.generate_new_buffer_name(
                    input_buffer_name, target_perm_seq
                )
                self._add_transpose(
                    new_input_buffer_name,
                    selected_input_buffer_name,
                    new_input_buffer_name,
                    selected_input_perm_seq,
                    target_perm_seq,
                    [
                        (input_buffer_name, op_graph.TraceType.TENSOR),
                        (src_op_name, op_graph.TraceType.OP),
                    ],
                )

                # Update transposed buffer into memo.
                layout_recorder.update_perm_seq(
                    input_buffer_name, new_input_buffer_name, target_perm_seq
                )

                new_input_buffer_names.append(new_input_buffer_name)

        return new_input_buffer_names
