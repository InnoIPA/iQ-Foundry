# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define layout inferers for layout-agnostic ops."""

import numpy as np

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.passes.layout_transform.layout_inferer import (
    layout_inferer_base,
)
from qti.aisw.converters.common.passes.layout_transform.layout_recorder import (
    LayoutRecorder,
)
from qti.aisw.converters.common.passes.layout_transform import util
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.converters.common.utils.converter_utils import log_assert


class AgnosticOpLayoutInfererBase(layout_inferer_base.LayoutInfererBase):
    """Inferer base for layout-agnostic op."""

    def infer_target_input_perm_seqs(
        self, input_buffer_names: list, input_shapes: list, layout_recorder: LayoutRecorder
    ):
        """Infer target input permute sequences.

        For layout-agnostic op, base implementation searches over existing permute sequences from
        layout memo for preferred one. Note that inherited classes must override this method for
        multi-inputs or special cases.
        """
        # Check for single input.
        log_assert(
            len(input_buffer_names) == 1,
            "Base implementation for agnostic op only support single input case, but this op has {} inputs.",
            len(input_buffer_names),
        )

        rank = len(input_shapes[0])
        existing_perm_seqs = layout_recorder.get_perm_seqs(input_buffer_names[0], rank)
        preferred_perm_seqs = layout_recorder.get_preferred_perm_seqs(rank)

        # Determine target permute sequence by searching over preferred ones.
        target_perm_seq = util.search_preferred_perm_seqs_in_order(
            preferred_perm_seqs, existing_perm_seqs
        )

        return [target_perm_seq]

    def infer_target_output_perm_seqs(
        self, target_input_perm_seqs: list, output_shapes: list
    ):
        """Infer target output permute sequences.

        By default, target output permue sequence directly reuses the one of the first input.
        Inherited classes may override this method for multi-outputs or rank changing cases.
        """
        return [target_input_perm_seqs[0]]

    def update_attr_with_target_input_perm_seqs(
        self, target_input_perm_seqs: list, attrs
    ):
        """Update attribute.

        Inherited classes may override this function if any attribute must be updated.
        """
        return {}


# ------------------------------------------------------------------------------
#   Arg
# ------------------------------------------------------------------------------
class ArgInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Arg op."""

    op_type = ir_graph.IR_OP_ARG

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder, src_attrs
    ):
        """Infer target input permute sequences.

        To avoid producing weird permute sequence, there are two cases to be considered.
            1. keep_dims=True -> Behave normally (i.e., layout agnostic).
            2. keep_dims=False -> Fallback to layout untrackable.
        """
        if src_attrs['keep_dims']:
            return super().infer_target_input_perm_seqs(
                input_buffer_names, input_shapes, layout_recorder
            )
        return list(map(util.get_src_perm_seq, util.get_ranks(input_shapes)))

    def infer_target_output_perm_seqs(self, target_input_perm_seqs, output_shapes, src_attrs):
        """Infer target output perm seqs."""
        if src_attrs['keep_dims']:
            return super().infer_target_output_perm_seqs(target_input_perm_seqs, output_shapes)
        return list(map(util.get_src_perm_seq, util.get_ranks(output_shapes)))

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        return {'axis': target_input_perm_seqs[0].index(attrs['axis'])}

    def infer_target_layouts_and_attrs(
        self, input_buffer_names, input_shapes, output_shapes, src_attrs, layout_recorder
    ):
        """Infer target layouts and attributes."""
        target_input_perm_seqs = self.infer_target_input_perm_seqs(
            input_buffer_names, input_shapes, layout_recorder, src_attrs
        )
        new_attrs = self.update_attr_with_target_input_perm_seqs(target_input_perm_seqs, src_attrs)
        target_output_perm_seqs = self.infer_target_output_perm_seqs(
            target_input_perm_seqs, output_shapes, src_attrs
        )
        return target_input_perm_seqs, target_output_perm_seqs, new_attrs


# ------------------------------------------------------------------------------
#   BatchPermutation
# ------------------------------------------------------------------------------
class BatchPermutationInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for BatchPermutation op."""

    op_type = ir_graph.QNN_OP_BATCH_PERMUTATION

    def infer_target_input_perm_seqs(self, input_buffer_names, input_shapes, layout_recorder):
        """Infer target input permute sequences.

        BatchPermutation has two inputs, input tensor and indices tensor. Input tensor is layout-
        agnostic as long as its target perm seq not permuting the batch dimension or layout-
        untrackable otherwise. On the other hand, indices tensor is layout-untrackable in all cases.
        """
        target_perm_seq = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )[0]
        src_perm_seqs = list(map(util.get_src_perm_seq, util.get_ranks(input_shapes)))

        # Check whether batch dimension is permuted.
        if target_perm_seq[0] == 0:
            return [target_perm_seq, src_perm_seqs[1]]
        return src_perm_seqs


# ------------------------------------------------------------------------------
#   Buffer
# ------------------------------------------------------------------------------
class BufferInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Buffer op."""

    op_type = ir_graph.QNN_OP_BUFFER

    def infer_target_input_perm_seqs(self, input_buffer_names, input_shapes, layout_recorder):
        """Infer target input permute sequences."""
        target_perm_seq = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )[0]
        if len(input_buffer_names) == 1:
            return [target_perm_seq]

        # The second input of Buffer is an optional boolean. Since 0D tensor supportiveness isn't
        # fully ready, the rank of this input is used to infer its perm seq instead of directly
        # setting to () or (0,).
        return [target_perm_seq, util.get_src_perm_seq(len(input_shapes[1]))]

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        return {'buffer_dim': target_input_perm_seqs[0].index(attrs['buffer_dim'])}


# ------------------------------------------------------------------------------
#   Cast
# ------------------------------------------------------------------------------
class CastInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Cast op."""

    op_type = 'cast'


# ------------------------------------------------------------------------------
#   Concat
# ------------------------------------------------------------------------------
class ConcatInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Concat op."""

    op_type = ir_graph.QNN_OP_CONCAT

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder
    ):
        """Infer target input permute sequences.

        In order to achieve fewest inserted Transpose (in a local minima manner), target input
        permute sequence is determined through voting.
        """
        ranks = util.get_ranks(input_shapes)
        # Rank are given to filter out imprimitive ranks.
        input_perm_seqs_table = layout_recorder.get_perm_seqs_table(
            input_buffer_names, ranks
        )

        # Vote for the target perm seq, where preferred perm seqs are adopted to untied the voting.
        preferred_perm_seqs = layout_recorder.get_preferred_perm_seqs(ranks[0])
        target_perm_seq = util.get_target_perm_seq_by_vote(
            input_perm_seqs_table, preferred_perm_seqs
        )

        return [target_perm_seq] * len(input_buffer_names)

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        return {"axis": target_input_perm_seqs[0].index(attrs["axis"])}

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names,
        input_shapes,
        output_shapes,
        src_attrs,
        layout_recorder,
    ):
        """Infer input/output permute sequences and update attributes."""
        # Ensure that input ranks are consistent.
        for input_shape in input_shapes:
            log_assert(
                len(input_shape) == len(input_shapes[0]),
                "Expect rank matches for ConcatOp. Got rank {}, Expected rank {}.",
                len(input_shape),
                len(input_shapes[0]),
            )
        return super().infer_target_layouts_and_attrs(
            input_buffer_names, input_shapes, output_shapes, src_attrs, layout_recorder
        )


# ------------------------------------------------------------------------------
#   Convert
# ------------------------------------------------------------------------------
class ConvertInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Convert op."""

    op_type = ir_graph.QNN_OP_CONVERT


# ------------------------------------------------------------------------------
#   CreateSparse
# ------------------------------------------------------------------------------
class CreateSparseInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for CreateSparse op."""

    op_type = ir_graph.QNN_OP_CREATE_SPARSE

    # TODO
    # Ideally, CreateSparse is catergorized as layout-agnostic. However, current support for
    # sparse tensor is limited in ONNX Conv3D, and thus below implementation is a workaround.

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder
    ):
        """Infer target input permute sequences."""
        converter_utils.log_assert(
            len(input_buffer_names) == 2,
            f'Expecting 2 inputs for CreateSparse but got {len(input_buffer_names)}.'
        )
        return list(map(util.get_src_perm_seq, util.get_ranks(input_shapes)))

    def infer_target_output_perm_seqs(self, target_perm_seqs, output_shapes):
        """Infer target output perm seqs."""
        rank = len(output_shapes[0])
        if rank == 5:
            return [(0, 2, 3, 4, 1)]
        elif rank == 4:
            return [(0, 2, 3, 1)]

    def update_attr_with_target_input_perm_seqs(self, target_perm_seqs, attrs):
        """Update attribute."""
        rank = len(attrs['shape'])
        if rank == 5:
            return {'shape': [attrs['shape'][axis] for axis in (0, 2, 3, 4, 1)]}
        elif rank == 4:
            return {'shape': [attrs['shape'][axis] for axis in (0, 2, 3, 1)]}



# ------------------------------------------------------------------------------
#   CumSum
# ------------------------------------------------------------------------------
class CumSumInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for CumSum op."""

    op_type = ir_graph.QNN_OP_CUMULATIVE_SUM

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        return {'axis': target_input_perm_seqs[0].index(attrs['axis'])}


# ------------------------------------------------------------------------------
#   Dequantize
# ------------------------------------------------------------------------------
class DequantizeInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Dequantize op."""

    op_type = ir_graph.QNN_OP_DEQUANTIZE


# ------------------------------------------------------------------------------
#   ElementwiseNeuron (Relu)
# ------------------------------------------------------------------------------
class ElementwiseNeuronInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for ElementwiseNeuron op."""

    op_type = ir_graph.QNN_OP_ELEMENT_WISE_NEURON


# ------------------------------------------------------------------------------
#   ElementwiseBinary
# ------------------------------------------------------------------------------
class ElementwiseBinaryInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for ElementwiseBianry op."""

    op_type = ir_graph.IR_OP_ELTWISE_BINARY

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder
    ):
        """Infer target input permute sequences.

        The infer process can be divided into two parts where the first one is collecting existing
        permute sequences and the second one is determining the target one.

        There are three cases for the first part:
            1. broadcast not required -> add existing permute sequences into candidates
            2. broadcast required
                2-1. source permute sequence not exists -> fallback
                2-1. source permute sequence exists -> add source permute sequence after broadcast
                     into candidates

        There are as well three cases for the second part:
            1. determine from candidates
                1-1. target permute sequence is source one -> fallback
                1-2. target permute sequence is not source one -> return
                    1-2-1. w/o scalar input -> return
                    1-2-2. w/ scalar input(s) -> change target permute sequence to (0,) for scalar
                           input before return
            2. fallback -> return source permute sequences before broadcast
        """
        ranks = util.get_ranks(input_shapes)
        input_perm_seqs_table = layout_recorder.get_perm_seqs_table(
            input_buffer_names, ranks
        )

        max_rank = max(ranks)
        # Candidate input permute sequences with max rank.
        broadcasted_input_perm_seqs_table = dict()
        fallback_all_inputs = False
        for (input_buffer_name, perm_seqs), shape in zip(input_perm_seqs_table.items(), input_shapes):
            rank = len(perm_seqs[0])
            src_perm_seq = util.get_src_perm_seq(rank)
            if rank < max_rank and src_perm_seq not in perm_seqs:
                fallback_all_inputs = True
                break
            elif shape == [1]:
                # scalar input won't be included in voting table
                continue
            elif rank < max_rank and src_perm_seq in perm_seqs:
                broadcasted_src_perm_seq = util.get_src_perm_seq(max_rank)
                broadcasted_input_perm_seqs_table[input_buffer_name] = [
                    broadcasted_src_perm_seq
                ]
            elif rank == max_rank:
                broadcasted_input_perm_seqs_table[input_buffer_name] = perm_seqs

        # Handle cases that all inputs are scalars.
        if not broadcasted_input_perm_seqs_table:
            fallback_all_inputs = True

        # Vote for the target perm seq, where preferred perm seqs are adopted to untied the voting.
        if not fallback_all_inputs:
            preferred_perm_seqs = layout_recorder.get_preferred_perm_seqs(max_rank)
            target_perm_seq = util.get_target_perm_seq_by_vote(
                broadcasted_input_perm_seqs_table, preferred_perm_seqs
            )
            fallback_all_inputs = target_perm_seq == util.get_src_perm_seq(max_rank)

        if fallback_all_inputs:
            # Adopt src perm seqs for all inputs.
            return [util.get_src_perm_seq(rank) for rank in ranks]

        # Change target perm seq to (0,) for scalar inputs as they can be automatically broadcasted.
        return [target_perm_seq if shape != [1] else (0,) for shape in input_shapes]

    def infer_target_output_perm_seqs(self, target_input_perm_seqs, output_shapes):
        """Infer target output permute sequences.

        This method is overridden to handle broadcast cases. The permute sequence with expected
        rank is adopted for target output.
        """
        rank = len(output_shapes[0])
        for target_perm_seq in target_input_perm_seqs:
            if len(target_perm_seq) == rank:
                return [target_perm_seq]


# ------------------------------------------------------------------------------
#   ElementWiseMuxOp
# ------------------------------------------------------------------------------
class ElementWiseMuxInferer(ElementwiseBinaryInferer):
    """Layout inferer for ElementWiseMuxOp op."""

    op_type = ir_graph.QNN_OP_ELEMENT_WISE_MUX


# ------------------------------------------------------------------------------
#   ElementwiseTernary
# ------------------------------------------------------------------------------
class ElementwiseTernaryInferer(ElementwiseBinaryInferer):
    """Layout inferer for ElementwiseTernary op."""

    op_type = ir_graph.IR_OP_ELTWISE_TERNARY


# ------------------------------------------------------------------------------
#   ElementwiseUnaryOp
# ------------------------------------------------------------------------------
class ElementwiseUnaryInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for ElementwiseUnaryOp op."""

    op_type = ir_graph.IR_OP_ELTWISE_UNARY


# ------------------------------------------------------------------------------
#   Gather
# ------------------------------------------------------------------------------
class GatherInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Gather op."""

    op_type = ir_graph.QNN_OP_GATHER

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder
    ):
        """Infer target input perm seqs.

        If the rank of the indices buffer is 1, then the gather ops is layout agnostic,
        Otherwise, keep the buffer in source format.
        """
        ranks = util.get_ranks(input_shapes)
        if ranks[1] == 1:
            target_perm_seq = super().infer_target_input_perm_seqs(
                [input_buffer_names[0]], [input_shapes[0]], layout_recorder
            )

            return [target_perm_seq[0], util.get_src_perm_seq(1)]
        else:
            return list(map(util.get_src_perm_seq, ranks))

    def infer_target_output_perm_seqs(self, target_input_perm_seqs, output_shapes):
        """Infer target output permute sequences.

        If the rank of the indices buffer is 1, then use the same permutation order as first input.
        Otherwise, keep the output buffer in source format.
        """
        if len(target_input_perm_seqs[1]) == 1:
            return [target_input_perm_seqs[0]]
        else:
            return [util.get_src_perm_seq(len(output_shapes[0]))]

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        # update attribute based on first target permute sequence
        return {"axis": target_input_perm_seqs[0].index(attrs["axis"])}

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names,
        input_shapes,
        output_shapes,
        src_attrs,
        layout_recorder,
    ):
        """Infer input/output permute sequences and update attributes."""
        log_assert(
            len(input_buffer_names) == 2,
            "Gather op inferer only support 2 inputs, but this op has {} inputs.",
            len(input_buffer_names),
        )
        return super().infer_target_layouts_and_attrs(
            input_buffer_names, input_shapes, output_shapes, src_attrs, layout_recorder
        )


# ------------------------------------------------------------------------------
#   Identity
# ------------------------------------------------------------------------------
class IdentityInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Identity op."""

    op_type = ir_graph.IR_OP_IDENTITY


# ------------------------------------------------------------------------------
#   IsInf
# ------------------------------------------------------------------------------
class IsInfInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for IsInf op."""

    op_type = ir_graph.QNN_OP_IS_INF


# ------------------------------------------------------------------------------
#   IsNan
# ------------------------------------------------------------------------------
class IsNanInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for IsNan op."""

    op_type = ir_graph.QNN_OP_IS_NAN


# ------------------------------------------------------------------------------
#   L2Norm
# ------------------------------------------------------------------------------
class L2NormInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for L2Norm op."""

    op_type = ir_graph.QNN_OP_L2_NORM

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        new_attrs = {}
        get_new_axis = lambda axis: target_input_perm_seqs[0].index(axis)

        if 'axis' in attrs:
            new_attrs['axis'] = get_new_axis(attrs['axis'])
        if 'axes' in attrs:
            new_attrs['axes'] = list(map(get_new_axis, attrs['axes']))

        return new_attrs


# ------------------------------------------------------------------------------
#   LayerNorm
# ------------------------------------------------------------------------------
class LayerNormInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for LayerNorm op."""

    op_type = ir_graph.QNN_OP_LAYER_NORM

    def infer_target_input_perm_seqs(
            self, input_buffer_names, input_shapes, layout_recorder, attrs
        ):
        """Infer target input perm seqs."""
        # LayerNorm requires gamma and beta to be unidirectional broadcastable to input. If target
        # perm seq breaks such constraint, this op must be treated like untrackable one. However,
        # the condition described above is expected not happening as such perm seq is considered
        # invalid as well. Thus, the implementation below simply ignores those cases.
        target_perm_seq = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )[0]

        # Ideally, there should be voting among input and params (gamma and beta) to reduce the
        # total number of inserted Transpose. However, the target perm seq of input is directly
        # applied to calculate the corresponding perm seq of params here for simplicity.
        param_perm_seq = tuple(np.argsort([target_perm_seq.index(axis) for axis in attrs['axes']]))

        return [target_perm_seq, param_perm_seq, param_perm_seq]

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        # Note that the sorting here is necessary since QNN expects sorted axes during validation.
        return {'axes': sorted(target_input_perm_seqs[0].index(axis) for axis in attrs['axes'])}

    def infer_target_layouts_and_attrs(
        self, input_buffer_names, input_shapes, output_shapes, src_attrs, layout_recorder
    ):
        """Infer input/output permute sequences and update attributes.

        This method is overridden to pass attribute `axes` for inferring target input perm seqs.
        """
        target_input_perm_seqs = self.infer_target_input_perm_seqs(
            input_buffer_names, input_shapes, layout_recorder, src_attrs
        )
        new_attrs = self.update_attr_with_target_input_perm_seqs(target_input_perm_seqs, src_attrs)
        target_output_perm_seqs = self.infer_target_output_perm_seqs(
            target_input_perm_seqs, output_shapes
        )
        return target_input_perm_seqs, target_output_perm_seqs, new_attrs


# ------------------------------------------------------------------------------
#   Logit
# ------------------------------------------------------------------------------
class LogitInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Logit op."""

    op_type = ir_graph.QNN_OP_LOGIT

    def infer_target_input_perm_seqs(self, input_buffer_names, input_shapes, layout_recorder):
        """Infer target input permute sequences."""
        target_perm_seq = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )[0]

        # No Epsilon case, LogitOp has only one input.
        if len(input_buffer_names) == 1:
            return [target_perm_seq]

        # The second input 1d tensor is an optional float epsilon, setting to (0,).
        return [target_perm_seq, (0,)]


# ------------------------------------------------------------------------------
#   LogSoftmax
# ------------------------------------------------------------------------------
class LogSoftmaxInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for LogSoftmax op."""

    op_type = ir_graph.QNN_OP_LOG_SOFTMAX

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        return {"axis": target_input_perm_seqs[0].index(attrs["axis"])}


# ------------------------------------------------------------------------------
#   Pad
# ------------------------------------------------------------------------------
class PadInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Pad op."""

    op_type = ir_graph.QNN_OP_PAD

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        old_pad_amount = attrs["pad_amount"]
        new_pad_amount = [old_pad_amount[axis] for axis in target_input_perm_seqs[0]]
        return {"pad_amount": new_pad_amount}


# ------------------------------------------------------------------------------
#   PreluOp
# ------------------------------------------------------------------------------
class PreluInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Prelu op."""

    op_type = ir_graph.QNN_OP_PRELU

    def infer_target_input_perm_seqs(self, input_buffer_names, input_shapes, layout_recorder):
        """Infer target input permute sequences.

        By QNN definition, Prelu slope must be unidirectionally broadcastable to input, and thus
        Prelu should be categorized as layout-agnostic. However, per-channel Prelu (especially for
        those in spatial-last format) may still exist as frontend translation does not insert
        Reshape around the slope to meet QNN spec. Therefore, such special case must be explicitly
        handled here.

        In brief, there are 3 cases, regarding slope's rank, to be considered.
            1. rank=0 -> input as agnostic, slope as (0,) to avoid explicit broadcast.
            2. rank=1 & per-channel -> input as heavily, slope as (0,) to avoid explicit broadcast.
            3. rank=others (including rank=1 but not per-channel) -> input and slope as agnostic.

        Note that since rank=0 (i.e., scalar) is represented by shape [1], a rank=1 slope is
        considered to be per-channel if and only if its shape is equal to input's channel dimension
        (i.e., dim=1) and does not match with input's last dimension. If the shape is equal to
        input's last dimension, the slope is in fact broadcastable and thus remains in
        layout-agnostic.
        """
        input_rank, slope_rank = util.get_ranks(input_shapes)
        if (
            slope_rank == 1
            and input_shapes[1] != [1]
            and input_shapes[1][0] != input_shapes[0][-1]
        ):
            # Case 2.
            # TODO: Remove this case once translations and optimizations are cleaned-up to fully
            # matching QNN spec.

            # Assure Prelu is per-channel by comparing slope's shape to input's second dimension.
            converter_utils.log_assert(
                input_shapes[1][0] == input_shapes[0][1],
                f'Expecting per-channel Prelu but got shapes {input_shapes}.'
            )

            # Borrow the logic from HeavilyOpLayoutInfererBase.
            src_layout = layout_recorder.get_src_layouts(self.op_type, input_rank)[0]
            desired_layout = layout_recorder.get_desired_layouts(self.op_type, input_rank)[0]
            target_perm_seq = tuple(src_layout.index(axis) for axis in desired_layout)

            return [target_perm_seq, (0,)]

        target_perm_seq = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )[0]
        if input_shapes[1] == [1]:
            # Case 1.
            return [target_perm_seq, (0,)]

        # Case 3.
        # Adopt identical perm seq as input for slope even when slope has smaller rank. Explicit
        # broadcast (i.e., inserting Reshape) will be applied later in mutator if necessary.
        return [target_perm_seq, target_perm_seq]


# ------------------------------------------------------------------------------
#   Quantize
# ------------------------------------------------------------------------------
class QuantizeInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Quantize op."""

    op_type = ir_graph.QNN_OP_QUANTIZE


# ------------------------------------------------------------------------------
#   RandomUniformLike
# ------------------------------------------------------------------------------
class RandomUniformLikeInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for RandomUniformLike op."""

    op_type = ir_graph.QNN_OP_RANDOM_UNIFORM_LIKE


# ------------------------------------------------------------------------------
#   RMSNorm
# ------------------------------------------------------------------------------
class RMSNormInferer(LayerNormInferer):
    """Layout inferer for RMSNorm op."""

    op_type = ir_graph.QNN_OP_RMS_NORM


# ------------------------------------------------------------------------------
#   Reduce
# ------------------------------------------------------------------------------
class ReduceInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Reduce op."""

    op_type = ir_graph.IR_OP_REDUCE

    def infer_target_output_perm_seqs(
        self, target_input_perm_seqs, output_shapes, src_attrs, new_attrs
    ):
        """Infer target output perm seqs."""
        if src_attrs['keep_dims']:
            return super().infer_target_output_perm_seqs(target_input_perm_seqs, output_shapes)

        def reduce_perm_seq(perm_seq, axes):
            """Simulate reduction on perm seq."""
            return tuple(axis for idx, axis in enumerate(perm_seq) if idx not in axes)

        # Infer by comparing reduced perm seqs between original and current ones.
        src_perm_seq = util.get_src_perm_seq(len(target_input_perm_seqs[0]))
        src_reduced_seq = reduce_perm_seq(src_perm_seq, src_attrs['axes'])

        if not src_reduced_seq:
            # Handle scalar case which is transformed to 1D instead.
            return [(0,)]

        tgt_reduced_seq = reduce_perm_seq(target_input_perm_seqs[0], new_attrs['axes'])
        return [tuple(src_reduced_seq.index(axis) for axis in tgt_reduced_seq)]

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        return {'axes': [target_input_perm_seqs[0].index(axis) for axis in attrs['axes']]}

    def infer_target_layouts_and_attrs(
        self, input_buffer_names, input_shapes, output_shapes, src_attrs, layout_recorder
    ):
        """Infer target layouts and attributes."""
        target_input_perm_seqs = self.infer_target_input_perm_seqs(
            input_buffer_names, input_shapes, layout_recorder
        )
        new_attrs = self.update_attr_with_target_input_perm_seqs(target_input_perm_seqs, src_attrs)
        target_output_perm_seqs = self.infer_target_output_perm_seqs(
            target_input_perm_seqs, output_shapes, src_attrs, new_attrs
        )
        return target_input_perm_seqs, target_output_perm_seqs, new_attrs


# ------------------------------------------------------------------------------
#   Softmax
# ------------------------------------------------------------------------------
class SoftmaxInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Softmax op."""

    op_type = ir_graph.QNN_OP_SOFTMAX

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        return {"axis": target_input_perm_seqs[0].index(attrs["axis"])}


# ------------------------------------------------------------------------------
#   SparseToDense
# ------------------------------------------------------------------------------
class SparseToDenseInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for SparseToDense op."""

    op_type = ir_graph.QNN_OP_SPARSE_TO_DENSE


# ------------------------------------------------------------------------------
#   Split
# ------------------------------------------------------------------------------
class SplitInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Split op."""

    op_type = ir_graph.QNN_OP_SPLIT

    def infer_target_output_perm_seqs(self, target_input_perm_seqs, output_shapes):
        """Infer target output permute sequences.

        This method is overridden to match the number of outputs.
        """
        return [target_input_perm_seqs[0]] * len(output_shapes)

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        return {"axis": target_input_perm_seqs[0].index(attrs["axis"])}


# ------------------------------------------------------------------------------
#   StridedSlice
# ------------------------------------------------------------------------------
class StridedSliceInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for StridedSlice op."""

    op_type = ir_graph.QNN_OP_STRIDED_SLICE

    # TODO: Revisit for possible optimization on `shrink_axes` and `new_axes_mask` cases.

    def infer_target_input_perm_seqs(
            self, input_buffer_names, input_shapes, layout_recorder, attrs
    ):
        """Infer target input permute sequences.

        For cases with `shrink_axes != 0` or `new_axes_mask != 0`, StridedSlice is fallen-back to
        layout-untrackable as the ranks are changed.
        """
        if attrs['shrink_axes'] == 0 and attrs['new_axes_mask'] == 0:
            return super().infer_target_input_perm_seqs(
                input_buffer_names, input_shapes, layout_recorder
            )
        return list(map(util.get_src_perm_seq, util.get_ranks(input_shapes)))

    def infer_target_output_perm_seqs(self, target_input_perm_seqs, output_shapes, attrs):
        """Infer target output permute sequences.

        For cases with `shrink_axes != 0` or `new_axes_mask != 0`, StridedSlice is fallen-back to
        layout-untrackable as the ranks are changed.
        """
        if attrs['shrink_axes'] == 0 and attrs['new_axes_mask'] == 0:
            return super().infer_target_output_perm_seqs(target_input_perm_seqs, output_shapes)
        return list(map(util.get_src_perm_seq, util.get_ranks(output_shapes)))

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute.

        For cases with `shrink_axes != 0` or `new_axes_mask != 0`, StridedSlice is fallen-back to
        layout-untrackable, and thus no attributes are updated.
        """
        if attrs['shrink_axes'] != 0 or attrs['new_axes_mask'] != 0:
            return {}

        begins, ends, strides = list(map(list, zip(*attrs['ranges'])))
        permute = lambda target: [target[axis] for axis in target_input_perm_seqs[0]]
        return {'ranges': list(map(list, zip(permute(begins), permute(ends), permute(strides))))}

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names,
        input_shapes,
        output_shapes,
        src_attrs,
        layout_recorder,
    ):
        """Infer input/output permute sequences and update attributes.

        This method is overridden to pass attributes for determining target perm seqs.
        """
        target_input_perm_seqs = self.infer_target_input_perm_seqs(
            input_buffer_names, input_shapes, layout_recorder, src_attrs
        )
        new_attrs = self.update_attr_with_target_input_perm_seqs(
            target_input_perm_seqs, src_attrs
        )
        target_output_perm_seqs = self.infer_target_output_perm_seqs(
            target_input_perm_seqs, output_shapes, src_attrs
        )
        return target_input_perm_seqs, target_output_perm_seqs, new_attrs


# ------------------------------------------------------------------------------
#   Tile
# ------------------------------------------------------------------------------
class TileInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Tile op."""

    op_type = ir_graph.QNN_OP_TILE

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attr):
        """Update attribute."""
        return {'multiples': [attr['multiples'][axis] for axis in target_input_perm_seqs[0]]}


# ------------------------------------------------------------------------------
#   Transpose
# ------------------------------------------------------------------------------
class TransposeInferer(AgnosticOpLayoutInfererBase):
    """Layout inferer for Transpose op."""

    op_type = ir_graph.QNN_OP_TRANSPOSE

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder
    ):
        """Infer target input permute sequences.

        This method is overridden to make the source permute sequence having the highest priority
        during searching over the preferred ones while the rest is identical to the base class.
        """
        rank = util.get_ranks(input_shapes)[0]

        existing_perm_seqs = layout_recorder.get_perm_seqs(input_buffer_names[0], rank)
        src_perm_seq = util.get_src_perm_seq(rank)
        preferred_perm_seqs = layout_recorder.get_preferred_perm_seqs(rank)

        # Note that source permute sequence is deliberately prepanded to the preferred permute
        # sequences.
        target_perm_seq = util.search_preferred_perm_seqs_in_order(
            [src_perm_seq] + preferred_perm_seqs, existing_perm_seqs
        )
        if not target_perm_seq:
            target_perm_seq = existing_perm_seqs[0]

        return [target_perm_seq]

    def infer_target_output_perm_seqs(self, target_input_perm_seqs, output_shapes):
        """Infer target output permute sequences.

        This method is overridden to adopt source permute sequence as target output one.
        """
        rank = len(output_shapes[0])
        return [util.get_src_perm_seq(rank)]

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute.

        As current target output permute sequence is the source one, attribute perm must be
        updated to make this Transpose logically identical to the source one. Therefore, there are
        two steps to calculate the new attribute:
            1. Transpose input back to source format.
            2. Combine two permute sequences into one.
        """
        org_perm = attrs["perm"]

        # Step 1.
        rank = len(target_input_perm_seqs[0])
        fallback_perm = util.calculate_perm_seq(
            target_input_perm_seqs[0], util.get_src_perm_seq(rank)
        )

        # Step 2.
        combined_perm = [fallback_perm[i] for i in org_perm]

        return {"perm": combined_perm}

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names,
        input_shapes,
        output_shapes,
        src_attrs,
        layout_recorder,
    ):
        """Infer input/output permute sequences and update attributes."""
        # Ensure input shape and permute order having the same rank.
        log_assert(
            len(input_shapes[0]) == len(src_attrs["perm"]),
            "Input rank doesn't matches rank of permute order.\nInput rank {}, rank of permute order {}.",
            len(input_shapes[0]),
            len(src_attrs["perm"]),
        )
        return super().infer_target_layouts_and_attrs(
            input_buffer_names, input_shapes, output_shapes, src_attrs, layout_recorder
        )
