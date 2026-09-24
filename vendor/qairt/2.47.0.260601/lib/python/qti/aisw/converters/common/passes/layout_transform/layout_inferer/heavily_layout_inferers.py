# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define layout inferers for heavily layout-sensitive ops."""

from qti.aisw.converters.common.passes.layout_transform import util
from qti.aisw.converters.common.passes.layout_transform.layout_inferer import (
    layout_inferer_base,
)
from qti.aisw.converters.common.passes.layout_transform.layout_recorder import (
    LayoutRecorder,
)
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.passes.layout_transform import util
from qti.aisw.converters.common.utils.converter_utils import log_assert


class HeavilyOpLayoutInfererBase(layout_inferer_base.LayoutInfererBase):
    """Inferer base for heavily layout-sensitive op."""

    def infer_target_input_perm_seqs(
        self, input_buffer_names: list, input_shapes: list, layout_recorder: LayoutRecorder
    ):
        """Infer target input permute sequences.

        For heavily layout-sensitive op, target input permute sequences are determined by comparing
        source and desired layouts from layout tables. Therefore, corresponding op type is expected
        to be defined in `layout_defs.py`.
        """
        rank = len(input_shapes[0])
        src_input_layouts = layout_recorder.get_src_layouts(self.op_type, rank)
        desired_input_layouts = layout_recorder.get_desired_layouts(self.op_type, rank)

        target_input_perm_seqs = []
        for src_layout, desired_layout in zip(src_input_layouts, desired_input_layouts):
            target_input_perm_seqs.append(
                tuple(int(src_layout.index(axis)) for axis in desired_layout)
            )
        return target_input_perm_seqs

    def infer_target_output_perm_seqs(self, target_perm_seqs: list, output_shapes: list):
        """Infer target output permute sequences.

        By default, target output permute sequece is identical to the first input.
        Inherited classes may override this method for multi-outputs or rank changing cases.
        """
        return [target_perm_seqs[0]]

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs: list, attrs):
        """Update attribute.

        Inherited classes may override this function if any attribute must be updated.
        """
        return {}


# ------------------------------------------------------------------------------
#   Batchnorm
# ------------------------------------------------------------------------------
class BatchnormInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Batchnorm op."""

    op_type = ir_graph.QNN_OP_BATCHNORM

# ------------------------------------------------------------------------------
#   BatchToSpace
# ------------------------------------------------------------------------------
class BatchToSpaceInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for BatchToSpace op."""

    op_type = ir_graph.QNN_OP_BATCH_TO_SPACE


# ------------------------------------------------------------------------------
#   ChannelShuffleOp
# ------------------------------------------------------------------------------
class ChannelShuffleInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for ChannelShuffle op."""

    op_type = ir_graph.QNN_OP_CHANNEL_SHUFFLE

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs):
        """Update attribute."""
        return {'axis': target_input_perm_seqs[0].index(attrs['axis'])}


# ------------------------------------------------------------------------------
#   ColorTransform
# ------------------------------------------------------------------------------
class ColorTransformInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for ColorTransform op.

    Note that ColorTransform represents QNN ArgbtoRgb and Nv12ToRgb/Nv21ToRgb operations and is
    differentiated by the attribute input_encoding_in during translation. Therefore, the inferer
    supports input rank 4 and 2 cases, respectively (refer to layout table for layout definition).
    Additionally, since ColorTransform rank 2 case is layout-heavily for output only but not input,
    output perm seq and attribute shape must be handled correspondingly.
    """

    # TODO
    # Replace with QNN macro once ColorTransform is aligned with QNN definition.
    op_type = 'color_transform'

    def infer_target_output_perm_seqs(self, target_perm_seqs, output_shapes, layout_recorder):
        """Infer target output perm seqs."""
        if len(target_perm_seqs[0]) == 4:
            return super().infer_target_output_perm_seqs(target_perm_seqs, output_shapes)

        # Exploit logic in inferring target input perm seqs for input rank 2 case.
        return self.infer_target_input_perm_seqs(None, output_shapes, layout_recorder)

    def update_attr_with_target_input_perm_seqs(self, target_output_perm_seqs, attrs):
        """Update attribute."""
        return {'shape': [attrs['shape'][axis] for axis in target_output_perm_seqs[0]]}

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names,
        input_shapes,
        output_shapes,
        src_attrs,
        layout_recorder,
    ):
        """Infer target layouts and attributes."""
        target_input_perm_seqs = self.infer_target_input_perm_seqs(
            input_buffer_names, input_shapes, layout_recorder
        )
        target_output_perm_seqs = self.infer_target_output_perm_seqs(
            target_input_perm_seqs, output_shapes, layout_recorder
        )
        # Note that target output perm seq is deliberatively used for transposing attribute shape.
        new_attrs = self.update_attr_with_target_input_perm_seqs(
            target_output_perm_seqs, src_attrs
        )
        return target_input_perm_seqs, target_output_perm_seqs, new_attrs


# ------------------------------------------------------------------------------
#   Conv1d
# ------------------------------------------------------------------------------
class Conv1dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Conv1d op."""

    op_type = ir_graph.IR_OP_CONV_1D


# ------------------------------------------------------------------------------
#   Conv2d
# ------------------------------------------------------------------------------
class Conv2dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Conv2d op."""

    op_type = ir_graph.QNN_OP_CONV_2D


# ------------------------------------------------------------------------------
#   Conv3d
# ------------------------------------------------------------------------------
class Conv3dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Conv3d op."""

    op_type = ir_graph.QNN_OP_CONV_3D


# ------------------------------------------------------------------------------
#   CropAndResize
# ------------------------------------------------------------------------------
class CropAndResizeInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for CropAndResize op."""

    op_type = ir_graph.QNN_OP_CROP_AND_RESIZE

    def infer_target_input_perm_seqs(self, input_buffer_names, input_shapes, layout_recorder):
        """Infer target input perm seqs."""
        target_perm_seq = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )[0]
        return [
            target_perm_seq,
            util.get_src_perm_seq(len(input_shapes[1])),
            util.get_src_perm_seq(len(input_shapes[2]))
        ]


# ------------------------------------------------------------------------------
#   DepthToSpace
# ------------------------------------------------------------------------------
class DepthToSpaceInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for DepthToSpace op."""

    op_type = ir_graph.QNN_OP_DEPTH_TO_SPACE


# ------------------------------------------------------------------------------
#   DepthWiseConv1d
# ------------------------------------------------------------------------------
class DepthWiseConv1dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for DepthWiseConv1d op."""

    op_type = ir_graph.IR_OP_DEPTH_WISE_CONV_1D


# ------------------------------------------------------------------------------
#   DepthWiseConv2d
# ------------------------------------------------------------------------------
class DepthWiseConv2dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for DepthWiseConv2d op."""

    op_type = ir_graph.QNN_OP_DEPTH_WISE_CONV_2D


# ------------------------------------------------------------------------------
#   GenerateProposals
# ------------------------------------------------------------------------------
class GenerateProposalsInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for GenerateProposals op."""

    op_type = ir_graph.QNN_OP_GENERATE_PROPOSALS

    def infer_target_input_perm_seqs(self, input_buffer_names, input_shapes, layout_recorder):
        """Infer target input permute sequences.

        GenerateProposals has inputs in shapes of (assuming in spatial-first format):
            - input0: [batch, height, width, num_anchors]
            - input1: [batch, height, width, num_anchors * 4]
            - input2: [num_anchors, 4]
            - input3: [batch, 2]

        Therefore, the first two inputs are considered as layout-heavily while the rest ones are
        categorized as layout-untrackable.
        """
        target_perm_seqs = super().infer_target_input_perm_seqs(
            input_buffer_names[:2], input_shapes[:2], layout_recorder
        )
        return [
            *target_perm_seqs,
            *list(map(util.get_src_perm_seq, util.get_ranks(input_shapes[2:])))
        ]

    def infer_target_output_perm_seqs(self, target_perm_seqs, output_shapes):
        """Infer target output permute sequences."""
        return list(map(util.get_src_perm_seq, util.get_ranks(output_shapes)))


# ------------------------------------------------------------------------------
#   GridSampleOp
# ------------------------------------------------------------------------------
class GridSampleInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for GridSample op."""

    op_type = ir_graph.QNN_OP_GRID_SAMPLE

    def infer_target_input_perm_seqs(self, input_buffer_names, input_shapes, layout_recorder):
        """Infer target input permute sequences.

        Grid Sample Op has two input but only first input is layout sensitive.
        Second input should be Kept in Source format.
        """
        ranks = util.get_ranks(input_shapes)
        target_perm_seq = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )
        return [target_perm_seq[0], util.get_src_perm_seq(ranks[1])]

    def infer_target_layouts_and_attrs(
            self,
            input_buffer_names,
            input_shapes,
            output_shapes,
            src_attrs,
            layout_recorder
    ):
        """Infer input/output permute sequences and update attributes."""
        log_assert(
            len(input_buffer_names) == 2,
            "GridSample op inferer only support 2 inputs, but this op has {} inputs.",
            len(input_buffer_names)
        )
        log_assert(
            len(input_shapes[0]) == 4 or len(input_shapes[0]) == 5,
            "GridSample op inferer only support 4D or 5D inputs, but this op has {}D inputs.",
            len(input_shapes[0])
        )
        return super().infer_target_layouts_and_attrs(
            input_buffer_names, input_shapes, output_shapes, src_attrs, layout_recorder
        )


# ------------------------------------------------------------------------------
#   GroupNorm
# ------------------------------------------------------------------------------
class GroupNormInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for GroupNorm op."""

    op_type = ir_graph.QNN_OP_GROUP_NORM


# ------------------------------------------------------------------------------
#   Gru
# ------------------------------------------------------------------------------
class GruInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Gru op."""
    op_type = ir_graph.QNN_OP_GRU

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder
    ):
        """Infer target input permute sequences."""
        # First input_buffer will be transfromed based on src layouts and desired layouts
        target_input_perm_seqs = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )

        # Other inputs should not be transformed, so append src_perm_seq for them
        for input_rank in util.get_ranks(input_shapes[1:]):
            target_input_perm_seqs.append(util.get_src_perm_seq(input_rank))

        return target_input_perm_seqs

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs={}):
        """Update attribute. based on input permute sequence."""
        # TODO: Check whether it's better to has "False" as default value in master opdef
        return {"time_major": False}

    def infer_target_output_perm_seqs(self, target_perm_seqs, output_shapes):
        """Infer target output permute sequences."""
        ranks = util.get_ranks(output_shapes)

        # 2nd output remains as src-format
        return [target_perm_seqs[0], util.get_src_perm_seq(ranks[1])]


# ------------------------------------------------------------------------------
#   InstanceNormOp
# ------------------------------------------------------------------------------
class InstanceNormInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for InstanceNorm op."""

    op_type = ir_graph.QNN_OP_INSTANCE_NORM


# ------------------------------------------------------------------------------
#   LrnOp
# ------------------------------------------------------------------------------
class LrnInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Lrn op."""

    op_type = ir_graph.QNN_OP_LRN


# ------------------------------------------------------------------------------
#   Lstm
# ------------------------------------------------------------------------------
class LstmInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Lstm op."""

    op_type = ir_graph.QNN_OP_LSTM

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder
    ):
        """Infer target input permute sequences."""
        # First input_buffer will be transfromed based on src layouts and desired layouts
        target_input_perm_seqs = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )

        # Other inputs should not be transformed, so append src_perm_seq for them
        for input_rank in util.get_ranks(input_shapes[1:]):
            target_input_perm_seqs.append(util.get_src_perm_seq(input_rank))

        return target_input_perm_seqs

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs={}):
        """Update attribute. based on input permute sequence."""
        # TODO: Check whether it's better to has "False" as default value in master opdef
        return {"time_major": False}

    def infer_target_output_perm_seqs(self, target_perm_seqs, output_shapes):
        """Infer target output permute sequences."""
        ranks = util.get_ranks(output_shapes)

        # Except 1st output, other outputs remains as src-format
        return [
            target_perm_seqs[0],
            util.get_src_perm_seq(ranks[1]),
            util.get_src_perm_seq(ranks[2]),
        ]


# ------------------------------------------------------------------------------
#   MergedWeightsGru
# ------------------------------------------------------------------------------
class MergedWeightsGruInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for MergedWeightsGru op."""

    op_type = ir_graph.IR_OP_MERGED_WEIGHTS_GRU

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder
    ):
        """Infer target input permute sequences."""
        # First input_buffer will be transfromed based on src layouts and desired layouts
        target_input_perm_seqs = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )

        # Other inputs should not be transformed, so append src_perm_seq for them
        for input_rank in util.get_ranks(input_shapes[1:]):
            target_input_perm_seqs.append(util.get_src_perm_seq(input_rank))

        return target_input_perm_seqs

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs={}):
        """Update attribute. based on input permute sequence."""
        # TODO: Check whether it's better to has "False" as default value in master opdef
        return {"time_major": False}

    def infer_target_output_perm_seqs(self, target_perm_seqs, output_shapes):
        """Infer target output permute sequences."""
        ranks = util.get_ranks(output_shapes)

        # 2nd output remains as src-format
        return [target_perm_seqs[0], util.get_src_perm_seq(ranks[1])]


# ------------------------------------------------------------------------------
#   Pool1d
# ------------------------------------------------------------------------------
class Pool1dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Pool1d op."""

    op_type = ir_graph.IR_OP_POOL1D


# ------------------------------------------------------------------------------
#   Pool2d
# ------------------------------------------------------------------------------
class Pool2dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Pool2d op."""

    op_type = ir_graph.IR_OP_POOL2D


# ------------------------------------------------------------------------------
#   Pool3d
# ------------------------------------------------------------------------------
class Pool3dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Pool3d op."""

    op_type = ir_graph.IR_OP_POOL3D


# ------------------------------------------------------------------------------
#   ResizeOp
# ------------------------------------------------------------------------------
class ResizeInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for Resize op."""

    op_type = ir_graph.QNN_OP_RESIZE


# ------------------------------------------------------------------------------
#   RoiAlignOp
# ------------------------------------------------------------------------------
class RoiAlignInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for RoiAlign op."""

    # TODO
    # Replace with QNN macro once RoiAlignOp is aligned with QNN.
    op_type = 'roi_align'

    def infer_target_input_perm_seqs(self, input_buffer_names, input_shapes, layout_recorder):
        """Infer target input perm seqs."""
        ranks = util.get_ranks(input_shapes)
        converter_utils.log_assert(
            ranks == [4, 2, 1],
            f'Expecting input ranks [4, 2, 1] for {self.op_type} but got {ranks}.'
        )

        target_perm_seq = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )[0]
        return [target_perm_seq, (0, 1), (0,)]


# ------------------------------------------------------------------------------
#   RoiPoolingOp
# ------------------------------------------------------------------------------
class RoiPoolingInferer(RoiAlignInferer):
    """Layout inferer for RoiPooling op."""

    # TODO
    # Replace with QNN macro once RoiPoolingOp is aligned with QNN.
    op_type = 'roi_pooling'


# ------------------------------------------------------------------------------
#   RolledLstm
# ------------------------------------------------------------------------------
class RolledLstmInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for RolledLstm op."""

    op_type = ir_graph.IR_OP_ROLLED_LSTM

    def infer_target_input_perm_seqs(
        self, input_buffer_names, input_shapes, layout_recorder
    ):
        """Infer target input permute sequences."""
        # First input_buffer will be transfromed based on src layouts and desired layouts
        target_input_perm_seqs = super().infer_target_input_perm_seqs(
            [input_buffer_names[0]], [input_shapes[0]], layout_recorder
        )

        # Other inputs should not be transformed, so append src_perm_seq for them
        for input_rank in util.get_ranks(input_shapes[1:]):
            target_input_perm_seqs.append(util.get_src_perm_seq(input_rank))

        return target_input_perm_seqs

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs, attrs={}):
        """Update attribute. based on input permute sequence."""
        # TODO: Check whether it's better to has "False" as default value in master opdef
        return {"time_major": False}

    def infer_target_output_perm_seqs(self, target_perm_seqs, output_shapes):
        """Infer target output permute sequences."""
        ranks = util.get_ranks(output_shapes)

        # Except 1st output, other outputs remains as src-format
        return [
            target_perm_seqs[0],
            util.get_src_perm_seq(ranks[1]),
            util.get_src_perm_seq(ranks[2]),
        ]

# ------------------------------------------------------------------------------
#   SpaceToBatch
# ------------------------------------------------------------------------------
class SpaceToBatchInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for SpaceToBatch op."""

    op_type = ir_graph.QNN_OP_SPACE_TO_BATCH


# ------------------------------------------------------------------------------
#   SpaceToDepth
# ------------------------------------------------------------------------------
class SpaceToDepthInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for SpaceToDepth op."""

    op_type = ir_graph.QNN_OP_SPACE_TO_DEPTH


# ------------------------------------------------------------------------------
#   TransposeConv1d
# ------------------------------------------------------------------------------
class TransposeConv1dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for TransposeConv1d op."""

    op_type = ir_graph.IR_OP_TRANSPOSE_CONV_1D


# ------------------------------------------------------------------------------
#   TransposeConv2d
# ------------------------------------------------------------------------------
class TransposeConv2dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for TransposeConv2d op."""

    op_type = ir_graph.QNN_OP_TRANSPOSE_CONV_2D


# ------------------------------------------------------------------------------
#   TransposeConv3d
# ------------------------------------------------------------------------------
class TransposeConv3dInferer(HeavilyOpLayoutInfererBase):
    """Layout inferer for TransposeConv3d op."""

    op_type = ir_graph.QNN_OP_TRANSPOSE_CONV_3D
