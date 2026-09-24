# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define dummy layout inferers for ops that don't require actual layout inferer implementation.

Note:
    When adding a new op, the recommended approach is to implement a new layout inferer
    for the op inheriting base class from one of "HeavilyOpLayoutInfererBase",
    "AgnosticOpLayoutInfererBase", or "UntrackableOpLayoutInfererBase".

    However, if it can be guaranteed that the new op will never appear in the input graph
    to layout transform stage during graph optimization, it is possible
    to register "dummy implementation" without real layout-inferring logic for the op
    in this file.

    Before implementing a dummy inferer for an op, the author is responsible for running
    validation tests on modelzoo for qairt-converter to make sure
    the previously mentioned condition holds.

    Do not blindly add new ops to this file without proper validation. This can lead
    to failure during graph optimization in qairt-converter.
"""

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.passes.layout_transform.layout_inferer import (
    layout_inferer_base,
)
from qti.aisw.converters.common.passes.layout_transform.layout_recorder import (
    LayoutRecorder,
)


class DummyOpLayoutInfererBase(layout_inferer_base.LayoutInfererBase):
    """Base for dummy inferers."""

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names: list,
        input_shapes: list,
        output_shapes: list,
        src_attrs,
        layout_recorder: LayoutRecorder,
    ):
        """Raise NotImplementedError because dummy layout inferer should never be used"""
        raise NotImplementedError(
            f"Unexpected invocation of dummy implemention {self.__class__.__name__}."
            "Please provide real layout-inferer implementation."
        )


class BoxDecoderDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for BoxDecoder."""

    op_type = ir_graph.IR_OP_BOX_DECODER_TYPE


class DetectionOutputDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for DetectionOutput."""

    op_type = ir_graph.QNN_OP_DETECTION_OUTPUT


class ErfDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for ERF."""

    op_type = ir_graph.IR_OP_ERF


class ExtractGlimpseDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for ExtractGlimpse."""

    op_type = ir_graph.QNN_OP_EXTRACT_GLIMPSE


class ExtractPatchesDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for ExtractPatches."""

    op_type = ir_graph.QNN_OP_EXTRACT_PATCHES


class GetSparseIndicesDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for GetSparseIndices."""

    op_type = ir_graph.QNN_OP_GET_SPARSE_INDICES


class GetSparseValuesDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for GetSparseValues."""

    op_type = ir_graph.QNN_OP_GET_SPARSE_VALUES


class ImageProjectionTransformDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for ImageProjectionTransform."""

    op_type = ir_graph.QNN_OP_IMAGE_PROJECTION_TRANSFORM


class InverseDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for Inverse."""

    op_type = ir_graph.IR_OP_INVERSE


class MeanDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for Mean."""

    op_type = "Mean"


class MomentsDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for Moments."""

    op_type = ir_graph.QNN_OP_MOMENTS


class ProposalDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for Proposal."""

    op_type = "proposal"


class ThresholdReluDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for ThresholdRelu."""

    op_type = "ThresholdedRelu"


class UdlDummyLayoutInferer(DummyOpLayoutInfererBase):
    """Dummy LayoutInferer for udl."""

    op_type = "udl"
