# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Definitions related to QNN that are LWQ Wrapper and IRGraphUpdater"""

import numpy as np

# pylint: disable=import-error
from qti.aisw.converters.common import ir_graph as ir_graph_lib

# QNN op types which correspond to data movement ops
DATA_MOVEMENT_OP_TYPES = {'BatchPermutation', 'ChannelShuffle', 'CropAndResize', 'DepthToSpace', 'Gather',
                          'GatherElements', 'GatherNd', 'Pad', 'Pool', 'Pool3d', 'Reduce', 'Reshape', 'Resize',
                          'ResizeNearestNeighbor', 'ResizeBilinear', 'SpaceToDepth', 'Split', 'StridedSlice', 'TopK',
                          'Transpose', 'BatchToSpace', 'SpaceToBatch', 'Tile', 'CreateSparse', 'Buffer'}

INTEGER_DTYPES = [ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_INT_8, ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_INT_16,
                  ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_INT_32, ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_INT_64,
                  ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_UINT_8, ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_UINT_16,
                  ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_UINT_32, ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_UINT_64,
                  ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_BOOL_8]

FXP_DTYPES = [ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8, ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_16,
              ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_32,
              ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8, ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_16,
              ir_graph_lib.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_32]

VALID_ENCODING_TYPES = [ir_graph_lib.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
                        ir_graph_lib.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                        ir_graph_lib.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET]

VALID_AXIS_ENCODING_TYPES = [ir_graph_lib.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
                             ir_graph_lib.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET]

OPS_WITH_BIAS = {'Batchnorm', 'Conv1d', 'Conv2d', 'Conv3d', 'DepthWiseConv1d', 'DepthWiseConv2d',
                 'FullyConnected', 'TransposeConv1d', 'TransposeConv2d', 'TransposeConv3d',
                 'InstanceNorm', 'LayerNorm', 'GroupNorm'}

def is_qnn_data_movement_op(op: ir_graph_lib.IrOp):
    """
    Return true if op is a data movement op.

    :param op: Op to check
    :return: True if op is a data movement op, False otherwise
    """
    if op.type not in DATA_MOVEMENT_OP_TYPES:
        return False
    if op.type == 'Reduce':
        return op.attrs_dict['reduce_type'] in {'ReduceMin', 'ReduceMax'}
    if op.type == 'Pool':
        return op.attrs_dict['pool_type'] in {'PoolMax2d'}
    return True


def is_integer_dtype(dtype: ir_graph_lib.Qnn_DataType_t):
    """
    Check if the data type is an integer
    :param dtype:  dtype to check
    :return: True if the datatype is quantized, False otherwise
    """
    return dtype in INTEGER_DTYPES


def is_fxp_qnn_dtype(dtype: ir_graph_lib.Qnn_DataType_t):
    """
    Check if the data type is fixed point
    :param dtype:  dtype to check
    :return: True if the datatype is fixed point, False otherwise
    """
    return dtype in FXP_DTYPES


def has_dummy_bias(op: ir_graph_lib.IrOp, input_tensor: ir_graph_lib.IrTensor) -> bool:
    """
    Check if given op has dummy bias (bias vector with all zeros) or not.
    :param op: Given IR Op
    :input_tensor: Tensor of given op to check
    """
    if op.type in OPS_WITH_BIAS and input_tensor == op.inputs()[2]:
        bias_vector = input_tensor.get_data()
        is_dummy_bias = not np.any(bias_vector)
        return is_dummy_bias

    return False
