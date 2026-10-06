# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""MaskedSoftmax op properties

MaskedSoftmax has 2 inputs (data, mask), 1 parameter (mode).  The mode parameter
is an int and the default value is defined here.

:var NUM_INPUTS: Number of inputs.
:type NUM_INPUTS: int
:var NUM_PARAMS: Number of parameters.
:type NUM_PARAMS: int
:var MODE_PARAM_IDX: The index of the mode parameter if parameters are in an ordered
    list instead of a dictionary (e.g., PyTorch).
:type MODE_PARAM_IDX: int
:var MODE_UNCOMPRESSED_VAL: The value that represents Uncompressed mode.
:type MODE_UNCOMPRESSED_VAL: int
:var MODE_COMPRESSED_VAL: The value that represents Compressed mode.
:type MODE_COMPRESSED_VAL: int
:var MODE_DEFAULT_VAL: The default value for the mode parameter.
:type MODE_DEFAULT_VAL: int
"""

from qti.aisw.converters.common import ir_graph

# These fields define a contract between PyTorch, TorchScript, ONNX, and
# the Converter. Any changes here should impact all four and require
# verification that the translation Framework->QNN IR occurs correctly.

NUM_INPUTS = 2
NUM_PARAMS = 1
MODE_PARAM_IDX = 0
MODE_UNCOMPRESSED_VAL = ir_graph.QNN_OP_MASKED_SOFTMAX_MODE_UNCOMPRESSED
MODE_COMPRESSED_VAL = ir_graph.QNN_OP_MASKED_SOFTMAX_MODE_COMPRESSED
MODE_DEFAULT_VAL = MODE_UNCOMPRESSED_VAL
