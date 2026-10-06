# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""Buffer op properties

Buffer has 2 inputs (input, reset), 4 parameter (buffer_size, buffer_dim, mode and stride).
The mode parameter is an int and the default value is defined here.

"""

from qti.aisw.converters.common import ir_graph

# These fields define a contract between PyTorch, TorchScript, ONNX, and
# the Converter. Any changes here should impact all four and require
# verification that the translation Framework->QNN IR occurs correctly.

NUM_INPUTS = 2
NUM_PARAMS = 4
BUFFER_MODE_BLOCKING = ir_graph.QNN_OP_BUFFER_MODE_BLOCKING
BUFFER_MODE_NON_BLOCKING_LEFT = ir_graph.QNN_OP_BUFFER_MODE_NON_BLOCKING_LEFT
BUFFER_MODE_NON_BLOCKING_RIGHT = ir_graph.QNN_OP_BUFFER_MODE_NON_BLOCKING_RIGHT
BUFFER_MODE_DEFAULT_VAL = BUFFER_MODE_BLOCKING
BUFFER_STRIDE_DEFAULT_VAL = 1
BUFFER_PADDING_DEFAULT_VAL = 0
