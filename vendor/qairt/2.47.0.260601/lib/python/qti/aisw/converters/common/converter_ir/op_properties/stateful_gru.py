# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""StatefulGru op properties

:var NUM_INPUTS: Number of inputs.
:type NUM_INPUTS: int
:var NUM_REQUIRED_INPUTS: Number of required inputs.
:type NUM_REQUIRED_INPUTS: int
:var NUM_OUTPUTS: Number of outputs.
:type NUM_OUTPUTS: int
"""

NUM_INPUTS = 7
NUM_REQUIRED_INPUTS = 3
NUM_OUTPUTS = 2

# Indices of inputs after initial conversion to RolledLstm node
IR_NUM_INPUTS = 6
IR_INPUT_IDX = 0
IR_INITIAL_H_IDX = 1
IR_INPUT_WEIGHTS_IDX = 2
IR_HIDDEN_STATE_WEIGHTS_IDX = 3
IR_GATE_BIASES_IDX = 4
IR_RESET_IDX = 5

# Map from IR indices to ONNX indices
IR_TO_ONNX_INDICES = {
    IR_INPUT_IDX: 0,
    IR_INPUT_WEIGHTS_IDX: 1,
    IR_HIDDEN_STATE_WEIGHTS_IDX: 2,
    IR_GATE_BIASES_IDX: 3,
    # ONNX input 4 is "sequence_lens", which is unused
    IR_INITIAL_H_IDX: 5,
    # Projection weights and biases combined in ONNX
    IR_RESET_IDX: 6,
}
