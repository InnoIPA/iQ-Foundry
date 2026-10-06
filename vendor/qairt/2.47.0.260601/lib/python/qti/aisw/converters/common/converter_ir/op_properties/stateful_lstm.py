# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""StatefulLstm op properties

:var NUM_INPUTS: Number of inputs.
:type NUM_INPUTS: int
:var NUM_PARAMS: Number of parameters.
:type NUM_PARAMS: int
"""

NUM_INPUTS = 9
NUM_REQUIRED_INPUTS = 3
NUM_OUTPUTS = 3

# Indices of inputs after initial conversion to RolledLstm node
IR_NUM_INPUTS = 11
IR_INPUT_IDX = 0
IR_INITIAL_H_IDX = 1
IR_INITIAL_C_IDX = 2
IR_INPUT_WEIGHTS_IDX = 3
IR_HIDDEN_STATE_WEIGHTS_IDX = 4
IR_GATE_BIASES_IDX = 5
IR_NORM_WEIGHTS_IDX = 6
IR_CELL_STATE_WEIGHTS_IDX = 7
IR_PROJ_WEIGHTS_IDX = 8
IR_PROJ_BIAS_IDX = 9
IR_RESET_IDX = 10

# Map from IR indices to ONNX indices
IR_TO_ONNX_INDICES = {
    IR_INPUT_IDX : 0,
    IR_INPUT_WEIGHTS_IDX : 1,
    IR_HIDDEN_STATE_WEIGHTS_IDX : 2,
    IR_GATE_BIASES_IDX : 3,
    # ONNX input 4 is "sequence_lens", which is unused
    IR_INITIAL_H_IDX : 5,
    IR_INITIAL_C_IDX : 6,
    # Projection weights and biases combined in ONNX
    IR_PROJ_WEIGHTS_IDX : 7,
    IR_PROJ_BIAS_IDX : 7,
    IR_RESET_IDX : 8,
}

# Indices of inputs after initial conversion to LSTM node (QAIRT multi-time-step flavor)
# These indices expand monolithic ONNX LSTM inputs into per-gate tensors to align with backend expectations.
LSTM_IR_NUM_INPUTS = 25
LSTM_IR_DATA_INPUT_IDX = 0
LSTM_INPUT_WEIGHT_IDX = 1
LSTM_INPUT_WEIGHT_TO_FORGET_GATE_IDX = 1
LSTM_INPUT_WEIGHT_TO_CELL_GATE_IDX = 2
LSTM_INPUT_WEIGHT_TO_OUTPUT_GATE_IDX = 3
LSTM_HIDDEN_WEIGHT_IDX = 4
LSTM_RECURRENT_WEIGHT_TO_FORGET_GATE_IDX = 4
LSTM_RECURRENT_WEIGHT_TO_CELL_GATE_IDX = 5
LSTM_RECURRENT_WEIGHT_TO_OUTPUT_GATE_IDX = 6
LSTM_BIAS_IDX = 7
LSTM_BIAS_TO_FORGET_GATE_IDX = 7
LSTM_BIAS_TO_CELL_GATE_IDX = 8
LSTM_BIAS_TO_OUTPUT_GATE_IDX = 9
LSTM_IR_INITIAL_H_IDX = 10
LSTM_IR_INITIAL_C_IDX = 11
LSTM_NORM_WEIGHT_TO_INPUT_GATE_IDX = 12
LSTM_NORM_WEIGHT_TO_FORGET_GATE_IDX = 13
LSTM_NORM_WEIGHT_TO_CELL_GATE_IDX = 14
LSTM_NORM_WEIGHT_TO_OUTPUT_GATE_IDX = 15
LSTM_INPUT_WEIGHT_TO_INPUT_GATE_IDX = 16
LSTM_RECURRENT_WEIGHT_TO_INPUT_GATE_IDX = 17
LSTM_CELL_WEIGHT_TO_INPUT_GATE_IDX = 18
LSTM_CELL_WEIGHT_TO_FORGET_GATE_IDX = 19
LSTM_CELL_WEIGHT_TO_OUTPUT_GATE_IDX = 20
LSTM_BIAS_TO_INPUT_GATE_IDX = 21
LSTM_PROJ_W_IDX = 22
LSTM_PROJ_B_IDX = 23
LSTM_IR_RESET_IDX = 24


# Map from LSTM IR indices to ONNX indices
# Notes:
# - ONNX input 4 is 'sequence_lens' (unused)
# - Projection weights and biases are combined at index 7 in ONNX
# - Reset input appears at index 8 in ONNX for the stateful flavor
LSTM_IR_TO_ONNX_INDICES = {
    LSTM_IR_DATA_INPUT_IDX : 0,
    LSTM_INPUT_WEIGHT_IDX : 1,
    LSTM_HIDDEN_WEIGHT_IDX : 2,
    LSTM_BIAS_IDX : 3,
    # ONNX input 4 is "sequence_lens", which is unused
    LSTM_IR_INITIAL_H_IDX : 5,
    LSTM_IR_INITIAL_C_IDX : 6,
    # Projection weights and biases combined in ONNX
    LSTM_PROJ_W_IDX : 7,
    LSTM_PROJ_B_IDX : 7,
    LSTM_IR_RESET_IDX : 8
}

# Map from per-gate IR indices to gate positions assuming ifoc gate order:
# input(i), forget(f), output(o), cell(c)
LSTM_IR_TO_GATE_INDICES = {
    LSTM_INPUT_WEIGHT_TO_FORGET_GATE_IDX : 1,
    LSTM_INPUT_WEIGHT_TO_CELL_GATE_IDX : 3,
    LSTM_INPUT_WEIGHT_TO_OUTPUT_GATE_IDX : 2,
    LSTM_INPUT_WEIGHT_TO_INPUT_GATE_IDX : 0,
    LSTM_RECURRENT_WEIGHT_TO_FORGET_GATE_IDX : 1,
    LSTM_RECURRENT_WEIGHT_TO_CELL_GATE_IDX : 3,
    LSTM_RECURRENT_WEIGHT_TO_OUTPUT_GATE_IDX : 2,
    LSTM_RECURRENT_WEIGHT_TO_INPUT_GATE_IDX : 0,
    LSTM_BIAS_TO_FORGET_GATE_IDX : 1,
    LSTM_BIAS_TO_CELL_GATE_IDX : 3,
    LSTM_BIAS_TO_OUTPUT_GATE_IDX : 2,
    LSTM_BIAS_TO_INPUT_GATE_IDX  : 0
}
