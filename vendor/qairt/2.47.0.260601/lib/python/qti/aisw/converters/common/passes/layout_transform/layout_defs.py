# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define expected layouts for heavily layout-sensitive ops."""

from qti.aisw.converters.common import ir_graph

# Table for default input layout in graph.
DEFAULT_INPUT_LAYOUT_TABLE = {
    "NCDHW": "NDHWC",
    "NDHWC": "NDHWC",
    "NCHW": "NHWC",
    "NHWC": "NHWC",
    "OIHW": "HWIO",
    "HWIO": "HWIO",
    "NCF": "NFC",
    "NFC": "NFC",
    "TNF": "NTF",
    "NTF": "NTF",
}

# Layout table records the expected layouts for heavily layout-sensitive ops with various ranks.
# For each op entry, ranks are mapped to lists of layouts in strs for each input. The ranks (key)
# are expected to be the possible ranks for the first input of heavily layout-sensitive ops, and
# the layouts (value) are the corresponding layouts for each input when the first input has the
# rank specified by key.
#
# For example of an entry in layout table:
#
#     <QNN_OP type>: {
#         rank1: [<1st input layout>, <2nd input layout>, ...],
#         rank2: [<1st input layout>, <2nd input layout>, ...],
#         ...
#     }

# Table for spatial-last layout.
SPATIAL_LAST_LAYOUT_TABLE = {
    ir_graph.QNN_OP_BATCHNORM: {
        5: ["NCDHW", "C", "C"],
        4: ["NCHW", "C", "C"],
        3: ["NCW", "C", "C"],
        2: ["NC", "C", "C"]
    },
    ir_graph.QNN_OP_BATCH_TO_SPACE: {
        4: ["NCHW"]
    },
    ir_graph.QNN_OP_CHANNEL_SHUFFLE: {
        4: ["NCHW"]
    },
    # TODO
    # Replace with QNN macro once ColorTransform is aligned with QNN definition.
    'color_transform': {
        2: ["NC"],
        4: ["NCHW"]
    },
    ir_graph.IR_OP_CONV_1D: {
        3: ["NCW", "OIW", "C"]
    },
    ir_graph.QNN_OP_CONV_2D: {
        4: ["NCHW", "OIHW", "C"]
    },
    ir_graph.QNN_OP_CONV_3D: {
        5: ["NCDHW", "OIDHW", "C"]
    },
    ir_graph.QNN_OP_CROP_AND_RESIZE: {
        4: ['NCHW']
    },
    ir_graph.QNN_OP_DEPTH_TO_SPACE: {
        4: ["NCHW"]
    },
    ir_graph.IR_OP_DEPTH_WISE_CONV_1D: {
        3: ["NCW", "OIW", "C"]
    },
    ir_graph.QNN_OP_DEPTH_WISE_CONV_2D: {
        4: ["NCHW", "OIHW", "C"]
    },
    ir_graph.QNN_OP_GENERATE_PROPOSALS: {
        4: ["NCHW", "NCHW"]
    },
    ir_graph.QNN_OP_GRID_SAMPLE: {
        4: ["NCHW"],
        5: ["NCDHW"]
    },
    ir_graph.QNN_OP_GROUP_NORM: {
        5: ["NCDHW", "C", "C"],
        4: ["NCHW", "C", "C"],
        3: ["NCW", "C", "C"]
    },
    ir_graph.QNN_OP_GRU: {
        3: ["TNF"]
    },
    ir_graph.QNN_OP_INSTANCE_NORM: {
        5: ["NCDHW", "C", "C"],
        4: ["NCHW", "C", "C"],
        3: ["NCW", "C", "C"]
    },
    ir_graph.QNN_OP_LRN: {
        5: ["NCDHW"],
        4: ["NCHW"],
        3: ["NCW"]
    },
    ir_graph.QNN_OP_LSTM: {
        3: ["TNF"]
    },
    ir_graph.IR_OP_MERGED_WEIGHTS_GRU: {
        3: ["TNF"]
    },
    ir_graph.IR_OP_POOL1D: {
        3: ["NCW"]
    },
    ir_graph.IR_OP_POOL2D: {
        4: ["NCHW"]
    },
    ir_graph.IR_OP_POOL3D: {
        5: ["NCDHW"]
    },
    ir_graph.QNN_OP_PRELU: {
        5: ["NCDHW"],
        4: ["NCHW"],
        3: ["NCW"]
    },
    ir_graph.QNN_OP_RESIZE: {
        5: ["NCDHW"],
        4: ["NCHW"],
        3: ["NCW"]
    },
    # TODO
    # Replace with QNN macro once RoiAlign is aligned with QNN.
    'roi_align': {
        4: ["NCHW"]
    },
    # TODO
    # Replace with QNN macro once RoiPooling is aligned with QNN.
    'roi_pooling': {
        4: ["NCHW"]
    },
    ir_graph.IR_OP_ROLLED_LSTM: {
        3: ["TNF"]
    },
    ir_graph.QNN_OP_SPACE_TO_BATCH: {
        4: ["NCHW"]
    },
    ir_graph.QNN_OP_SPACE_TO_DEPTH: {
        4: ["NCHW"]
    },
    ir_graph.IR_OP_TRANSPOSE_CONV_1D: {
        3: ["NCW", "IOW", "C"]
    },
    ir_graph.QNN_OP_TRANSPOSE_CONV_2D: {
        4: ["NCHW", "IOHW", "C"]
    },
    ir_graph.QNN_OP_TRANSPOSE_CONV_3D: {
        5: ["NCDHW", "IODHW", "C"]
    }
}

# Table for spatial-first layout.
SPATIAL_FIRST_LAYOUT_TABLE = {
    ir_graph.QNN_OP_BATCHNORM: {
        5: ["NDHWC", "C", "C"],
        4: ["NHWC", "C", "C"],
        3: ["NWC", "C", "C"],
        2: ["NC", "C", "C"]
    },
    ir_graph.QNN_OP_BATCH_TO_SPACE: {
        4: ["NHWC"]
    },
    ir_graph.QNN_OP_CHANNEL_SHUFFLE: {
        4: ["NHWC"]
    },
    # TODO
    # Replace with QNN macro once ColorTransform is aligned with QNN definition.
    'color_transform': {
        2: ["NC"],
        4: ["NHWC"]
    },
    ir_graph.IR_OP_CONV_1D: {
        3: ["NWC", "WIO", "C"]
    },
    ir_graph.QNN_OP_CONV_2D: {
        4: ["NHWC", "HWIO", "C"],
    },
    ir_graph.QNN_OP_CONV_3D: {
        5: ["NDHWC", "DHWIO", "C"],
    },
    ir_graph.QNN_OP_CROP_AND_RESIZE: {
        4: ['NHWC']
    },
    ir_graph.QNN_OP_DEPTH_TO_SPACE: {
        4: ["NHWC"]
    },
    ir_graph.IR_OP_DEPTH_WISE_CONV_1D: {
        3: ["NWC", "WIO", "C"]
    },
    ir_graph.QNN_OP_DEPTH_WISE_CONV_2D: {
        4: ["NHWC", "HWIO", "C"]
    },
    ir_graph.QNN_OP_GENERATE_PROPOSALS: {
        4: ["NHWC", "NHWC"]
    },
    ir_graph.QNN_OP_GRID_SAMPLE: {
        4: ["NHWC"],
        5: ["NDHWC"]
    },
    ir_graph.QNN_OP_GROUP_NORM: {
        5: ["NDHWC", "C", "C"],
        4: ["NHWC", "C", "C"],
        3: ["NWC", "C", "C"]
    },
    ir_graph.QNN_OP_GRU: {
        3: ["NTF"]
    },
    ir_graph.QNN_OP_INSTANCE_NORM: {
        5: ["NDHWC", "C", "C"],
        4: ["NHWC", "C", "C"],
        3: ["NWC", "C", "C"]
    },
    ir_graph.QNN_OP_LRN: {
        5: ["NDHWC"],
        4: ["NHWC"],
        3: ["NWC"]
    },
    ir_graph.QNN_OP_LSTM: {
        3: ["NTF"]
    },
    ir_graph.IR_OP_MERGED_WEIGHTS_GRU: {
        3: ["NTF"]
    },
    ir_graph.IR_OP_POOL1D: {
        3: ["NWC"]
    },
    ir_graph.IR_OP_POOL2D: {
        4: ["NHWC"]
    },
    ir_graph.IR_OP_POOL3D: {
        5: ["NDHWC"]
    },
    ir_graph.QNN_OP_PRELU: {
        5: ["NDHWC"],
        4: ["NHWC"],
        3: ["NWC"]
    },
    ir_graph.QNN_OP_RESIZE: {
        5: ["NDHWC"],
        4: ["NHWC"],
        3: ["NWC"]
    },
    # TODO
    # Replace with QNN macro once RoiAlign is aligned with QNN.
    'roi_align': {
        4: ["NHWC"]
    },
    # TODO
    # Replace with QNN macro once RoiPooling is aligned with QNN.
    'roi_pooling': {
        4: ["NHWC"]
    },
    ir_graph.IR_OP_ROLLED_LSTM: {
        3: ["NTF"]
    },
    ir_graph.QNN_OP_SPACE_TO_BATCH: {
        4: ["NHWC"]
    },
    ir_graph.QNN_OP_SPACE_TO_DEPTH: {
        4: ["NHWC"]
    },
    ir_graph.IR_OP_TRANSPOSE_CONV_1D: {
        3: ["NWC", "WIO", "C"]
    },
    ir_graph.QNN_OP_TRANSPOSE_CONV_2D: {
        4: ["NHWC", "HWIO", "C"]
    },
    ir_graph.QNN_OP_TRANSPOSE_CONV_3D: {
        5: ["NDHWC", "DHWIO", "C"]
    }
}
