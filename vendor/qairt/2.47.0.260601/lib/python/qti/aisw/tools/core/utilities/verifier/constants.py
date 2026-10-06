# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from enum import Enum


class AxisFormat(Enum):
    axis_format_mappings = {
        # (spatial-last format, spatial-first format): ((reshape order), (transpose order))
        ('NCDHW', 'NDHWC'): ((0, 4, 1, 2, 3), (0, 2, 3, 4, 1)),
        ('NCHW', 'NHWC'): ((0, 3, 1, 2), (0, 2, 3, 1)),
        ('NCF', 'NHWC'): ((0, 3, 1, 2), (0, 2, 3, 1)),
        ('NCDHW', 'NHWC'): ((0, 3, 1, 2), (0, 2, 3, 1)),
        ('NCDHW', 'NCHW'): ((0, 1, 2, 3), (0, 1, 2, 3)),
        ('NCF', 'NCHW'): ((0, 1, 2, 3), (0, 1, 2, 3)),
        ('NCS', 'NSC'): ((0, 3, 1, 2), (0, 2, 3, 1)),
        ('NCF', 'NFC'): ((0, 2, 1), (0, 2, 1)),
        ('TNF', 'NTF'): ((1, 0, 2), (1, 0, 2)),
        ('IODHW', 'DHWIO'): ((2, 3, 4, 0, 1), (2, 3, 4, 0, 1)),
        ('OIDHW', 'DHWOI'): ((3, 4, 0, 1, 2), (2, 3, 4, 1, 0)),
        ('OIDHW', 'DHWIO'): ((4, 3, 0, 1, 2), (0, 1, 2, 4, 3)),
        ('IOHW', 'HWIO'): ((2, 3, 0, 1), (2, 3, 0, 1)),
        ('OIHW', 'HWOI'): ((2, 3, 0, 1), (2, 3, 0, 1)),
        ('OIHW', 'HWIO'): ((3, 2, 0, 1), (2, 3, 1, 0)),
        ('IOF', 'FIO'): ((1, 2, 0), (2, 0, 1)),
        ('OIF', 'FOI'): ((1, 2, 0), (2, 0, 1)),
        ('OIF', 'FIO'): ((2, 1, 0), (2, 1, 0))
        # need to add below axis formats in mapping
        # NF: ir_graph.AxisFormat.NF,
        # NC: ir_graph.AxisFormat.NC,
        # ANY: ir_graph.AxisFormat.ANY,
        # NONTRIVIAL: ir_graph.AxisFormat.NONTRIVIAL,
    }

