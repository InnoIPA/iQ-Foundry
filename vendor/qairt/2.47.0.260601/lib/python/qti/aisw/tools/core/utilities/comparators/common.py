# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from enum import Enum


class COMPARATORS(str, Enum):
    L1_NORM = "l1_norm"
    L2_NORM = "l2_norm"
    AVERAGE = "average"
    COSINE = "cosine"
    STANDARD_DEVIATION = "standard_deviation"
    MSE = "mse"
    SNR = "snr"
    KLD = "kl_divergence"
    RTOL_ATOL = "rtol_atol"
    L1_ERROR = "l1_error"
    MSE_REL = "mse_rel"
    TOPK = "topk"
    ADJUSTED_RTOL_ATOL = "adjusted_rtol_atol"
    MAE = "mae"


class TensorShapeError(Exception):
    pass


class ComparisonError(Exception):
    pass
