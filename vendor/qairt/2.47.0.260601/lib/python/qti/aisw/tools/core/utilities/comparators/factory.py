# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from importlib import import_module
from typing import Optional

from .common import COMPARATORS
from .comparator import Comparator, ComparatorParams


comparator_module = "qti.aisw.tools.core.utilities.comparators"
comparators_mapping = {
    COMPARATORS.KLD: ["kld", "KLDComparator", "KLDParams"],
    COMPARATORS.L1_NORM: ["l1_norm", "L1NormComparator", "L1NormParams"],
    COMPARATORS.L2_NORM: ["l2_norm", "L2NormComparator", "L2NormParams"],
    COMPARATORS.MSE: ["mse", "MSEComparator", "MSEParams"],
    COMPARATORS.SNR: ["snr", "SNRComparator", "SNRParams"],
    COMPARATORS.STANDARD_DEVIATION: ["std", "STDComparator", "STDParams"],
    COMPARATORS.AVERAGE: ["average", "AverageComparator", "AverageParams"],
    COMPARATORS.COSINE: ["cosine", "CosineComparator", "CosineParams"],
    COMPARATORS.L1_ERROR: ["l1_error", "L1ErrorComparator", "L1ErrorParams"],
    COMPARATORS.MSE_REL: ["mse_rel", "MSERelComparator", "MSERelParams"],
    COMPARATORS.TOPK: ["topk", "TopKComparator", "TopKParams"],
    COMPARATORS.ADJUSTED_RTOL_ATOL: ["adjusted_rtol_atol", "AdjustedRtolAtolComparator", "AdjustedRtolAtolParams"],
    COMPARATORS.RTOL_ATOL: ["rtol_atol", "RtolAtolComparator", "RtolAtolParams"],
    COMPARATORS.MAE: ["mae", "MAEComparator", "MAEParams"],
}


def get_comparator_param(comparator: COMPARATORS) -> ComparatorParams:
    """This method returns the instance of the param class of the corresponding comparator passed

    Args:
        comparator: an enum value that represents the name of the comparator to be created.

    Returns:
        ComparatorParams: An instance of the corresponding param class
    """
    comparator_file, _, params_class = comparators_mapping[comparator]
    comparator_file = comparator_module + "." + comparator_file
    param_instance = getattr(import_module(comparator_file), params_class)
    return param_instance()


def get_comparator(
    comparator: COMPARATORS, params: Optional[ComparatorParams] = None
) -> Comparator:
    """This method creates a new instance of a specified comparator given parameters.

    Args:
        comparator: an enum value that represents the name of the comparator to be created.
        params: optional parameters specifying arguments for creating a specific type of comparator.

    Returns:
        Comparator: An instance of the requested comparator class.
    """
    if comparator in comparators_mapping.keys():
        comparator_file, comparator_class, _ = comparators_mapping[comparator]
        comparator_file = comparator_module + "." + comparator_file
        comparator_instance = getattr(import_module(comparator_file), comparator_class)
        if params:
            return comparator_instance(params=params)
        return comparator_instance()
    raise NotImplementedError(f"Comparator, '{comparator}' is not implemented.")
