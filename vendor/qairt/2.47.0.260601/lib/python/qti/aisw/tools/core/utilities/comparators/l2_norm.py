# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from .common import COMPARATORS
from .comparator import Comparator, ComparatorParams


class L2NormParams(ComparatorParams):
    """This class defines the params required for L2Norm Comparator"""

    pass


class L2NormComparator(Comparator):
    """This class defines a comparator for comparing two input tensors based on L2 Norm."""

    def __init__(self, params: L2NormParams = L2NormParams()):
        """L2Norm Comparator Initialization
        Args:
            params (L2NormComparator) : Params required for L2Norm Comparator
        """
        super().__init__(name=COMPARATORS.L2_NORM.value, params=params)

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using L2 norm.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        # Flatten the input tensors and round them upto defined decimal place.
        tensor1, tensor2 = self._flatten_round_tensors(tensor1, tensor2)

        ref_l2 = np.linalg.norm(tensor1, ord=2)
        if float(ref_l2) == 0.0:
            error = 0.0 if np.all(tensor2==0.0) else 1.0
        else:
            error = np.linalg.norm(tensor2 - tensor1, ord=2) / ref_l2
        return float(error)
