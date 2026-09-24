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


class L1NormParams(ComparatorParams):
    """This class defines the params required for L1Norm Comparator"""

    pass


class L1NormComparator(Comparator):
    """This class defines a comparator for comparing two input tensors based on L2 Norm."""

    def __init__(self, params: L1NormParams = L1NormParams()):
        """L1Norm Comparator Initialization
        Args:
            params (L1NormParams) : Params required for  L1Norm Comparator
        """
        super().__init__(name=COMPARATORS.L1_NORM.value, params=params)

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using L1 norm.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        # Flatten the input tensors and round them upto defined decimal place.
        tensor1 = tensor1.flatten()
        tensor2 = tensor2.flatten()

        tensor1 = np.around(tensor1, decimals=self._decimals)
        tensor2 = np.around(tensor2, decimals=self._decimals)

        ref_l1 = np.linalg.norm(tensor1, ord=1)
        if float(ref_l1) == 0.0:
            error = 0.0 if np.all(tensor2 == 0.0) else 1.0
        else:
            # Apply l1 norm on difference and compare it with l1 norm of reference.
            error = np.linalg.norm(tensor2 - tensor1, ord=1) / ref_l1
        return float(error)
