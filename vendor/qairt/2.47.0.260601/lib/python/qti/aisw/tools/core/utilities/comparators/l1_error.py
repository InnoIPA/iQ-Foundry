# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


import numpy as np
from pydantic import Field

from .common import COMPARATORS
from .comparator import Comparator, ComparatorParams


class L1ErrorParams(ComparatorParams):
    """This class defines the params required for L1 Error comparator"""

    multiplier: float = Field(default=1.0)
    scale: float = Field(default=1.0)


class L1ErrorComparator(Comparator):
    """This class implements a L1Error comparator to compare the tensors
    and compute the absolute error
    """

    def __init__(self, params: L1ErrorParams = L1ErrorParams()):
        """L1Error Comparator Initialization
        Args:
            params (L1ErrorParams) : Params required for  L1 Error Comparator
        """
        super().__init__(name=COMPARATORS.L1_ERROR.value, params=params)
        self._multiplier = params.multiplier
        self._scale = params.scale

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using L1Error
        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: percent from 0 to 1 of how many elements did not match
        """
        tensor1 = tensor1.flatten()
        tensor2 = tensor2.flatten()

        tensor1 = np.around(tensor1, decimals=self._decimals)
        tensor2 = np.around(tensor2, decimals=self._decimals)
        delta = tensor1 - tensor2
        l1_sum = np.sum(np.absolute(delta))
        l1_error = l1_sum * self._multiplier * self._scale
        return l1_error
