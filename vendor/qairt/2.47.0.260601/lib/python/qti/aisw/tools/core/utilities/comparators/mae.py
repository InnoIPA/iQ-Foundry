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


class MAEParams(ComparatorParams):
    """This class defines the params required for MAE Comparator"""

    pass


class MAEComparator(Comparator):
    """This class implements a comparator to compare the tensors
    using Mean Absolute Error (MAE).
    """

    def __init__(self, params: MAEParams = MAEParams()):
        """MAE Comparator Initialization
        Args:
            params (MAEParams) : Params required for  MAE Comparator
        """
        super().__init__(name=COMPARATORS.MAE.value, params=params)

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using MAE.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        tensor1 = tensor1.flatten()
        tensor2 = tensor2.flatten()

        tensor1 = np.around(tensor1, decimals=self._decimals)
        tensor2 = np.around(tensor2, decimals=self._decimals)
        delta = tensor1 - tensor2
        l1_mean = np.mean(np.absolute(delta))
        return l1_mean
