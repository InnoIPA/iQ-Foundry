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


class MSERelParams(ComparatorParams):
    """This class defines the params required for relative MSE Comparator"""

    pass


class MSERelComparator(Comparator):
    """This class implements a comparator to compare the tensors
    using relative Mean Square Error (MSE).
    """

    def __init__(self, params: MSERelParams = MSERelParams()):
        """Relative MSE Comparator Initialization
        Args:
            params (MSERelParams) : Params required for relative MSE Comparator
        """
        super().__init__(name=COMPARATORS.MSE_REL.value, params=params)

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using relative MSE.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        # Flatten and round the input tensors
        tensor1, tensor2 = self._flatten_round_tensors(tensor1, tensor2)

        # calculate mean square error between two tensors
        mse_tensor1 = np.mean(np.square(tensor1))
        if float(mse_tensor1) == 0.0:
            error = 0.0 if np.all(tensor2==0) else 100.0
        else:
            mse_diff = np.mean(np.square(tensor1 - tensor2))
            error = mse_diff / mse_tensor1

        return float(error)
