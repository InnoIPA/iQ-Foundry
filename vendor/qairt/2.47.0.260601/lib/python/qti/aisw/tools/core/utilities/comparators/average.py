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


class AverageParams(ComparatorParams):
    """This class defines the params required for Average comparator"""

    pass


class AverageComparator(Comparator):
    """This class implements a comparator to compare the tensors
    by averaging the element wise errors.
    """

    def __init__(self, params: AverageParams = AverageParams()):
        """Average Comparator Initialization
        Args:
            params (AverageParams) : Params required for  Average Comparator
        """
        super().__init__(name=COMPARATORS.AVERAGE.value, params=params)

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using average.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        # Flatten and round the input tensors
        tensor1, tensor2 = self._flatten_round_tensors(tensor1, tensor2)

        ref_avg = max(np.average(np.absolute(tensor1)), np.average(np.absolute(tensor2)))
        if float(ref_avg) == 0.0:
            average_error = 0.0 if np.all(tensor2 == 0) else 1.0
        else:
            # compute the absolute difference between the tensors
            diff = np.average(np.absolute(tensor1 - tensor2))
            # compute the mean value of the absolute difference
            diff_avg = np.average(diff)
            average_error = diff_avg / ref_avg
        return float(average_error)
