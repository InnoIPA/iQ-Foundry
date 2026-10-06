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


class STDParams(ComparatorParams):
    """This class defines the params required for STD comparator"""

    pass


class STDComparator(Comparator):
    """This class implements a comparator that calculates the standard deviation
    difference between two tensors.
    """

    def __init__(self, params: STDParams = STDParams()):
        """STD Comparator Initialization
        Args:
            params (STDParams) : Params required for STD Comparator
        """
        super().__init__(name=COMPARATORS.STANDARD_DEVIATION.value, params=params)

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using standard deviation comparison.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        # Flatten and round the input tensors
        tensor1, tensor2 = self._flatten_round_tensors(tensor1, tensor2)
    
        
        if np.all(tensor1 == tensor2):
            return 0.0 # Perfect match, no error

        # finds the std deviation for tensor1 and tensor2
        std1 = np.std(tensor1)
        std2 = np.std(tensor2)

        if float(std1) == 0:
            return 100.0  # Reference has no variation, but tensors differ
        else:
            error = abs((std2 - std1)) / abs(std1)
            return float(error)
