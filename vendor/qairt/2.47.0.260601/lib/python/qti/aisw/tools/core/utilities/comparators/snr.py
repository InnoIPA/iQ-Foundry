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


class SNRParams(ComparatorParams):
    """This class defines the params required for AdjustedRtolAtol comparator"""

    pass


class SNRComparator(Comparator):
    """This class implements a comparator to compare the tensors based on Signal to Noise Ratio."""

    def __init__(self, params: SNRParams = SNRParams()):
        """SNR Comparator Initialization
        Args:
            params (SNRParams) : Params required for  SNR Comparator
        """
        super().__init__(name=COMPARATORS.SNR.value, params=params)

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using SNR comparison.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        # Flatten and round the input tensors
        tensor1, tensor2 = self._flatten_round_tensors(tensor1, tensor2)

        diff = tensor1 - tensor2
        mse_diff = np.mean(np.square(diff))

        if float(mse_diff) == 0.0:
            snr = 100.0
        else:
            mse_tensor1 = np.mean(np.square(tensor1))
            snr = 10.0 * np.log10((mse_tensor1 / mse_diff))

        return float(snr)
