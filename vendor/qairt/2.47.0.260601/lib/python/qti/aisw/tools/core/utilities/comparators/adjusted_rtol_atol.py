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


class AdjustedRtolAtolParams(ComparatorParams):
    """This class defines the params required for AdjustedRtolAtol comparator"""

    levels_num: int = Field(default=4, ge=1)


class AdjustedRtolAtolComparator(Comparator):
    """This class implements a comparator to compare the tensors
    using Adjusted Rtol Atol Comparator.
    """

    def __init__(self, params: AdjustedRtolAtolParams = AdjustedRtolAtolParams()):
        """AdjustedRtolAtol Comparator Initialization
        Args:
            params (AdjustedRtolAtolParams) : Params required for  AdjustedRtolAtol Comparator
        """
        super().__init__(name=COMPARATORS.ADJUSTED_RTOL_ATOL.value, params=params)
        self._levels_num = params.levels_num

    def _calculate_margins(self, input_tensor: np.array) -> tuple[float, float]:
        """This method is used to generate adjusted rtol and atol margin
        Args:
            input_tensor: Numpy Array for which margin is to be generated
        Reuturns:
            tuple[float,float] : Adjusted Atol and Rtol Margin
        """
        min_value = np.min(input_tensor)
        max_value = np.max(input_tensor)
        step = (max_value - min_value) / 255
        # allow at least four step sizes of absolute tolerance which is only 1.5 % (4/255)
        adjusted_atol_margin = self._levels_num * step
        # and add some 10% rtol, mainly for very small values
        adjusted_rtol_margin = 1e-1
        return adjusted_atol_margin, adjusted_rtol_margin

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using AdjustedRtolAtol.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        # Flatten and round the input tensors
        tensor1, tensor2 = self._flatten_round_tensors(tensor1, tensor2)
        atol_margin, rtol_margin = self._calculate_margins(tensor1)
        match_array = np.isclose(tensor2, tensor1, atol=atol_margin, rtol=rtol_margin)
        percent_not_close = (len(match_array) - np.count_nonzero(match_array)) / len(match_array)
        percent_not_close = percent_not_close * 100
        return percent_not_close
