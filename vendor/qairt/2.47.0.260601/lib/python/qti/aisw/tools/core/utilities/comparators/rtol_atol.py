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


class RtolAtolParams(ComparatorParams):
    """This class defines the params required for RtolAtolParams comparator"""

    rtol_margin: float = Field(default=1e-2)
    atol_margin: float = Field(default=1e-2)


class RtolAtolComparator(Comparator):
    """This class implements a RtolAtol comparator to compare the tensors
    and compute the percentatge of the number of elements that are not close
    """

    def __init__(self, params: RtolAtolParams = RtolAtolParams()):
        """AdjustedRtolAtol Comparator Initialization
        Args:
            params (RtolAtolParams) : Params required for RtolAtol Comparator
        """
        super().__init__(name=COMPARATORS.RTOL_ATOL.value, params=params)
        self._rtol_margin = params.rtol_margin
        self._atol_margin = params.atol_margin

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using RtolAtol Comparator
        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: percent from 0 to 1 of how many elements did not match
        """
        match_array = np.isclose(tensor1, tensor2, atol=self._atol_margin, rtol=self._rtol_margin)
        percent_not_close = (len(match_array) - np.count_nonzero(match_array)) / len(match_array)
        percent_not_close = percent_not_close * 100
        return percent_not_close
