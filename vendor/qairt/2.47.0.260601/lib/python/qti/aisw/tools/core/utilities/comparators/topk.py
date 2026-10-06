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


class TopKParams(ComparatorParams):
    """This class defines the params required for TopK comparator"""

    k: int = Field(default=1)
    ordered: bool = Field(default=False)


class TopKComparator(Comparator):
    """This class implements a comparator to compare the tensors
    using TopK.
    """

    def __init__(self, params: TopKParams = TopKParams()):
        """TopK Comparator Initialization
        Args:
            params (TopKParams) : Params required for TopK Comparator
        """
        super().__init__(name=COMPARATORS.TOPK.value, params=params)
        self._k = params.k
        self._ordered = params.ordered

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using TopK comparator.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        # Flatten and round the input tensors
        tensor1, tensor2 = self._flatten_round_tensors(tensor1, tensor2)

        top_k_indices_from_golden_output = np.flip(tensor1.argsort()[-self._k :])
        top_k_indices_from_inference_output = np.flip(tensor2.argsort()[-self._k :])

        if not self._ordered:
            top_k_indices_from_golden_output.sort()
            top_k_indices_from_inference_output.sort()

        number_of_diff_indices = 0
        for index_gold, index_inf in zip(
            top_k_indices_from_golden_output, top_k_indices_from_inference_output
        ):
            if self._ordered:
                if index_gold != index_inf:
                    number_of_diff_indices += 1
            else:
                if index_inf not in top_k_indices_from_golden_output:
                    number_of_diff_indices += 1

        percent_diff = number_of_diff_indices / min(self._k, len(tensor1))
        percent_diff = percent_diff * 100
        return percent_diff
