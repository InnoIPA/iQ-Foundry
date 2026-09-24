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


class CosineParams(ComparatorParams):
    """This class defines the params required for Cosine Comparator"""

    pass


class CosineComparator(Comparator):
    """This class implements comparator for comparing tensors based on Cosine similarity."""

    def __init__(self, params: CosineParams = CosineParams()):
        """Cosine Comparator Initialization
        Args:
            params (CosineParams) : Params required for Cosine Comparator
        """
        super().__init__(name=COMPARATORS.COSINE.value, params=params)

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using Cosine similarity.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        # Flatten and round the input tensors
        tensor1, tensor2 = self._flatten_round_tensors(tensor1, tensor2)

        tensor1_l2norm = np.linalg.norm(tensor1, ord=2)
        tensor2_l2norm = np.linalg.norm(tensor2, ord=2)

        if float(tensor1_l2norm) == 0.0 or float(tensor2_l2norm) == 0.0:
            similarity = 0
        else:
            similarity = np.dot(tensor1, tensor2) / (tensor1_l2norm * tensor2_l2norm)
        return float(similarity)
