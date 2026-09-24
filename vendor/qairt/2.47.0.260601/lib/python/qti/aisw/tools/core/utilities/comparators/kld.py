# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from scipy.stats import entropy

from .common import COMPARATORS
from .comparator import Comparator, ComparatorParams


class KLDParams(ComparatorParams):
    pass


class KLDComparator(Comparator):
    """This class implements comparator for comparing tensors based on
    Kullback-Leibler Divergence.
    """
    def __init__(self, params: KLDParams = KLDParams()):
        super().__init__(name=COMPARATORS.KLD.value, params=params)

    @Comparator.check_shape
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """This method compares two numpy array using KL divergence similarity.

        Args:
            tensor1: Numpy array 1. This is used as reference tensor.
            tensor2: Numpy array 2.

        Returns:
            float: a float representing error.
        """
        # Flatten and round the input tensors
        tensor1, tensor2 = self._flatten_round_tensors(tensor1, tensor2)

        ref_norm = np.linalg.norm(tensor1, ord=2)
        inf_norm = np.linalg.norm(tensor2, ord=2)

        if np.all(tensor1 == tensor2):
            return 0.0  # Perfect match, no divergence
        elif float(ref_norm) == 0.0 or float(inf_norm) == 0.0:
            return 1.0  # Max divergence to avoid division-by-zero error
        else:
            tensor1 = tensor1 / ref_norm
            tensor2 = tensor2 / inf_norm
            kld = entropy(tensor1, tensor2, base=2)
            return float(kld)
