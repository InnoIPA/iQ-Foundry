# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from typing import Optional

import numpy as np
from qti.aisw.tools.core.utilities.data_processing import NDArrayRepresentation
from qti.aisw.tools.core.utilities.data_processing.metrics import Metric


class TopKMetric(Metric):
    """Metric plugin to calculate the number of times where the correct label
    is among the top k predicted labels.

    Attributes:
        k (int | list[int]): Values for k to calculate.
        label_offset (Optional[int]): Offset to apply to label index.
        decimal_places (Optional[int]): Rounding factor for results.
    """

    def __init__(
        self,
        k: int | list[int] = [1, 5],
        label_offset: Optional[int] = 0,
        decimal_places: Optional[int] = 7,
    ):
        """Initializes the TopKMetric.

        Args:
            k (int | list[int]): Values for k to calculate.
            label_offset (Optional[int]): Offset to apply to label index.
            decimal_places (Optional[int]): Rounding factor for results.
        """
        self.top_k_values = k
        if isinstance(self.top_k_values, int):
            self.top_k_values = [self.top_k_values]
        # Map top-k values to a dictionary for easier access later
        self.top_k_dict = {k_val: [] for k_val in self.top_k_values}

        self.label_offset_value = label_offset
        self.decimal_places = decimal_places

    def calculate(self, result: NDArrayRepresentation) -> None:
        """Calculates the top-k metric for a given sample containing data and
        annotations.

        Args:
            result (NDArrayRepresentation): A representation containing data and annotations.
        """
        ground_truth = result.annotation.data
        raw_data = result.data[0]
        # Assuming raw_data.shape: [1, num_classes + offset]
        top_labels = np.argsort(raw_data[0, self.label_offset_value :])[::-1].tolist()

        for k_val in self.top_k_values:
            self.top_k_dict[k_val].append(ground_truth in top_labels[:k_val])

    def finalize(self) -> dict[str, float]:
        """Finalizes the metric calculation and returns the result.

        Returns:
            dict[str, float]: A dictionary containing the final metric results.
        """
        total_count = len(self.top_k_dict[self.top_k_values[0]])
        result = {}
        for k_val in self.top_k_values:
            assert len(self.top_k_dict[k_val]) == total_count
            score = sum(self.top_k_dict[k_val]) / (total_count or 1)
            result[f"top{k_val}"] = score
        # Add a count key to the result dictionary
        result["count"] = total_count
        return result
