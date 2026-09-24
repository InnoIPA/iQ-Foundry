# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.tools.core.utilities.data_processing.metrics import Metric


class Precision(Metric):
    """Calculate precision metric for image classification tasks.

    This class calculates precision, which is the ratio of correct predictions
    to total predictions.

    Attributes:
        output_index (int): Index of the output to be used from the data provided.
                             Defaults to 0.
        decimal_places (int): Number of decimal places to round the precision value to.
                            Defaults to 7.
    """

    def __init__(self, output_index: int = 0, decimal_places: int = 7) -> None:
        """Initialize the Precision metric.

        Args:
            output_index (int, optional): Index of the output to be used from the data provided.
                             Defaults to 0.
            decimal_places (int, optional): Number of decimal places to round the precision value to.
                            Defaults to 7.
        """
        # Note: super init is called to setup metric_state
        super().__init__()
        self.output_index = output_index
        self.decimal_places = decimal_places
        self.validate()

    def validate(self):
        """Validate the metric inputs."""
        if not (type(self.output_index) is int and self.output_index >= 0):
            raise ValueError(f"output index must be an non negative integer >=0. Given: {self.output_index}")
        if not (type(self.decimal_places) is int and self.decimal_places >= 0):
            raise ValueError(
                f"decimal places must be an non negative integer >=0. Given: {self.decimal_places}"
            )

    def finalize(self) -> dict:
        """Calculate and return the precision metric.
        self.metric_state is a list containing all model outputs as representation object.

        Returns:
            dict: A dictionary containing the precision value and count of total inputs.
        """
        max_inputs = len(self.metric_state)
        save_results = {}
        if max_inputs > 0:
            # compute precision
            correct = sum(
                output.data[self.output_index] == output.annotation.data for output in self.metric_state
            )
            precision = round(correct / max_inputs, self.decimal_places)
            save_results = {"precision": precision, "count": max_inputs}
        return save_results
