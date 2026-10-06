# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np
from qti.aisw.tools.core.utilities.data_processing import Annotation, Representation
from qti.aisw.tools.core.utilities.data_processing.metrics import Metric
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class Perplexity(Metric):
    """A class for calculating perplexity of a model's output.

    Attributes:
        logits_index (int, optional): Index of the logits in the model output. Defaults to 0.
    """

    def __init__(self, logits_index: int = 0) -> None:
        """Initializes the Perplexity class.

        Args:
            logits_index (int, optional): Index of the logits in the model output. Defaults to 0.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")  # noqa: F841
        self.logits_index = logits_index
        self._losses = []
        self.validate()

    def validate(self):
        """Validate the Perplexity parameters provided"""
        if not isinstance(self.logits_index, int) or not self.logits_index >= 0:
            raise ValueError("logits_index must be a non negative integer")

    def validate_input(self, input_sample: Representation) -> Representation:
        """Validates the input sample by checking its length and annotation.

        Args:
            input_sample (Representation): The input data to be validated.

        Returns:
            Representation: The validated input sample.

        Raises:
            ValueError: If logits_index is out of range for the given input.
            ValueError: If input_sample.annotation is not an instance of Annotation or lacks label data.
        """
        if len(input_sample.data) < self.logits_index:
            raise ValueError(f"logits_index:{self.logits_index} is index out of range for the given input")
        if not isinstance(input_sample.annotation, Annotation) or input_sample.annotation.data is None:
            raise ValueError("Perplexity score requires labels information for computing the loss.")
        return input_sample

    @Metric.validate_input_output
    def calculate(self, model_output: Representation) -> None:
        """Calculates the perplexity of a model's output.

        Args:
            model_output (Representation): The output of the model.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")  # noqa: F841
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
        input_data = model_output.data[self.logits_index]
        labels = model_output.annotation.data
        shift_logits = torch.tensor(input_data[..., :-1, :])
        shift_labels = torch.tensor(labels[..., 1:])
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self._losses.append(loss.item())

    def finalize(self) -> dict[str, float]:
        """Finalizes the calculation of perplexity.

        Returns:
            dict[str, float]: A dictionary containing the calculated perplexity.
        """
        result = {}
        if len(self._losses) > 0:
            loss = np.mean(self._losses)
            ppl = np.exp(loss)
            result["perplexity"] = ppl
        return result
