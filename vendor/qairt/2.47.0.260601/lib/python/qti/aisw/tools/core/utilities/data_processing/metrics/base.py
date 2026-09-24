# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from abc import abstractmethod
from typing import Any, Optional

from qti.aisw.tools.core.utilities.data_processing import Representation


class Metric:
    """QAIRT Metric Base Class.

    This class serves as a base for all custom metric plugins. It
    provides a basic structure for computing metrics and storing their
    intermediate results.
    """

    def __init__(self, **kwargs):
        """Initializer (Optional): Required only if custom control over the
        metric state.

        Args:
            kwargs:  Allows developer to supply various parameters

        Attributes:
            metric_state (list[Any]): A list of intermediate results for this metric instance
        """
        self._metric_state = []
        # Needed to make sure parameters supplied during creation are made attributes
        self.__dict__.update(kwargs)

    @property
    def metric_state(self) -> list[Any]:
        """Getter: Returns the current state of the metric.

        Returns:
            list[Any]: A list of intermediate results for this metric instance
        """
        return self._metric_state

    @metric_state.setter
    def metric_state(self, value: list[Any]) -> None:
        """Setter: Updates the current state of the metric.

        Args:
            value (list[Any]): A list of intermediate results for this metric instance.
        """
        self._metric_state = value

    def __call__(self, data):
        """This method calculates a metric from the given data and appends it to self.metric_state.

        Args:
            data (object): The input data to calculate the metric for.

        Returns:
            None

        Raises:
            RuntimeError: If an exception occurs during calculation of the metric.
        """
        try:
            result = self.calculate(data)
            if result is not None:
                self.metric_state.append(result)
        except Exception as e:
            raise RuntimeError(f"Failed to run calculate in plugin : {self.__class__.__name__}. Reason: {e}")

    def validate_input(self, input_sample):
        """Validates the input sample before the calling the calculate method.

        Args:
           input_sample: Data representation that needs validation.

        Returns:
           Representation: Data representation post the input validation
        """
        return input_sample

    def validate_output(self, output_sample):
        """Validates the output sample before returning it from the Metric.

        Args:
           output_sample: Data representation that needs validation.

        Returns:
           Representation: Data representation post the output validation
        """
        return output_sample

    @staticmethod
    def validate_input_output(calculate) -> "calculate":
        """This decorator function is used to validate the input data before
        calling the calculate method.

        Args:
            calculate: Method being decorated.

        Returns:
            Decorated calculate method.
        """

        def wrap(self, data):
            validated_input = self.validate_input(data)
            output = calculate(self, validated_input)
            validated_output = self.validate_output(output)
            return validated_output

        return wrap

    @validate_input_output
    def calculate(self, data: Representation) -> Optional[tuple | float]:
        """Describes the logic for computing metric score for single input.
        If user does not override this function, we preserve the entire data state.
        User can optionally prepare the data and meta metric computation
        data  -> A Representation object which contains model outputs and annotation info
                 (per sample or reference to entire dataset).
        Returns :
            tuple | float: A tuple containing intermediate data & annotation information
             that would be used during finalize call for computing metric or per sample metric score.

        Note: Implementing the calculate method is optional for any subclass.
         It can be utilized in scenarios where the metric score can be computed on a per-sample basis.
        """
        return data

    @abstractmethod
    def finalize(self) -> dict[str, float]:
        """Compute metric for use cases where per sample metric score cannot be
        computed. If it is possible to compute per sample metric score, this
        method will aggregate metric score computed across inputs.
        By default, metric_state will contain data required to compute metric score.
        Note: It must return the results as a python dictionary
        """
        pass
