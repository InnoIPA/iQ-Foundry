# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
import math
from typing import List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from pydantic import Field
from .common import TensorShapeError, ComparisonError
from qti.aisw.tools.core.modules.api.definitions import AISWBaseModel


class ComparatorParams(AISWBaseModel):
    """This class defines base class for comparator specific params classes"""
    tol: float = Field(default=1e-5, ge=1e-10, le=1e-2)

class Comparator(ABC):
    """
    This class defines an interface to perform comparison on two set of input tensors.
    The derived classes implement this interface by defining '_compare' function which
    is used by 'compare'
    """
    def __init__(self, name: str, params: Optional[ComparatorParams] = None):
        """
        Constructor.
        Args:
            name: Name of the comparator.
            params: A named tuple that contains parameters required by the comparator implementation.
                    For example, tolerance value can be specified here.
        """
        if not params:
            params = ComparatorParams()
        self.name = name
        self._tol = params.tol
        self._decimals = int(abs(round(math.log(self._tol, 10))))

    @abstractmethod
    def _compare(self, tensor1: np.array, tensor2: np.array) -> float:
        """
        This function implements the actual logic to compute difference between two tensors.
        Comparator implementation must override this function.
        Args:
            tensor1: First Numpy array.
            tensor2: Second Numpy array.
        Returns:
            float: a float representing the comparison result.
        """
        pass

    @staticmethod
    def check_shape(_compare) -> _compare:
        """
        This decorator function is used to validate the dimensions of the two input tensors before
        calling the _compare method.
        Args:
            _compare: Method being decorated.
        Returns:
            Decorated method.
        """
        def wrap(self, tensor1, tensor2):
            if not (tensor1.shape == tensor2.shape):
                raise TensorShapeError("Mismatch in shape of the input arrays")
            return _compare(self, tensor1, tensor2)
        return wrap

    def compare(self, tensor1: List[np.array], tensor2: List[np.array]) -> List[Any]:
        """
        This function compares two tensor lists using '_compare' method defined in subclass and,
        returns a list of errors between each pair of tensors in inputs
        Both the tensor lists should be of same size.
        Args:
            tensor1: List of Numpy arrays. These tensors are considered as reference during comparison.
            tensor2: List of Numpy arrays.

        Returns:
            List: A list of comparison results.
        """
        if not (tensor1 and tensor2):
            raise ValueError("Empty data list.")
        if len(tensor1) != len(tensor2):
            raise Exception("Length of list1 and list2 are not equal.")

        errors = []
        for t1, t2 in zip(tensor1, tensor2):
            if t1.size != t2.size:
                raise TensorShapeError("tensor1 and tensor2 are of different size.")
            try:
                error = self._compare(t1, t2)
                errors.append(error)
            except Exception as ep:
                raise ComparisonError("Failed to compare tensors: {}".format(ep))
        return errors

    def get_tolerance(self):
        """
        Getter function to access the tolerance attribute.
        Returns:
            Float: Tolerance value.
        """
        return self._tol

    def _flatten_round_tensors(self, tensor1: np.array, tensor2: np.array) -> Tuple[np.array, np.array]:
        """
        This function flattens both the tensors and rounds them off up to decimal places
        Args:
            tensor1: Numpy array representing first tensor
            tensor2: Numpy array representing second tensor

        Returns:
            tensor1: Flattened and rounded off version of first tensor
            tensor2: Flattened and rounded off version of second tensor
        """
        tensor1 = tensor1.flatten()
        tensor2 = tensor2.flatten()

        tensor1 = np.around(tensor1, decimals=self._decimals)
        tensor2 = np.around(tensor2, decimals=self._decimals)

        return tensor1, tensor2

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
