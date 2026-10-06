# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import ABC, abstractmethod
from os import PathLike
from typing import Dict, List, Optional, Union

import numpy as np
from qti.aisw.tools.core.utilities.framework.utils.constants import (
    FrameworkExecuteReturn,
    FrameworkModels,
)


class BaseFramework(ABC):
    """Abstract base class representing a framework.

    Attributes:
        None
    """

    def __init__(self):
        """Initializes the BaseFramework class attributes.

        Args:
            None

        Returns:
            None
        """

    @abstractmethod
    def load_model(self, input_model: Union[str, PathLike], **kwargs) -> FrameworkModels:
        """Load a machine learning inference model into the class.

        Takes in model paths (relative or absolute paths) to the model files,
        and loads the model into the class.

        Args:
            input_model (Union[str, PathLike]): The input model path.
            **kwargs (dict): Keyword arguments specific to the framework.

        Returns:
            FrameworkModels: A handle to the loaded model.
        """
        raise NotImplementedError("Method load_model must be implemented to use this base class")

    @abstractmethod
    def validate_model(self, input_model: str | PathLike | FrameworkModels) -> bool:
        """Validates the model.

        Args:
            input_model (str | PathLike | FrameworkModels): Path to the model file or a
                Framework object.

        Returns:
            bool: True if the model is valid, False otherwise.

        Raises:
            None
        """
        raise NotImplementedError("Method validate_model must be implemented to use this " "base class")

    @abstractmethod
    def run_inference(
        self,
        input_model: FrameworkModels,
        input_data: List[np.ndarray] | Dict[str, np.ndarray],
        return_numpy: bool = True,
        output_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, FrameworkExecuteReturn]:
        """Run the inference of given input_model.

        Args:
            input_model (FrameworkModels): The input_model to be used for inference.
            input_data (List[np.ndarray] | Dict[str, np.ndarray]): A list containing tensor data,
                or a dictionary containing input tensor names as keys and corresponding
                tensor data as values.
            return_numpy (bool): Flag used to control whether to return the results
                as numpy arrays or in the native framework type.
            output_names (Optional[List[str]]): Optional list of output tensor names for inference.
            kwargs (dict): Additional keyword arguments passed to run_inference method.

        Returns:
            Dict[str, FrameworkExecuteReturn]:  Dict containing output tensor name as
                key and its computed output as value.
                The output can either be a numpy array or native framework type.
        """
        raise NotImplementedError("Method run_inference must be implemented to use this base class")

    @abstractmethod
    def get_model_batch_size(self, input_model: FrameworkModels, input_name: str) -> int:
        """Return batch size for given input_model from the input node's shape.

        Args:
            input_model (FrameworkModels): The input_model to retrieve the batch size from.
            input_name (str): Name of input tensor whose batch size is to be retrieved

        Returns:
            int: Batch size for the model
        """
        raise NotImplementedError("Method get_model_batch_size must be implemented to use this " "base class")

    @abstractmethod
    def get_intermediate_output_tensors(self, input_model: FrameworkModels) -> List[str]:
        """Return list of intermediate output tensors for given input_model.

        Args:
            input_model (FrameworkModels): The input_model to be get the intermediate
                output tensors from.

        Returns:
            List[str]: List of intermediate output tensors for the model
        """
        raise NotImplementedError(
            "Method get_intermediate_output_tensors must be implemented " "to use this base class"
        )
