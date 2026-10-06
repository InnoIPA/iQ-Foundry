# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging
from os import PathLike
from typing import Dict, List, Optional

import numpy as np
import torch
from qti.aisw.tools.core.utilities.framework.frameworks.base_framework import BaseFramework
from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger


pytorch_framework_log_area = LogAreas.register_log_area("PytorchFramework")


class PytorchFramework(BaseFramework):
    """Class representing the Pytorch framework.

    This class provides methods for loading and running inference on Pytorch models.

    Attributes:
        logger (QAIRTLogger): Logger instance for the class.
    """

    def __init__(self, parent_logger: logging.Logger = None):
        """Init function for PytorchFramework class."""
        super().__init__()
        if parent_logger:
            self.logger = QAIRTLogger.register_area_logger(pytorch_framework_log_area, parent_logger=parent_logger)
        else:
            self.logger = QAIRTLogger.register_area_logger(
                pytorch_framework_log_area, level="INFO", formatter_val="extended", handler_list=["dev_console"]
            )

    def load_model(self, input_model: str | PathLike, **kwargs) -> torch.nn.Module:
        """Load a machine learning inference model into the class.

        Takes in model paths (relative or absolute paths) to the model files, and
        loads the model into the class.

        Args:
            input_model (str | PathLike): The input model path.
            kwargs (dict): Additional keyword arguments passed to the load_model method.

        Returns:
            torch.nn.Module: The loaded model.

        Raises:
            ValueError: If input_model is neither a valid path.
        """
        # Check if input_model is a path (string or PathLike object)
        if isinstance(input_model, (str, PathLike)):
            # Load the model
            try:
                # Try loading as a TorchScript model
                model = torch.jit.load(input_model, map_location="cpu")
            except Exception as exc:
                # Handle other exceptions
                self.logger.error(f"torch.load failed with Exception: {str(exc)}")
                raise
        else:
            raise ValueError("input_model must be a valid path")

        return model

    def validate_model(self, input_model: str | PathLike | torch.nn.Module) -> bool:
        """Validates the model file by checking if it is a valid Pytorch model.

        Args:
            input_model (str | PathLike | torch.nn.Module): Path to the model file or
                a torch.nn.Module object.

        Returns:
            bool: Returns False since not implemented for Pytorch

        Raises:
            None
        """
        self.logger.warning("validate_model not implemented for Pytorch")
        return False

    def run_inference(
        self,
        model: torch.nn.Module,
        input_data: List[np.ndarray] | Dict[str, np.ndarray],
        return_numpy: bool = True,
        output_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray | torch.Tensor]:
        """Runs inference using the given model.

        Args:
            model (torch.nn.Module): The model to be used for inference.
            input_data (List[np.ndarray] | Dict[str, np.ndarray]): A list containing tensor data,
                or a dictionary containing input tensor names as keys and corresponding
                tensor data as values.
            return_numpy (bool): Flag used to control whether to return the results as
                numpy arrays or in the native framework type.
            output_names (Optional[List[str]]): Optional list of output tensor names for inference.
            kwargs (dict): Additional keyword arguments passed to run_inference method.

        Returns:
            Dict[str, np.ndarray | torch.Tensor]: A dictionary containing output tensor names
                as keys and their computed outputs as values. The output can either be a
                numpy array or native pytorch framework type.

        Raises:
            None
        """
        model.eval()  # Set the model to evaluation mode

        # Convert input data to PyTorch tensors
        if isinstance(input_data, dict):
            # Input data is a dictionary
            input_tensors = [torch.from_numpy(data) for _, data in input_data.items()]
        else:
            # Input data is a list
            input_tensors = [torch.from_numpy(data) for data in input_data]

        # Run inference
        try:
            with torch.no_grad():
                output_tensors = model(*input_tensors, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"PyTorch inference execution failed with error: {exc}")

        # Prepare output data
        output_data = {}

        if not output_names:
            output_names = [f"output_{i}" for i in range(len(output_tensors))]
        elif len(output_names) != len(output_tensors):
            raise ValueError("The lengths of output_names provided do not match the number " "of output tensors.")

        for output_name, output_value in zip(output_names, output_tensors):
            if return_numpy:
                output_data[output_name] = output_value.numpy()
            else:
                output_data[output_name] = output_value

        return output_data

    def get_model_batch_size(self, model: torch.nn.Module, input_name: str) -> int:
        """Not applicable to pytorch, returns -1."""
        self.logger.warning("get_model_batch_size not implemented for Pytorch")
        return -1

    def get_intermediate_output_tensors(self, input_model: torch.nn.Module) -> List[str]:
        """Return list of intermediate output tensors for given input_model.

        Args:
            input_model (torch.nn.Module): The input_model to be get the
                intermediate output tensors from.

        Returns:
            List[str]: List of intermediate output tensors for the model
        """
        self.logger.warning("get_intermediate_output_tensors not implemented yet for Pytorch")
        return []
