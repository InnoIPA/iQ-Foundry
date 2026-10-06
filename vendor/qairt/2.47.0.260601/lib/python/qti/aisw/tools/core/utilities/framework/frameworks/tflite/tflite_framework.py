# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import logging
from os import PathLike
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from qti.aisw.tools.core.utilities.framework.frameworks.base_framework import BaseFramework
from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger
from tensorflow.lite.python.interpreter import Interpreter


tflite_framework_log_area = LogAreas.register_log_area("TFLiteFramework")


class TFLiteFramework(BaseFramework):
    """Class representing the TFLite framework.

    This class provides methods for loading and running inference on TFLite models.

    Attributes:
        logger (QAIRTLogger): Logger instance for the class.
    """

    def __init__(self, parent_logger: logging.Logger = None):
        """Init function for TFLiteFramework class."""
        super().__init__()
        if parent_logger:
            self.logger = QAIRTLogger.register_area_logger(tflite_framework_log_area, parent_logger=parent_logger)
        else:
            self.logger = QAIRTLogger.register_area_logger(
                tflite_framework_log_area, level="INFO", formatter_val="extended", handler_list=["dev_console"]
            )

    def load_model(self, input_model: str | PathLike, **kwargs) -> Interpreter:
        """Load a machine learning inference model into the class.

        Takes in model paths (relative or absolute paths) to the model files, and
        loads the model into the class.

        Args:
            input_model (str | PathLike): The input model path.
            kwargs (dict): Additional keyword arguments passed to the load_model method.

        Returns:
            Interpreter: The loaded TFLite model interpreter.

        Raises:
            ValueError: If input_model is neither a valid path.
        """
        # Check if input_model is a path (string or PathLike object)
        if isinstance(input_model, (str, PathLike)):
            # Load the model
            try:
                interpreter = Interpreter(model_path=input_model, **kwargs)
                interpreter.allocate_tensors()
            except Exception as exc:
                self.logger.error(f"tflite load failed with Exception: {str(exc)}")
                raise
        else:
            raise ValueError("input_model must be a valid path")

        return interpreter

    def validate_model(self, input_model: str | PathLike | Interpreter) -> bool:
        """Validates the model file by checking if it is a valid TFLite model.

        Args:
            input_model (str | PathLike | Interpreter): Path to the model file or a
                TFLite Interpreter object.

        Returns:
            bool: Returns False since not implemented for TFLite

        Raises:
            None
        """
        self.logger.warning("validate_model not implemented for TFLite")
        return False

    def _validate_input_data(
        self, model: Interpreter, input_data: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], str]:
        """Validates the input data for inference against the model's input nodes.

        Args:
            model (Interpreter): The loaded TFLite model interpreter.
            input_data (Dict[str, np.ndarray]): A dictionary containing input data
                where keys are input node names and values are NumPy arrays.

        Returns:
            Tuple (Dict[str, np.ndarray], str): A tuple containing
                A filtered dictionary of input data containing only the input nodes present
                    in the model and None if any input node is missing.
                A missing input node name if any input node is missing.

        Raises:
            None
        """
        input_details = model.get_input_details()

        filtered_input_data = {}

        for input_name, data in input_data.items():
            input_index = next((i for i, detail in enumerate(input_details) if detail["name"] == input_name), None)
            if input_index is not None:
                filtered_input_data[input_name] = data
            else:
                self.logger.error(
                    f"Error: Inference input data for input: {input_name} is "
                    "not found. Please provide the same for inference."
                )
                return {}, input_name

        return filtered_input_data, None

    def _validate_output_names(self, model: Interpreter, output_names: Optional[List[str]]) -> Tuple[List[str], str]:
        """Validates the provided output names against the model's output nodes.

        Args:
            model (Interpreter): The loaded TFLite model interpreter.
            output_names (Optional[List[str]]): List of output node names to validate.
                If None, all model output names are used.

        Returns:
            Tuple (List[str], str): A tuple containing
                A filtered list of output names containing only the output nodes present
                    in the model and None if any output node is missing.
                A missing output node name if any output node is missing.

        Raises:
            None
        """
        output_details = model.get_output_details()
        model_output_names = [output["name"] for output in output_details]

        filtered_output_names = []

        if output_names:
            for output_name in output_names:
                if output_name in model_output_names:
                    filtered_output_names.append(output_name)
                else:
                    self.logger.error(
                        f"Error: Given output: {output_name} is not found in "
                        "model's outputs. Please create that tensor as model's "
                        "output and then supply for inference."
                    )
                    return [], output_name
        else:
            filtered_output_names = model_output_names

        return filtered_output_names, None

    def _generate_input_dict(self, model: Interpreter, input_data: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert a list of input data to a dictionary using input tensor names.

        Args:
            model (Interpreter): The loaded TFLite model interpreter.
            input_data (List[np.ndarray]): A list containing tensor data.

        Returns:
            Dict[str, np.ndarray]: A dictionary with input tensor names as keys and corresponding
                tensor data as values.
        """
        input_details = model.get_input_details()
        return {input_details[i]["name"]: data for i, data in enumerate(input_data)}

    def run_inference(
        self,
        model: Interpreter,
        input_data: List[np.ndarray] | Dict[str, np.ndarray],
        return_numpy: bool = True,
        output_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray | tf.Tensor]:
        """Run inference on the input data using the loaded model.

        Args:
            model (Interpreter): The loaded TFLite model interpreter.
            input_data (List[np.ndarray] | Dict[str, np.ndarray]): A list containing tensor data,
                or a dictionary containing input tensor names as keys and corresponding
                tensor data as values.
            return_numpy (bool): Whether to return the output as numpy arrays.
            output_names (Optional[List[str]]): Names of the output tensors.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Dict[str, np.ndarray | tf.Tensor]: The inference results.
        """
        """
        If input_data is provided as a List[np.ndarray], convert it to dict to include
        input tensor names as keys, and tensor data as values.
        """
        if isinstance(input_data, list):
            input_data = self._generate_input_dict(model, input_data)

        input_data, _missing_input_name = self._validate_input_data(model, input_data)
        if not input_data:
            raise ValueError(
                "_validate_input failed: Inference input data not found for " "input: {missing_input_name}"
            )

        output_names, missing_output_name = self._validate_output_names(model, output_names)
        if not output_names:
            raise ValueError(
                f"_validate_output failed: Output name: {missing_output_name} " "not found in model outputs"
            )

        # Set input tensors
        input_details = model.get_input_details()
        for input_name, data in input_data.items():
            input_index = next(i for i, detail in enumerate(input_details) if detail["name"] == input_name)
            model.set_tensor(input_details[input_index]["index"], data)

        # Run inference
        try:
            model.invoke()
        except Exception as exc:
            raise RuntimeError(f"TFLite inference execution failed with error: {exc}")

        # Get output tensors
        results = {}

        output_details = model.get_output_details()

        # Iterate over each output tensor
        for output_detail in output_details:
            output_data = model.get_tensor(output_detail["index"])
            if return_numpy:
                results[output_detail["name"]] = output_data
            else:
                results[output_detail["name"]] = tf.convert_to_tensor(output_data)

        return results

    def get_model_batch_size(self, model: Interpreter, input_name: str) -> int:
        """Get the batch size of the model for a given input tensor.

        Args:
            model (Interpreter): The loaded TFLite model interpreter.
            input_name (str): The name of the input tensor.

        Returns:
            int: Batch size for the model.
                - Returns the actual batch size if found in the input shape.
                - Returns 1 if batch size is not found in the input shape.
                - Returns -1 if the provided input_name is not found in the model.
        """
        try:
            # Get input details
            input_details = model.get_input_details()
            for input_detail in input_details:
                if input_detail["name"] == input_name:
                    input_shape = input_detail["shape"]
                    if len(input_shape) > 0:
                        # Return the first dimension which represents the batch size
                        return input_shape[0]
                    else:
                        # If no dimensions are available, assume a single sample
                        return 1
            # If the input_name is not found in the model, return -1
            return -1
        except Exception as e:
            # Handle exceptions here
            self.logger.error(f"An error occurred: {e}")
            return -1

    def get_intermediate_output_tensors(self, input_model: Interpreter) -> List[str]:
        """Return list of intermediate output tensors for given input_model.

        Args:
            input_model (Interpreter): The input_model to be get the intermediate
                output tensors from.

        Returns:
            List[str]: List of intermediate output tensors for the model
        """
        self.logger.warning("get_intermediate_output_tensors not implemented yet for TFLite")
        return []
