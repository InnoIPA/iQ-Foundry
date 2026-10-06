# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import logging
import os
from importlib import import_module
from os import PathLike
from typing import Any, Dict, List, Optional

from qti.aisw.tools.core.utilities.framework.utils.constants import (
    ExecuteInputData,
    FrameworkExecuteReturn,
    FrameworkModels,
    OnnxFrameworkInfo,
    PytorchFrameworkInfo,
    TensorflowFrameworkInfo,
    TFLiteFrameworkInfo,
)
from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger


framework_mgr_log_area = LogAreas.register_log_area("FrameworkManager")


class FrameworkManager:
    """Framework Manager class - Performs generic framework level functionality.

    Attributes:
        framework_type (str): Inferred framework type based on files in input_model
            if it is given as a path, or in-memory model object itself.
        framework_instance: Instance of the specific framework (initialized later).
        _available_frameworks (dict): Mapping of framework names to their corresponding modules.
        logger (QAIRTLogger): Logger instance for the class.
    """

    logger = QAIRTLogger.register_area_logger(
        framework_mgr_log_area, level="DEBUG", formatter_val="extended", handler_list=["dev_console"]
    )

    def __init__(self, parent_logger: logging.Logger = None):
        """Initializes the FrameworkManager class attributes."""
        self.framework_type = None
        self.framework_instance = None
        self._available_frameworks = {
            "onnx": ["frameworks.onnx.onnx_framework", "OnnxFramework"],
            "pytorch": ["frameworks.pytorch.pytorch_framework", "PytorchFramework"],
            "tensorflow": ["frameworks.tensorflow.tensorflow_framework", "TensorFlowFramework"],
            "tflite": ["frameworks.tflite.tflite_framework", "TFLiteFramework"],
        }
        if parent_logger:
            self.logger = QAIRTLogger.register_area_logger(framework_mgr_log_area, parent_logger=parent_logger)
        else:
            self.logger = QAIRTLogger.register_area_logger(
                framework_mgr_log_area, level="INFO", formatter_val="extended", handler_list=["dev_console"]
            )

        # TODO: Add support for version viability.
        #       Min supported version needs to be compared against model's/runtime's version
        # self.version = None

    @property
    def available_frameworks(self):
        """Get the mapping of framework names to their corresponding modules."""
        return self._available_frameworks

    @classmethod
    def _validate_model_path(cls, input_model_path: str | PathLike) -> None:
        """Validates if the input_model_path exists.

        Args:
            input_model_path: str | PathLike: Input model path.

        Returns:
            None

        Raises:
            FileNotFoundError: If the input_model_path path does not exist.

        Usage:
            Call this method within the FrameworkManager instance to ensure that the
                specified model path exists.
        """
        if not os.path.exists(input_model_path):
            raise FileNotFoundError(f"Model path '{input_model_path}' does not exist.")

    def _load_framework_instance(self, input_model: str | PathLike | FrameworkModels) -> None:
        """Obtain the framework for the model to be loaded and instantiate the framework class.

        Args:
            input_model (str | PathLike | FrameworkModels]): Input model path or model object.

        Returns:
            None (sets framework_instance attribute in self)

        Raises:
            ImportError: If the framework module cannot be imported.

        Usage:
            Call this method within the FrameworkManager instance to load the appropriate framework.
        """
        # Infer framework type
        self.framework_type = FrameworkManager.infer_framework_type(input_model)
        framework_util_dir = "qti.aisw.tools.core.utilities.framework"

        module, framework_class_name = self._available_frameworks[self.framework_type]
        module = framework_util_dir + "." + module

        try:
            framework_class = getattr(import_module(module), framework_class_name)
            self.framework_instance = framework_class(self.logger)
        except ImportError as exc:
            self.logger.error(f"Unable to import framework class {exc}")
            raise

    @classmethod
    def infer_framework_type(cls, input_model: str | PathLike | FrameworkModels) -> str:
        """Class method to infer the framework type based on the input_model.

        Args:
            input_model: str | PathLike | FrameworkModels : Input model path or model object.

        Returns:
            str: Inferred framework type ('onnx', 'tensorflow', 'pytorch', or 'tflite').

        Raises:
            Exception: If an invalid model format is specified (not .onnx/.pb/.tflite/.pt).

        Usage:
            Call this method to determine the framework type.
        """
        # Mapping of file extensions to framework types
        input_model_to_framework = {".onnx": "onnx", ".pb": "tensorflow", ".pt": "pytorch", ".tflite": "tflite"}

        def is_onnx_model(model: Any) -> bool:
            try:
                from onnx import ModelProto

                return isinstance(model, ModelProto)
            except ImportError:
                return False

        def is_tensorflow_model(model: Any) -> bool:
            try:
                import tensorflow as tf

                return isinstance(model, (tf.compat.v1.GraphDef, tf.compat.v1.Graph, tf.keras.Model, tf.Graph))
            except ImportError:
                return False

        def is_pytorch_model(model: Any) -> bool:
            try:
                import torch

                return isinstance(model, torch.nn.Module)
            except ImportError:
                return False

        def is_tflite_model(model: Any) -> bool:
            try:
                from tensorflow.lite.python.interpreter import Interpreter

                return isinstance(model, Interpreter)
            except ImportError:
                return False

        # Input model path handling
        if isinstance(input_model, (str, PathLike)):
            FrameworkManager._validate_model_path(input_model)

            model_path, model_ext = os.path.splitext(input_model)

            # For TensorFlow 2, input is a folder containing the ".pb" file
            if model_ext not in input_model_to_framework:
                model_files = os.listdir(model_path)
                for file in model_files:
                    file_ext = os.path.splitext(file)[1]
                    if file_ext == ".pb":
                        model_ext = ".pb"
                        break

            # Validate if the framework type is present in the list of available framework types
            if model_ext not in input_model_to_framework:
                raise Exception(
                    "Invalid model format specified. Supported types are\
                                    .onnx/.pb/.tflite/.pt"
                )

            framework_type = input_model_to_framework[model_ext]

        # Framework-specific checks
        elif is_onnx_model(input_model):
            framework_type = OnnxFrameworkInfo.name
        elif is_tensorflow_model(input_model):
            framework_type = TensorflowFrameworkInfo.name
        elif is_pytorch_model(input_model):
            framework_type = PytorchFrameworkInfo.name
        elif is_tflite_model(input_model):
            framework_type = TFLiteFrameworkInfo.name
        else:
            raise TypeError("Invalid model format specified.")

        cls.logger.debug(f"Framework type determined: {framework_type}")

        return framework_type

    def load(self, input_model: str | PathLike, **kwargs) -> FrameworkModels:
        """Load the specified framework model.

        Args:
            input_model (str | PathLike): Input model path.
            kwargs: Additional keyword arguments specific to the framework (e.g., session, device).

        Returns:
            FrameworkModels
                Should return one of the following -
                    - Onnx - ModelProto
                    - Tensorflow - tf.compat.v1.Graph
                    - Pytorch - torch.nn.Module
                    - TFLite - tf.lite.Interpreter

        Raises:
            None

        Usage:
            Call this method within the FrameworkManager instance to load the model.
        """
        # Instantiate the framework class if not done before
        if not self.framework_instance:
            self._load_framework_instance(input_model)
        self.logger.debug(f"Loading input_model: {input_model}")

        return self.framework_instance.load_model(input_model, **kwargs)

    def validate(self, input_model: str | PathLike | FrameworkModels) -> bool:
        """Verifies if the provided model is valid.

        Args:
            input_model: str | PathLike | FrameworkModels: Input model path or model object.

        Returns:
            bool: True if the model is valid, False otherwise.

        Raises:
            NotImplementedError: If validation is not implemented for the specified framework.

        Usage:
            Call this method within the FrameworkManager instance to check the validity
                of the loaded model.
        """
        # Instantiate the framework class if not done before
        if not self.framework_instance:
            self._load_framework_instance(input_model)

        return self.framework_instance.validate_model(input_model)

    def execute(
        self,
        input_model: FrameworkModels,
        input_data: ExecuteInputData | List[ExecuteInputData],
        return_numpy: bool = True,
        output_tensor_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, FrameworkExecuteReturn] | List[Dict[str, FrameworkExecuteReturn]]:
        """Run inference on the loaded model using the specified input data.

        Args:
            input_model (FrameworkModels): Input model object.
            input_data (ExecuteInputData | List[ExecuteInputData]):
                Input tensor data for a single inference or a list of data for multiple inferences.
                The input data for a single inference can either be a list or a dictionary
                containing input tensor names as keys and corresponding tensor data as values.
            return_numpy (bool): Flag used to control whether to return the results as
                numpy arrays or in the native framework type. Default is True.
            output_tensor_names (Optional[List[str]]): List of output tensor names to
                retrieve predictions.
            **kwargs: Additional keyword arguments specific to the framework
                (e.g., session, device).

        Keyword Args:
            Onnx:
                run_options (RunOptions): Options for running the inference session.
                restore_back (bool): Restore back the external data.
                    Applicable for models with size > 2 GB.
                include_attributes (bool): Whether to include tensors that are part of node
                    attributes. Defaults to True.

        Returns:
            Dict[str, FrameworkExecuteReturn] | List[Dict[str, FrameworkExecuteReturn]]:
                A dictionary containing output tensor names as keys and their computed outputs as values,
                or a list of dictionaries of the same format.
                The output can either be a numpy array or native framework type.

        Raises:
           None

        Usage:
            Call this method within the FrameworkManager instance to perform inference
                on the loaded model.
        """
        # Instantiate the framework class if not done before
        if not self.framework_instance:
            self._load_framework_instance(input_model)

        # Check if there will be multiple inferences
        if all(isinstance(single_inference_inputs, (list, dict)) for single_inference_inputs in input_data):
            outputs = []
            for single_inference_inputs in input_data:
                try:
                    output = self.framework_instance.run_inference(
                        input_model, single_inference_inputs, return_numpy, output_tensor_names, **kwargs
                    )
                    outputs.append(output)
                except (ValueError, RuntimeError) as err:
                    self.logger.error(
                        f"Unable to run inference on the model using the input data \
                        {single_inference_inputs} due to error {err}."
                    )
                    outputs.append(None)
            return outputs

        # Single inference
        return self.framework_instance.run_inference(
            input_model, input_data, return_numpy, output_tensor_names, **kwargs
        )

    def get_model_batch_size(self, input_model: str | PathLike | FrameworkModels, input_name: str) -> int:
        """Get the batch size of the loaded model.

        Args:
            input_model: str | PathLike | FrameworkModels: Input model path or model object.
            input_name: str: Name of the input tensor.

        Returns:
            int: Batch size of the loaded model.

        Raises:
            None

        Usage:
            Call this method within the FrameworkManager instance to get the model's batch size
        """
        # Instantiate the framework class if not done before
        if not self.framework_instance:
            self._load_framework_instance(input_model)

        return self.framework_instance.get_model_batch_size(input_model, input_name)

    def generate_intermediate_outputs(
        self,
        input_model: FrameworkModels,
        input_data: ExecuteInputData | List[ExecuteInputData],
        return_numpy: bool = True,
        output_tensor_names: Optional[List[str]] = None,
        intermediate_output_tensors: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, FrameworkExecuteReturn] | List[Dict[str, FrameworkExecuteReturn]]:
        """Get the intermediate layer outputs of the loaded model.

        Args:
            input_model: FrameworkModels: Input model object.
            input_data (ExecuteInputData | List[ExecuteInputData]):
                Input tensor data for a single inference or a list of data for multiple inferences.
                The input data for a single inference can either be a list or a dictionary
                containing input tensor names as keys and corresponding tensor data as values.
            return_numpy: bool: Flag used to control whether to return the results as
                numpy arrays or in the native framework type. Default is True.
            output_tensor_names: Optional[List[str]]: List of output tensor names to
                retrieve predictions.
            intermediate_output_tensors: Optional[List[str]]: List of intermediate layer names
                to retrieve outputs.
            **kwargs: Additional keyword arguments specific to the framework
                (e.g., session, device).

        Keyword Args:
            Onnx:
                run_options (RunOptions): Options for running the inference session.
                restore_back (bool): Restore back the external data.
                    Applicable for models with size > 2 GB.
                include_attributes (bool): Whether to include tensors that are part of node
                    attributes. Defaults to True.
                infer_shape (bool): Whether to perform shape inference for the model.

        Returns:
            Dict[str, FrameworkExecuteReturn] | List[Dict[str, FrameworkExecuteReturn]]:
                A dictionary containing output tensor names as keys and their computed outputs as values,
                or a list of dictionaries of the same format.
                The output can either be a numpy array or native framework type.

        Raises:
            None

        Usage:
            Call this method within the FrameworkManager instance to add intermediate tensors
                as output tensors, and run inference on the updated model to
                generate intermediate outputs.
        """
        # Instantiate the framework class if not done before
        if not self.framework_instance:
            self._load_framework_instance(input_model)

        if (output_tensor_names is None) and (self.framework_type == OnnxFrameworkInfo.name):
            """
            If output_tensor_names is None, and the framework type is Onnx, use the onnx helper
            method get_output_names to get the output names, and append the
            intermediate_output_tensors to this list
            """
            from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model_helper import OnnxModelHelper

            output_tensor_names = OnnxModelHelper.get_output_names(input_model)

        if (self.framework_type == OnnxFrameworkInfo.name):
            if not intermediate_output_tensors:
                intermediate_output_tensors = self.framework_instance.get_intermediate_output_tensors(input_model)
            input_model = self.framework_instance.add_outputs(input_model=input_model, output_tensor_names=intermediate_output_tensors, infer_shape=kwargs.get("infer_shape", True))
        else:
            intermediate_output_tensors = self.framework_instance.get_intermediate_output_tensors(input_model)
            input_model = self.framework_instance.add_outputs(input_model=input_model, output_tensor_names=intermediate_output_tensors)

        # Combine output_tensor_names and intermediate_output_tensors to provide to execute method
        output_tensor_names = output_tensor_names + intermediate_output_tensors
        output_tensor_names = list(dict.fromkeys(output_tensor_names))

        # Execute the model
        return self.execute(input_model, input_data, return_numpy, output_tensor_names, **kwargs)
