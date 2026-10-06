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

# Onnx imports
import onnx
from onnx import (
    ModelProto,
)
from qti.aisw.tools.core.utilities.framework.frameworks.base_framework import BaseFramework
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model_helper import OnnxModelHelper
from qti.aisw.tools.core.utilities.framework.utils.constants import OnnxExecuteReturn
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger


onnx_framework_log_area = LogAreas.register_log_area("OnnxFramework")


class OnnxFramework(BaseFramework):
    """Class representing the Onnx framework.

    This class provides methods for loading and running inference on Onnx models.

    Attributes:
        _session (onnxruntime.InferenceSession): The loaded inference session.
        logger (QAIRTLogger): Logger instance for the class.
    """

    def __init__(self, parent_logger: logging.Logger = None):
        """Init function for OnnxFramework class."""
        super().__init__()
        self._session = None

        if parent_logger:
            self.logger = QAIRTLogger.register_area_logger(onnx_framework_log_area, parent_logger=parent_logger)
        else:
            self.logger = QAIRTLogger.register_area_logger(
                onnx_framework_log_area, level="INFO", formatter_val="extended", handler_list=["dev_console"]
            )

    def load_model(self, input_model: str | PathLike, **kwargs) -> ModelProto:
        """Load a machine learning inference model into the class.

        Takes in model paths (relative or absolute paths) to the model files, and
        loads the model into the class.

        Args:
            input_model (str | PathLike): The input model path.
            kwargs (dict): Additional keyword arguments passed to the load_model method.

        Returns:
            ModelProto: The loaded model.

        Raises:
            ValueError: If input_model is not a valid path.
        """
        # Check if input_model is a path (string or PathLike object)
        if isinstance(input_model, (str, PathLike)):
            # Load the model
            try:
                model = onnx.load(input_model)
                model = OnnxModelHelper._fix_constant_data_range(model)
            except Exception as exc:
                self.logger.error(f"onnx.load failed with Exception: {str(exc)}")
                raise
        else:
            raise ValueError("input_model must be a valid path")
        self.logger.debug(f"Loaded input_model: {input_model} successfully")

        return model

    def validate_model(self, input_model: str | PathLike | ModelProto) -> bool:
        """Validates the model file by checking if it is a valid ONNX model.

        Args:
            input_model (str | PathLike | ModelProto): Path to the model file or a
                ModelProto object.

        Returns:
            bool: True if the model is valid, False otherwise.

        Raises:
            None
        """
        if isinstance(input_model, ModelProto) and OnnxModelHelper.check_model_size(input_model):
            self.logger.error("input_model size > 2 GB, please provide model path in order to" "validate model")
            return False

        try:
            onnx.checker.check_model(input_model)
        except onnx.checker.ValidationError as e:
            self.logger.error(f"Invalid model: {e}")
            return False

        if isinstance(input_model, ModelProto):
            model_info = f"Model name: {input_model.graph.name}, Number of nodes: {len(input_model.graph.node)}"
        else:
            model_info = input_model

        self.logger.debug(f"Validated input_model: {model_info} successfully")

        return True

    def _validate_input_data(self, input_data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], str]:
        """Validates the input data for inference against the model's input nodes.

        Args:
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
        try:
            input_node_names = [x.name for x in self._session.get_inputs()]
        except AttributeError:
            self.logger.error("Error: ONNX Runtime session is not created." "Cannot validate output names.")
            return {}, None

        filtered_input_data = {}

        for input_name in input_node_names:
            if input_name in input_data:
                filtered_input_data[input_name] = input_data[input_name]
            else:
                self.logger.error(
                    f"Error: Inference input data for input: {input_name} is "
                    "not found. Please provide the same for inference."
                )
                return {}, input_name

        return filtered_input_data, None

    def _validate_output_names(self, output_names: Optional[List[str]]) -> Tuple[List[str], str]:
        """Validates the provided output names against the model's output nodes.

        Args:
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
        try:
            model_output_names = [x.name for x in self._session.get_outputs()]
        except AttributeError:
            self.logger.error("Error: ONNX Runtime session is not created." "Cannot validate output names.")
            return [], None

        if output_names is None:
            output_names = model_output_names
        else:
            for output_name in output_names:
                if output_name not in model_output_names:
                    self.logger.error(
                        f"Error: Given output: {output_name} is not found in "
                        "model's outputs. Please create that tensor as model's "
                        "output and then supply for inference."
                    )
                    return [], output_name

        return output_names, None

    def run_inference(
        self,
        input_model: ModelProto,
        input_data: List[np.ndarray] | Dict[str, np.ndarray],
        return_numpy: bool = True,
        output_names: Optional[List[str]] = None,
        custom_lib: Optional[str | PathLike] = None,
        optimize: Optional[bool] = False,
        **kwargs,
    ) -> Dict[str, np.ndarray | OnnxExecuteReturn]:
        """Runs inference using the given input_model.

        Steps:
            1. Create session
            2. Validate input and output
            3. Register custom ops (if applicable)
            4. Check input_model size (if applicable)
            5. Run session
            6. Obtain results
            7. Return results

        Args:
            input_model (ModelProto): The input_model to be used for inference.
            input_data (List[np.ndarray] | Dict[str, np.ndarray]): A list containing tensor data,
                or a dictionary containing input tensor names as keys and corresponding
                tensor data as values.
            return_numpy (bool): Flag used to control whether to return the results as
                numpy arrays or in the native framework type.
            output_names (Optional[List[str]]): Optional list of output tensor names for inference.
            custom_lib (Optional[str | PathLike]): Path to the custom operator library.
                Defaults to None.
            optimize (Optional[bool]): Optimize the model using Onnx Runtime and return the
                optimized model. Defaults to False.
            kwargs (dict): Additional keyword arguments passed to run_inference method.

        Keyword Args:
            run_options (RunOptions): Options for running the inference session.

        Returns:
            Dict[str, np.ndarray | OnnxExecuteReturn]: A dictionary containing
                output tensor names as keys and their computed outputs as values.
                The output can either be a numpy array or native onnx framework type.

        Raises:
            None
        """
        self._session, _ = OnnxModelHelper.create_inference_session(
            input_model, optimize=optimize, custom_lib=custom_lib, **kwargs
        )

        """
        If input_data is provided as a List[np.ndarray], convert it to dict to include
        input tensor names as keys, and tensor data as values.

        This is done in order to provide input data in the Dict[str, np.ndarray] format
        required by onnx.run method
        """
        if isinstance(input_data, list):
            input_data = OnnxModelHelper.generate_input_dict(input_model, input_data)

        input_data, _missing_input_name = self._validate_input_data(input_data)
        if not input_data:
            raise ValueError(
                "_validate_input failed: Inference input data not found for " "input: {missing_input_name}"
            )

        output_names, missing_output_name = self._validate_output_names(output_names)
        if not output_names:
            raise ValueError(
                f"_validate_output failed: Output name: {missing_output_name} not " "found in model outputs"
            )

        run_options = kwargs.get("run_options")

        # Run inference
        try:
            outputs = self._session.run(output_names, input_data, run_options=run_options)
        except Exception as exc:
            raise RuntimeError(f"ONNX inference execution failed with error: {exc}")

        if return_numpy and isinstance(outputs[0], np.ndarray):
            return {name: np.array(output) for name, output in zip(output_names, outputs)}
        else:
            return {name: output for name, output in zip(output_names, outputs)}

    def get_model_batch_size(self, input_model: ModelProto, input_name: str) -> int:
        """Retrieves the batch size for the given input_model from the input node's shape.

        NOTE: Call transform_node_names on the input_name before calling get_model_batch_size and
            passing input_name

        Args:
            input_model (ModelProto): The input_model to retrieve the batch size from.
            input_name (str): Name of the input tensor whose batch size is to be retrieved.

        Returns:
            int: Batch size for the model.
                - Returns the actual batch size if found in the input shape.
                - Returns 1 if batch size is not found in the input shape.
                - Returns -1 if the provided input_name is not found in the model.

        Raises:
            None
        """
        for inp in input_model.graph.input:
            # Check if the input name matches the provided input name
            if inp.name == input_name:
                try:
                    return inp.type.tensor_type.shape.dim[0].dim_value
                except Exception:
                    # Log a warning if the batch size is not found
                    self.logger.warning("Batch size not found for model: " "{input_model.graph.name}. Exception: {e}")
                    self.logger.warning("Setting model batch size to 1")
                    return 1
        # If the input name was not found in the graph inputs, raise an exception
        self.logger.error(
            "Incorrect model input name provided in config. "
            "Given: {input_name}, "
            "Expected: {[i.name for i in input_model.graph.input]}"
        )
        return -1

    def get_intermediate_output_tensors(self, input_model: ModelProto) -> List[str]:
        """Return list of intermediate output tensors for given input_model.

        Args:
            input_model (ModelProto): The input_model to be get the intermediate
                output tensors from.

        Returns:
            List[str]: List of intermediate output tensors for the model
        """
        intermediate_output_tensors = []

        for node in input_model.graph.node:
            for output in node.output:
                intermediate_output_tensors.append(onnx.ValueInfoProto(name=output).name)

        return intermediate_output_tensors

    def add_outputs(
        self, input_model: ModelProto, output_tensor_names: List[str], infer_shape: bool = True
    ) -> ModelProto:
        """Adds additional output tensors to the model.

        Args:
            input_model (ModelProto): The input_model to which the additional output tensors
                should be added.
            output_tensor_names (List[str]): List of output tensor names to be added to the model.
            infer_shape (bool): Perform shape inference before output creation.
                Defaults to True.

        Raises:
            ValueError: If the given tensor can't be made the graph output.

        Returns:
            ModelProto: The updated model with additional output tensors.
        """
        if infer_shape:
            try:
                input_model = OnnxModelHelper.native_shape_inference(input_model, delete_existing_shapes=True)
            except Exception as e:
                self.logger.error(f"Onnx shape inference failed due to: {e}")

        tensor_val_info_dict = OnnxModelHelper.get_value_info_proto_mappings(input_model)
        input_val_info_dict = OnnxModelHelper.get_inputs(input_model)
        output_val_info_dict = OnnxModelHelper.get_outputs(input_model)
        all_val_info_dict = {
            **tensor_val_info_dict,
            **input_val_info_dict,
            **output_val_info_dict,
        }

        model_outputs = OnnxModelHelper.get_output_names(input_model)

        filtered_output_names = [name for name in output_tensor_names if name not in model_outputs]

        if not filtered_output_names:
            # All the elements of output_names are already output of model.
            return input_model

        # Create a mapping of tensor names to their producing nodes
        tensor_to_node = {}
        for node in input_model.graph.node:
            for output in node.output:
                tensor_to_node[output] = node

        # Check and create ValueInfoProto for tensors not in value info
        for name in filtered_output_names:
            if name not in all_val_info_dict:
                value_info = self._get_or_create_value_info(name, tensor_to_node, all_val_info_dict)
                all_val_info_dict[name] = value_info

        # If all output_names are available in value info then update the input_model.
        for name in filtered_output_names:
            input_model.graph.output.append(all_val_info_dict[name])

        return input_model

    def _get_or_create_value_info(
        self,
        name: str,
        tensor_to_node: Dict[str, onnx.NodeProto],
        all_val_info_dict: Dict[str, onnx.ValueInfoProto],
    ) -> onnx.ValueInfoProto:
        """Gets or creates a ValueInfoProto for a tensor that's not in the value info dict.

        Args:
            name (str): The name of the tensor.
            tensor_to_node (Dict[str, onnx.NodeProto]): Mapping of tensor names to their producing nodes.
            all_val_info_dict (Dict[str, onnx.ValueInfoProto]): Dictionary of existing value info protos.

        Returns:
            onnx.ValueInfoProto: The created or retrieved ValueInfoProto.

        Raises:
            ValueError: If the tensor has no producing node and is not in value info.
        """
        # Try to infer type information from the producing node
        if name in tensor_to_node:
            node = tensor_to_node[name]
            value_info = self._create_value_info_from_node(node, name)
            if value_info:
                self.logger.debug(f"Created ValueInfoProto for {name} from node {node.name}")
                return value_info
            else:
                # If we can't infer the type, create a minimal ValueInfoProto
                # without type information - just the name
                value_info = onnx.ValueInfoProto()
                value_info.name = name
                self.logger.debug(
                    f"Created minimal ValueInfoProto for {name} (type will be inferred by ONNX Runtime)"
                )
                return value_info
        else:
            raise ValueError(
                f"{name} can't be made an output as it is not present in value info "
                f"and no producing node was found."
            )

    def _create_value_info_from_node(
        self, node: onnx.NodeProto, output_name: str
    ) -> Optional[onnx.ValueInfoProto]:
        """Creates a ValueInfoProto for a tensor based on its producing node.

        Args:
            node (onnx.NodeProto): The node that produces the tensor.
            output_name (str): The name of the output tensor.

        Returns:
            Optional[onnx.ValueInfoProto]: The created ValueInfoProto, or None if it cannot be created.
        """
        try:
            # Handle Constant nodes specially - we can extract type info directly
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                        tensor = attr.t
                        # Create ValueInfoProto from the constant tensor
                        value_info = onnx.helper.make_tensor_value_info(
                            output_name, tensor.data_type, list(tensor.dims) if tensor.dims else []
                        )
                        return value_info

            # For other node types, we cannot reliably infer the type
            # Return None so the caller can create a minimal ValueInfoProto
            return None

        except Exception as e:
            self.logger.warning(
                f"Failed to create ValueInfoProto for {output_name} from node {node.name}: {e}"
            )
            return None


class OnnxTransformModel:
    """Class containing methods for Onnx model transformation.

    Attributes:
        None
    """

    onnx_tm_log_area = LogAreas.register_log_area("OnnxTransformModel")
    logger = QAIRTLogger.register_area_logger(
        onnx_tm_log_area, level="INFO", formatter_val="extended", handler_list=["dev_console"]
    )

    @classmethod
    def optimize_by_simplifier(
        cls,
        model: str | PathLike | ModelProto,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        perform_optimization: bool = True,
        custom_lib: Optional[str | PathLike] = None,
        **simplify_args,
    ) -> ModelProto:
        """Simplifies an already loaded Onnx model in memory using the onnxsim library.

        Args:
            model (str | PathLike | ModelProto): Onnx ModelProto object or path to the model.
            input_shapes (Optional[Dict[str, List[int]]]): A dictionary with mapping of input names
                to their corresponding static shapes. Defaults to None.
            perform_optimization (bool): Whether to perform optimization during simplification.
                Defaults to True.
            custom_lib (Optional[str]): Path to the custom operator library. Defaults to None.
            simplify_args: Additional keyword arguments to pass to the onnxsim.simplify function.

        Keyword Args:
            custom_op_lib (str): Path to the custom operator library. Defaults to None.
            perform_optimization (bool): Whether to perform optimization during simplification.
                Defaults to True.

        Returns:
            ModelProto: The simplified Onnx model.
        """
        simplified_model = None

        try:
            from onnxsim import simplify

            # Update custom_op_lib and perform_optimization if provided in kwargs
            if "custom_lib" in simplify_args:
                custom_lib = simplify_args["custom_op_lib"]
            if "perform_optimization" in simplify_args:
                perform_optimization = simplify_args["perform_optimization"]

            simplified_model, ret_val = simplify(
                str(model),
                input_shapes=input_shapes,
                perform_optimization=perform_optimization,
                custom_lib=custom_lib,
                **simplify_args,
            )

            if not ret_val:
                cls.logger.warning("ONNX model simplification failed.")

        except ImportError:
            cls.logger.error("onnxsim package not found in current environment. " "Simplification will be skipped")
            return None

        except Exception as e:
            cls.logger.error(f"Onnx model simplification failed due to: {e}")

        return simplified_model

    @classmethod
    def transform_graph_nodes(cls, model: ModelProto) -> ModelProto:
        """Transforms the names of input and output nodes in the graph.

        Args:
            model (ModelProto): The model to be transformed.

        Returns:
            ModelProto: The transformed Onnx model.

        Example:
            Suppose we have an ONNX model with the following input and output node names:
            - Input nodes: ["input@1", "1input_2", "#input_3"]
            - Output nodes: ["output!", "output#2"]

            After applying this method, the node names will be transformed, e.g.:
            - Transformed input nodes: ["input_1", "_input_2", "_input_3"]
            - Transformed output nodes: ["output_", "output_2"]

            The method ensures consistent naming conventions for input and output nodes.
        """
        graph = model.graph
        for node in graph.node:
            # Transform input node names
            for i in range(len(node.input)):
                node.input[i] = Helper.transform_node_names(node.input[i])
            # Transform output node names
            for i in range(len(node.output)):
                node.output[i] = Helper.transform_node_names(node.output[i])

        return model

    @classmethod
    def transform_graph_inputs_and_outputs(cls, model: ModelProto) -> ModelProto:
        """Transforms the names of input and output nodes in the graph.

        This method modifies the node names within the input ONNX model:
        - It transforms input node names using transform_node_names helper method.
        - It transforms output node names using transform_node_names.
        - It also updates value-info names and initializer names if present.

        Args:
            model (ModelProto): The model to be transformed.

        Returns:
            modelProto: The transformed Onnx model.
        """
        graph = model.graph

        # Transform input node names
        for input_node in graph.input:
            input_node.name = Helper.transform_node_names(input_node.name)

        # Transform output node names
        for output_node in graph.output:
            output_node.name = Helper.transform_node_names(output_node.name)

        # Transform value-info names
        for value_info_node in graph.value_info:
            value_info_node.name = Helper.transform_node_names(value_info_node.name)

        # Transform initializer names
        for initializer_node in graph.initializer:
            initializer_node.name = Helper.transform_node_names(initializer_node.name)

        return model

    @classmethod
    def transform_native_tensor_names(cls, tensor_names: str) -> str:
        """Transforms tensor names to follow the converter's node naming conventions.

        Args:
            tensor_names (str): A string containing tensor names in the format
                'graphName0:tensorName0,tensorName1;graphName1:tensorName0,tensorName1'.

        Returns:
            str: Transformed tensor names in the same format.
        """
        tensor_names_list = tensor_names.split(";")
        transformed_tensor_names = ""

        for tensor_name in tensor_names_list:
            # Split the tensor name into graph name and individual tensor names
            tensor_name_list = tensor_name.split(":", 1)
            graph_name = tensor_name_list[0]
            tensors = tensor_name_list[1].split(",")
            for idx, tensor in enumerate(tensors):
                tensors[idx] = Helper.transform_node_names(tensor)
            transformed_tensors = ",".join(tensors)
            transformed_tensor_names += graph_name + ":" + transformed_tensors + ";"

        # Remove the last ';' from the transformed tensor names
        return transformed_tensor_names[:-1]

    @classmethod
    def transform_dynamic_shapes(cls, model: ModelProto, symbols: Dict[str, int]) -> ModelProto:
        """Replaces dynamic shapes with actual values.

        Args:
            model (ModelProto): The model to be transformed.
            symbols (Dict[str, int]): A dictionary containing the replacement values.

        Returns:
            ModelProto: The transformed Onnx model.
        """
        graph = model.graph

        for ip in graph.input:
            dim_len = len(ip.type.tensor_type.shape.dim)
            for i in range(dim_len):
                if len(ip.type.tensor_type.shape.dim[i].dim_param) > 0:
                    _symbol = ip.type.tensor_type.shape.dim[i].dim_param
                    if _symbol in symbols:
                        ip.type.tensor_type.shape.dim[i].dim_value = symbols[_symbol]
                    else:
                        ip.type.tensor_type.shape.dim[i].dim_value = 1
                    cls.logger.debug(
                        "Replaced symbol {} with value {}".format(_symbol, ip.type.tensor_type.shape.dim[i].dim_value)
                    )

        for op in graph.output:
            dim_len = len(op.type.tensor_type.shape.dim)
            for i in range(dim_len):
                if len(op.type.tensor_type.shape.dim[i].dim_param) > 0:
                    _symbol = op.type.tensor_type.shape.dim[i].dim_param
                    if _symbol in symbols:
                        op.type.tensor_type.shape.dim[i].dim_value = symbols[_symbol]
                    else:
                        op.type.tensor_type.shape.dim[i].dim_value = 1
                    cls.logger.debug(
                        "Replaced symbol {} with value {}".format(_symbol, op.type.tensor_type.shape.dim[i].dim_value)
                    )

        return model
