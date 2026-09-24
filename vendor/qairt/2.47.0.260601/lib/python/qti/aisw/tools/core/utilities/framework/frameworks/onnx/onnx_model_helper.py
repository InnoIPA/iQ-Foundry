# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import itertools
import logging
import os
import sys
import tempfile
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Onnx imports
import onnx
import onnxruntime
from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    TensorProto,
    ValueInfoProto,
    onnx_pb,
)
from onnx.external_data_helper import (
    load_external_data_for_tensor,
    save_external_data,
    set_external_data,
    uses_external_data,
)
from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger


onnx_model_helper_log_area = LogAreas.register_log_area("OnnxModelHelper")

# Represents model size threshold of 2 GB
THRESHOLD_MODEL_SIZE = 2
ONNX_EXTERNAL_DATA_THRESHOLD = 1024
INT_MAX = (1 << 31) - 1  # 2**31 - 1
FP16_MAX = 65504 # 2**15 − 2**(−10)
SYMBOLIC_SHAPE_INFER_MIN_VER = 7
QAIRT_TMP_DIR = os.getenv("QAIRT_TMP_DIR", tempfile.gettempdir())

class OnnxModelHelper:
    """Utility class for working with ONNX models."""

    logger = QAIRTLogger.register_area_logger(
        onnx_model_helper_log_area, level="WARN", formatter_val="extended", handler_list=["dev_console"]
    )

    @classmethod
    def check_model_size(cls, model: ModelProto) -> bool:
        """Checks if the size of the ONNX model exceeds the THRESHOLD_MODEL_SIZE.

        Args:
            model (ModelProto): An ONNX model proto instance.

        Returns:
            bool: True if the model size exceeds the threshold, False otherwise.
        """
        return OnnxModelHelper.get_model_size(model) > THRESHOLD_MODEL_SIZE

    @classmethod
    def convert_model_to_external_data(
        cls, model: ModelProto, file_name: str | PathLike, include_attributes: bool = True
    ) -> None:
        """Converts the tensors in the given model by updating their data location and
        data offset parameters. Note: This API will not convert data to external data,
        but it will populate external data fields in each tensor. Actual conversion
        to external data will happen via the onnx.save API.

        Args:
            model (ModelProto): The ONNX model proto instance.
            file_name (str): Path of the ONNX external data file.
            include_attributes (bool): Whether to include tensors that are part of node attributes.
                Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        for tensor in OnnxModelHelper.get_all_tensors(model, include_attributes):
            if tensor.HasField("raw_data") and sys.getsizeof(tensor.raw_data) >= ONNX_EXTERNAL_DATA_THRESHOLD:
                set_external_data(tensor, file_name)

    @classmethod
    def create_inference_session(
        cls,
        model: ModelProto,
        optimize: bool = False,
        execution_providers: str = "CPUExecutionProvider",
        custom_lib: Optional[str | PathLike] = None,
        **kwargs,
    ) -> Tuple[onnxruntime.InferenceSession, Optional[ModelProto]]:
        """Creates an Onnx Runtime session and verifies the correctness of the model.

        Args:
            model (ModelProto): The model to be used for inference.
            optimize (bool): Optimize the model using Onnx Runtime and return the optimized model.
                Defaults to False.
            execution_providers (str): Execution provider to use.
                Defaults to "CPUExecutionProvider".
            custom_lib (Optional[str | PathLike]): Path to the custom operator library.
                Defaults to None.
            kwargs (dict): Additional keyword arguments passed to the session options.

        Keyword Args:
            restore_back (bool): Restore back the external data.
                Applicable for models with size > 2 GB.
            include_attributes (bool): Whether to include tensors that are part of node attributes.
                Defaults to True.

        Returns:
            Tuple (onnxruntime.InferenceSession,Optional[ModelProto]): A tuple containing the
                created InferenceSession and the optimized ModelProto (if optimization is enabled).

        Raises:
            Exception: If the creation of the ORT session fails.
        """
        try:
            optimized_model = None
            sess_options = onnxruntime.SessionOptions()

            # Set log severity level to match OnnxFramework class logLevel
            sess_options.log_severity_level = OnnxModelHelper.get_log_severity_level(name=cls.__name__)

            # Load the custom lib if it's provided, and the path exists
            if custom_lib and os.path.exists(custom_lib):
                sess_options.register_custom_ops_library(custom_lib)
            elif custom_lib:
                cls.logger.warning(f"Path does not exist: {custom_lib}")

            # Create the inference session

            with tempfile.TemporaryDirectory(dir=QAIRT_TMP_DIR) as tmpdirname:
                temp_model_path = os.path.join(tmpdirname, "model.onnx")

                if optimize:
                    temp_model_path = os.path.join(tmpdirname, "model_optimized.onnx")

                restore_back = kwargs.get("restore_back", False)
                include_attributes = kwargs.get("include_attributes", True)

                OnnxModelHelper.save_model(
                    model, temp_model_path, restore_back=restore_back, include_attributes=include_attributes
                )

                session = onnxruntime.InferenceSession(
                    temp_model_path, sess_options, providers=[execution_providers], **kwargs
                )

                if optimize:
                    optimized_model = onnx.load(temp_model_path)

            return session, optimized_model

        except Exception as e:
            cls.logger.error(f"Creation of ORT session is failed! with Exception: {str(e)}")
            raise

    @classmethod
    def generate_input_dict(cls, model: onnx.ModelProto, input_data: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Generates input dictionary with keys as input tensor names and values as tensors,
        from given input tensors list.

        model (ModelProto): The ONNX model proto instance.
        input_data (List[np.ndarray]): Input tensors, as a list
        """
        # Extract the input tensor information from the model
        input_tensors = {input.name: input for input in model.graph.input}

        # Initialize the dictionary to hold the validated input data
        input_dict = {}

        # Iterate over the input data and validate against the model's input tensors
        for idx, data in enumerate(input_data):
            # Get the corresponding input tensor name
            input_name = list(input_tensors.keys())[idx]
            input_tensor = input_tensors[input_name]

            # Validate the shape
            expected_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            if data.shape != tuple(expected_shape):
                raise ValueError(
                    f"Shape mismatch for input '{input_name}': expected " f"{expected_shape}, got {data.shape}"
                )

            # Validate the type
            expected_type = input_tensor.type.tensor_type.elem_type
            expected_np_dtype = OnnxModelHelper.tensor_dtype_to_np_dtype(expected_type)

            if data.dtype != expected_np_dtype:
                raise ValueError(
                    f"Type mismatch for input '{input_name}': expected "
                    f"{expected_np_dtype}, got {data.dtype}"
                )

            # Assign the validated data to the input name
            input_dict[input_name] = data

        return input_dict

    @classmethod
    def get_all_tensors(cls, model: ModelProto, include_attributes: bool = True) -> List[TensorProto]:
        """Returns a list of all tensors (e.g., Initializer and constant attribute tensors)
        from the given ONNX model.

        Args:
            model (ModelProto): The ONNX model proto instance.
            include_attributes (bool): Whether to include tensors that are part of node attributes.
                                       Defaults to True.

        Returns:
            List[TensorProto]: List of all tensors in the model.
        """
        tensors = []

        for graph in OnnxModelHelper.get_graphs(model):
            tensors.extend(graph.initializer)

            if not include_attributes:
                continue
            for node in graph.node:
                for attribute in node.attribute:
                    if attribute.HasField("t"):
                        tensors.append(attribute.t)
                    tensors.extend(attribute.tensors)

        return tensors

    @classmethod
    def get_full_model(
        cls,
        model: ModelProto,
        external_data_path: str | PathLike,
        remove_file: bool = True,
        include_attributes: bool = True,
    ) -> ModelProto:
        """Get the entire model with weights, reloads external data into given model
            from provided external_data_path.

        Args:
            model (ModelProto): Onnx model proto.
            external_data_path (str | PathLike): Path to external data file.
            remove_file (bool): Remove external data file after loading.
                Defaults to True.
            include_attributes (bool): Include tensors which are part of node's attributes.
                Defaults to True

        Returns:
            ModelProto: Onnx model proto with weights.

        Raises:
            None
        """
        full_model = copy.deepcopy(model)

        full_model = OnnxModelHelper.load_external_data_for_model(full_model, os.path.sep, include_attributes)

        if remove_file:
            os.remove(external_data_path)

        return full_model

    @classmethod
    def get_graphs(cls, model: ModelProto) -> List[GraphProto]:
        """Returns a list of all graphs present in the given ONNX model.

        Args:
            model (ModelProto): The ONNX model graph proto.

        Returns:
            List[GraphProto]: A list of ONNX graphs.

        Raises:
            None
        """
        all_graphs = []
        graph_queue = [model.graph]

        while graph_queue:
            graph = graph_queue.pop(0)
            all_graphs.append(graph)

            for node in graph.node:
                for attr in node.attribute:
                    if attr.type == AttributeProto.AttributeType.GRAPH:
                        if not isinstance(attr.g, onnx_pb.GraphProto):
                            cls.logger.error(f"{attr.g} is not an instance of onnx_pb.GraphProto")
                        else:
                            graph_queue.append(attr.g)
                    if attr.type == AttributeProto.AttributeType.GRAPHS:
                        for g in attr.graphs:
                            if not isinstance(g, onnx_pb.GraphProto):
                                cls.logger.error(f"{g} is not an instance of onnx_pb.GraphProto")
                            else:
                                graph_queue.append(g)

        return all_graphs

    @classmethod
    def get_inputs(cls, model: ModelProto) -> Dict[str, ValueInfoProto]:
        """Get the graph inputs tensors except initializers.

        Args:
            model (ModelProto): The ONNX model proto instance.

        Returns:
            Dict[str, ValueInfoProto]: Dict mapping input tensor names to input tensors.
        """
        initializer_names = [x.name for x in model.graph.initializer]
        return {ipt.name: ipt for ipt in model.graph.input if ipt.name not in initializer_names}

    @classmethod
    def get_log_severity_level(cls, name: Optional[str] = None) -> int:
        """Maps Python logger levels to ONNX Runtime session log severity levels.

        Returns:
            int: ONNX Runtime session log severity level.
        """
        # Map logger levels to ONNX Runtime session log severity levels
        level_mapping = {
            logging.DEBUG: 0,  # Verbose
            logging.INFO: 1,  # Info
            logging.WARNING: 2,  # Warning
            logging.ERROR: 3,  # Error
            logging.CRITICAL: 4,  # Fatal
        }

        # Get the logger's effective log level
        logger_level = logging.getLogger(name or cls.__name__).getEffectiveLevel()

        # Map to ONNX Runtime session log severity level
        return level_mapping.get(logger_level)

    @classmethod
    def get_model_size(cls, model: ModelProto) -> float:
        """Provides the size of the ONNX model in gigabytes (GB).

        Args:
            model (ModelProto): An ONNX model proto instance.

        Returns:
            float: Size of the model in GB.

        Raises:
            RuntimeError: If no default domains are found in the model.

        Example:
            For an ONNX model with a size of 2.5 GB, the return value is 2.5.
        """
        NUM_GB_BYTES = 1024**3
        size_bytes = model.ByteSize()
        size_gb = size_bytes / NUM_GB_BYTES
        return size_gb

    @classmethod
    def get_opset_version(cls, model: ModelProto) -> int:
        """Returns the model opset version for the default domain.

        Args:
            model (ModelProto): An ONNX model proto instance.

        Returns:
            int: Opset version of the ONNX domain.

        Raises:
            RuntimeError: If no default domains are found in the model.

        Example:
            For an ONNX model with opset version 11 for the default domain, the return value is 11.
        """
        for opset in model.opset_import:
            if opset.domain in ["", "ai.onnx", "qti_aisw"]:
                return opset.version
        raise RuntimeError("The ONNX model has no opset for the default domain.")

    @classmethod
    def get_outputs(cls, model: ModelProto) -> Dict[str, ValueInfoProto]:
        """Get the graph outputs tensors.

        Args:
            model (ModelProto): The ONNX model proto instance.

        Returns:
            Dict[str, ValueInfoProto]: Dict mapping output tensor names to output tensors.
        """
        initializer_names = [x.name for x in model.graph.initializer]
        return {ipt.name: ipt for ipt in model.graph.output if ipt.name not in initializer_names}

    @classmethod
    def get_output_names(cls, model: ModelProto) -> List[str]:
        """Get the Onnx Model output names.

        Args:
            model (ModelProto): The ONNX model proto instance.

        Returns:
            List[str]: List of output names.
        """
        initializer_names = [x.name for x in model.graph.initializer]
        return [ipt.name for ipt in model.graph.output if ipt.name not in initializer_names]

    @classmethod
    def get_value_info_proto_mappings(cls, model: ModelProto) -> Dict[str, ValueInfoProto]:
        """Returns a dictionary mapping value info names to ValueInfoProtos.

        Args:
            model (ModelProto): The ONNX model proto instance.

        Returns:
            Dict[str, onnx.ValueInfoProto]: Dict mapping value info names to ValueInfoProtos.
        """
        return {v.name: v for v in model.graph.value_info}

    @classmethod
    def load_external_data_for_model(
        cls, model: ModelProto, base_dir: str, include_attributes: bool = True
    ) -> ModelProto:
        """Loads external tensors into model.

        Args:
            model (ModelProto): An ONNX model proto instance.
            base_dir (str): Base directory where external data is stored.
            include_attributes (bool): Whether to include tensors that are part of node attributes.
                                       Defaults to True.

        Returns:
            ModelProto: Updated onnx model

        Raises:
            None
        """
        for tensor in OnnxModelHelper.get_all_tensors(model, include_attributes):
            if uses_external_data(tensor):
                load_external_data_for_tensor(tensor, base_dir)
                # After loading raw_data from external_data, change the state of tensors
                tensor.data_location = TensorProto.DEFAULT
                # and remove external data
                del tensor.external_data[:]

        return model

    @classmethod
    def native_shape_inference(cls, model: ModelProto, delete_existing_shapes: bool = False) -> ModelProto:
        """Performs shape inference on the given ONNX model.

        Args:
            model (ModelProto): The ONNX model proto instance.
            delete_existing_shapes (bool): Delete existing shapes before performing shape inference.
                                            Defaults to False.

        Returns:
            ModelProto: The updated ONNX model proto instance with inferred shapes.
        """
        try:
            """
            As a first step try to run symbolic shape inference.
            If this fails then as a fallback mechanism use normal shape inference.
            """
            model = OnnxModelHelper.symbolic_shape_inference(model)
        except Exception as e:
            """
            Note: Symbolic shape inference will fail for CustomOps.
            So as a fall back we call normal shape inference.
            """

            cls.logger.warning("Symbolic shape inference failed. " f"Exception: {e}. Running normal shape inference.")

            try:
                model = OnnxModelHelper.shape_inference(model, delete_existing_shapes=delete_existing_shapes)
            except Exception as e:
                cls.logger.warning(f"Shape inference failed With exception: {e}.")
                raise

        cls.logger.debug("Shape inference successful")

        return model

    @classmethod
    def onnx_type_to_numpy(cls, onnx_type: str) -> Tuple[np.dtype, int]:
        """Returns the corresponding NumPy datatype for a given ONNX tensor element type.

        Args:
            onnx_type (str): ONNX tensor element type (e.g., '1', '2', '3', ...).

        Returns:
            tuple: A tuple containing the corresponding NumPy datatype and size (in bytes).

        Raises:
            KeyError: If the provided type is not supported.

        Example:
            For type '1', the return value is (np.float32, 4).
        """
        onnx_to_numpy = {
            "1": (np.float32, 4),
            "2": (np.uint8, 1),
            "3": (np.int8, 1),
            "4": (np.uint16, 2),
            "5": (np.int16, 2),
            "6": (np.int32, 4),
            "7": (np.int64, 8),
            "9": (np.bool_, 1),
        }
        try:
            return onnx_to_numpy[onnx_type]
        except KeyError:
            raise KeyError(f"Unsupported type: {onnx_type}")

    @classmethod
    def remove_shapes(cls, model: ModelProto) -> ModelProto:
        """Remove shape information from the onnx model.

        Args:
            model (ModelProto): The ONNX model proto instance.

        Returns:
            ModelProto: The updated ONNX model proto instance without shapes.
        """
        input_val_info = OnnxModelHelper.get_inputs(model)
        output_val_info = OnnxModelHelper.get_outputs(model)

        # Use names instead of ValueInfoProto instances
        input_names = set(input_val_info.keys())
        output_names = set(output_val_info.keys())

        for val_info in itertools.chain(model.graph.value_info, model.graph.output):
            if val_info.name in input_names or val_info.name in output_names:
                continue
            if val_info.type.HasField("tensor_type"):
                val_info.type.tensor_type.ClearField("shape")

        return model

    @classmethod
    def save_model(
        cls, model: ModelProto, filename: str | PathLike, restore_back: bool = True, include_attributes: bool = True
    ) -> None:
        """Saves the ONNX model to disk.

        Args:
            model (ModelProto): An ONNX model proto instance.
            filename (str | PathLike): Path at which the model is to be saved.
            restore_back (bool): Restore back the external data.
                Applicable for models with size > 2 GB.
            include_attributes (bool): Whether to include tensors that are part of node attributes.
                Defaults to True.

        Returns:
            None

        Raises:
            RuntimeError: If the model size exceeds 2 GB.

        Example:
            To save the ONNX model to a file named "my_model.onnx":
            OnnxModelHelper.save_model(my_model, "my_model.onnx")
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Using model.graph.name to create a unique model_name
        model_name = model.graph.name + ".onnx"

        if os.path.isfile(path):
            model_name = os.path.splitext(os.path.basename(path))[0]

        weight_file_name = model_name.split(".")[0] + ".data"

        # onnx.save doesn't support saving the model > THRESHOLD_MODEL_SIZE
        if OnnxModelHelper.check_model_size(model):
            model_dir = os.path.abspath(os.path.dirname(path))
            OnnxModelHelper.convert_model_to_external_data(
                model, file_name=weight_file_name, include_attributes=include_attributes
            )
            onnx.save(model, path)
            if restore_back:
                """
                onnx.save doesn't directly save models larger than 2GB.
                To work around this limitation, the method first converts the model to
                external data format (splitting weights into separate files) and then saves it.
                After saving, the external data files need to be loaded using
                load_external_data_for_model to ensure correct functionality.
                """
                OnnxModelHelper.load_external_data_for_model(model, model_dir)
        else:
            onnx.save(model, path)

    @classmethod
    def shape_inference(cls, model: ModelProto, delete_existing_shapes: bool = False) -> ModelProto:
        """Performs shape inference on the given ONNX model.

        Args:
            model (ModelProto): The ONNX model proto instance.
            delete_existing_shapes (bool): Delete existing shapes before performing shape inference.
                                            Defaults to False.

        Returns:
            ModelProto: The updated ONNX model proto instance with inferred shapes.
        """
        if delete_existing_shapes:
            model = OnnxModelHelper.remove_shapes(model)

        try:
            shapes_model = onnx.shape_inference.infer_shapes(model)
        except (ValueError, RuntimeError, TypeError) as e:
            cls.logger.debug(f"Direct shape inference failed: {e}. Attempting file-based approach.")

            with tempfile.TemporaryDirectory(dir=QAIRT_TMP_DIR) as tmpdir:
                temp_model_path = os.path.join(tmpdir, "temp.onnx")
                shapes_model_path = os.path.join(tmpdir, "shapes.onnx")

                try:
                    # Save model with external data
                    onnx.save(model, temp_model_path, save_as_external_data=True)

                    # Perform shape inference on file
                    onnx.shape_inference.infer_shapes_path(temp_model_path, shapes_model_path)

                    # Load the result
                    shapes_model = onnx.load(shapes_model_path)

                    # CRITICAL: Load external data before temp directory is cleaned up
                    shapes_model = OnnxModelHelper.load_external_data_for_model(
                        shapes_model, tmpdir, include_attributes=True
                    )

                except Exception as fallback_error:
                    cls.logger.error(f"File-based shape inference also failed: {fallback_error}")
                    raise RuntimeError(
                        f"Shape inference failed. Direct inference error: {e}. "
                        f"File-based inference error: {fallback_error}"
                    ) from fallback_error

        return shapes_model

    @classmethod
    def symbolic_shape_inference(cls, model: ModelProto) -> ModelProto:
        """Adds the symbolic shape info to the model file.

        Args:
            model (ModelProto): The ONNX model proto instance.

        Returns:
            ModelProto: The updated ONNX model proto instance with inferred
        """
        # Symbolic shape inference works for both ModelProtos < 2GB and > 2GB.
        try:
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference, get_opset

            # Symbolic Shape Inference doesn't need to explicitly remove
            # existing shapes.

            # Symbolic shape inference needs model with weights.
            onnx_opset = get_opset(model)
            if (not onnx_opset) or onnx_opset < SYMBOLIC_SHAPE_INFER_MIN_VER:
                raise RuntimeError(f"Symbolic Shape Inference only supports models of onnx opset \
                        {SYMBOLIC_SHAPE_INFER_MIN_VER} and above.")
            symbolic_shape_inference = SymbolicShapeInference(
                int_max=INT_MAX, auto_merge=False, guess_output_rank=False, verbose=0
            )

            all_shapes_inferred = False

            symbolic_shape_inference._preprocess(model)

            while symbolic_shape_inference.run_:
                all_shapes_inferred = symbolic_shape_inference._infer_impl()

            symbolic_shape_inference._update_output_from_vi()

            if not all_shapes_inferred:
                raise RuntimeError("Incomplete symbolic shape inference.")

            return symbolic_shape_inference.out_mp_
        except ImportError:
            raise ImportError(
                "Onnxruntime package not found in current environment. \
                    Symbolic Shape Inference will be skipped."
            )

        return None

    @classmethod
    def tensor_dtype_to_np_dtype(cls, expected_type: int) -> np.dtype:
        """
        Convert an ONNX tensor elem_type to a numpy dtype.
        Uses onnx.helper.tensor_dtype_to_np_dtype when available (>= 1.19.1),
        otherwise falls back to onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.

        Args:
            expected_type (int): ONNX tensor elem_type.
        Returns:
            np.dtype: Corresponding numpy dtype.
        """
        # For onnx >= 1.19.1
        helper = getattr(onnx, "helper", None)
        if helper is not None and hasattr(helper, "tensor_dtype_to_np_dtype"):
            return onnx.helper.tensor_dtype_to_np_dtype(expected_type)

        # Fallback to old mapping (onnx <= 1.18.x)
        mapping_mod = getattr(onnx, "mapping", None)
        if mapping_mod is not None and hasattr(mapping_mod, "TENSOR_TYPE_TO_NP_TYPE"):
            return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[expected_type]

        # If both are missing, something is wrong with the ONNX installation
        raise RuntimeError(
            "No ONNX API available to convert tensor dtype to numpy dtype."
        )

    @classmethod
    def unload_external_data(
        cls, model: ModelProto, temp_path: Optional[str | PathLike] = None, include_attributes: bool = True
    ) -> str | PathLike:
        """Unload external data from the given model into a temporary file.

        Args:
            model (ModelProto): An ONNX model proto instance.
            temp_path (Optional[str | PathLike]): Path to the temporary directory where
                external data will be stored.
                If not provided, a random directory will be created in the system's
                    temporary folder.
            include_attributes (bool): Whether to include tensors that are part of node attributes.
                                    Defaults to True.

        Returns:
            str | PathLike: Path to the file containing external data.

        Raises:
            None
        """
        if temp_path is None:
            temp_data_dir = os.path.join(os.getcwd(), ".tmp")
        else:
            temp_data_dir = os.path.abspath(temp_path)

        os.makedirs(temp_data_dir, exist_ok=True)

        temp_data_file = os.path.join(temp_data_dir, "model.data")

        OnnxModelHelper.convert_model_to_external_data(
            model, file_name=temp_data_file, include_attributes=include_attributes
        )
        OnnxModelHelper.write_external_data_tensors(model, temp_data_dir, include_attributes=include_attributes)

        return temp_data_file

    # TODO: Update filepath to dir
    @classmethod
    def write_external_data_tensors(
        cls, model: ModelProto, filepath: str, include_attributes: bool = True
    ) -> ModelProto:
        """Serializes data for all the tensors which have data location set to
        TensorProto.External.

        Note: This function also strips basepath information from all tensors'
        external_data fields.

        Args:
            model (ModelProto): An ONNX model proto instance.
            filepath (str): Path where the external data file is stored.
            include_attributes (bool): Whether to include tensors that are part of node attributes.
                                       Defaults to True.

        Returns:
            ModelProto: Updated onnx model

        Raises:
            None
        """
        for tensor in OnnxModelHelper.get_all_tensors(model, include_attributes):
            # Writing to external data happens in 2 passes:
            # 1. Tensors with raw data which pass the necessary conditions (size threshold etc)
            #    are marked for serialization
            # 2. The raw data in these tensors is serialized to a file
            # Thus serialize only if tensor has raw data and it was marked for serialization
            if uses_external_data(tensor) and tensor.HasField("raw_data"):
                save_external_data(tensor, filepath)
                tensor.ClearField("raw_data")

        return model

    @classmethod
    def _fix_constant_data_range(cls, model: ModelProto) -> ModelProto:
        """
        Fix constant out-of-range data in ONNX graph.
        Replaces scalar constants equal to INT_MAX or FP16_MAX with -1.
        
        Args:
            model (ModelProto): An ONNX model proto instance.
        
        Returns:
            ModelProto: Updated onnx model
        """
        for node in model.graph.node:
            if node.op_type != "Constant" or not node.attribute:
                continue

            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    tensor = attr.t
                    np_tensor = onnx.numpy_helper.to_array(tensor)

                    # Check for scalar tensor and out-of-range values
                    if np_tensor.ndim == 0 and np_tensor.item() in [FP16_MAX, INT_MAX]:
                        tensor.raw_data = np.array(-1, dtype=np_tensor.dtype).tobytes()
                        cls.logger.debug(f"Fixed constant data {np_tensor.item()} with -1.")

        cls.logger.debug("Completed fixing constant out-of-range data in ONNX graph.")
        return model

    @classmethod
    def create_minimal_onnx_model(
        cls,
        input_name: str = "input",
        output_name: str = "output",
        dtype: int = TensorProto.FLOAT,
        shape: Optional[List[int]] = None,
    ) -> ModelProto:
        """Create a minimal ONNX model consisting of a single Identity node.

        Args:
            input_name (str): Name of the input tensor. Defaults to "input".
            output_name (str): Name of the output tensor. Defaults to "output".
            dtype (int): ONNX TensorProto element type. Defaults to TensorProto.FLOAT.
            shape (Optional[List[int]]): Static shape for the input/output tensor. Defaults to [1].

        Returns:
            ModelProto: An ONNX model with graph: output = Identity(input).

        Raises:
            onnx.checker.ValidationError: If the generated model is invalid.
        """
        if shape is None:
            shape = [1]

        input_vi = onnx.helper.make_tensor_value_info(input_name, dtype, shape)
        output_vi = onnx.helper.make_tensor_value_info(output_name, dtype, shape)
        node = onnx.helper.make_node("Identity", inputs=[input_name], outputs=[output_name], name="Identity")
        graph = onnx.helper.make_graph([node], "SeedGraph", [input_vi], [output_vi])
        model = onnx.helper.make_model(graph, producer_name="onnx-seed")
        onnx.checker.check_model(model)
        return model
