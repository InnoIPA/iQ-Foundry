# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import yaml
from qti.aisw.core.model_level_api.utils.subprocess_executor import (
    get_np_dtype_from_qnn_dtype,
)

from qti.aisw.core.model_level_api.utils.native_executor import (
    input_list_to_in_memory_input,
)


def validate_input_tensor_dimensions(parsed_data: dict) -> bool:
    """
    Checks if all input tensors have dimensions.

    Args:
        parsed_data (dict): The parsed YAML data containing the input tensor information.

    Returns:
        bool: True if all input tensors have dimensions, False otherwise.
    """
    graphs = parsed_data.get("graphs", [])

    if not graphs:
        raise ValueError("No graphs found in the YAML data.")

    return all(
        tensor.get("dimensions", None) is not None
        and len(tensor.get("dimensions", [])) > 0
        for graph in graphs
        for tensor_set in graph.get("tensorSets", [])
        for tensor in tensor_set.get("inputTensors", [])
    )


def extract_and_load_input_tensors(
    yaml_path: str,
    native_inputs: bool = False,
    native_input_tensor_names: list = None,
    graph_input_name_dtype_pairs: List[Dict[str, List[Tuple[str, str]]]] = None,
) -> Union[list, dict[str, list]]:
    """
    Parses a YAML input file, and determines whether input tensors are dynamic or static and extracts them accordingly.
    It basically checks if the input tensors have dimensions specified in the YAML data.
    If dimensions are present, it loads the input tensors as dynamic tensors. Otherwise,
    it loads them as static tensors.

    Args:
        yaml_path (str): The path to the YAML input file.
        native_inputs (bool): Whether to use native inputs.
        native_input_tensor_names (list): List of native input tensor names.
        graph_input_name_dtype_pairs (list): List of input name and data type pairs for
        each graph in the parsed_yaml file.

    Returns:
        dict or list: Either a dictionary mapping input tensor names to their corresponding NumPy arrays
            (for dynamic tensors), or a list of dictionaries (for static tensors).

    Note: A Static Tensor can have dimensions, if network specialization is enabled. If
    Network Specialization isn't enabled, Static Tensors should contain only one graph.
    """
    with open(yaml_path, "r") as f:
        parsed_data = yaml.safe_load(f)
    num_graphs_present = len(parsed_data.get("graphs", []))
    if num_graphs_present == 1 or not validate_input_tensor_dimensions(parsed_data):
        # TODO: Add check for confirming network specialization isnt enabled in static tensors,
        # in that case - user must provided dimensions else error is thrown.
        # To be addressed as part of [AISW-147090]
        return load_static_input_tensors(
            yaml_path,
            native_inputs,
            native_input_tensor_names,
            graph_input_name_dtype_pairs,
        )
    else:
        return load_dynamic_input_tensors(
            parsed_data,
            native_inputs,
            native_input_tensor_names,
            graph_input_name_dtype_pairs,
        )


def extract_all_graph_input_dtypes(
    input_name_dtype_mapping: Dict[str, List[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    """
    Extracts all (input_name, dtype) pairs from a dictionary mapping graph names
    to a list of their input tensor names and data types. This is specially required for
    static_input_tensors, that are in format as specified in input_name_dtype_mapping.

    Args:
        input_name_dtype_mapping: A dictionary where keys are graph names (str)
                                  and values are lists of (input_name, dtype) tuples.

    Returns:
        A flattened list of all (input_name, dtype) tuples found across all graphs.
    """
    all_inputs_with_dtypes = []
    for _, input_list_for_graph in input_name_dtype_mapping.items():
        # input_list_for_graph is already a List[Tuple[str, str]]
        all_inputs_with_dtypes.extend(input_list_for_graph)
    return dict(all_inputs_with_dtypes)


def load_static_input_tensors(
    yaml_path: str,
    native_inputs: bool = False,
    native_input_tensor_names: Optional[list] = None,
    graph_input_name_dtype_pairs: List[Dict[str, List[Tuple[str, str]]]] = None,
) -> list[dict[str, np.ndarray]]:
    """
    Loads static input tensors from the YAML file. It is used to parse a YAML input file,
    extract the input tensor information,and load the input tensors into memory.
    It supports both native and non-native input data types.


    Args:
        yaml_path (str): The path to the YAML input file.
        native_inputs (bool): Whether to use native inputs.
        native_input_tensor_names (list): List of native input tensor names.
        graph_input_name_dtype_pairs (list): List of input name and data type pairs for
            each graph in the parsed_yaml file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
            the tensor name and its corresponding NumPy array.
    """
    graph_input_name_dtype_pairs_dict = {}
    if graph_input_name_dtype_pairs and isinstance(graph_input_name_dtype_pairs, list):
        # This function handles classic static inputs, where per-tensor dimensions are not explicitly provided in the YAML.
        # In this specific scenario, the input YAML is expected to describe a single graph.
        # Multi-graph configurations in network specialization (which require explicit dimensions)
        # are processed by `load_dynamic_input_tensors`. Hence, we extract the input-dtype mapping
        # from the first (and only expected) element of the list.
        graph_input_name_dtype_pairs_dict = extract_all_graph_input_dtypes(
            graph_input_name_dtype_pairs[0]
        )
    return input_list_to_in_memory_input(
        Path(yaml_path),
        native_inputs,
        native_input_tensor_names,
        graph_input_name_dtype_pairs_dict,
    )


def load_dynamic_input_tensors(
    parsed_data: dict,
    native_inputs: bool = False,
    native_input_tensor_names: list = None,
    graph_input_name_dtype_pairs: List[Dict[str, List[Tuple[str, str]]]] = None,
) -> dict[str, list[dict[str, np.ndarray]]]:
    """
    Loads dynamic input tensors with dimensions from the parsed YAML data.

    It iterates over the input tensors in the parsed YAML data, loads each
    tensor from its file path, and reshapes it according to its specified dimensions.

    Args:
        parsed_data (dict): The parsed YAML data containing the input tensor information.
        native_inputs (bool): Whether to use native inputs.
        native_input_tensor_names (list): List of native input tensor names.
        graph_input_name_dtype_pairs (list): List of input name and data type pairs for
        each graph in the parsed_yaml file.

    Returns:
        dict: A dictionary mapping graph names to lists of dictionaries, where each dictionary
            maps input tensor names to their corresponding NumPy arrays.

    Sample output dictionary format:
    {
        'graph_name1': [
            {'input_tensor_name1': numpy_array1},
            {'input_tensor_name2': numpy_array2},
            ...
        ],
        'graph_name2': [
            {'input_tensor_name3': numpy_array3},
            {'input_tensor_name4': numpy_array4},
            ...
        ],
        ...
    }
    """
    input_file_map = {}
    graphs = parsed_data.get("graphs", [])

    if not graphs:
        raise ValueError("No graphs found in the YAML data.")

    for graph in graphs:
        graph_name = graph.get("name")
        input_file_map[graph_name] = []
        for tensor_set in graph.get("tensorSets", []):
            inference_input_dict = {}
            for input_tensor in tensor_set.get("inputTensors", []):
                input_name = input_tensor.get("name")
                file_path = input_tensor.get("filePath")
                dimensions = input_tensor.get("dimensions")

                if not input_name or not file_path or not dimensions:
                    raise ValueError(
                        f"Invalid tensor configuration: missing or empty fields: Received: name={input_name}, file_path={file_path}, dimensions={dimensions}"
                    )

                dimensions = [int(dim) for dim in dimensions]

                # Determining the numpy datatype based on the input data type
                # this logic is referenced from input_list_to_in_memory_input
                if native_inputs or (
                    native_input_tensor_names
                    and input_name in native_input_tensor_names
                ):
                    try:
                        graph_input_name_dtype_pairs_dict = next(
                            d for d in graph_input_name_dtype_pairs if graph_name in d
                        )
                    except StopIteration:
                        raise ValueError(f"Graph name not found: {graph_name}")

                    input_dtype = next(
                        (
                            dtype
                            for name, dtype in graph_input_name_dtype_pairs_dict[
                                graph_name
                            ]
                            if name == input_name
                        ),
                        None,
                    )
                    if input_dtype is None:
                        raise ValueError(f"Input dtype not found for {input_name}")
                    np_dtype = get_np_dtype_from_qnn_dtype(input_dtype)
                else:
                    # Note: The dtype will be hardcoded to float32, since
                    # in qnn-net-run, the inputs are assumed to be fp32 unless the user marks
                    # the input as native using –use_native_input_files
                    # (which indicates that all inputs are their native datatype) or
                    # –native_input_tensor_names (which indicates that the named inputs are native, and all others are fp32)
                    np_dtype = np.float32

                data = np.fromfile(file_path, dtype=np_dtype)

                # Checking if the size of the data matches the size of the shape
                expected_size = np.prod(dimensions)
                actual_size = data.size

                if expected_size != actual_size:
                    raise ValueError(
                        f"Size mismatch for {input_name}. Expected {expected_size}, got {actual_size}"
                    )

                data = data.reshape(dimensions)
                inference_input_dict[input_name] = data
            input_file_map[graph_name].append(inference_input_dict)

    return input_file_map
