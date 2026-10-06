# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from os import path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pydantic import DirectoryPath, FilePath
from qti.aisw.tools.core.utilities.comparators.comparator import Comparator
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger
from qti.aisw.tools.core.utilities.verifier.layout import TensorLayout
from qti.aisw.tools.core.utilities.verifier.utils import (
    get_tensor_paths,
    permute_tensor_data_axis_order,
)


class Verifier:
    """This class implements the functionality of comparing reference and inference output tensors.
    For the comparators used, the corresponding comparator object should be created and passed.

    Examples:
       Example-1:
       >>> dir_path_ref_tensor = '/path/to/ref/tensors/'
       >>> dir_path_inf_tensor = '/path/to/inf/tensors/'
       >>> mse_comparator = MSEComparator()
       >>> std_comparator = STDComparator()
       >>> verifier = Verifier([mse_comparator,std_comparator])
       >>> verify_result = verifier.verify_directory_of_tensors(dir_path_ref_tensor,
       >>>                                                      dir_path_inf_tensor)
       Example-2:
       >>> list_of_ref_raw_files = List[FilePath]
       >>> list_of_inf_raw_files = List[FilePath]
       >>> cosine comparator = CosineComparator()
       >>> verifier = Verifier([cosine_comparator])
       >>> verify_result = verifier.verify_list_of_arrays(list_of_ref_raw_files,
       >>>                                                list_of_inf_raw_files)
       Example-3:
       >>> dict_ref_tensors = output_from_framework_utils
       >>> dict_inf_tensors = output_from_net-run module
       >>> l2_norm_comparator = L2NormComparator()
       >>> verifier = Verifier([l2_norm_comparator])
       >>> verify_result = verifier.verify_dictionary_of_tensors(dict_ref_tensors,
       >>>                                                       dict_inf_tensors)

    """

    def __init__(self, comparators: list[Comparator], logger: Any = None) -> None:
        """Args:
        comparators: List of comparators to use for verification.
        logger: Logger object.
        """
        if logger:
            self.logger = logger
        else:
            self.log_area = LogAreas.register_log_area("Verifier")
            self.logger = QAIRTLogger.register_area_logger(self.log_area, level="INFO")

        self._comparator_obj = comparators

    def verify_list_of_arrays(
        self, reference_tensors: List[NDArray], inference_tensors: List[NDArray]
    ) -> List[Dict[str, Any]]:
        """This function takes a list of reference tensors, an inference tensor, their respective data
        types.

        Args:
            reference_tensors: List of reference tensors.
            inference_tensors: List of inference tensors.
        Returns: List of dictionaries where each dictionary represents the comparison result of each
                 pair of reference and inference tensors.
        """
        if reference_tensors is None or inference_tensors is None:
            raise ValueError("No input tensors found to perform verification!")

        if len(reference_tensors) != len(inference_tensors):
            raise ValueError("Both input tensors should be of same length")

        compare_results = []
        for ref, inf in zip(reference_tensors, inference_tensors):
            compare_results.append(
                {
                    comparator.name: comparator.compare(
                        [ref.astype(np.float32)], [inf.astype(np.float32)]
                    )[0]
                    for comparator in self._comparator_obj
                }
            )
        return compare_results

    def verify_list_of_files(
        self,
        reference_tensors: List[FilePath],
        inference_tensors: List[FilePath],
        reference_dtype: DTypeLike = None,
        inference_dtype: DTypeLike = None,
    ) -> List[Dict[str, Any]]:
        """The function iterates over each pair of reference and inference tensors, compares them
        using given comparators and produces a list of dictionaries, where each dictionary represents
        the comparison result of each pair of reference and inference tensors.

        Args:
            reference_tensors: List of reference tensors.
            inference_tensors: List of inference tensors.
            reference_dtype: Data type of the reference tensors.
            inference_dtype: Data type of the inference tensors.
        Returns: List of dictionaries where each dictionary represents the comparison result of each
                 pair of reference and inference tensors.
        """
        if reference_tensors is None or inference_tensors is None:
            raise ValueError("No input tensors found to perform verification!")

        if len(reference_tensors) != len(inference_tensors):
            raise ValueError("Both input tensors should be of same length")

        compare_results = []
        for ref_file, inf_file in zip(reference_tensors, inference_tensors):
            ref_tensor = np.fromfile(ref_file, reference_dtype).astype(np.float32)
            inf_tensor = np.fromfile(inf_file, inference_dtype).astype(np.float32)
            compare_results.append(
                {
                    comparator.name: comparator.compare([ref_tensor], [inf_tensor])[0]
                    for comparator in self._comparator_obj
                }
            )
        return compare_results

    def verify_dictionary_of_tensors(
        self,
        reference_tensors: Dict[str, NDArray],
        inference_tensors: Dict[str, NDArray],
        dlc_file: FilePath = None,
        graph_info: Dict = None,
        disable_layout_transform: bool = False,
    ) -> (Dict)[Tuple[str, str], Dict[str, Any]]:
        """This function takes a dictionaries where each dictionary contains both a reference
        tensor and an inference tensor.

        Args:
            reference_tensors: List of dictionaries where each dictionary contains reference tensor.
            inference_tensors: List of dictionaries where each dictionary contains inference tensor.
            dlc_file: Path to DLC file.
            graph_info: Dictionary containing graph information like, tensor mapping,
                        graph structure and layout information.
            disable_layout_transform: When True, skips tensor layout permute step
        Returns: List of dictionaries where each dictionary represents the comparison result of each
                 pair of reference and inference tensors.
        """
        if reference_tensors is None or inference_tensors is None:
            raise ValueError("No input tensors found to perform verification!")

        if not dlc_file and not graph_info:
            raise ValueError("Both DLC file path and graph info cannot be none.")

        result = self._verify(
            reference_tensors,
            inference_tensors,
            None,
            None,
            dlc_file,
            graph_info,
            in_memory=True,
            disable_layout_transform=disable_layout_transform,
        )

        return result

    def verify_directory_of_tensors(
        self,
        reference_tensors: DirectoryPath,
        inference_tensors: DirectoryPath,
        reference_dtype: DTypeLike = None,
        inference_dtype: DTypeLike = None,
        dlc_file: FilePath = None,
        graph_info: Dict = None,
        disable_layout_transform: bool = False,
    ) -> (Dict)[Tuple[str, str], Dict[str, Any]]:
        """This function compares two sets of tensors stored on disk based on the specified comparators
        and produces a dictionary containing the comparison results.

        Args:
            reference_tensors: Dictionary containing reference_tensors.
            inference_tensors: Dictionary containing inference_tensors:
            reference_dtype: Data type of reference tensors
            inference_dtype: Data type of inference tensors
            dlc_file: Path to DLC file.
            graph_info: Dictionary containing graph information like, tensor mapping,
                        graph structure and layout information.
            disable_layout_transform: When True, skips tensor layout permute step
        Returns: List of dictionaries where each dictionary represents the comparison result of each
                 pair of reference and inference tensors.
        """
        if reference_tensors is None or inference_tensors is None:
            raise ValueError("No input tensors found to perform verification!")

        if not path.exists(reference_tensors) or not path.exists(inference_tensors):
            raise ValueError("Tensor directory path doesn't exist.")

        result = self._verify(
            reference_tensors,
            inference_tensors,
            reference_dtype,
            inference_dtype,
            dlc_file,
            graph_info,
            in_memory=False,
            disable_layout_transform=disable_layout_transform,
        )

        return result

    def _verify(
        self,
        reference_tensors: Dict[str, NDArray] | DirectoryPath,
        inference_tensors: Dict[str, NDArray] | DirectoryPath,
        reference_dtype: DTypeLike = None,
        inference_dtype: DTypeLike = None,
        dlc_file: FilePath = None,
        graph_info: dict = None,
        in_memory: bool = True,
        disable_layout_transform: bool = False,
    ) -> Dict[Tuple[str, str], Dict[str, np.array]]:
        """This function compares two sets of tensors either loaded from files or directly passed as
        Numpy arrays.

        Args:
            reference_tensors: Either a dictionary of reference tensors or a directory containing
                               reference tensors.
            inference_tensors: Either a dictionary of inference tensors or a directory containing
                               inference tensors.
            reference_dtype: Datatype of reference tensors.
            inference_dtype: Datatype of inference tensors.
            dlc_file: Path to DLC file.
            graph_info: Dictionary containing graph information like, tensor mapping,
                        graph structure and layout information.
            in_memory: Boolean specifying whether the tensors are in memory or on disk.
            disable_layout_transform: When True, skips tensor layout permute step

        Returns:
        """
        if in_memory:
            reference_tensor_list = list(reference_tensors.keys())
            inference_tensor_list = list(inference_tensors.keys())
            tensor_mapping = self._get_tensor_mapping(dlc_file, graph_info=graph_info)
            reference_tensors_dict = reference_tensors
            inference_tensors_dict = inference_tensors
        else:
            reference_dtype = reference_dtype if reference_dtype else np.float32
            inference_dtype = inference_dtype if inference_dtype else np.float32
            reference_tensors_dict = get_tensor_paths(reference_tensors)
            inference_tensors_dict = get_tensor_paths(inference_tensors)

            inference_tensor_list = list(inference_tensors_dict.keys())
            reference_tensor_list = list(reference_tensors_dict.keys())
            tensor_mapping = self._get_tensor_mapping(
                ref_tensor_dir=reference_tensors,
                inf_tensor_dir=inference_tensors,
                graph_info=graph_info,
            )

        graph_structure_obj = self._get_graph_structure(dlc_file, graph_info)
        if graph_structure_obj:
            inference_tensors_op_types = graph_structure_obj.all_types
            tensor_dimensions = graph_structure_obj.tensor_dimension_dict
            inference_tensor_list = [
                tensor
                for tensor in graph_structure_obj.all_tensors
                if tensor in inference_tensor_list
            ]
        else:
            inference_tensors_op_types = None
            tensor_dimensions = None

        layout_info = None if disable_layout_transform else self._get_layout(dlc_file, graph_info)

        inference_to_golden_tensor_map = self._generate_inference_to_golden_map(
            inference_tensor_list, tensor_mapping
        )

        compare_result: Dict[Tuple[str, str], Dict[str, np.array]] = {}

        for inference_tensor_name, reference_tensor_name in inference_to_golden_tensor_map.items():
            if (
                inference_tensor_name in inference_tensor_list
                and reference_tensor_name in reference_tensor_list
            ):
                if in_memory:
                    inf_tensor_dim = inference_tensors[inference_tensor_name].shape
                    ref_tensor = reference_tensors_dict[reference_tensor_name].flatten()
                    inf_tensor = inference_tensors_dict[inference_tensor_name].flatten()
                    # Typecasting both reference and inference tensors to fp32 just in case they
                    # of different datatypes
                    ref_tensor = ref_tensor.astype(np.float32)
                    inf_tensor = inf_tensor.astype(np.float32)
                else:
                    ref_tensor = np.fromfile(
                        reference_tensors_dict[reference_tensor_name], reference_dtype
                    ).astype(np.float32)
                    inf_tensor = np.fromfile(
                        inference_tensors_dict[inference_tensor_name], inference_dtype
                    ).astype(np.float32)
                    inf_tensor_dim = (
                        tensor_dimensions[inference_tensor_name]
                        if (tensor_dimensions and (inference_tensor_name in tensor_dimensions))
                        else None
                    )

                if layout_info and inference_tensor_name in layout_info:
                    inf_tensor = permute_tensor_data_axis_order(
                        inf_tensor, layout_info[inference_tensor_name]
                    )

                result = self._compare_tensors(
                    ref_tensor,
                    inf_tensor,
                    inference_tensors_op_types[inference_tensor_name]
                    if inference_tensors_op_types
                    else None,
                    inf_tensor_dim,
                )

                compare_result[(inference_tensor_name, reference_tensor_name)] = result

        return compare_result

    def _compare_tensors(
        self, ref_tensor: np.array, inf_tensor: np.array, op_type: str = "", dimensions: str = ""
    ) -> dict:
        """This method compares two tensors using one or more comparators and returns the error metrics.

        Args:
            ref_tensor: Tensor 1 used as reference.
            inf_tensor: Tensor 2
            op_type: Type of operator
            dimensions: Dimensions of tensors
        Returns:
            Dictionary: Comparison results
        """
        compare_outputs = {}
        for comparator in self._comparator_obj:
            try:
                compare_outputs[comparator.name] = comparator.compare([ref_tensor], [inf_tensor])[0]
            except Exception as e:
                self.logger.warning(f"Tensor comparison failed with error '{e}'")
                compare_outputs[comparator.name] = float("nan")
        compare_outputs["op_type"] = op_type
        compare_outputs["dimensions"] = dimensions
        return compare_outputs

    @staticmethod
    def _generate_inference_to_golden_map(inference_tensors: list, mapping: dict) -> dict:
        """This method generates a map of inference to golden tensor names.

        Args:
          inference_tensors: list of inference tensor name
          mapping: Tensor mapping from inference tensors to reference tensors.
        Returns: Filtered mapping from inference tensors to reference tensors.
        """
        return {
            inference: mapping[inference]
            if inference in mapping and mapping[inference] is not None
            else inference
            for inference in inference_tensors
        }

    @staticmethod
    def _get_tensor_mapping(
        dlc_file: FilePath = None,
        graph_info: dict = None,
        ref_tensor_dir: DirectoryPath = None,
        inf_tensor_dir: DirectoryPath = None,
    ) -> dict:
        """This method returns tensor mapping from reference outputs to inference outputs.

        Args:
            dlc_file: Path to dlc file.
            graph_info: Dictionary containing graph information like, tensor mapping,
                        graph structure and layout information.

        Returns:
            Dictionary: Dictionary containing mapping of inference and reference outputs.
        """
        tensor_mapping = (
            graph_info["tensor_mapping"]
            if graph_info and ("tensor_mapping" in graph_info)
            else None
        )
        if tensor_mapping:
            # If tensor mapping file exists then load it into dictionary.
            return tensor_mapping
        else:
            from qti.aisw.tools.core.utilities.tensor_mapping.tensor_mapping import (
                TensorMapper,
                TensorMapperInputConfig,
            )
            # if DLC file is provided, generate tensor mapping using tensor mapping utility.
            if dlc_file:
                tensor_mapping_input = TensorMapperInputConfig(dlc_path=dlc_file)
                tensor_mapping_obj = TensorMapper()
                tensor_mapping = tensor_mapping_obj.run(tensor_mapping_input).tensor_mapping_output

            elif ref_tensor_dir and inf_tensor_dir:
                tensor_mapping_input = TensorMapperInputConfig(
                    golden_reference_output=ref_tensor_dir, inference_output=inf_tensor_dir
                )
                tensor_mapping_obj = TensorMapper()
                tensor_mapping = tensor_mapping_obj.run(tensor_mapping_input).tensor_mapping_output

        return tensor_mapping

    @staticmethod
    def _get_graph_structure(dlc_file: FilePath, graph_info: dict):
        """This method loads graph structure either from previously generated JSON file or extract it
        from DLC file and returns graph structure object.

        Args:
            dlc_file: Path to dlc file.
            graph_info: Dictionary containing graph information like, tensor mapping,
                        graph structure and layout information.

        Returns:
            Dict: Graph structure
        """
        graph_structure = (
            graph_info["graph_structure"]
            if graph_info and ("graph_structure" in graph_info)
            else None
        )
        if graph_structure:
            # Load graph structure from Json.
            from qti.aisw.tools.core.utilities.verifier.graph_structure import GraphStructure
            graph_structure_obj = GraphStructure(graph_structure)
        elif dlc_file:
            from qti.aisw.tools.core.utilities.verifier.graph_structure import GraphStructure
            graph_structure_dict = GraphStructure.get_graph_structure_from_dlc(str(dlc_file))
            graph_structure_obj = GraphStructure(graph_structure_dict)
        else:
            graph_structure_obj = None
        return graph_structure_obj

    @staticmethod
    def _get_layout(dlc_file: FilePath, graph_info: dict) -> dict:
        """This method generates layout information from DLC file.

        Args:
            dlc_file: Path to dlc file.
            graph_info: Dictionary containing graph information like, tensor mapping,
                        graph structure and layout information.

        Returns:
            Dict: Layout Information. Key representing inference tensor.
        """
        layout_info = (
            graph_info["layout_info"] if graph_info and "layout_info" in graph_info else None
        )
        if layout_info:
            return layout_info
        elif dlc_file:
            layout_obj = TensorLayout()
            layout_info = layout_obj.get_layout_info_from_dlc(dlc_file)
        return layout_info
