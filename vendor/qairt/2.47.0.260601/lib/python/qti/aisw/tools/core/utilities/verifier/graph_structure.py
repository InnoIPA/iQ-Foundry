# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from pydantic import FilePath
from collections import OrderedDict
import json
import os
from typing import Dict, List

from qti.aisw.dlc_utils.snpe_dlc_utils import ModelInfo
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.dlc_utils.snpe_dlc_utils import OpRow

NUM_ELEMENTS_IN_OP_INFO = 3


class Component:
    """
    Component represents a layer for an engine (such as SNPE) along with their
    associated input and output tensors.
    """

    def __init__(self, key, info):
        """
        Args:
           key: Name of Op.
           info: input, output and encoding information of Op.
        """
        self.name = key
        self.type_ = info[0]
        if len(info[1:]) == NUM_ELEMENTS_IN_OP_INFO:
            self.input_tensors, self.output_tensors, self.output_encodings = info[1:]
        else:
            self.input_tensors, self.output_tensors = info[1:]
            self.output_encodings = None
        self.all_tensors = self.input_tensors.copy()
        self.all_tensors.update(self.output_tensors)


class GraphStructure:
    """
    This class implements functionality of extracting the graph structure from DLC file or
    loading it from Json file.
    """

    def __init__(self, graph_structure: dict):
        """
        Instantiation of GraphStructure.
        Args:
            graph_structure: OrderedDict of op/layer name to list of two
            lists (input tensors, output tensors).
        """

        if not isinstance(graph_structure, dict):
            raise TypeError("Expected argument 'graph_structure' to be a dictionary but got {}".format(
                type(graph_structure)))

        self._components = []
        self._tensor_list = []
        self._types = OrderedDict()
        self._tensor_dimension_dict = OrderedDict()
        self._all_output_tensor_encoding_dict = OrderedDict()
        self._initialize(graph_structure)

    @staticmethod
    def get_graph_structure_from_dlc(dlc: FilePath) -> Dict[str, OpRow]:
        """
        This static method extracts layers information from a dlc file
        The function returns dictionary having key as node-name and value as list containing tuple
        of type, input dimensions and output dimensions.
        Args:
           dlc: Path to dlc file.
        Returns:
            dict : Dictionary containing all nodes and their details like type, input dimensions and
            output dimensions.
        """

        def _get_structure(dlc_info: List[OpRow]) -> OrderedDict:
            """
            This function populates the graph structure by iterating over each OpRow.
            Args:
                dlc_info: List of OpRows that contains info about each layer present in the graph.
            Returns: Ordered dictionary containing layer names and details of layers like,
                     type, input tensors, output tensors, dimensions.
            """
            graph_list_structure = [(layer.name, [
                layer.type,
                dict(
                    zip([Helper.transform_node_names(input_name)
                         for input_name in layer.input_names], layer.input_dims)),
                dict(
                    zip([
                        Helper.transform_node_names(output_name) for output_name in
                        layer.output_names
                    ], layer.output_dims_list))
            ]) for layer in dlc_info]
            return OrderedDict(graph_list_structure)

        model_info = ModelInfo(str(dlc))

        if len(model_info.graph_names) > 1:
            # If incase in future multiple graphs are created then raise an exception to catch this
            # scenario
            raise Exception(
                "Multiple graphs found in DLC {}. Graph structure generation is not supported.".
                format(dlc))
        for graph_name in model_info.graph_names:
            (dlc_info, total_params, total_macs, total_memory_cpu, total_memory_fxp, total_op_types,
             is_stripped) = model_info.extract_model_info(graph_name)
            graph_structure = _get_structure(dlc_info)
        return graph_structure

    @classmethod
    def load_graph_structure_from_file(cls, json_graph_file: FilePath) -> "GraphStructure":
        """
        Retrieves the structure of a graph from file.
        Args:
           json_graph_file: Json file path to retrieve graph structure from.
        Returns:
            Returns GraphStructure
        """

        graph_structure = json.load(open(json_graph_file), object_pairs_hook=OrderedDict)
        return cls(graph_structure)

    def _initialize(self, graph_structure: dict):
        """Initialize internal state based on graph structure.
        Args:
            graph_structure: OrderedDict of op/layer name to list of two
            lists (input tensors, output tensors).
        """
        self._components = [
            Component(key, info) for key, info in graph_structure.items()
        ]

        for component in self._components:
            # Initialize list of all tensors in graph structure without duplicates
            self._tensor_list = self._tensor_list + list(component.output_tensors.keys())
            for output_name in component.output_tensors:
                self._types[output_name] = component.type_
            # Initialize dictionary of all tensors to their dimensions
            self._tensor_dimension_dict.update(component.all_tensors)
            if component.output_encodings:
                self._all_output_tensor_encoding_dict.update(component.output_encodings)

    @staticmethod
    def save_graph_structure(graph_file: FilePath, graph_struct: dict):
        """Save the structure of a graph to file.
        Args:
            graph_file: Filename of where to store graph structure.
            graph_struct: OrderedDict of op/layer name to list of two
            lists (input tensors, output tensors). The value of an
            inference engine or framework's get_graph_structure should
            be passed in.
        Returns:
            None
        """
        os.makedirs(os.path.dirname(graph_file), exist_ok=True)

        with open(graph_file, 'w+') as f:
            json.dump(graph_struct, f, indent=4)

    @property
    def all_types(self):
        return self._types

    @property
    def all_tensors(self):
        return self._tensor_list

    @property
    def tensor_dimension_dict(self):
        return self._tensor_dimension_dict

    @property
    def output_tensor_encodings_dict(self):
        return self._all_output_tensor_encoding_dict
