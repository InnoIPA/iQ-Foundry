# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


import os
import numpy as np
import json
import shutil
from typing import List
import copy
from qti.aisw.converters.common.tensor_transforms.operator import Operator
from qti.aisw.converters.common import modeltools
from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning


class Transform:
    def __init__(self, src_tensors, dest_tensors, operator, tensor_dtypes=None):
        """
        Initialize a Transform object that represents a tensor transformation.

        Args:
            src_tensors (list[str]): Names of the input tensors
            dest_tensors (list[str]): Names of the output tensors
            operator (Operator): Operator object containing the ONNX operator information
            tensor_dtypes (dict): Dict containing tensor_name: dtype mapping
        """
        self.src_tensors = src_tensors
        self.dest_tensors = dest_tensors
        self.tensor_dtypes = tensor_dtypes
        self.operator = operator

    def __str__(self):
        """String representation of the Transform."""
        return f"Transform(src_tensors={self.src_tensors}, " \
               f"dest_tensors={self.dest_tensors}, operator={self.operator})"


class TransformManager:
    SUPPORTED_DTYPES = {
        'float32', 'float16', 'int8', 'int16', 'int32', 'int64',
        'uint8', 'uint16', 'uint32', 'uint64'
    }
    class TransformGraph:
        """Inner class to hold a set of transforms"""
        def __init__(self, order_num, concurrency_name, split_num, ar_n):
            self.transforms = []  # List to store Transform objects
            self.input_tensors = set()  # Tensors that are only inputs (no producers)
            self.output_tensors = set()  # Tensors that are only outputs (no consumers)
            self.produced_tensors = set()  # All tensors that appear as destinations
            self.consumed_tensors = set()  # All tensors that are used as inputs
            self.execution_order = order_num
            self.concurrency_name = concurrency_name
            self.split_num = split_num
            self.ar_n = ar_n
            self.tensor_dtype_info = dict()

    def __init__(self):
        """
        Initialize a TransformManager to manage multiple graphs of tensor transformations.
        Each graph is identified by a string ID, with 'base' as the default graph.
        """
        self.transform_graphs = {}  # Dictionary mapping IDs to TransformGraph objects

    def get_transform_graph(self, graph_id):
        """Get the TransformGraph for an ID - throws error if it doesn't exist."""
        if graph_id not in self.transform_graphs:
            raise ValueError(f"Transform graph {graph_id} doesn't exist")
        return self.transform_graphs[graph_id]

    # Creates a new graph, and returns TransformGraph object for the corresponding new graph name
    def create_transform_graph(self, graph_id, order_num, concurrency_name=None, split_num=None, ar_n=None):
        if graph_id in self.transform_graphs:
            raise ValueError(f"Transform_graph {graph_id} already exists!")

        self.transform_graphs[graph_id] = self.TransformGraph(order_num, concurrency_name, split_num, ar_n)
        return self.transform_graphs[graph_id]


    def get_graph_ids(self):
        """Get a list of all transform graph IDs.

        Returns:
            list[str]: List of transform graph IDs
        """
        return list(self.transform_graphs.keys())


    def copy_graph(self, source_id, new_id):
        """
        Copy an existing transform graph with a new ID.

        Args:
            source_id (str): ID of the existing transform graph to copy
            new_id (str): ID for the new copied transform graph

        Raises:
            KeyError: If the source transform graph doesn't exist
        """
        if source_id not in self.transform_graphs:
            raise KeyError(f"Transform graph '{source_id}' not found.")

        # Create a deep copy of the source transform graph
        copied_state = copy.deepcopy(self.transform_graphs[source_id])
        self.transform_graphs[new_id] = copied_state


    def rename_graph(self, source_id, new_id):
        """
        Rename an existing transform graph with a new ID.

        Args:
            source_id (str): ID of the existing transform graph to rename
            new_id (str): ID for the renamed transform graph

        Raises:
            KeyError: If the source transform graph doesn't exist
            KeyError: If the new transform graph already exists
        """
        if source_id not in self.transform_graphs:
            raise KeyError(f"Transform graph '{source_id}' not found.")
        if source_id == new_id:
            log_warning(f"Graph source_id and new_id are identical. Skipping rename for {source_id}")
        elif new_id in self.transform_graphs:
            raise KeyError(f"Transform graph '{new_id}' already exists. Cannot rename {source_id}")
        else:
            # Rename the transform graph
            transform_graph = self.transform_graphs.pop(source_id)
            self.transform_graphs[new_id] = transform_graph


    def add_transform(self, src_tensors, dest_tensors, operator, graph_id, no_track=False, tensor_dtypes=None):
        """Add a new transformation to the manager and update input/output tensor sets."""
        def _add_tensor_dtypes():
            """Helper to add dtypes for tensors in transform"""
            src_tensors_set = set(src_tensors)
            dest_tensors_set = set(dest_tensors)
            transform_tensors_set = src_tensors_set.union(dest_tensors_set)
            for tensor_name, dtype in tensor_dtypes.items():
                if tensor_name not in transform_tensors_set:
                    raise ValueError(f"Tensor {tensor_name} not found in transform src or dest tensors. Cannot set datatype")

                existing_dtype = self._get_tensor_dtype(tensor_name, graph_id)

                if not existing_dtype:
                    self._set_tensor_dtype(tensor_name, dtype, graph_id)

                elif existing_dtype != dtype:
                    raise ValueError(f"Datatype mismatch for tensor: {tensor_name}. Already found datatype: {existing_dtype}, Newly specified datatype: {dtype}")

        if self.has_transform(src_tensors, dest_tensors, graph_id):
            raise ValueError(f"Transform already exists with src tensors {src_tensors} and dest_tensors {dest_tensors} in graph {graph_id}")

        transform_graph = self.get_transform_graph(graph_id)
        transform = Transform(src_tensors, dest_tensors, operator, tensor_dtypes)
        transform_graph.transforms.append(transform)

        if no_track:
            return transform

        if src_tensors == dest_tensors and operator.op_type == "Identity":
            transform_graph.input_tensors.update(src_tensors)
            transform_graph.output_tensors.update(dest_tensors)
            if tensor_dtypes:
                _add_tensor_dtypes()
            return transform

        # Add any new source tensors as inputs if they've never been produced
        for src in src_tensors:
            if src not in transform_graph.produced_tensors:
                transform_graph.input_tensors.add(src)

        # Track all tensors that are produced as destinations
        transform_graph.produced_tensors.update(dest_tensors)

        # If any destination tensor was previously an input, it's no longer an input
        transform_graph.input_tensors.difference_update(dest_tensors)

        # Track which tensors are consumed as inputs
        transform_graph.consumed_tensors.update(src_tensors)

        # For each destination tensor in this new transform
        for dest in dest_tensors:
            # Add it as an output initially
            transform_graph.output_tensors.add(dest)
            # But remove it if it's already been consumed by another transform
            if dest in transform_graph.consumed_tensors:
                transform_graph.output_tensors.remove(dest)

        # Remove this transform's source tensors from outputs since they're now consumed
        transform_graph.output_tensors.difference_update(src_tensors)

        if tensor_dtypes:
            _add_tensor_dtypes()

        return transform

    def _set_tensor_dtype(self, tensor_name, dtype, graph_id):
        if dtype not in TransformManager.SUPPORTED_DTYPES:
            raise ValueError(f"Invalid dtype '{dtype}' for tensor '{tensor_name}'. Supported dtypes: {list(TransformManager.SUPPORTED_DTYPES)}")

        self.get_transform_graph(graph_id).tensor_dtype_info[tensor_name] = dtype

    def _get_tensor_dtype(self, tensor_name, graph_id):
        return self.get_transform_graph(graph_id).tensor_dtype_info.get(tensor_name)

    def _remove_tensor_dtype(self, tensor_name, graph_id):
        return self.get_transform_graph(graph_id).tensor_dtype_info.pop(tensor_name, None)

    def get_metadata(self):
        """
        Get transformation metadata as a serializable dictionary for all transform graphs.

        Returns:
            dict: A dictionary with format:
                {
                    "version": "1.0.0",
                    "graphs": {
                        id1: metadata_dict1,
                        id2: metadata_dict2,
                        ...
                    }
                }
                Each metadata_dict has the same format as before.
        """
        def parse_transform_graph(transform_graph, graph_id):
            """Helper function to serialize a single transform set"""
            transforms_list = self._sort_transforms_topologically(graph_id)
            transforms_data = []
            for transform in transforms_list:
                transform_dict = {
                    "src_tensors": transform.src_tensors,
                    "dest_tensors": transform.dest_tensors
                }

                if transform.tensor_dtypes is not None:
                    transform_dict["tensor_dtypes"] = transform.tensor_dtypes

                transform_dict["operator"] = {
                                                "op_type": transform.operator.op_type,
                                                "attributes": transform.operator.attributes,
                                                "input_shapes": transform.operator.input_shapes,
                                                "output_shapes": transform.operator.output_shapes
                                            }
                transforms_data.append(transform_dict)

            result = dict()
            result["execution_order"] = transform_graph.execution_order

            # Add optional fields only if they're not None
            if transform_graph.concurrency_name is not None:
                result["concurrency_name"] = transform_graph.concurrency_name

            if transform_graph.split_num is not None:
                result["split_num"] = transform_graph.split_num

            if transform_graph.ar_n is not None:
                result["ar_n"] = transform_graph.ar_n

            result["input_tensors"] = list(transform_graph.input_tensors)
            result["output_tensors"] = list(transform_graph.output_tensors)
            result["transforms"] = transforms_data

            return result

        metadata_content = {}
        for graph_id, transform_graph in self.transform_graphs.items():
            metadata_content[graph_id] = parse_transform_graph(transform_graph, graph_id)

        # Create the final metadata structure
        metadata = {
            "version": "1.0.0",
            "graphs": metadata_content
        }

        return metadata


    def save_metadata(self, output_file_path):
        """
        Save transformation metadata to a json file

        Args:
            output_path (str): Path to save the metadata JSON
        """

        # Ensure directory exists
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        metadata = self.get_metadata()
        with open(output_file_path, 'w') as fp:
            json.dump(metadata, fp, indent=4)
        log_info(f"metadata saved at: {output_file_path}")


    def load_metadata(self, input_file_path):
        """
        Import transforms from a metadata dictionary.

        Args:
            input_file_path (str): Path to json containing transform metadata in the format:
                {
                    "graph": {
                        "transforms": [
                            {
                                "src_tensors": [...],
                                "dest_tensors": [...],
                                "operator": {
                                    "op_type": str,
                                    "attributes": {...},
                                    "input_shapes": [...],
                                    "output_shapes": [...]
                                }
                            },
                            ...
                        ],
                        "input_tensors": [...],
                        "output_tensors": [...]
                    },
                    ...
                }
        """
        def load_graph(metadata_dict, graph_id):
            """Helper function to load a single transform set"""
            # Throw error if graph already exists with the same name
            if graph_id in self.transform_graphs:
                raise ValueError(f"Unable to load graph \'{graph_id}\', Graph already exists with same name!")

            execution_order = metadata_dict.get('execution_order')
            if execution_order == None:
                raise ValueError(f"Metadata transform graph: {graph_id}, is missing execution order field")

            # Optional fields
            concurrency_name = metadata_dict.get('concurrency_name')
            split_num = metadata_dict.get('split_num')
            ar_n = metadata_dict.get('ar_n')

            # Create new graph
            self.create_transform_graph(graph_id,
                                        order_num=execution_order,
                                        concurrency_name=concurrency_name,
                                        split_num=split_num,
                                        ar_n=ar_n
                                        )

            # Process each transform in the metadata
            for operator_data in metadata_dict["transforms"]:
                # Create Operator object from the metadata
                operator = Operator(
                    op_type=operator_data["operator"]["op_type"],
                    attributes=operator_data["operator"]["attributes"],
                    input_shapes=operator_data["operator"]["input_shapes"],
                    output_shapes=operator_data["operator"]["output_shapes"]
                )

                # Add the transform using existing method to maintain dependency tracking
                self.add_transform(
                    src_tensors=operator_data["src_tensors"],
                    dest_tensors=operator_data["dest_tensors"],
                    operator=operator,
                    graph_id=graph_id,
                    tensor_dtypes=operator_data.get("tensor_dtypes")
                )

            # Set input and output tensors from metadata if they exist
            if "input_tensors" in metadata_dict and "output_tensors" in metadata_dict:
                transform_graph = self.get_transform_graph(graph_id)
                transform_graph.input_tensors = set(metadata_dict["input_tensors"])
                transform_graph.output_tensors = set(metadata_dict["output_tensors"])

        with open(input_file_path, 'r') as fp:
            metadata = json.load(fp)

        for graph_id, graph_metadata in metadata['graphs'].items():
            load_graph(graph_metadata, graph_id)


    def serialize_onnx(self, output_dir, opset=18):
        """Serialize each transform group into an ONNX model using the ONNX helper API."""

        import onnx
        from onnx import helper, TensorProto

        DTYPE_TO_TENSORPROTO = {
            'float32' : TensorProto.FLOAT,
            'float16' : TensorProto.FLOAT16,
            'int8'    : TensorProto.INT8,
            'int16'   : TensorProto.INT16,
            'int32'   : TensorProto.INT32,
            'int64'   : TensorProto.INT64,
            'uint8'   : TensorProto.UINT8,
            'uint16'  : TensorProto.UINT16,
            'uint32'  : TensorProto.UINT32,
            'uint64'  : TensorProto.UINT64
        }

        if set(DTYPE_TO_TENSORPROTO.keys()) != TransformManager.SUPPORTED_DTYPES:
            raise ValueError(f"Mismatch between SUPPORTED_DTYPES and DTYPE_TO_TENSORPROTO keys")

        def _get_tensorproto_dtype(dtype):
            tp_dtype = DTYPE_TO_TENSORPROTO.get(dtype)
            if not tp_dtype:
                raise ValueError(f"Invalid dtype specified. \'{dtype}\' not supported. Supported dtypes: {list(DTYPE_TO_TENSORPROTO.keys())}")

            return tp_dtype

        def create_node(transform):
            """Create an ONNX node from a Transform object"""
            node_name = transform.dest_tensors[0] + "/producer"
            origin_attributes = {k:v for k,v in transform.operator.attributes.items()}
            initializers = []
            origin_inputs = list(transform.src_tensors)
            inputs = []
            if onnx.defs.has(transform.operator.op_type, opset):
                op_schema = onnx.defs.get_schema(transform.operator.op_type, opset, "")

                for input in op_schema.inputs:
                    if input.name in origin_attributes:
                        # create an initializer
                        new_init = onnx.numpy_helper.from_array(
                                        np.array(origin_attributes[input.name]),
                                        name=node_name+"/cst_"+input.name
                                    )
                        initializers.append(new_init)
                        del origin_attributes[input.name]
                        inputs.append(new_init.name)
                    elif len(origin_inputs) > 0:
                        # Check if this is a variadic input (like Concat's inputs)
                        if input.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
                            # Add ALL remaining inputs for variadic parameters
                            inputs.extend(origin_inputs)
                            origin_inputs.clear()
                        else:
                            # For non-variadic, add just one input
                            inputs.append(origin_inputs.pop(0))
                    elif len(origin_attributes) > 0:
                        inputs.append("")
                    else:
                        break
            else:
                # This branch can be triggered when using a custom operator.
                # In such cases, we can't automatically configure attributes or inputs based on the operator definition.
                # Therefore, we retain the original attributes and inputs as-is.
                inputs = origin_inputs

            return helper.make_node(
                transform.operator.op_type,
                inputs=inputs,
                outputs=transform.dest_tensors,
                **origin_attributes
            ), initializers

        # Process each transform graph
        for graph_id, transform_graph in self.transform_graphs.items():
            try:
                # Get all transforms in topological order
                sorted_transforms = self._sort_transforms_topologically(graph_id)

                # Create map of tensor name to its shape for all tensors
                tensor_shapes = {}
                for transform in sorted_transforms:
                    # Map input tensor names to their shapes
                    for idx, tensor_name in enumerate(transform.src_tensors):
                        tensor_shapes[tensor_name] = transform.operator.input_shapes[idx]

                    # Map output tensor names to their shapes
                    for idx, tensor_name in enumerate(transform.dest_tensors):
                        tensor_shapes[tensor_name] = transform.operator.output_shapes[idx]

                # Create value infos for all tensors
                value_infos = []
                for name, shape in tensor_shapes.items():
                    tensor_dtype = self._get_tensor_dtype(name, graph_id)

                    # default to float if dtype not specified
                    if tensor_dtype:
                        tensor_tp_dtype = _get_tensorproto_dtype(tensor_dtype)
                    else:
                        tensor_tp_dtype = TensorProto.FLOAT

                    value_infos.append(helper.make_tensor_value_info(name, tensor_tp_dtype, shape))

                # Split value infos into inputs, outputs, and intermediates
                inputs = [vi for vi in value_infos if vi.name in transform_graph.input_tensors]
                outputs = [vi for vi in value_infos if vi.name in transform_graph.output_tensors]
                intermediates = [vi for vi in value_infos
                                if vi.name not in transform_graph.input_tensors
                                and vi.name not in transform_graph.output_tensors]

                # Create nodes from transforms in topological order
                initializers = []
                nodes = []
                for transform in sorted_transforms:
                    n, new_inits = create_node(transform)
                    nodes.append(n)
                    initializers += new_inits

                # Create graph with value info for all tensors
                graph = helper.make_graph(
                    nodes=nodes,
                    name=f"transform_graph_{graph_id}",
                    inputs=inputs,
                    outputs=outputs,
                    value_info=intermediates,
                    initializer=initializers
                )

                # Create model
                model = helper.make_model(graph, producer_name='transform_manager',
                                          opset_imports=[
                                              onnx.OperatorSetIdProto(
                                                    domain="",
                                                    version=opset)
                                          ]
                                        )

                # Save model
                output_path = os.path.join(output_dir, f"{graph_id}.onnx")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                onnx.save(model, output_path)
                log_info(f"model saved at: {output_path}")
            except Exception as e:
                log_info(f"Error serializing transform graph '{graph_id}': {str(e)}")


    def _sort_transforms_topologically(self, graph_id):
        """Sort transforms in topological order using breadth-first search."""
        transform_graph = self.get_transform_graph(graph_id)

        # Build a map of tensor names to the transforms that produce them
        tensor_producers = {}
        for transform in transform_graph.transforms:
            for dest in transform.dest_tensors:
                tensor_producers[dest] = transform

        # Start with transforms that only use input tensors
        sorted_transforms = []
        seen_transforms = set()
        queue = []

        # Find initial transforms (those that only use input tensors)
        for transform in transform_graph.transforms:
            if all(src in transform_graph.input_tensors or src not in transform_graph.produced_tensors
                  for src in transform.src_tensors):
                queue.append(transform)
                seen_transforms.add(transform)

        # BFS through the transform graph
        while queue:
            current = queue.pop(0)
            sorted_transforms.append(current)

            # Find transforms that consume the outputs of the current transform
            for dest in current.dest_tensors:
                for transform in transform_graph.transforms:
                    if (transform not in seen_transforms and dest in transform.src_tensors):
                        # Only add if all prerequisites are met
                        if all(src in transform_graph.input_tensors or
                              tensor_producers.get(src) in seen_transforms
                              for src in transform.src_tensors):
                            queue.append(transform)
                            seen_transforms.add(transform)

        return sorted_transforms


    def has_graph(self, graph_id):
        """
        Check if a transform graph ID exists in the manager.

        Args:
            graph_id (str): ID of the transform graph to check

        Returns:
            bool: True if the graph exists, False otherwise
        """
        return graph_id in self.transform_graphs


    def remove_graph(self, graph_id):
        """
        Remove a transform graph ID from the manager.

        Args:
            graph_id (str): ID of the transform graph to remove

        Raises:
            KeyError: If the graph_id doesn't exist
        """

        if graph_id not in self.transform_graphs:
            raise KeyError(f"Transform graph '{graph_id}' not found")
        del self.transform_graphs[graph_id]

    def rename_tensor(self, graph_id, old_name, new_name, skip_graph_input=False, skip_graph_output=False):
        transform_graph = self.get_transform_graph(graph_id)
        graph_sets = [
            transform_graph.produced_tensors,
            transform_graph.consumed_tensors
        ]
        if not skip_graph_input:
            graph_sets.append(transform_graph.input_tensors)

        if not skip_graph_output:
            graph_sets.append(transform_graph.output_tensors)

        for graph_set in graph_sets:
            if old_name in graph_set:
                graph_set.discard(old_name)
                graph_set.add(new_name)

        for transform in transform_graph.transforms:
            if old_name in transform.src_tensors:
                if not skip_graph_input or old_name not in transform_graph.input_tensors:
                    index = transform.src_tensors.index(old_name)
                    transform.src_tensors[index] = new_name
            if old_name in transform.dest_tensors:
                index = transform.dest_tensors.index(old_name)
                transform.dest_tensors[index] = new_name

        old_dtype = self._get_tensor_dtype(old_name, graph_id)
        if old_dtype:
            self._set_tensor_dtype(new_name, old_dtype, graph_id)
            self._remove_tensor_dtype(old_name, graph_id)

    def has_transform(self, src_tensors: List[str], dest_tensors: List[str], graph_id: str):
        transform_graph = self.get_transform_graph(graph_id)
        sorted_src = sorted(src_tensors)
        sorted_dest = sorted(dest_tensors)

        for transform in transform_graph.transforms:
            transform_src = sorted(transform.src_tensors)
            transform_dest = sorted(transform.dest_tensors)

            if transform_src == sorted_src and transform_dest == sorted_dest:
                return True

        return False

    def save_od_transforms_dlc(self, json_file_paths: List[str | os.PathLike], output_dir: str | os.PathLike, dlc_name: str, dump_onnx=False):
        #Added imports here to prevent circular import
        from qti.aisw.tools.core.modules.converter.converter_module import QAIRTConverter, ConverterInputConfig
        from qti.aisw.tools.core.modules.converter.optimizer_module import QAIRTOptimizer, OptimizerInputConfig
        from qti.aisw.tools.core.modules.converter.serializer_module import SerializerInputConfig, get_dlc_backend

        # Clear all existing graphs in TransformManager
        existing_graph_ids = self.get_graph_ids()
        for existing_id in existing_graph_ids:
            self.remove_graph(existing_id)

        if not dlc_name.endswith('.dlc'):
            # Remove any existing extension and add .dlc
            dlc_name = os.path.splitext(dlc_name)[0] + '.dlc'

        # Create a temp directory for ONNX files
        temp_onnx_dir = os.path.join(output_dir, "temp_onnx")
        os.makedirs(temp_onnx_dir, exist_ok=True)

        def get_cpp_graph(py_graph, backend):
            backend.prepare_py_graph(py_graph)

            cpp_graph = backend.get_ir_graph(py_graph)
            backend.prepare_cpp_graph(py_graph, cpp_graph)

            return cpp_graph


        for path in json_file_paths:
            self.load_metadata(path)

        graph_exec_order = dict()
        for graph_id in self.get_graph_ids():
            graph = self.get_transform_graph(graph_id)
            exec_order = graph_exec_order.get(graph.execution_order)
            if exec_order:
                graph_exec_order[graph.execution_order].append(graph_id)
            else:
                graph_exec_order[graph.execution_order] = [graph_id]


        num_stages = len(graph_exec_order.keys())

        # Renaming of input/outputs between stages, starting from the latest stage
        for order_num in range(num_stages - 1,  0, -1):
            # change all graph names for each order
            stage_graph_ids = graph_exec_order.get(order_num)
            if not stage_graph_ids:
                raise ValueError(f"Execution order: {order_num} not found!")

            rename_mapping = {}

            # Rename all input tensors of current stage graphs, with _inter_{order_num}
            for stage_graph_id in stage_graph_ids:
                stage_graph = self.get_transform_graph(stage_graph_id)
                stage_graph_old_inputs = set()
                for transform in stage_graph.transforms:
                    for idx, src_tensor in enumerate(transform.src_tensors):
                        if src_tensor in stage_graph.input_tensors:
                            # For capturing indices, we use the same src/dest tensors, in such case, we don't want to remove tensor dtype
                            indices_transform = False
                            if transform.src_tensors == transform.dest_tensors:
                                indices_transform = True
                            ori_name = src_tensor
                            new_name = ori_name + f"_inter_{order_num}"
                            transform.src_tensors[idx] = new_name
                            rename_mapping[ori_name] = new_name
                            stage_graph_old_inputs.add(ori_name)

                            old_dtype = self._get_tensor_dtype(ori_name, stage_graph_id)
                            if old_dtype:
                                self._set_tensor_dtype(new_name, old_dtype, stage_graph_id)
                                if not indices_transform:
                                    self._remove_tensor_dtype(ori_name, stage_graph_id)

                # Rename input_tensors field for each stage_graph, with updated name
                for old_input in stage_graph_old_inputs:
                    stage_graph.input_tensors.remove(old_input)
                    stage_graph.input_tensors.add(rename_mapping[old_input])

            # Rename all output tensors of previous stage
            prev_stage_graph_ids = graph_exec_order.get(order_num - 1)
            for prev_stage_graph_id in prev_stage_graph_ids:
                prev_stage_graph = self.get_transform_graph(prev_stage_graph_id)
                for transform in prev_stage_graph.transforms:
                    for idx, dest_tensor in enumerate(transform.dest_tensors):
                        if dest_tensor in prev_stage_graph.output_tensors:
                            # Rename transform dest_tensor and remove/add from output_tensors field
                            transform.dest_tensors[idx] = rename_mapping[dest_tensor]

                            old_dtype = self._get_tensor_dtype(dest_tensor, prev_stage_graph_id)
                            if old_dtype:
                                self._set_tensor_dtype(rename_mapping[dest_tensor], old_dtype, prev_stage_graph_id)
                                self._remove_tensor_dtype(dest_tensor, prev_stage_graph_id)

                            prev_stage_graph.output_tensors.remove(dest_tensor)
                            prev_stage_graph.output_tensors.add(rename_mapping[dest_tensor])

        graph_names = self.get_graph_ids()
        #rename all graphs using info from schema
        for graph_name in graph_names:
            transform_graph = self.get_transform_graph(graph_name)

            new_graph_name = ""
            if transform_graph.concurrency_name is not None:
                new_graph_name += transform_graph.concurrency_name + "_"

            if transform_graph.ar_n is not None:
                new_graph_name += f"ar{transform_graph.ar_n}_"

            if transform_graph.split_num is not None:
                new_graph_name += f"split{transform_graph.split_num}_"

            new_graph_name += f"stage{transform_graph.execution_order}"
            self.rename_graph(graph_name, new_graph_name)

        self.serialize_onnx(output_dir=temp_onnx_dir)
        dlc_path = os.path.join(output_dir, dlc_name)

        # get renamed graph names
        graph_names = self.get_graph_ids()

        # converter cmd needs to have " " otherwise it won't serialize
        dlc_writer = modeltools.IrDlcSerializer(dlc_path, "", "", " ", "")
        dlc_writer.initialize()

        qairt_converter = QAIRTConverter()
        qairt_optimizer = QAIRTOptimizer()

        for graph_name in graph_names:
            onnx_file_path = os.path.join(temp_onnx_dir, f"{graph_name}.onnx")

            # Step 1: Convert ONNX to Python IR graph
            converter_config = ConverterInputConfig(
                input_network=onnx_file_path,
                onnx_simplification=False,
            )
            converter_output = qairt_converter.convert(converter_config)

            # Step 2: Optimize the Python IR graph
            optimizer_config = OptimizerInputConfig(
                ir_graph=converter_output.ir_graph,
                framework=converter_output.framework,
            )
            optimizer_output = qairt_optimizer.optimize(optimizer_config)

            # Step 3: Convert Python IROpGraph to C++ IrGraph
            serializer_config = SerializerInputConfig(
                output_dlc=os.path.join(output_dir, graph_name + '.dlc'),
                framework=converter_output.framework,
                optimized_graph=optimizer_output.optimized_graph,
                optimizer_args=optimizer_output.optimizer_args,
            )
            backend = get_dlc_backend(serializer_config)
            cpp_graph = get_cpp_graph(optimizer_output.optimized_graph, backend)

            # Step 4: Serialize to multi-graph DLC
            dlc_writer.serialize(cpp_graph)
            backend.finish()

        dlc_writer.finish()
        log_info("On-device transform DLC saved at: " + dlc_path)

        # cleanup onnx files if dump_onnx is False
        if dump_onnx:
            # Move ONNX files to output_dir
            for filename in os.listdir(temp_onnx_dir):
                if filename.endswith('.onnx'):
                    src_path = os.path.join(temp_onnx_dir, filename)
                    dest_path = os.path.join(output_dir, filename)
                    shutil.move(src_path, dest_path)
                    log_info(f"ONNX file saved at: {dest_path}")


            # Remove temp dir
            shutil.rmtree(temp_onnx_dir)
        else:
            # Remove all onnx files
            shutil.rmtree(temp_onnx_dir)
            log_info("Temporary ONNX files removed")