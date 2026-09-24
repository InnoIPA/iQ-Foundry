# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, ValueInfoProto
from qti.aisw.converters.common.framework_optimizer import FrameworkOptimizer
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils.converter_utils import (
    log_debug1,
    log_error,
)
from qti.aisw.converters.common.utils.framework_utils import generate_test_data
from qti.aisw.converters.common.utils.translation_utils import compare_values
from qti.aisw.converters.onnx.util import (
    get_initializer_mappings,
    get_initializer_value,
    get_node_by_input_name,
    get_node_by_output_name,
    get_nodes_by_op_type,
    get_shape_from_value_info_proto,
    get_unique_ops,
    get_value_info_proto_mappings,
    run_ort_inference,
)


class ONNXConstantFoldingOptimization(FrameworkOptimizer):
    """ONNXConstantFoldingOptimization class is responsible for:
    - Identifying conversion time constant nodes
    - Run an inference to get the actual values of those nodes
    - Replacing the nodes' outputs with new initializer values in graph
    - Cleaning up redandunt nodes and initializers from the graph.
    """

    def __init__(self, loader: FrameworkModelLoader):
        """Constructor of ONNXConstantFoldingOptimization"""
        self.loader = loader

    def is_const_shape_node(
        self, node: NodeProto, value_info_dict: Dict[str, ValueInfoProto]
    ) -> Tuple[bool, List[int]]:
        """Check whether a given shape node is conversion time constant node or not.
        This can be checked by checking the shape of its parent node. If the parent
        node's shape is known then the given shape node can be treated as const node.

        :param NodeProto node: Shape node reference.
        :param Dict[str, ValueInfoProto] value_info_dict: Dict mapping of tensor
            name to its value info proto reference.
        :return bool: Boolean value representing whether the given node is
            const node or not.
        """
        if node.op_type != "Shape":
            return False

        shape_input = node.input
        assert len(shape_input) == 1, "Shape node shall have only one input."
        shape_input = shape_input[0]
        if shape_input not in value_info_dict:
            return False
        value_info = value_info_dict[shape_input]
        input_shape = get_shape_from_value_info_proto(
            value_info, use_onnx_def_symbols=True
        )
        is_static_shape = all(isinstance(s, int) for s in input_shape)

        return is_static_shape

    def is_node_const(
        self,
        node: NodeProto,
        const_nodes: List[NodeProto],
        node_by_output_name: Dict[str, NodeProto],
        init_dict: Dict[str, TensorProto],
        val_info_dict: Dict[str, ValueInfoProto],
    ) -> bool:
        """Check whether the given node is a conversion time constant node or not.

        :param NodeProto node: Node reference.
        :param List[NodeProto] const_nodes: List of const nodes.
        :param Dict[str, NodeProto] node_by_output_name: Mapping of node's
            output tensor name to the node.
        :param Dict[str, TensorProto] init_dict: Mapping of each initializer's
            name to initializer reference.
        :param Dict[str, ValueInfoProto] val_info_dict: Mapping of each tensor's
            name to its value info proto reference.
        :return bool: Boolean value indicating whether the given node is a const
            node or not.
        """
        node_inputs = node.input
        is_const_node = self.is_const_shape_node(node, val_info_dict)
        if is_const_node:
            return True

        # Dont include if node as const node, as we can't replace the output of if
        # node in the original graph using onnxruntime. Instead we should be checking
        # till if node and not its children nodes. Once the conditional input is
        # identified as constant then we can replace if nodes in update_graph function.
        if node.op_type == "If":
            return False

        const_input_ctr = 0
        for node_input in node_inputs:
            if node_input in init_dict:
                const_input_ctr += 1
            if node_input in node_by_output_name:
                _node = node_by_output_name[node_input]
                if _node in const_nodes:
                    const_input_ctr += 1

        if len(node_inputs) == const_input_ctr:
            return True
        return False

    def identify_const_nodes(self) -> List[NodeProto]:
        """Function to identify all the conversion time constant nodes in the graph.

        :return List[NodeProto]: List of all the const nodes.
        """
        value_info_dict = get_value_info_proto_mappings(self.loader.model)
        input_val_info_dict = self.loader.get_inputs()
        output_val_info_dict = self.loader.get_outputs()
        comb_val_info_dict = {
            **value_info_dict,
            **input_val_info_dict,
            **output_val_info_dict,
        }

        init_dict = get_initializer_mappings(self.loader.model)
        node_by_input_name = get_node_by_input_name(self.loader.model)
        node_by_output_name = get_node_by_output_name(self.loader.model)

        input_nodes = []
        for tensor_name in input_val_info_dict:
            nodes = node_by_input_name.get(tensor_name, [])
            input_nodes.extend(nodes)

        # Sort the nodes before traversing so that the constant input
        # dependencies can be identified correctly.
        self.loader.utils.topological_sort()
        const_nodes = []
        for _node in self.loader.model.graph.node:
            if self.is_node_const(
                _node, const_nodes, node_by_output_name, init_dict, comb_val_info_dict
            ):
                const_nodes.append(_node)

        return const_nodes

    def filter_const_nodes(self, const_nodes: List[NodeProto]) -> List[NodeProto]:
        """
        Filter const nodes whose output has a shape defined so that those can be
        made graph outputs. This is to make the graph a valid graph. Onnxruntime
        only requires the output to be of correct dtype.

        :param List[NodeProto] const_nodes: List of conversion time constant nodes.
        :return List[NodeProto]: Updated list of conversion time constant nodes.
        """
        tensor_val_info_dict = get_value_info_proto_mappings(self.loader.model)
        input_val_info_dict = self.loader.get_inputs()
        output_val_info_dict = self.loader.get_outputs()
        all_val_info_dict = {
            **tensor_val_info_dict,
            **input_val_info_dict,
            **output_val_info_dict,
        }

        # node_by_input_name = get_node_by_input_name(self.loader.model)
        model_outputs = self.loader.get_output_names()

        exclude_nodes = []
        for node in const_nodes:
            # FIXME: Add below lines after testing on a real model and after
            #        adding some test cases.
            # all_childrens = True
            # for node_output in node.output:
            #     children_nodes = node_by_input_name[node_output]
            #     all_childrens.extend(children_nodes)

            # all_childrens_are_const = all([True if node in const_nodes else False for node in all_childrens])
            # if all_childrens_are_const:
            #     exclude_nodes.append(node)

            for node_output in node.output:
                if node_output in model_outputs:
                    continue

                if node_output not in all_val_info_dict:
                    exclude_nodes.append(node)

        return [node for node in const_nodes if node not in exclude_nodes]

    def create_const_outputs(self, const_nodes: List[NodeProto]):
        """Function to convert the output of all the constant nodes into graph
        outputs.

        :param List[NodeProto] const_nodes: List of const nodes.
        :raises RuntimeError: If the node's output shape is not found then
            RuntimeError will be thrown.
        """
        new_outputs = [
            node_output for node in const_nodes for node_output in node.output
        ]
        self.loader.utils.add_outputs(new_outputs, infer_shape=False)

    def update_if_nodes(self):
        """Function to update the if nodes in the graph if their condition
        input is known.
        """
        init_dict = get_initializer_mappings(self.loader.model)

        if_nodes = get_nodes_by_op_type(self.loader.model, "If")
        for node in if_nodes:
            if node.input[0] in init_dict:
                condition_value = get_initializer_value(init_dict[node.input[0]])
                branch_check = "then_branch" if condition_value else "else_branch"
                for attr in node.attribute:
                    if attr.name == branch_check:
                        # Replace the subgraph's output nodes' output tensor names
                        # with if node's output tensor names.
                        if_node_outputs = node.output
                        subgraph_outputs = [output.name for output in attr.g.output]
                        subgraph_nodes = attr.g.node
                        subgraph_val_info_dict = {
                            val_info.name: val_info for val_info in attr.g.value_info
                        }
                        for subgraph_node in subgraph_nodes:
                            for i, node_output in enumerate(subgraph_node.output):
                                if node_output in subgraph_outputs:
                                    subgraph_outputs_idx = subgraph_outputs.index(
                                        node_output
                                    )
                                    subgraph_node.output[i] = if_node_outputs[
                                        subgraph_outputs_idx
                                    ]
                                    if node_output in subgraph_val_info_dict:
                                        subgraph_val_info = subgraph_val_info_dict[
                                            node_output
                                        ]
                                        subgraph_val_info.name = if_node_outputs[
                                            subgraph_outputs_idx
                                        ]

                        # Add the subgraph nodes, initializers and value_infos
                        # into the main graph.
                        self.loader.model.graph.node.extend(attr.g.node)
                        self.loader.model.graph.value_info.extend(attr.g.value_info)
                        self.loader.model.graph.initializer.extend(attr.g.initializer)

                # Remove the if node from the graph.
                self.loader.model.graph.node.remove(node)

    def update_graph(self, const_nodes: List[NodeProto]) -> bool:
        """Function to add the computed outputs of all the conversion time constant
        nodes into the graph as initializers.

        :param List[NodeProto] const_nodes: List of const nodes.
        :return bool: Boolean value indicating whether the graph is updated or not.
        """
        filtered_output_names = []
        for node in const_nodes:
            filtered_output_names.extend(node.output)

        status, ort_outputs = self.get_outputs(filtered_output_names)
        if not status:
            log_error("Failed to get outputs using model inference.")
            return status

        for tensor_name, tensor_value in ort_outputs.items():
            new_tensor = onnx.numpy_helper.from_array(tensor_value, name=tensor_name)
            self.loader.model.graph.initializer.append(new_tensor)

        # Remove the node connections here. Later on cleanup of the nodes will
        # take care of dangling nodes.
        for node in const_nodes:
            del node.output[:]

        # Update if nodes whose condition input is known.
        self.update_if_nodes()

        return True

    def get_outputs(self, output_names: List = []) -> Dict[str, np.ndarray]:
        """Function to get the outputs of output_names graph outputs using
        onnxruntime session by generating random input data.

        :param List output_names: List of output tensors to compute, defaults to []
        :return Dict[str, np.ndarray]: Mapping of output tensor name to its
            computed value.
        """
        input_info_dict = self.loader.get_input_info()
        ort_inputs = generate_test_data(input_info_dict)
        model_path = self.loader.model_wrapper.get_model_path()
        status, ort_outputs = run_ort_inference(
            model_path, ort_inputs, output_names
        )
        return status, ort_outputs

    def compare_outputs(
        self,
        before_outputs: Dict[str, np.ndarray],
        after_outputs: Dict[str, np.ndarray],
    ):
        """Compare the onnxruntime results for original graph outputs.

        :param Dict[str, np.ndarray] before_outputs: Mapping of graph outputs
        :param Dict[str, np.ndarray] after_outputs: _description_
        :raises RuntimeError: _description_
        :raises RuntimeError: _description_
        """
        if before_outputs.keys() != after_outputs.keys():
            raise RuntimeError(
                "Number of outputs in the graph has changed "
                "due to constant folding. Outputs before: "
                f"{len(before_outputs.keys())}, outputs after:{len(after_outputs.keys())}"
            )

        for tensor_name in before_outputs:
            status = compare_values(
                before_outputs[tensor_name], after_outputs[tensor_name]
            )
            if not status:
                raise RuntimeError(
                    "Due to constant folding optimization the result of model execution has changed."
                )

    def __str__(self) -> str:
        """String representation of ONNXEinsumPatternOptimizer class.
        :returns str: String representation of class.
        """
        return "ONNX - Constant Folding Optimizer"

    def optimize(self, max_iter: int = 50) -> FrameworkModelLoader:
        """Apply Constant folding optimization to identify and remove conversion
        time constant nodes from the onnx graph.

        :param int max_iter: Maximum number of iterations to be performed.
            If graph converges before that then further iterations will be skipped.
        :return FrameworkModelLoader: Updated loader instance.
        """
        if self.loader.has_custom_op:
            log_debug1(
                "The model has custom ops due to which constant "
                "folding optimization is skipped."
            )
            return self.loader

        # FIXME: Currently enabling this for If opeartor only.
        if_nodes = get_nodes_by_op_type(self.loader.model, "If")
        if len(if_nodes) == 0:
            log_debug1(
                "The model doesn't contain any If operators. Skipping "
                "constant folding."
            )
            return self.loader

        # 1. Perform shape inference so that shape node's inputs have proper shapes.
        self.loader.utils.native_shape_inference()
        before_loader = self.loader.clone_loader()
        status, before_outputs = self.get_outputs()
        if not status:
            log_error("Failed to get outputs using model inference.")
            return before_loader

        graph_changed = True
        iter_count = 0
        while (graph_changed) and (iter_count < max_iter):
            # 2. Identify all the conversion time constant nodes.
            const_nodes = self.identify_const_nodes()

            if len(const_nodes) == 0:
                break

            nodes_before = len(self.loader.get_nodes())
            nodes_dict_before = get_unique_ops(self.loader.model)

            # 2.a. Filter const node based on whether the node's output
            # can be made graph output or not.
            const_nodes = self.filter_const_nodes(const_nodes)

            # 3. Make the output of const nodes as graph's outputs.
            self.create_const_outputs(const_nodes)

            # 4. Run the ort session to get the values of const nodes' outputs.
            #    and Replace the values in the original graph with computed outputs.
            status = self.update_graph(const_nodes)
            if not status:
                log_error(
                    "Failed to update the graph based on model inference. "
                    "Returning original model reference."
                )
                return before_loader

            # 6. Get rid of dangling nodes and unused initializers by calling clean_model.
            self.loader.utils.clean_model().topological_sort()

            # 7. Check the validity of the model after the optimization.
            try:
                self.loader.native_checker()
            except Exception as e:
                log_error(
                    "During constant folding optimization, the graph "
                    "has become invalid. Returning original graph."
                )
                return before_loader

            # 8. Perform shape inference after newly added nodes or initializer.
            # This will enable scope for further optimizations.
            self.loader.utils.native_shape_inference()

            nodes_after = len(self.loader.get_nodes())
            nodes_dict_after = get_unique_ops(self.loader.model)

            log_debug1(
                "Change in nodes after constant folding onnx optimization: "
                f"Before: {nodes_before}, After: {nodes_after}"
            )

            # 9. Perform the optimization again if there is change in number of
            # nodes after optimization.
            graph_changed = nodes_dict_before != nodes_dict_after
            iter_count += 1

        status, after_outputs = self.get_outputs()
        if not status:
            log_error("Failed to get outputs using model inference.")
            return before_loader
        try:
            self.compare_outputs(before_outputs, after_outputs)
            log_debug1("Constant folding successful and outputs are matching.")
        except RuntimeError as e:
            log_error(f"Constant folding optimization failed due to: {e}")
            return before_loader

        return self.loader
