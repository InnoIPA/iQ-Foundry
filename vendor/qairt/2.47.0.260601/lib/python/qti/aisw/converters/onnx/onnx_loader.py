# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import os
import sys
from typing import Any, Dict, List, Optional, Text, Union

import onnx
import qti.aisw.converters.onnx.util as Utils
from onnx import ModelProto, NodeProto
from qti.aisw.converters.common.custom_ops.op_factory import CustomOpFactory
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning
from qti.aisw.converters.common.utils.framework_utils import (
    FrameworkSummary,
    TensorInfo,
)
from qti.aisw.converters.onnx import model_evaluator
from qti.aisw.converters.onnx.onnx_model_api import ONNXModelUtils
from qti.aisw.converters.onnx.util import ModelWrapper


class ONNXLoader(FrameworkModelLoader):
    def __init__(
        self,
        args: Any,
        custom_op_factory: Optional[CustomOpFactory],
        **kwargs
    ) -> None:
        """
        Creates the onnx loader instance.
        :param custom_op_factory: CustomOpFactory Instance, defaults to None
        """
        self.defer_loading = kwargs.get("defer_loading", False)
        self.skip_simplification = kwargs.get("skip_simplification", False)
        self.init(args.input_network)
        super(ONNXLoader, self).__init__(
            converter_type="onnx", custom_op_factory=custom_op_factory, args=args
        )

    def init(
        self,
        path_or_proto: Union[Text, ModelProto]
    ):
        """
        Initialize the ONNX Loader class.

        :param Union[Text,ModelProto] path_or_proto: Path of onnx model or model proto.
        :param bool defer_loading: If False, the model will be loaded eagerly. If True, the model will be loaded lazily.
        """
        self.model_wrapper = ModelWrapper(model_or_path = path_or_proto, include_attributes = True, defer_loading = self.defer_loading, skip_simplification=self.skip_simplification)
        self.utils = ONNXModelUtils(self)

    def clone_loader(self):
        """
        Clone the onnx loader safely

        :return ONNXLoader: Returns the deep copied ONNXLoader instance.
        """
        temp_loader = copy.deepcopy(self)
        model_w_weights = self.model_wrapper.get_full_model()
        temp_loader.init(model_w_weights)
        return temp_loader

    def load_model(self,inplace=True):
        """
        Loads the model based on the defer_loading flag.

        If defer_loading is True, only updates the model's tensor proto paths,
        which is typically used for lazy or deferred loading scenarios where
        the full model is not immediately needed.

        If defer_loading is False, loads the full model into memory in-place.

        This method abstracts the conditional logic for model loading strategy.
        """
        if self.defer_loading:
            # Perform deferred loading by updating only the tensor proto paths
            self.model_wrapper.update_model_tensorproto_paths()
        else:
            # Load the complete model into memory
            self.model_wrapper.get_full_model(inplace=inplace)


    def update_model(self, model: ModelProto) -> None:
        """
        Update the current object's ModelProto with given ModelProto.

        :param ModelProto model: ModelProto instance to be copied.
        """
        self.model_wrapper.update_model(model)

    def update_model_wrapper(self, model_wrapper: ModelWrapper) -> None:
        """
        Update the current object's wrapper with given wrapper.

        :param ModelWrapper model: ModelWrapper instance to be copied.
        """
        self.model_wrapper.move_wrapper(model_wrapper)

    def get_model_with_no_weights(self) -> ModelProto:
        """
        Creates a copy of the ModelProto instance without the weights.

        :return ModelProto: Onnx ModelProto instance copy without weights
        """
        temp_model = copy.deepcopy(self.model)
        if temp_model.graph.initializer:
            temp_model.graph.ClearField("initializer")
        return temp_model

    @property
    def model(self) -> ModelProto:
        """
        Get the ModelProto instance from loader.

        :return ModelProto: Onnx ModelProto instance.
        """
        return self.model_wrapper.model

    @property
    def const_op_types(self) -> List[str]:
        """
        Get a list of constant operators types provided by onnx.

        :return List[str]: List of onnx operator types.
        """
        return ["Constant"]

    def get_op_type(self, node: NodeProto) -> str:
        """
        Get the op type of a given node.

        :param NodeProto node: Onnx node reference.
        :return str: Op type of a given node.
        """
        return node.op_type

    def get_inputs(self) -> Dict[str, onnx.ValueInfoProto]:
        """
        Get the ONNX Model input tensors dict.

        :return Dict[str, onnx.ValueInfoProto]: Dict with input tensor name as
            key and input tensor as value.
        """
        return {inp.name: inp for inp in Utils.get_inputs(self.model)}

    def get_outputs(self) -> Dict[str, onnx.ValueInfoProto]:
        """
        Get the ONNX Model output tensors dict.

        :return Dict[str, onnx.ValueInfoProto]: Dict with output tensor name as
            key and output tensor as value.
        """
        return {inp.name: inp for inp in Utils.get_outputs(self.model)}

    def get_input_names(self) -> List[str]:
        """
        Get the ONNX Model input names.

        :returns:List[str]: list of input names.
        """
        return [inp.name for inp in Utils.get_inputs(self.model)]

    def get_output_names(self) -> List[str]:
        """
        Get the Onnx Model output names.

        :returns:List[str]: list of output names.
        """
        return [out.name for out in Utils.get_outputs(self.model)]

    def get_nodes(self, include_subgraphs=True) -> List[NodeProto]:
        """
        Get all the nodes from underlying onnx model.

        :param bool include_subgraphs: If True, will include nodes from subgraphs also, default: True
        :returns:Underlying onnx nodes.
        """
        return Utils.get_nodes(self.model, include_subgraphs=include_subgraphs)

    def get_input_nodes_of_graph(self) -> List[NodeProto]:
        """
        Get the list of input nodes of onnx graph.

        :return List[NodeProto]: List of input nodes.
        """
        input_tensors = self.get_input_names()
        dangling_input_tensors = input_tensors.copy()

        input_nodes = []
        for node in self.get_nodes():
            for node_input in node.input:
                if node_input in input_tensors:
                    input_nodes.append(node)
                    if node_input in dangling_input_tensors:
                        dangling_input_tensors.remove(node_input)

        for input in dangling_input_tensors:
            log_warning(
                f"Input tensor {input} is not connected to any node in the graph."
            )
        return input_nodes

    def get_output_nodes_of_graph(self) -> List[NodeProto]:
        """
        Get the list of output nodes of onnx graph.

        :return List[NodeProto]: List of output nodes.
        """
        output_tensors = self.get_output_names()
        dangling_output_tensors = self.get_output_names()

        output_nodes = []
        for node in self.get_nodes():
            for node_output in node.output:
                if node_output in output_tensors:
                    output_nodes.append(node)
                    dangling_output_tensors.remove(node_output)

        for output in dangling_output_tensors:
            log_warning(
                f"Output tensor {output} is not connected to any node in the graph."
            )
        return output_nodes

    def is_input_node(self, node: NodeProto) -> bool:
        """
        Check whether the given node is input node of the graph or not.

        :param NodeProto node: Reference onnx node.
        :return bool: Boolean value indicating whether the node is input node or not.
        """
        return node in self.get_input_nodes_of_graph()

    def is_output_node(self, node: NodeProto) -> bool:
        """
        Check whether the given node is output node of the graph or not.

        :param NodeProto node: Reference onnx node.
        :return bool: Boolean value indicating whether the node is output node or not.
        """
        return node in self.get_output_nodes_of_graph()

    def get_input_op_types(self, node: NodeProto) -> List[str]:
        """
        Get the op types of the parent nodes of a given node.

        :param NodeProto node: Reference onnx node.
        :return List[str]: List of op types of parent nodes.
        """
        if self.is_input_node(node):
            model_inputs = self.get_input_names()
            get_node_by_output = Utils.get_node_by_output_name(self.model)

            op_types = []
            for tensor_name in node.input:
                if tensor_name in model_inputs:
                    op_types.append("Input")
                elif tensor_name in get_node_by_output:
                    par_node = get_node_by_output[tensor_name]
                    op_types.append(self.get_op_type(par_node))
            return op_types

        parent_nodes = self.get_parent_nodes(node)
        return [self.get_op_type(node) for node in parent_nodes]

    def get_output_op_types(self, node: NodeProto) -> List[str]:
        """
        Get the op types of the children nodes of a given node.

        :param NodeProto node: Reference onnx node.
        :return List[str]: List of op types of children nodes.
        """
        if self.is_output_node(node):
            model_outputs = self.get_output_names()
            get_node_by_input = Utils.get_node_by_input_name(self.model)

            op_types = []
            for tensor_name in node.output:
                if tensor_name in model_outputs:
                    op_types.append("Output")
                elif tensor_name in get_node_by_input:
                    children_nodes = get_node_by_input[tensor_name]
                    op_types.extend([self.get_op_type(n) for n in children_nodes])
            return op_types
        children_nodes = self.get_children_nodes(node)
        return [self.get_op_type(node) for node in children_nodes]

    def get_parent_nodes(self, node: NodeProto) -> List[NodeProto]:
        """
        Get list of parent node on which the given node depends.

        :param NodeProto node: Reference onnx node.
        :return List[NodeProto]: List of parent nodes.
        """
        get_node_by_output = Utils.get_node_by_output_name(self.model)

        parent_nodes = []
        for tensor_name in node.input:
            if tensor_name in get_node_by_output:
                _node = get_node_by_output[tensor_name]
                parent_nodes.append(_node)
        return parent_nodes

    def get_children_nodes(self, node: NodeProto) -> List[NodeProto]:
        """
        Get list of children node which depend on given node.

        :param NodeProto node: Reference onnx node.
        :return List[NodeProto]: List of children nodes.
        """
        get_node_by_input = Utils.get_node_by_input_name(self.model)

        children_nodes = []
        for tensor_name in node.output:
            if tensor_name in get_node_by_input:
                _nodes = get_node_by_input[tensor_name]
                children_nodes.extend(_nodes)
        return children_nodes

    def get_input_info(self) -> Dict[Text, TensorInfo]:
        """
        Get input name to TensorInfo Mappings. e.g. shape, dtype, layout etc.

        :return Dict[Text, TensorInfo]: TensorInfo mapping for inputs.
        """
        return Utils.get_input_info(self.model)

    def get_output_info(self) -> Dict[Text, TensorInfo]:
        """
        Get output name to TensorInfo Mappings. e.g. shape, dtype, layout etc.

        :return Dict[Text, TensorInfo]: TensorInfo mapping for outputs.
        """
        return Utils.get_output_info(self.model)

    def get_node_inputs(self, node: NodeProto) -> List[str]:
        """
        Get the list of input tensors of a given node.

        :param NodeProto node: Reference onnx node.
        :return List[str]: List of input tensors of given node.
        """
        return node.input

    def get_node_outputs(self, node: NodeProto) -> List[str]:
        """
        Get the list of output tensors of a given node.

        :param NodeProto node: Reference onnx node.
        :return List[str]: List of output tensors of given node.
        """
        return node.output

    def get_supported_operators(self) -> List[str]:
        """
        Get the list of supported operators by onnx.

        :return List[str]: List of supported operators.
        """
        all_ops = [n.name for n in onnx.defs.get_all_schemas()]
        return all_ops

    def native_checker(self, dry_run=None) -> bool:
        """
        This method will return the result of onnx model checker as well as evaluate the model.
        :return: Boolean indicating the success/failure of the Native Onnx checker
        """
        if self.defer_loading and self.skip_simplification:
            log_warning("Skipping Native checker in Defer Loading and ONNX Simplification not invoked")
            return False

        success = True
        # Calling graph checker for sanity checking about the graph's node names,
        # initializer names etc.
        graph_check_status = Utils.graph_checker(self.model)
        if not graph_check_status:
            log_warning("Duplicate naming found in the graph.")

        try:
            # Checker needs weights to be present in the model.
            if self.model_wrapper.has_external_data:
                model_path = self.model_wrapper.get_model_path()
                onnx.checker.check_model(model_path)
            else:
                onnx.checker.check_model(self.model)
        except RuntimeError as e:
            # If get_model_path is not able to save the model on disk.
            log_warning(f"Onnx checker failed due to exception: {e}")
            return False
        except Exception as e:
            log_warning("The model is invalid: %s" % e)
            return False

        if dry_run:
            log_info(
                "Proceeding with model evaluation in dry run mode...................................\n"
            )
            model_evaluator.setup_dry_run(self.model, dry_run)
            log_info(
                "Exiting conversion process in dry run mode...................................\n"
            )
            sys.exit(0)

        return success

    def save_model(self, path: Text) -> None:
        """
        Save The ONNX model on to the disc.

        :param Text path: Path where the model is to be saved.
        """
        if not path.endswith(".onnx"):
            prepared_name = self.model_wrapper.model_name
            path = os.path.join(path, prepared_name)
        model_w_weights = self.model_wrapper.get_full_model()
        Utils.save_model(model_w_weights, path, restore_back=False)

    def summarize_model(self) -> FrameworkSummary:
        """
        Populates summary of the onnx model.

        :return FrameworkSummary: Returns the framework summary object.
        """
        summary = FrameworkSummary()
        summary.total_parameters = Utils.get_model_params(self.model)
        summary.ops_counter = Utils.get_unique_ops(self.model)
        summary.ir_version = self.model.ir_version
        summary.model_name = self.model_wrapper.model_name

        inp_specs = self.get_input_info()
        out_specs = self.get_output_info()

        summary.inp_specs = {
            k: (v.shape, "input", v.dtype) for k, v in inp_specs.items()
        }
        summary.out_specs = {
            k: (v.shape, "output", v.dtype) for k, v in out_specs.items()
        }
        return summary
