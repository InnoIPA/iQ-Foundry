# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import shutil
import tempfile
from typing import Dict, List, Optional, Text, Tuple, Union

import numpy as np
import onnx
import qti.aisw.converters.onnx.util as Utils
from onnx import GraphProto, NodeProto, TensorProto, ValueInfoProto, helper
from packaging import version
from qti.aisw.converters.common.framework_model_api import FrameworkModelAPI
from qti.aisw.converters.common.utils.converter_utils import (
    log_debug1,
    log_error,
    log_warning,
    log_info,
)
from qti.aisw.converters.common.utils.framework_utils import spawn_process_and_exec
from qti.aisw.converters.onnx.optimizations.const_fold import (
    ONNXConstantFoldingOptimization,
)
from qti.aisw.converters.onnx.optimizations.einsum import ONNXEinsumPatternOptimizer
from qti.aisw.converters.onnx.optimizations.nms_utils import (
    get_supported_ssd_models,
    get_supported_yolo_models,
)
from qti.aisw.converters.onnx.optimizations.sequence_construct import (
    ONNXSequenceConstructPatternOptimizer,
)
from qti.aisw.converters.onnx.optimizations.ssd_nms_optimizer import OnnxSsdNmsOptimizer
from qti.aisw.converters.onnx.optimizations.yolo_nms_optimizer import (
    OnnxYoloNmsOptimizer,
)
from qti.aisw.converters.onnx.util import (
    MAX_INT_32,
    SYMBOLIC_SHAPE_INFER_MIN_OPSET_VER,
    ModelWrapper,
    cleanup,
    create_node_name,
    create_tensor_name,
    get_graph_by_name,
    get_graphs,
    get_initializer_value,
    get_type_dims_info,
    make_node,
    np_dtype_to_onnx_tensor_dtype,
)


class ONNXModelUtils(FrameworkModelAPI):
    def __init__(self, loader):
        self.loader = loader
        self.assign_names_to_empty_nodes()
        self.cache_op_tensor_names()

    def cache_op_tensor_names(self) -> None:
        """
        Setup the node naming dict and tensor name set for new name creation.
        It is to be used in create_node function.
        """
        self.__node_name_suffix = {}
        self.__tensor_name_set = set()

        for node in self.loader.get_nodes():
            for act_name in node.output:
                self.__tensor_name_set.add(act_name)
            for in_act in node.input:
                self.__tensor_name_set.add(in_act)
            self.__tensor_name_set.add(node.name)

    def get_new_node_name(self, op_type: str) -> str:
        """
        Get new unique node name based on op_type of the node.

        :param str op_type: Op type of the node.
        :return str: New node name.
        """
        node_name, self.__node_name_suffix = create_node_name(
            self.loader.model.graph, op_type, self.__node_name_suffix
        )
        return node_name

    def get_new_tensor_name(self, node_name: str) -> str:
        """
        Get new unique tensor name based on node name of the node.

        :param str node_name: Name of the node.
        :return str: New tensor name.
        """
        output_name, self.__tensor_name_set = create_tensor_name(
            node_name, self.__tensor_name_set
        )
        return output_name

    def remove_initializer_from_input(self):
        """
        Function to remove initializers from graph inputs

        :return ONNXModelUtils: Self instance of the class.
        """
        if self.loader.model.ir_version < 4:
            log_warning(
                "Can't remove initializer from inputs for models having ir version < 4"
            )
            return self

        init_dict = Utils.get_initializer_mappings(self.loader.model)
        input_dict = {i.name: i for i in self.loader.model.graph.input}
        for ip_name in input_dict:
            if ip_name in init_dict:
                input_ = input_dict[ip_name]
                self.loader.model.graph.input.remove(input_)
        return self

    def _optimize_by_ort_helper(
        self, opt_level, disabled_optimizers
    ) -> Union[None, ModelWrapper]:
        """
        Function to optimize the model using the Onnx-Runtime Optimizers

        :param Optional[Text] opt_level: Graph optimization level
            (Default is BASIC), available options ["BASIC", "EXTENDED", "ALL"]
        :param Optional[List] disabled_optimizers: A list of names of
            disabled optimizers., defaults to []
        :return Union[None, ModelWrapper]: ModelWrapper object if operation is
            successful else None.
        """
        SUPPORTED_OPT = ["BASIC", "EXTENDED", "ALL"]
        try:
            import onnxruntime
            from onnxruntime.GraphOptimizationLevel import (
                ORT_ENABLE_ALL,
                ORT_ENABLE_BASIC,
                ORT_ENABLE_EXTENDED,
            )

        except ImportError:
            log_warning(
                "Onnxruntime package not found in current "
                "environment. Optimization using onnxruntime will be skipped."
            )
            return None

        if opt_level not in SUPPORTED_OPT:
            log_warning(
                f"Provided optimization level '{opt_level}' for "
                + "onnxruntime optimizations is not supported. "
                + "Optimizing model using BASIC optimizations."
            )
            opt_level = "BASIC"
        if opt_level == "BASIC":
            graph_optimization_level = ORT_ENABLE_BASIC
        elif opt_level == "EXTENDED":
            graph_optimization_level = ORT_ENABLE_EXTENDED
        else:
            graph_optimization_level = ORT_ENABLE_ALL
        kwargs = {}

        if disabled_optimizers:
            kwargs["disabled_optimizers"] = disabled_optimizers

        optimized_model = None
        model_path = self.loader.model_wrapper.get_model_path()
        status, _, optimized_model = Utils.create_ort_session(
            model_path,
            optimize=True,
            graph_optimization_level=graph_optimization_level,
            **kwargs,
        )
        if not status:
            return None

        return ModelWrapper(optimized_model)

    def optimize_by_ort(
        self,
        opt_level: Optional[Text] = "BASIC",
        disabled_optimizers: Optional[List] = [],
    ):
        """
        Function to optimize the model using the Onnx-Runtime Optimizers

        :param Optional[Text] opt_level: Graph optimization level (Default is BASIC), available options ["BASIC", "EXTENDED", "ALL"]
        :param Optional[List] disabled_optimizers: A list of names of disabled optimizers., defaults to []
        :return ONNXModelUtils: Self instance of the class.
        """

        log_debug1("Performing Optimization using ORT")
        status, model_wrapper = spawn_process_and_exec(
            self._optimize_by_ort_helper,
            opt_level,
            disabled_optimizers,
            process_name="ORT Optimizer",
        )
        if not status:
            log_warning("Optimization with ORT failed.")
            return self
        self.loader.update_model_wrapper(model_wrapper)
        return self

    def _optimize_by_simplifier_helper(
        self, static_input_shapes, skip_optimization=False
    ) -> ModelWrapper:
        """
        Function to optimize the model inplace using the Onnx Simplifier Tools

        :param Optional[Dict] static_input_shapes: Dict with mapping of
            input names to their corresponding static shapes., defaults to None
        :return ModelWrapper: Instance of onnx model in the form of ModelWrapper.
        """
        simplified_model = None
        try:
            from onnxsim import simplify

            # FIXME: Add a fix in onnx-simplifier repo where we can either pass
            # full model (<2gb) or model path (>2gb). For model with >2gb if we
            # pass path then it should not read the model and again dump to the
            # disk.
            model_w_weights = self.loader.model_wrapper.get_full_model()
            if static_input_shapes:
                # Covered the simplifier call inside contextlib to suppress the logs incase
                # the outputs are not matched.
                simplified_model, check = simplify(
                    model_w_weights,
                    input_shapes=static_input_shapes,
                    perform_optimization=not skip_optimization,
                )
            else:
                simplified_model, check = simplify(
                    model_w_weights, perform_optimization=not skip_optimization
                )

        except Exception as e:
            log_warning(f"Onnx model simplification failed due to: {e}")
            check = False

        if check == False:
            log_warning("Simplified model validation failed")
            return None
        else:
            log_debug1("Simplified model validation is successful")
            return ModelWrapper(simplified_model)

    def optimize_by_simplifier(self, static_input_shapes: Optional[Dict] = None):
        """
        Function to optimize the model inplace using the Onnx Simplifier Tools

        :param Optional[Dict] static_input_shapes: Dict with mapping of
            input names to their corresponding static shapes., defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        status, model_wrapper = spawn_process_and_exec(
            self._optimize_by_simplifier_helper,
            static_input_shapes,
            process_name="ONNX Simplifier Optimizer",
        )
        clean_model_req = False
        if not status:
            clean_model_req = True
            log_debug1("Running Onnx Simplifier Optimizer without any optimizations.")
            status, model_wrapper = spawn_process_and_exec(
                self._optimize_by_simplifier_helper,
                static_input_shapes,
                skip_optimization=True,
                process_name="ONNX Simplifier Optimizer",
            )
            if not status:
                return self

        log_debug1("Optimization using Onnx Simplifier applied successfully.")
        self.loader.update_model_wrapper(model_wrapper)

        # Since, onnx-simplifier fuses/replaces some nodes, the new nodes are
        # not having any names. Hence calling below line.
        self.assign_names_to_empty_nodes()

        if clean_model_req:
            # Need to call clean_model API as for few models it is observed that
            # when onnxsim is executed without any optimization, at that time
            # only constant folding and shape inference is executed. In that
            # there are some redundant initializers still present in the model
            # because these were expected to be removed due to
            # eliminate_unused_initializer pass in onnx-optimizer. As these are
            # not removed by simplifier these actually gets translated to IR.
            # This we need to prevent by clean_model() call.
            self.loader.utils.clean_model()
        return self

    def optimize_einsum(self, einsum_nodes, skip_shape_inference: bool):
        """
        Apply Einsum optimization to the loader.

        :param bool skip_shape_inference: Skip shape inference at the end of
            Einsum optimization.
        :return ONNXModelUtils: Self instance of the class.
        """
        einsum_opt = ONNXEinsumPatternOptimizer(self.loader)
        einsum_opt.optimize(einsum_nodes=einsum_nodes, skip_shape_inference=skip_shape_inference)

    def optimize_sequence_construct_ops(self, sequence_construct_nodes, skip_shape_inference: bool):
        """
        Apply Sequence Construct optimization to the loader.

        :param bool skip_shape_inference: Skip shape inference at the end of
            Sequence Construct optimization.
        :return ONNXModelUtils: Self instance of the class.
        """
        sequence_construct_opt = ONNXSequenceConstructPatternOptimizer(self.loader)
        sequence_construct_opt.optimize(sequence_construct_nodes=sequence_construct_nodes, skip_shape_inference=skip_shape_inference)

    def optimize_const_fold(self):
        """
        Apply Constant folding optimization to the loader.

        :return ONNXModelUtils: Self instance of the class.
        """
        const_fold_opt = ONNXConstantFoldingOptimization(self.loader)
        const_fold_opt.optimize()

    def optimize_nms(self, **nms_params: Dict):
        """
        Apply NMS optimization to the loader.

        :kwargs nms_type (str) : Type of NMS optimization to be applied. It can
            be either Host or Device.
        :kwargs arch_type (str) : Type of model architecture for the loaded model.
        :kwargs max_boxes (int) : Max number of boxes the model should generate.
        :kwargs max_boxes_per_class (int) : Max number of boxes to be selected
            per class as a part of NonMaxSuppression operation.
        :kwargs iou_threshold (float) : IOU threshold to be used for NMS computation.
        :kwargs score_threshold (float) : Score threshold to be used for NMS computation.
        :kwargs num_classes (int) : Number of classes the model is predicting. It
            should include background class as well for SSD kind of models.
        :kwargs class_specific_nms (bool) : Whether to apply class specific NMS
            or class agnostic NMS in the model.
        :kwargs background_class_idx (int) : Index of the background class in
            the predicted classes.
        :kwargs map_coco_80_to_90 (bool) : Map the coco 80 class indices to 90
            class indices.
        :kwargs anchor_data (str) : Path of the anchor data.
        :kwargs boxes_format (str) : Format of the raw boxes predicted by the
            model. This is specific to SSD kind of models only. It can be either
            xywh or yxhw.
        :kwargs scale_xy (float) : Scaling factor for xy prediction for SSD kind
            of models.
        :kwargs scale_wh (float) : Scaling factor for wh prediction for SSD kind
            of models.
        :kwargs scores_activation (str) : Activation function to be applied to
            raw scores predicted by model.

        :return ONNXModelUtils: Self instance of the class.
        """
        if nms_params.get("nms_type") is None:
            log_debug1("Skipping NMS optimization pass.")
            return

        if nms_params.get("arch_type") is None:
            log_error(
                "Fail to apply NMS optimization. 'onnx_nms_arch_type' parameter "
                "is not provided for NMS optimization."
            )
            return

        arch_type = nms_params["arch_type"]
        try:
            if arch_type.value in get_supported_yolo_models():
                nms_optimizer = OnnxYoloNmsOptimizer(self.loader, nms_params)
            elif arch_type.value in get_supported_ssd_models():
                nms_optimizer = OnnxSsdNmsOptimizer(self.loader, nms_params)
            else:
                raise RuntimeError(
                    f"Given model type '{arch_type}' is not supported in NMS Optimization."
                )
            nms_optimizer.optimize()
        except Exception as e:
            log_error(f"NMS Optimization failed due to exception: {e}.")

    def optimize(self, **kwargs: Dict):
        """
        Optimize the onnx graph inplace with all the default optimization.

        :kwargs static_input_shapes (List) : Optional Input shapes.
        :kwargs skip_simplifier (bool) : Flag to skip onnx runtime and
            simplifier optimizations.
        :kwargs opt_level (Text) : Graph optimization level (Default is BASIC),
            available options ["BASIC", "EXTENDED", "ALL"]
        :kwargs nms_params (Dict) : Dict containing all the parameters required
            for NMS optimization.

        :return ONNXModelUtils: Self instance of the class.
        """
        # FIXME: Divide this function in 2 parts so that weight load unload can be minimized.
        #       - Weight dependent optimizations    (simplifier)
        #       - Weight independent optimizations (einsum)

        static_input_shapes = kwargs.get("static_input_shapes", None)
        skip_simplifier = kwargs.get("skip_simplifier", False)
        perform_sequence_construct_optimizer = kwargs.get("perform_sequence_construct_optimizer", False)
        nms_params = kwargs.get("nms_params", {})

        if not skip_simplifier:
            self.optimize_by_simplifier(static_input_shapes)

            if not self.loader.has_custom_op:
                opt_level = kwargs.get("opt_level", "BASIC")
                disabled_optimizers = kwargs.get("disabled_optimizers", [])
                # FIXME: Some models are failing due to onnx runtime optimization at IR
                # level but working with onnx runtime.
                # self.optimize_by_ort(opt_level, disabled_optimizers)

        nodes_by_op_types = Utils.get_nodes_by_op_types(self.loader.model, ["Einsum", "SequenceConstruct"])
        einsum_nodes = nodes_by_op_types["Einsum"]
        sequence_construct_nodes = nodes_by_op_types["SequenceConstruct"]

        # Apply Einsum optimizer
        self.optimize_einsum(einsum_nodes, skip_shape_inference=skip_simplifier)

        # Apply Sequence construct optimization pass
        if perform_sequence_construct_optimizer:
            self.optimize_sequence_construct_ops(sequence_construct_nodes, skip_shape_inference=skip_simplifier)

        # Apply constant folding optimization
        # self.optimize_const_fold()

        # Apply NMS Optimizer for AIC backend only. For AIC backend nms_params
        # might be populated if user has provided those arguments.
        self.optimize_nms(**nms_params)
        return self

    def update_input_node(self, input_names: List[Text], input_dims: List[Text]):
        """
        Updates input nodes dimensions if node is present as input else delete
        the existing graph inputs and add new inputs with provided shape.

        :param List[Text] input_names: List of input names.
        :param List[Text] input_dims: List of input dims to be used while updating.
        :raises ValueError: If the update input node is not supported.
        :raises ValueError: If the input name is one of the initializer.
        :raises ValueError: input_dims command not for any input.
        :raises ValueError: If the any of the input can't be updated.
        :return ONNXModelUtils: Self instance of the class.
        """

        # FIXME: Instead of taking input_names and input_dims as 2 different
        #        lists we shall take one dict which has mapping from name to dim.

        graph = self.loader.model.graph
        if version.parse(onnx.version.version) < version.parse("1.6.0"):
            raise ValueError(
                "update input node does not supported with ONNX versions < 1.6.0"
            )

        input_names = list(input_names)
        input_names_copy = input_names.copy()
        input_dims = list(input_dims)
        initializers = [init.name for init in graph.initializer]
        original_inputs = {model_input.name: model_input for model_input in graph.input}
        original_types = {
            model_input.name: model_input.type.tensor_type.elem_type
            for model_input in graph.input
        }
        new_inputs = {name: dim for name, dim in zip(input_names, input_dims)}

        # Step 1: remove original graph inputs
        for node_name in original_inputs:
            if node_name not in initializers:
                graph.input.remove(original_inputs[node_name])

        # Step 2: If input specified is part of graph inputs, update its dimensions
        for name in new_inputs:
            if name in initializers:
                raise ValueError(
                    "--input_dim command not supported with initializer " + name
                )
            elif name in original_inputs:
                dim = new_inputs[name]
                dims = tuple(map(int, dim.split(",")))
                type_new = original_types[name]
                input_new = onnx.helper.make_tensor_value_info(name, type_new, dims)
                graph.input.append(input_new)
                input_names.remove(name)
                input_dims.remove(dim)
            else:
                continue

        # Check if all inputs are accounted for, if Yes nothing more to be done. Return
        if len(input_names) == 0 and len(input_dims) == 0:
            return self

        # Get the type of each model input
        input_types = {}
        for input_name in input_names:
            input_found, input_type, _ = get_type_dims_info(
                self.loader.model.graph.input, input_name
            )
            input_types[input_name] = (
                input_type if input_found else onnx.TensorProto.FLOAT
            )

        # Step 3: If input specified is intermittent graph output,
        #         a.  Add this buffer to a list for removal later
        #         b.  Create input TensorProto with this name and dimension
        bufs_to_remove = set()
        for i, src_op in enumerate(graph.node):
            for output_buf_name in src_op.output:
                if output_buf_name in input_names:
                    position = input_names.index(output_buf_name)
                    dim = input_dims[position]
                    dims = tuple(map(int, dim.split(",")))
                    input_new = onnx.helper.make_tensor_value_info(
                        output_buf_name, input_types[output_buf_name], dims
                    )
                    graph.input.append(input_new)
                    bufs_to_remove.add(output_buf_name)
                    input_names.remove(output_buf_name)
                    input_dims.remove(dim)

        # Check if all inputs specified are accounted for
        if len(input_names) != 0 and len(input_dims) != 0:
            invalid_names = ", ".join(input_names)
            raise ValueError(
                "input_dim command input name(s) not found: {}".format(invalid_names)
            )

        # Step 4: Find all nodes to be removed from the graph. These include:
        #   a. Nodes that produce the buffers cached for removal
        #   b. All nodes that precede them in the graph
        nodes_to_remove = []
        while bufs_to_remove:
            buf_name = bufs_to_remove.pop()
            if buf_name in original_inputs or buf_name in initializers:
                # This was already removed or does not need to be handled
                continue

            # Find node that produces the buffer or is named after the buffer
            node_list = [node for node in graph.node if buf_name in node.output]
            if not node_list:
                raise KeyError("Node that produces {} not found".format(buf_name))
            elif len(node_list) != 1:
                raise KeyError(
                    "Multiple nodes {} found for output buffer {}".format(
                        node_list, buf_name
                    )
                )

            node = node_list[0]
            # Add all inputs of this node as also to be removed
            bufs_to_remove.update(set(node.input))
            # Add this node to be removed if not already added
            if node not in nodes_to_remove:
                nodes_to_remove.append(node)

        # Step 5: Remove the nodes marked in Step 4
        # Check that all buffers in a slice were specified, if not Throw Error
        remaining_nodes = [node for node in graph.node if node not in nodes_to_remove]
        remaining_buffers = set()
        for remaining_node in remaining_nodes:
            remaining_buffers.update(remaining_node.input)
        for node in nodes_to_remove:
            for output in node.output:
                if output in remaining_buffers and output not in input_names_copy:
                    raise ValueError(
                        "Cannot disconnect node with outputs: {} as output buffer"
                        ": {} is still in use and was not specified as input to the Model".format(
                            str(node.output), str(output)
                        )
                    )
            graph.node.remove(node)

        return self

    def update_output_names(self, output_names: List[Text]):
        """
        Update the model output names inplace.
        This will make the provided output_names as output and remove the rest
        of the outputs from the model.

        :param List[Text] output_names: List of output names to update.
        :return ONNXModelUtils: Self instance of the class.
        """
        # Nodes to delete is map with node.name as Key and node as value
        nodes_to_delete = {}
        queue = list(output_names)
        visited = set(queue)
        map_node_output = {}
        model = self.loader.model

        # traverse the graph once and build a map with (node, output name) for the graph
        for node in model.graph.node:
            # same node might appear for more than one output_name
            for output_name in node.output:
                map_node_output[output_name] = node
            # add all nodes in the graph to nodes_to_delete, later retained nodes will be deleted from it
            nodes_to_delete[node.name] = node
        log_debug1("Total nodes in nodes_to_delete {}".format(len(nodes_to_delete)))
        log_debug1("Total nodes in map_node_output {}".format(len(map_node_output)))

        while queue:
            name = queue.pop(0)
            # find the node who has same output name from the map
            if name not in map_node_output:
                log_debug1("tensor name {} is an input or constant tensor".format(name))
                continue
            current_node = map_node_output.get(name)

            if current_node.name not in nodes_to_delete:
                # if node does not exist in nodes_to_delete, it might have appeared for more than one output
                log_debug1(
                    "{} node is not present. Might have already removed from map".format(
                        current_node.name
                    )
                )
                continue
            nodes_to_delete.pop(current_node.name)
            # save the input tensor name in queue for finding producer node for the same
            for input_name in current_node.input:
                if input_name in visited:
                    continue
                queue.append(input_name)
                visited.add(input_name)

        log_debug1(
            "Total nodes remained in nodes_to_delete {}".format(len(nodes_to_delete))
        )
        # Remove nodes that are not retained
        for node in nodes_to_delete.values():
            model.graph.node.remove(node)

        # Get the output dimensions of the new output nodes
        new_output_value_infos = []
        for output_name in output_names:
            # First check the graph outputs for info on outputs
            output_found, output_type, output_dims = get_type_dims_info(
                model.graph.output, output_name
            )

            # Fallback to using optional value_info field for info on new outputs
            if not output_found and model.graph.value_info:
                output_found, output_type, output_dims = get_type_dims_info(
                    model.graph.value_info, output_name
                )

            # Finally, fallback to using graph inputs for info on new outputs
            if not output_found:
                output_found, output_type, output_dims = get_type_dims_info(
                    model.graph.input, output_name
                )

            if output_found:
                output_value_info = onnx.helper.make_tensor_value_info(
                    output_name, output_type, output_dims
                )
            else:
                output_value_info = onnx.helper.ValueInfoProto()
                output_value_info.name = output_name

            new_output_value_infos.append(output_value_info)

        # Remove old output nodes
        for output_node in [_ for _ in model.graph.output]:
            model.graph.output.remove(output_node)

        # Add new output info
        model.graph.output.extend(new_output_value_infos)
        return self

    def update_onnx_define_symbols(
        self, define_symbols: Optional[Dict] = None, batch: Text = None
    ):
        """
        Update the onnx define symbols as well as batch in place.

        :param Optional[Dict] define_symbols: onnx define symbols dictionary, defaults to None
        :param Text batch: Value for batch symbol to be imputed, defaults to None
        :raises ValueError: In case of not being able to update the onnx defined symbols.
        :return ONNXModelUtils: Self instance of the class.
        """
        if not (define_symbols or batch):
            return self

        graph = self.loader.model.graph
        if version.parse(onnx.version.version) < version.parse("1.6.0"):
            raise ValueError(
                "Not able to add batch size and define symbols for ONNX versions < 1.6.0"
            )

        # Override any symbols present in the input shapes with values passed by the client
        original_inputs = {node.name: node for node in graph.input}
        new_inputs = Utils.get_all_dims_info(Utils.get_inputs(self.loader.model))
        for name, dtype, dims in new_inputs:
            log_debug1(
                "Proccessing overrides for input {} with dims {}".format(name, dims)
            )
            modified = False
            if define_symbols:
                for i, dim in enumerate(dims):
                    if isinstance(dim, str):
                        if dim in define_symbols:
                            log_debug1(
                                'Overriding "{}" with {}'.format(
                                    dim, int(define_symbols[dim])
                                )
                            )
                            dims[i] = int(define_symbols[dim])
                            modified = True
                        else:
                            log_warning(
                                f"For input: {name}, the onnx defined "
                                f"symbol {dim} is not provided."
                            )

            # Override the batch dimension of all inputs with the passed value
            # TODO At some point make this batch logic common for all converters
            if batch:
                log_debug1(
                    "Overriding batch dim of {} from {} to {}".format(
                        name, dims[0], int(batch)
                    )
                )
                dims[0] = int(batch)
                modified = True

            # Remove the original input and add the new updated input
            if modified:
                new_input = onnx.helper.make_tensor_value_info(name, dtype, dims)
                log_debug1("Generated new input {} with dims {}".format(name, dims))
                graph.input.remove(original_inputs[name])
                graph.input.append(new_input)

        return self

    def add_inputs(self, input_names: List[Text], infer_shape: bool = True):
        """
        Function to convert the given tensor name into graph inputs.

        :param List[Text] input_names: List of tensor names which needs to be
            converted into graph inputs.
        :param bool infer_shape: Perform shape inference before input creation,
            defaults to True
        :raises Exception: If the given tensor can't be made the graph input.
        :return ONNXModelUtils: Self instance of the class.
        """
        if infer_shape:
            self.native_shape_inference(delete_existing_shapes=True)

        val_info_dict = Utils.get_value_info_proto_mappings(self.loader.model)
        init_dict = Utils.get_initializer_mappings(self.loader.model)
        node_by_output_tensor_name = Utils.get_node_by_output_name(self.loader.model)

        model_inputs = self.loader.get_input_names()
        filtered_input_names = [
            name for name in input_names if name not in model_inputs
        ]

        if not filtered_input_names:
            # Means all the elements of input_names are already inputs of model.
            return self

        # First check whether all the given input_names are available in value info.
        for name in filtered_input_names:
            if name not in init_dict and name not in val_info_dict:
                raise Exception(
                    f"{name} can't be made an input as it is an intermediate"
                    "tensor and it is not present in value info."
                )

        for name in filtered_input_names:
            # Remove the connection of this tensor with its parent node.
            if name in node_by_output_tensor_name:
                node = node_by_output_tensor_name[name]
                node_op_to_remove = [
                    node_op for node_op in node.output if node_op == name
                ]
                for node_op in node_op_to_remove:
                    node.output.remove(node_op)

                # Add the value info of the intermediate tensor in the graph's
                # input.
                self.loader.model.graph.input.append(val_info_dict[name])

            # Remove the instance of the new input from initializer if any.
            if name in init_dict:
                initializer = init_dict[name]
                initializer_numpy_data = get_initializer_value(initializer)
                self.loader.model.graph.initializer.remove(initializer)

                # Create a new value info for the initializer tensor and add the
                # same to graph's input.
                new_input_val_info = helper.make_tensor_value_info(
                    name,
                    np_dtype_to_onnx_tensor_dtype(initializer_numpy_data.dtype),
                    initializer_numpy_data.shape,
                )
                self.loader.model.graph.input.append(new_input_val_info)

        # Cleanup the model as after removing the connection with the parent
        # node the graph has dangling nodes.
        self.clean_model()
        return self

    def add_outputs(self, output_names: List[Text], infer_shape: bool = True):
        """
        Function to convert the given tensor name into graph outputs.

        :param List[Text] output_names: List of tensors names which needs to be
            converted into graph outputs.
        :param bool infer_shape: Perform shape inference before output creation,
            defaults to True
        :raises ValueError: If the given tensor can't be made the graph output.
        :return ONNXModelUtils: Self instance of the class.
        """
        if infer_shape:
            self.native_shape_inference(delete_existing_shapes=True)

        tensor_val_info_dict = Utils.get_value_info_proto_mappings(self.loader.model)
        input_val_info_dict = self.loader.get_inputs()
        output_val_info_dict = self.loader.get_outputs()
        all_val_info_dict = {
            **tensor_val_info_dict,
            **input_val_info_dict,
            **output_val_info_dict,
        }

        model_outputs = self.loader.get_output_names()

        filtered_output_names = [
            name for name in output_names if name not in model_outputs
        ]

        if not filtered_output_names:
            # Means all the elements of output_names are already output of model.
            return self

        # First check whether all the given output_names are available in value info.
        for name in filtered_output_names:
            if name not in all_val_info_dict:
                raise ValueError(
                    f"{name} can't be made an output as it is not present in value info."
                )

        # If all output_names are available in value info then make the change to self.loader.
        for name in filtered_output_names:
            self.loader.model.graph.output.append(all_val_info_dict[name])
        return self

    def remove_inputs(self, input_names_to_remove: List[str]):
        """
        Function to remove give inputs from the model.
        Note: This API should not be used to remove the initializer which are
        present in graph as inputs also. For such cases call
        remove_initializer_from_input() API.

        :param List[str] input_names_to_remove: List of input names to be
                removed from model's input.
        :return ONNXModelUtils: Self instance of the class.
        """
        get_node_by_input_tensor_name = Utils.get_node_by_input_name(self.loader.model)

        input_dict = self.loader.get_inputs()
        for input_name in input_names_to_remove:
            if input_name in input_dict:
                ip = input_dict[input_name]
                if input_name not in get_node_by_input_tensor_name:
                    # This means it is not used by any nodes. Means we are safe to
                    # remove this input.
                    self.loader.model.graph.input.remove(ip)
                else:
                    children_nodes = [
                        n.name for n in get_node_by_input_tensor_name[input_name]
                    ]
                    log_warning(
                        f"{input_name} is can't be removed from the "
                        f"graph as it is used by {children_nodes} nodes."
                    )
            else:
                log_debug1(
                    f"{input_name} can't be removed from model's inputs as it "
                    "is not a model's input."
                )
        return self

    def remove_outputs(self, output_names_to_remove: List[str]):
        """
        Function to remove given outputs from the model.

        :param List[str] output_names_to_remove: List of output names to be
                removed from model's outputs.
        :return ONNXModelUtils: Self instance of the class.
        """
        model_output_names = self.loader.get_output_names()
        output_dict = self.loader.get_outputs()
        for output_name in output_names_to_remove:
            if output_name in model_output_names:
                op = output_dict[output_name]
                # Remove the tensor's value info from graph's output
                self.loader.model.graph.output.remove(op)
                # Add the tensor's value info from graph's value info list
                # because the tensor still exists in the graph. Now it exists
                # only as a node's output node as graph's output.
                self.add_value_info(op)
            else:
                log_debug1(
                    f"{output_name} can't be removed from "
                    "model's outputs as it is not a model's output."
                )
        return self

    def _native_shape_inference_helper(
        self, delete_existing_shapes
    ) -> Union[None, ModelWrapper]:
        """
        Run the shape inference over the model and add shape information for all
        nodes and all outputs.

        :param bool delete_existing_shapes: Delete existing shape information
            before obtaining shapes, defaults to True
        :return Union[None, ModelWrapper]: ModelWrapper object if operation is
            successful else None.
        """
        try:
            # As a first step try to run symbolic shape inference. If this fails
            # then as a fallback mechanism use normal shape inference.
            model_wrapper = self.__symbolic_shape_inference()
            log_debug1("Symbolic Shape inference executed successfully.")
        except Exception as e:
            # Note: Symbolic shape inference will fail for CustomOps.
            #       So as a fall back we should be calling normal shape
            #       inference.
            log_warning(
                f"Symbolic shape inference Failed. Exception: {e}. Running "
                "normal shape inference."
            )

            try:
                model_wrapper = self.__shape_inference(delete_existing_shapes)
                log_debug1("Onnx Shape inference executed successfully.")
            except Exception as e:
                log_warning(f"Shape inference Failed With Exception: {e}.")
                return None

        return model_wrapper

    def native_shape_inference(self, delete_existing_shapes: bool = False):
        """
        Run the shape inference over the model and add shape information for all
        nodes and all outputs.

        :param bool delete_existing_shapes: Delete existing shape information
            before obtaining shapes, defaults to True
        :return ONNXModelUtils: Self instance of the class.
        """
        status, model_wrapper = spawn_process_and_exec(
            self._native_shape_inference_helper,
            delete_existing_shapes,
            process_name="Shape Inference",
        )
        if not status:
            return self

        log_debug1("Shape inference executed successfully.")
        self.loader.update_model_wrapper(model_wrapper)
        return self

    def __symbolic_shape_inference(self) -> ModelWrapper:
        """
        This method will add the symbolic shape info to the model file.

        :return ModelWrapper: Instance of onnx model in the form of ModelWrapper.
        """
        # Symbolic shape inference works for both ModelProtos < 2GB and > 2GB.
        try:
            from onnxruntime.tools.symbolic_shape_infer import (
                SymbolicShapeInference,
                get_opset,
            )

            # Symbolic Shape Inference doesn't need to explicitly remove
            # existing shapes.

            if len(self.loader.get_outputs()) == 0:
                raise RuntimeError(
                    "Unable to infer shapes because model has no outputs."
                )
            if self.loader.has_custom_op:
                raise RuntimeError(
                    "Unable to infer shapes because model has custom ops."
                )

            log_debug1("Running onnxrt symbolic shape inference.")
            # Symbolic shape inference needs model with weights.
            model_w_weights = self.loader.model_wrapper.get_full_model()
            onnx_opset = get_opset(model_w_weights)
            if (not onnx_opset) or onnx_opset < SYMBOLIC_SHAPE_INFER_MIN_OPSET_VER:
                raise RuntimeError(
                    "Symbolic Shape Inference only supports "
                    "models of onnx opset 7 and above."
                )
            symbolic_shape_inference = SymbolicShapeInference(
                int_max=MAX_INT_32, auto_merge=False, guess_output_rank=False, verbose=0
            )
            all_shapes_inferred = False
            symbolic_shape_inference._preprocess(model_w_weights)
            while symbolic_shape_inference.run_:
                all_shapes_inferred = symbolic_shape_inference._infer_impl()
            symbolic_shape_inference._update_output_from_vi()
            if not all_shapes_inferred:
                raise RuntimeError("Incomplete symbolic shape inference.")
            return ModelWrapper(symbolic_shape_inference.out_mp_)
        except ImportError as e:
            raise ImportError(
                "Onnxruntime package not found in current "
                "environment. Symbolic Shape Inference will be skipped."
            )

    def __shape_inference(self, delete_existing_shapes: bool) -> ModelWrapper:
        """
        This method will add the shape info to the model file.

        :param bool delete_existing_shapes: Delete existing shape information
            before obtaining shapes, defaults to True
        :return ModelWrapper: Instance of onnx model in the form of ModelWrapper.
        """
        log_debug1("Running onnx shape inference.")
        if delete_existing_shapes:
            self.remove_shapes()
        # onnx's shape inference works without weights, too. Hence using
        # self.loader.model which is not having weights in it.

        # FIXME: Here we are cloning existing wrapper which means for the cloned
        # version the remove_on_del is also True. Right now we are in child process
        # and for all child process remove_on_del should be false.
        model = onnx.shape_inference.infer_shapes(self.loader.model)
        # Note: Here we are changing self.loader.model_wrapper instead of creating
        # new one. Shape inference is executed in children process hence any change
        # to self variable won't be affected to parent process.
        self.loader.model_wrapper.update_model(model)
        return self.loader.model_wrapper

    def clean_model(self):
        """
        Clean the model from dangling or redundant nodes or input or outputs in
        the graph.

        :return ONNXModelUtils: Self instance of the class.
        """
        model = cleanup(self.loader.model)
        self.loader.update_model(model)
        return self

    def remove_node(self, node: NodeProto):
        """
        Function to remove the node from the graph

        :param NodeProto node: Node reference to be removed from graph.
        :return ONNXModelUtils: Self instance of the class.
        """
        for graph in get_graphs(self.loader.model):
            if node in graph.node:
                graph.node.remove(node)
        return self

    def remove_nodes(self, nodes_to_remove: NodeProto):
        """
        Function to remove the nodes from the graph

        :param NodeProto nodes_to_remove: List of nodes to remove from the graph
        :return ONNXModelUtils: Self instance of the class.
        """
        for node in nodes_to_remove:
            self.remove_node(node)
        return self

    def add_initializer(self, tensor: TensorProto, graph_name: Text = None):
        """
        Function to add single initializer to the given graph.

        :param TensorProto tensor: Onnx TensorProto instance.
        :param Text graph_name:  Name of the graph in which node will be added,
            defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        if graph_name is None or graph_name == self.loader.model.graph.name:
            self.loader.model.graph.initializer.extend([tensor])
        else:
            graph = get_graph_by_name(self.loader.model, graph_name)
            graph.initializer.extend([tensor])
        return self

    def add_initializers(
        self,
        tensors_to_add: List[TensorProto],
        tensor_name_to_graph_name: Dict[Text, Text] = None,
    ):
        """
        Function to add the list of initializers into the graph

        :param List[TensorProto] tensors_to_add: List of initializers to be
            added to the graph
        :param Dict[Text, Text] tensor_name_to_graph_name: Initializer name to
            graph name dict which represents which initializer name to be added
            in which graph, defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        if tensor_name_to_graph_name is None:
            for tensor in tensors_to_add:
                self.add_initializer(tensor)
        else:
            for tensor in tensors_to_add:
                graph_name = tensor_name_to_graph_name[tensor.name]
                self.add_initializer(tensor, graph_name)
        return self

    def __get_topological_insert_id(self, graph: GraphProto, outputs: List[str]) -> int:
        """
        Function to get the topological insert id of the node.

        :param GraphProto graph: Onnx graph reference.
        :param List[str] outputs: Node output names for given node.
        :return int: Insert id of the given node.
        """
        for idx, node in enumerate(graph.node):
            for input in node.input:
                if input in outputs:
                    return idx
        return len(graph.node)

    def add_node(self, node: NodeProto, graph_name: Text = None):
        """
        Function to add the single node to the given graph name.

        :param NodeProto node: Onnx node reference.
        :param Text graph_name: Name of the graph in which node will be added,
            defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        if graph_name is None or graph_name == self.loader.model.graph.name:
            graph_name = self.loader.model.graph.name

        graph = get_graph_by_name(self.loader.model, graph_name)
        insert_idx = self.__get_topological_insert_id(graph, node.output)
        graph.node.insert(insert_idx, node)
        return self

    def add_nodes(
        self,
        nodes_to_add: List[NodeProto],
        node_name_to_graph_name: Dict[Text, Text] = None,
    ):
        """
        Function to add the nodes to the graph

        :param List[NodeProto] nodes_to_add: List of node to add to the graph
        :param Dict[Text, Text] node_name_to_graph_name: Node name to graph name
            dict which represents which node to be added in which graph,
            defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        if node_name_to_graph_name is None:
            for node in nodes_to_add:
                self.add_node(node)
        else:
            for node in nodes_to_add:
                graph_name = node_name_to_graph_name[node.name]
                self.add_node(node, graph_name)
        return self

    def add_value_info(self, value_info: ValueInfoProto, graph_name: Text = None):
        """
        Function to add the single value info object to the given graph name.

        :param ValueInfoProto value_info: Onnx value info reference.
        :param Text graph_name: Name of the graph in which value_info will be added,
            defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        if (graph_name is None) or (graph_name == self.loader.model.graph.name):
            graph_name = self.loader.model.graph.name

        graph = get_graph_by_name(self.loader.model, graph_name)
        graph.value_info.extend([value_info])
        return self

    def add_value_infos(
        self,
        value_infos_to_add: List[ValueInfoProto],
        value_info_name_to_graph_name: Dict[Text, Text] = None,
    ):
        """
        Function to add list of value infos into the graph.

        :param List[ValueInfoProto] value_infos_to_add: List of value_info
            objects to be add to the graph.
        :param Dict[Text, Text] value_info_name_to_graph_name: Value info name
            to graph name dict which represents which value_info object to be
            added in which graph, defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        if value_info_name_to_graph_name is None:
            for val_info in value_infos_to_add:
                self.add_value_info(val_info)
        else:
            for val_info in value_infos_to_add:
                graph_name = value_info_name_to_graph_name[val_info.name]
                self.add_value_info(val_info, graph_name)
        return self

    def create_node(self, input_names: List[str], node_args: Dict, opset: int) -> Tuple:
        """
        Create a node based on input tensor names and node_args which contains
        node's attributes names and their values.
        Note: Graph shall be checked after adding this node using native_checker.

        :param List[str] input_names: List of input tensor names.
        :param Dict node_args: Dict mapping of node's attributes name and their values.
        :param int opset: Opset to be used while generating the node.
        :raises RuntimeError: If op_type key is missing in node_args.
        :raises RuntimeError: If an attribute provided in node_args in not an
            attribute for the node of given op_type.
        :raises RuntimeError: In case of unknown failure during node creation.
        :return Tuple: Tuple of onnx node and initializer which is used as
            attribute to the created node.
        """
        op_type = node_args.get("op_type")
        if op_type is None:
            raise RuntimeError("No op_type supplied for node creation.")
        node_name = self.get_new_node_name(op_type)
        output_names = []
        for _ in range(node_args.get("num_outputs", 1)):
            output_name = self.get_new_tensor_name(node_name)
            output_names.append(output_name)
        node_args["inputs"] = input_names
        node_args["name"] = node_name
        node_args["outputs"] = output_names

        valid_attribute_names = list(
            onnx.defs.get_schema(op_type, opset).attributes.keys()
        )
        valid_input_names = [
            inp.name for inp in onnx.defs.get_schema(op_type, opset).inputs
        ]
        new_initializers = []

        node_arg_keys = list(node_args.keys())
        for attr_name in node_arg_keys:
            if attr_name in ["inputs", "outputs", "name", "op_type"]:
                continue
            if attr_name in valid_input_names:
                # This is used to add constant inputs to the node.
                # If the node have variable inputs then user shall provide
                # that as input_names.
                node_val = node_args[attr_name]
                if not isinstance(node_val, np.ndarray):
                    node_val = np.array(node_val, dtype=np.int64)
                initializer = onnx.numpy_helper.from_array(
                    node_val, name=f"{node_name}.{attr_name}"
                )
                new_initializers.append(initializer)
                node_args["inputs"].append(initializer.name)
                node_args.pop(attr_name)
            elif attr_name in valid_attribute_names:
                valid_attribute_names.remove(attr_name)
            else:
                log_warning(
                    f"For op: {op_type}, given attribute/input '{attr_name}' "
                    "is redundant. Ignoring the same."
                )
                node_args.pop(attr_name)

        if len(valid_attribute_names) != 0:
            raise RuntimeError(
                f"For op: {op_type}, following attributes are not provided: "
                f"{valid_attribute_names}."
            )
        if op_type == "Concat":
            if len(node_args["inputs"]) == 0:
                raise RuntimeError(
                    f"For op: {op_type}, number of inputs shall be 1 or more."
                )
        else:
            if len(valid_input_names) != len(node_args["inputs"]):
                raise RuntimeError(
                    f"For op: {op_type}, number of inputs provided is "
                    f"incorrect. Expected: {len(valid_input_names)}, "
                    f"provided: {len(node_args['inputs'])}."
                )

        node = make_node(**node_args)
        # Ideally we should be calling check_node, but it is not reliably checking
        # due to some internal bug. Hence relying on the caller to use this generated
        # node, add it to graph and call check_graph.
        # onnx.checker.check_node(node)
        return node, new_initializers

    def create_value_info(
        self, name: str, dtype: np.dtype, shape: List[int]
    ) -> ValueInfoProto:
        """
        Get a value info proto based on provided name, dtype and shape of the tensor.

        :param str name: Name of the tensor.
        :param np.dtype dtype: Dtype of the tensor.
        :param List[int] shape: Shape of the tensor.
        :raises RuntimeError: If the provided dtype is invalid as there is no
            onnx dtype mapping is found.
        :return ValueInfoProto: ValueInfo representation of the tensor.
        """
        if dtype not in Utils.numpy_to_onnx_type:
            raise RuntimeError(
                f"No onnx conversion found for provided numpy dtype: {dtype}."
            )
        onnx_dtype = Utils.numpy_to_onnx_type[dtype]
        return onnx.helper.make_tensor_value_info(name, onnx_dtype, shape)

    def __graph_topological_sort(self, graph: GraphProto):
        """
        Function to get the topologically sorted graph from Graph proto

        :param GraphProto graph: GraphProto of the model.
        :return ONNXModelUtils: Self instance of the class.
        """
        deps_count = [0] * len(graph.node)  # dependency count of each node
        deps_to_nodes = {}  # input to node indices
        sorted_nodes = []  # initialize sorted_nodes

        for node_idx, node in enumerate(graph.node):
            # CANNOT use len(node.input) directly because input can be optional
            deps_count[node_idx] = sum(1 for _ in node.input if _)
            if deps_count[node_idx] == 0:  # Constant doesn't depend on any inputs
                sorted_nodes.append(graph.node[node_idx])
                continue
            for input_name in node.input:
                if input_name not in deps_to_nodes:
                    deps_to_nodes[input_name] = [node_idx]
                else:
                    deps_to_nodes[input_name].append(node_idx)

        # Note: this logic only applies to top level graph since a sub graph could use intializer from parent graph
        initializer_names = [init.name for init in graph.initializer]
        graph_input_names = [input.name for input in graph.input]
        input_names = initializer_names + graph_input_names

        intermediate_output_tensor_names = set()
        for n in graph.node:
            for n_op in n.output:
                if n_op not in intermediate_output_tensor_names:
                    intermediate_output_tensor_names.add(n_op)

        for n in graph.node:
            for n_ip in n.input:
                if n_ip == "":
                    # Skip the blank named input as that input is an attribute
                    # of node but its value is not present. Assuming this as an
                    # optional attribute of the node.
                    continue
                if (
                    n_ip not in initializer_names
                    and n_ip not in graph_input_names
                    and n_ip not in intermediate_output_tensor_names
                ):
                    # If a node's input is not output of any node in the present graph then
                    # add that name in input_names with an assumption that the name is part of
                    # parent graph's some node's output or the node has no dependency on input
                    # and it is a constant node.
                    input_names.append(n_ip)

        input_names.sort()
        prev_input_name = None
        for input_name in input_names:
            if prev_input_name == input_name:
                continue
            prev_input_name = input_name
            if input_name in deps_to_nodes:
                for node_idx in deps_to_nodes[input_name]:
                    deps_count[node_idx] = deps_count[node_idx] - 1
                    if deps_count[node_idx] == 0:
                        sorted_nodes.append(graph.node[node_idx])

        start = 0
        end = len(sorted_nodes)
        while start < end:
            for output in sorted_nodes[start].output:
                if output in deps_to_nodes:
                    for node_idx in deps_to_nodes[output]:
                        deps_count[node_idx] = deps_count[node_idx] - 1
                        if deps_count[node_idx] == 0:
                            sorted_nodes.append(graph.node[node_idx])
                            end = end + 1
            start = start + 1

        assert end == len(graph.node), "Graph is not a DAG"
        graph.ClearField("node")
        graph.node.extend(sorted_nodes)
        return self

    def topological_sort(self):
        """
        Function to get the topologically sorted graphs.

        :return Self instance of the class.
        """
        for graph in get_graphs(self.loader.model):
            self.__graph_topological_sort(graph)
        return self

    def assign_names_to_empty_nodes(self):
        """
        Add name to all the nodes whos name is "".

        :return ONNXModelUtils: Self instance of the class.
        """
        model = Utils.assign_names_to_empty_nodes(self.loader.model)
        self.loader.update_model(model)
        return self

    def remove_shapes(self):
        """
        Remove tensor shape information from model.

        :return ONNXModelUtils: Self instance of the class.
        """
        model = Utils.remove_shapes(self.loader.model)
        self.loader.update_model(model)
        return self
