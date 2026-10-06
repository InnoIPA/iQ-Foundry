# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx import NodeProto, TensorProto
from qti.aisw.converters.common.framework_optimizer import FrameworkOptimizer
from qti.aisw.converters.common.framework_pattern_matcher import FrameworkPatternMatcher
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils.converter_utils import log_debug1, log_error
from qti.aisw.converters.onnx.optimizations.nms_utils import (
    COMBINEDNMS_DOMAIN,
    COMBINEDNMS_OPSET,
    NMSType,
    get_host_nms_arch_type,
    get_supported_anchor_free_models,
    read_anchors,
    validate_nms_params,
)
from qti.aisw.converters.onnx.util import (
    get_extracted_model,
    get_ir_version,
    get_opset_version,
    make_node,
)


class OnnxBaseNMSOptimizer(FrameworkOptimizer):
    """
    OnnxBaseNMSOptimizer class is responsible for identifying the object detection
    algorithms' feature extractors, cutting the graph at the feature extractors and
    added subsequent nodes to make it compatible to Host / Device NMS model.
    """

    def __init__(self, loader: FrameworkModelLoader, nms_params: Dict):
        """
        Constructor of OnnxBaseNMSOptimizer

        :param FrameworkModelLoader loader: Onnx model loader reference.
        :param Dict nms_params: Dict containing useful NMS related parameters
            obtained from user via argparser.
        :raises RuntimeError: If the user provided NMS params or anchors are invalid.
        """
        self.loader = loader
        self.model_opset = get_opset_version(self.loader.model)
        self.model_ir_version = get_ir_version(self.loader.model)
        self.nms_params = nms_params
        if not validate_nms_params(self.nms_params):
            raise RuntimeError("Provided NMS parameters are invalid.")
        if not read_anchors(self.nms_params):
            raise RuntimeError("Provided anchors are invalid.")

    def convert_xywh_to_yxyx(
        self, loader: FrameworkModelLoader, xywh_tensor: str
    ) -> str:
        """
        Convert boxes from xywh format into yxyx format.

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :param str xywh_tensor: Name of the boxes tensor in xywh format.
        :return str: Name of output tensor representing boxes in yxyx format.
        """

        split_node, initializers = loader.utils.create_node(
            [xywh_tensor],
            {"op_type": "Split", "split": [2, 2], "axis": 2, "num_outputs": 2},
            self.model_opset,
        )
        loader.utils.add_node(split_node)
        loader.utils.add_initializers(initializers)
        [split_xy_output, split_wh_output] = split_node.output

        mul_node_wh, initializers = loader.utils.create_node(
            [split_wh_output],
            {"op_type": "Mul", "B": np.array([0.5], dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(mul_node_wh)
        loader.utils.add_initializers(initializers)

        sub_node_x1y1, initializers = loader.utils.create_node(
            [split_xy_output, mul_node_wh.output[0]],
            {"op_type": "Sub"},
            self.model_opset,
        )
        loader.utils.add_node(sub_node_x1y1)
        loader.utils.add_initializers(initializers)

        add_node_x2y2, initializers = loader.utils.create_node(
            [split_xy_output, mul_node_wh.output[0]],
            {"op_type": "Add"},
            self.model_opset,
        )
        loader.utils.add_node(add_node_x2y2)
        loader.utils.add_initializers(initializers)

        split_node, initializers = loader.utils.create_node(
            [sub_node_x1y1.output[0]],
            {"op_type": "Split", "split": [1, 1], "axis": 2, "num_outputs": 2},
            self.model_opset,
        )
        loader.utils.add_node(split_node)
        loader.utils.add_initializers(initializers)
        [split_x1_output, split_y1_output] = split_node.output

        split_node, initializers = loader.utils.create_node(
            [add_node_x2y2.output[0]],
            {"op_type": "Split", "split": [1, 1], "axis": 2, "num_outputs": 2},
            self.model_opset,
        )
        loader.utils.add_node(split_node)
        loader.utils.add_initializers(initializers)
        [split_x2_output, split_y2_output] = split_node.output

        concat_node, initializers = loader.utils.create_node(
            [split_y1_output, split_x1_output, split_y2_output, split_x2_output],
            {"op_type": "Concat", "axis": 2},
            self.model_opset,
        )
        loader.utils.add_node(concat_node)
        loader.utils.add_initializers(initializers)
        yxyx_tensor = concat_node.output[0]
        return yxyx_tensor

    def add_combined_nms_node(
        self, loader: FrameworkModelLoader, comb_nms_inputs: List[str], batch_size: int
    ) -> List:
        """
        Generate CombinedNms CustomOp node based on the inputs
        and nms_params.

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :param List[str] comb_nms_inputs: List of inputs to CombinedNMS node.
        :param int batch_size: Batch size of the model.
        :raises RuntimeError: If the generated node is incorrect.
        :return List[str]: List of output tensors of NMS node.
        """
        new_initializers = []

        op_type = "CombinedNms"
        num_outputs = 4
        node_name = loader.utils.get_new_node_name(op_type)
        output_names = []
        for _ in range(num_outputs):
            output_name = loader.utils.get_new_tensor_name(node_name)
            output_names.append(output_name)

        node_args = {
            "op_type": op_type,
            "name": node_name,
            "inputs": comb_nms_inputs,
            "outputs": output_names,
            "domain": COMBINEDNMS_DOMAIN,
        }
        node_args["clip_boxes"] = 0
        node_args["pad_per_class"] = 0
        node_args["max_boxes_per_class"] = self.nms_params["max_boxes_per_class"]
        node_args["max_total_boxes"] = self.nms_params["max_boxes"]
        node_args["iou_threshold"] = self.nms_params["iou_threshold"]
        node_args["score_threshold"] = self.nms_params["score_threshold"]

        detection_boxes_val_info = onnx.helper.make_tensor_value_info(
            output_names[0],
            TensorProto.FLOAT,
            [batch_size, self.nms_params["max_boxes"], 4],
        )
        detection_scores_val_info = onnx.helper.make_tensor_value_info(
            output_names[1],
            TensorProto.FLOAT,
            [batch_size, self.nms_params["max_boxes"]],
        )
        detection_classes_val_info = onnx.helper.make_tensor_value_info(
            output_names[2],
            TensorProto.INT32,
            [batch_size, self.nms_params["max_boxes"]],
        )
        num_detections_val_info = onnx.helper.make_tensor_value_info(
            output_names[3], TensorProto.INT32, [batch_size]
        )
        new_value_infos = [
            detection_boxes_val_info,
            detection_scores_val_info,
            detection_classes_val_info,
            num_detections_val_info,
        ]

        node = make_node(**node_args)
        try:
            special_context = c_checker.CheckerContext()
            special_context.ir_version = self.model_ir_version
            special_context.opset_imports = {
                "": self.model_opset,
                COMBINEDNMS_DOMAIN: COMBINEDNMS_OPSET,
            }
            onnx.checker.check_node(node, special_context)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create node of op_type: {op_type} due to exception: {e}."
            )

        loader.utils.add_node(node)
        loader.utils.add_initializers(new_initializers)
        loader.utils.add_value_infos(new_value_infos)
        return output_names

    def __str__(self) -> str:
        """
        String representation of the NMS optimization.

        :return str: String representation of the NMS optimization.
        """
        return "NMS Optimization"

    def cleanup_output_names(self, names: List[str]) -> List[str]:
        """
        Remove "/" or "." from names of the tensor.

        :param List[str] names: Given list of tensor name.
        :return List[str]: Updated list of tensor name.
        """
        for i, name in enumerate(names):
            new_name = name.replace("/", "_").replace(".", "_")
            names[i] = new_name
        return names

    def generate_hostnms_common_config(
        self, loader: FrameworkModelLoader, model_info: Dict
    ) -> Tuple[bool, Dict]:
        """
        Generate the host nms config with common fields populated.

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :param Dict model_info: Dict containing useful information about model
            such as model's input related info, model's identified feature
            extractors related info etc.
        :return Tuple[bool, Dict]: Tuple of 2 results.
            - boolean value indicating whether the hostnms config is generated
              correctly or not.
            - Hostnms config dict conatining all the common information about
              the host nms execution.
        """
        hostnms_config = {}
        status, host_nms_arch = get_host_nms_arch_type(self.nms_params["arch_type"])
        if not status:
            return False, {}
        hostnms_config["model-architecture"] = host_nms_arch

        hostnms_config["num-classes"] = self.nms_params["num_classes"]
        hostnms_config["num-landmark"] = 0
        hostnms_config["input-info"] = {}
        hostnms_config["input-info"]["input-layout"] = model_info["input_info"][
            "image_layout"
        ]
        hostnms_config["landmark-output-list"] = []
        hostnms_config["do-class-specific-nms"] = self.nms_params["class_specific_nms"]

        hostnms_config["map-classes-coco-81-to-91"] = self.nms_params[
            "map_coco_80_to_90"
        ]
        hostnms_config["profiling-per-iter"] = False

        hostnms_config["model-output-dir"] = (
            "<Update the directory where net-run outputs are saved.>"
        )
        hostnms_config["model-output-extension"] = ".raw"
        hostnms_config["model-num-outputs"] = 1
        hostnms_config["abp-output-dir"] = os.path.join(
            os.path.dirname(self.nms_params["dlc_path"]), "hostnms_output"
        )

        if self.nms_params["arch_type"] in get_supported_anchor_free_models():
            hostnms_config["prior-filepath"] = ""
        else:
            if not isinstance(self.nms_params["anchor_data"], np.ndarray):
                log_error(
                    "anchor_data should be properly processed and present as numpy array."
                )
                return False, {}

            anchor_path = os.path.join(
                os.path.dirname(self.nms_params["dlc_path"]), "anchors.raw"
            )
            self.nms_params["anchor_data"].tofile(anchor_path)
            hostnms_config["prior-filepath"] = anchor_path

        hostnms_config["model-run-time"] = "QNN"
        hostnms_config["qnn-model-info"] = {}
        hostnms_config["qnn-model-info"]["model-binary-path"] = (
            "<Update the directory where context binary is saved.>"
        )
        hostnms_config["qnn-model-info"]["model-output-suffix"] = "-"

        input_name = model_info["input_info"]["image_input_name"]
        input_shape = None
        for tensor_name, tensor_info in loader.get_input_info().items():
            if tensor_name == input_name:
                input_shape = tensor_info.shape
        if input_shape is None:
            log_error(f"No shape information found for model input: {input_name}.")
            return False, {}

        hostnms_config["score-threshold"] = self.nms_params["score_threshold"]
        hostnms_config["nms-threshold"] = self.nms_params["iou_threshold"]
        hostnms_config["max-detections-image"] = self.nms_params["max_boxes"]
        hostnms_config["max-boxes-class"] = self.nms_params["max_boxes_per_class"]
        return True, hostnms_config

    def partition_model(
        self, matched_nodes: List[List[NodeProto]], partition_node_idx: int = -1
    ) -> FrameworkModelLoader:
        """
        Partition the model based on matched nodes and partition node index.
        The model will be partitioned at the end of all the matched node.
        e.g. last node of matched pattern.

        :param List[List[NodeProto]] matched_nodes: List of pattern matched nodes.
        :param int partition_node_idx: Partition the model at the index from
            the last node in pattern, defaults to -1
        :return FrameworkModelLoader: Partitioned model loader instance.
        """
        end_tensors = set()
        for matched_fmap in matched_nodes:
            end_node = matched_fmap[partition_node_idx]
            end_tensors.update(end_node.output)

        # Get subgraph based on original model's input as subgraph inputs and
        # matched pattern's output tensors as subgraph outputs.
        start_tensors = self.loader.get_input_names()
        subgraph_loader = get_extracted_model(self.loader, start_tensors, end_tensors)
        return subgraph_loader

    def rewrite(self, matched_nodes: List[List[NodeProto]]) -> bool:
        """
        Apply the NMS optimization based on identified nodes.

        :param List[List[NodeProto]] matched_nodes: Identified nodes based on
            registered patterns.
        :return bool: Boolean status indicating success of graph modification.
        """
        status, model_info = self.get_model_info(matched_nodes)
        if not status:
            log_error("Failed to extract feature extractor information.")
            return False

        subgraph_loader = self.partition_model(matched_nodes)

        status, subgraph_loader = self.add_abp_nms_nodes(subgraph_loader, model_info)
        if not status:
            log_error(
                f"Failed to generate anchor box processing subgraph for model type {self.nms_params['arch_type']}."
            )
            return False

        if self.nms_params["nms_type"] == NMSType.HOST:
            status, path = self.generate_hostnms_config(subgraph_loader, model_info)
            if not status:
                log_error("Failed to generate the host nms config file.")
                return False
        self.loader.update_model(subgraph_loader.model)
        return True

    def optimize(self) -> bool:
        """
        Apply NMS optimization by identifying feature maps and adding anchor box
        processing nodes and NMS node in the graph.

        :return bool: Status indicating whether the optimization is applied
            correctly or not.
        """
        pattern_matched = False

        all_matched_nodes = []
        for pattern in self.patterns:
            matched_nodes = FrameworkPatternMatcher.match(pattern, self.loader)
            if not matched_nodes:
                continue
            all_matched_nodes.extend(matched_nodes)
            pattern_matched = True

        if not pattern_matched:
            log_error("No patterns are identified.")
            return False

        status = self.rewrite(all_matched_nodes)
        if status:
            log_debug1("NMS Optimization successfully applied.")
        else:
            log_error(f"NMS Optimization failed.")
        return status
