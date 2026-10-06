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
from onnx import NodeProto
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils.converter_utils import log_debug1, log_error
from qti.aisw.converters.onnx.optimizations.base_nms_optimizer import (
    OnnxBaseNMSOptimizer,
)
from qti.aisw.converters.onnx.optimizations.nms_patterns import patterns
from qti.aisw.converters.onnx.optimizations.nms_utils import (
    COMBINEDNMS_DOMAIN,
    COMBINEDNMS_OPSET,
    ModelArchType,
    NMSType,
    dump_yaml,
    get_image_input_details,
)
from qti.aisw.converters.onnx.util import (
    get_node_mappings,
    get_shape_from_value_info_proto,
    get_value_info_by_name,
)


class OnnxSsdNmsOptimizer(OnnxBaseNMSOptimizer):
    """
    OnnxSsdNmsOptimizer class responsible for applying NMS Optimization to SSD
    kind of models.
    """

    def __init__(self, loader: FrameworkModelLoader, nms_params: Dict):
        """
        Constructor for SSD NMS Optimizer.

        :param FrameworkModelLoader loader: Onnx model loader reference.
        :param Dict nms_params: Dict containing useful NMS related parameters
            obtained from user via argparser.
        :raises RuntimeError: If the user provided NMS params or anchors are invalid.
        """
        super().__init__(loader, nms_params)
        self.patterns = patterns["ssd"]
        # anchors are processed from yxyx user input into format as per below dict.
        self.host_nms_anchor_transforms = {
            ModelArchType.SSDVGG: lambda x: x,  # xywh -> xywh
            ModelArchType.SSDRESNET: lambda x: x,  # xywh -> xywh
            ModelArchType.SSDINCEPTION: lambda x: x,  # xywh -> xywh
            ModelArchType.SSDMOBILENET: lambda x: x,  # xywh -> xywh
            ModelArchType.EFFICIENTDET: self.convert_anchors_xywh_to_yxyx,  # xywh -> yxyx
            ModelArchType.RETINANET: self.convert_anchors_xywh_to_xyxy,  # xywh -> xyxy
        }

        # FIXME: We can use self.nms_params["boxes_format"] instead of below mapping.
        self.yxhw_prediction_map = {
            ModelArchType.SSDRESNET: False,  # checked and correct
            ModelArchType.RETINANET: False,  # checked and correct
            ModelArchType.SSDVGG: False,  # checked and correct
            ModelArchType.SSDMOBILENET: True,  # checked and correct
            ModelArchType.EFFICIENTDET: True,  # checked and correct
            ModelArchType.SSDINCEPTION: True,  # checked and correct
        }

    def convert_anchors_xywh_to_xyxy(self, anchors: np.ndarray) -> np.ndarray:
        """
        Convert the provided anchors from xywh format into xyxy format.

        :param np.ndarray anchors: Anchors in the form of numpy array with shape
            as [N x 4]
        :return np.ndarray: Updated anchors in xyxy format in [N x 4] shape.
        """
        anchors_xc = anchors[:, 0:1]
        anchors_yc = anchors[:, 1:2]
        anchors_w = anchors[:, 2:3]
        anchors_h = anchors[:, 3:4]
        anchors_xyxy = np.concatenate(
            [
                anchors_xc - anchors_w / 2,
                anchors_yc - anchors_h / 2,
                anchors_xc + anchors_w / 2,
                anchors_yc + anchors_h / 2,
            ],
            axis=1,
        )
        return anchors_xyxy

    def convert_anchors_xywh_to_yxyx(self, anchors: np.ndarray) -> np.ndarray:
        """
        Convert the provided anchors from xywh format into yxyx format.

        :param np.ndarray anchors: Anchors in the form of numpy array with shape
            as [N x 4]
        :return np.ndarray: Updated anchors in yxyx format in [N x 4] shape.
        """
        anchors_xc = anchors[:, 0:1]
        anchors_yc = anchors[:, 1:2]
        anchors_w = anchors[:, 2:3]
        anchors_h = anchors[:, 3:4]
        anchors_yxyx = np.concatenate(
            [
                anchors_yc - anchors_h / 2,
                anchors_xc - anchors_w / 2,
                anchors_yc + anchors_h / 2,
                anchors_xc + anchors_w / 2,
            ],
            axis=1,
        )
        return anchors_yxyx

    def check_and_update_anchors(self, nms_params: Dict, fmap_info: Dict) -> None:
        """
        Update the anchors as per the SSD model variant.
        Note: This will update the nms_params dict inplace.

        :param Dict nms_params: Dict containing useful NMS related parameters
            obtained from user via argparser.
        :param Dict fmap_info: Dict containing useful information related to
            feature maps.
        """
        anchors_yxyx = nms_params["anchor_data"]
        anchors_xywh = np.concatenate(
            [
                (anchors_yxyx[:, 1:2] + anchors_yxyx[:, 3:4]) / 2,  # xc
                (anchors_yxyx[:, 0:1] + anchors_yxyx[:, 2:3]) / 2,  # yc
                (anchors_yxyx[:, 3:4] - anchors_yxyx[:, 1:2]),  # w
                (anchors_yxyx[:, 2:3] - anchors_yxyx[:, 0:1]),  # h
            ],
            axis=1,
        )
        nms_params["anchor_data"] = anchors_xywh
        nms_params["total_boxes"] = anchors_xywh.shape[0]
        return

    def identify_boxes_and_scores(
        self, matched_nodes: List[List[NodeProto]]
    ) -> Tuple[bool, List]:
        """
        Identify boxes concat tensor and scores concat tensor from matched nodes
        of the model.

        :param List[List[NodeProto]] matched_nodes: List of matched nodes found
            from pattern matching.
        :return Tuple[bool, List]: Tuple of 2 results.
            - boolean value indicating whether the tensors are correctly
              identified or not.
            - List of below items
              - boxes concat tensor name
              - scores concat tensor name
              - boxes pattern nodes
              - scores pattern nodes
        """
        node_mapping = get_node_mappings(self.loader.model)
        boxes_pattern_nodes = None
        scores_pattern_nodes = None

        # In SSD kind of model, we will have same pattern for boxes and scores.
        # But, the patterns will have one common concat node for boxes and scores
        # each. To differentiate which matched_nodes are for boxes and which
        # ones are for scores, we will internally group the matched_nodes based
        # on that common concat node.
        grouped_node_lists = [
            [n.name for n in matched_nodes[0]]
        ]  # 1st pattern nodes are added.
        ref_node_list = set(
            [n.name for n in matched_nodes[0]]
        )  # Name of all the nodes in 1st pattern nodes.
        for node_list in matched_nodes[1:]:
            node_name_list = [n.name for n in node_list]
            common_nodes = ref_node_list.intersection(set(node_name_list))
            if len(common_nodes):
                # If there is a common node between current pattern nodes and
                # 1st pattern nodes then the current pattern nodes are similar
                # to 1st pattern nodes.
                grouped_node_lists.append(node_name_list)

        # Identify the common node amoung all the grouped nodes.
        intersection_set = set(grouped_node_lists[0])
        common_node_name = list(
            intersection_set.intersection(*map(set, grouped_node_lists))
        )
        if len(common_node_name) != 1:
            log_error(
                "There should be only one common node among the boxes/scores prediction nodes.."
            )
            return False, []
        # This common node can be for boxes or scores. We can differentiate
        # between these 2 by checking the shapes.
        common_node = node_mapping[common_node_name[0]]

        output_tensor_name = common_node.output[0]
        output_tensor = get_value_info_by_name(self.loader.model, output_tensor_name)
        if output_tensor is None:
            log_error(f"No shape information found for tensor: {output_tensor}.")
            return False, []
        output_shape = get_shape_from_value_info_proto(output_tensor)
        # boxes will be of shape: [B, N, 4] or [B, 4, N]
        # score will be of shape: [B, N, num_classes] or [B, num_classes, N]
        if self.nms_params["num_classes"] == 4:
            log_error(
                "Not able to differentiate between the boxes/scores "
                "prediction nodes in the given model. Please update the "
                "patterns to make this differentiation clear."
            )
            return False, []
        if len(output_shape) == 2:
            if 4 * self.nms_params["num_anchors"] == output_shape[1]:
                boxes_pattern_nodes = grouped_node_lists
                boxes_pattern_common_nodes = common_node
            elif (
                self.nms_params["num_classes"] * self.nms_params["num_anchors"]
                == output_shape[1]
            ):
                scores_pattern_nodes = grouped_node_lists
                scores_pattern_common_nodes = common_node
            else:
                log_error(
                    "Not able to identify the boxes/scores prediction nodes in the given model."
                )
                return False, []
        elif len(output_shape) == 3:
            if 4 in output_shape[1:]:
                boxes_pattern_nodes = grouped_node_lists
                boxes_pattern_common_nodes = common_node
            elif self.nms_params["num_classes"] in output_shape[1:]:
                scores_pattern_nodes = grouped_node_lists
                scores_pattern_common_nodes = common_node
            else:
                log_error(
                    "Not able to identify the boxes/scores prediction nodes in the given model."
                )
                return False, []
        else:
            log_error(
                "Not able to identify the boxes/scores prediction nodes in the given model."
            )
            return False, []

        # Now we have obained either boxes or scores tensor and their patterns.
        # The other one would be either scores or boxes tensor and patterns respectively.
        remaining_node_list = []
        for node_list in matched_nodes:
            node_name_list = [n.name for n in node_list]
            if node_name_list not in grouped_node_lists:
                remaining_node_list.append(node_name_list)

        intersection_set = set(remaining_node_list[0])
        common_node_name = list(
            intersection_set.intersection(*map(set, remaining_node_list))
        )
        if len(common_node_name) != 1:
            log_error(
                "There should be only one common node among the boxes/scores prediction nodes.."
            )
            return False, []
        common_node = node_mapping[common_node_name[0]]
        if boxes_pattern_nodes is None:
            boxes_pattern_nodes = remaining_node_list
            boxes_pattern_common_nodes = common_node
        elif scores_pattern_nodes is None:
            scores_pattern_nodes = remaining_node_list
            scores_pattern_common_nodes = common_node
        else:
            log_error(
                "Not able to identify the boxes/scores prediction nodes in the given model."
            )
            return False, []

        boxes_concat_tensor = get_value_info_by_name(
            self.loader.model, boxes_pattern_common_nodes.output[0]
        )
        scores_concat_tensor = get_value_info_by_name(
            self.loader.model, scores_pattern_common_nodes.output[0]
        )

        if (boxes_concat_tensor is None) or (scores_concat_tensor is None):
            log_error(
                "Not able to identify the boxes/scores prediction nodes in the given model."
            )
            return False, []

        if len(boxes_pattern_nodes) != len(scores_pattern_nodes):
            log_error(
                "Number of feature extractor branches are not same for boxes prediction and scores prediction."
            )
            return False, []

        return True, [
            boxes_concat_tensor,
            scores_concat_tensor,
            boxes_pattern_nodes,
            scores_pattern_nodes,
        ]

    def get_num_anchors(
        self, shape: List[int], coord_or_classes: int, tensor_type: str
    ) -> Tuple[bool, int]:
        """
        Get the number of anchors from the given shape and number of coordinate
        or number of classes.

        :param List[int] shape: Shape of the tensor to be checked.
        :param int coord_or_classes: Pass number of coordinates or number of
            class. This will be used to identify which axis is for anchors.
        :param str tensor_type: Type of the tensor which is being checked.
        :return Tuple[bool, int]: Tuple of 2 results.
            - boolean value indicating whether the anchors are found or not.
            - Number of anchors present in model.
        """
        if len(shape) == 2:
            if shape[1] == self.nms_params["num_anchors"] * coord_or_classes:
                return True, shape[1] // coord_or_classes
            else:
                log_error("Can not determine number of anchors for given model.")
                return False, None

        idx = [i for i, s in enumerate(shape[1:]) if s == coord_or_classes]
        if len(idx) > 1:
            if tensor_type == "boxes":
                log_error(
                    "Model has same number of anchors and number of coordinates, which is incompatible."
                )
            else:
                log_error(
                    "Model has same number of anchors and number of classes, which is incompatible."
                )
            return False, None
        elif len(idx) == 0:
            log_error(
                f"Failed to get the anchor index for {tensor_type} "
                f"tensor. The {tensor_type} shape should contain {coord_or_classes} value."
            )
            return False, None
        coord_or_classes_index = idx[0] + 1
        if len(shape) == 3:
            anchor_index = [
                i for i, _ in enumerate(shape) if i not in [0, coord_or_classes_index]
            ]
        elif len(shape) == 4:
            # Ignore the extra axis if the shape is [B, N, 1, 4]
            anchor_index = [
                i
                for i, s in enumerate(shape)
                if (i not in [0, coord_or_classes_index]) and (s != 1)
            ]
        else:
            log_error(
                f"Failed to get the anchor index for {tensor_type} "
                f"tensor. The {tensor_type} tensor should have 3d or 4d shape."
            )
            return False, None
        return True, shape[anchor_index[0]]

    def get_fmap_info(self, matched_nodes: List[List[NodeProto]]) -> Tuple[bool, Dict]:
        """
        Fetch SSD model specific feature map related information from the
        identified nodes.

        :param List[List[NodeProto]] matched_nodes: List of list of nodes that
            are found by pattern identification.
        :return Tuple[bool, Dict]: Tuple of two results.
            - Boolean status indicating whether the feature map information is
              identified correctly or not
            - Dict containing feature map related information.
        """
        feature_map_info = {}
        node_mapping = get_node_mappings(self.loader.model)

        status, identified_tensors = self.identify_boxes_and_scores(matched_nodes)
        if not status:
            return False, {}
        [boxes_tensor, scores_tensor, boxes_pattern_nodes, scores_pattern_nodes] = (
            identified_tensors
        )

        boxes_concat_shape = get_shape_from_value_info_proto(boxes_tensor)
        scores_concat_shape = get_shape_from_value_info_proto(scores_tensor)

        status, num_anchors_in_boxes = self.get_num_anchors(
            boxes_concat_shape, 4, "boxes"
        )
        if not status:
            return False, {}
        status, num_anchors_in_scores = self.get_num_anchors(
            scores_concat_shape, self.nms_params["num_classes"], "scores"
        )
        if not status:
            return False, {}

        if num_anchors_in_boxes != num_anchors_in_scores:
            log_error(
                "Number of anchors found in all box predictions and all score predictions is not same."
            )
            return False, {}

        if num_anchors_in_boxes != self.nms_params["num_anchors"]:
            log_error(
                "Number of anchors found in all model is not same as provided anchors."
            )
            return False, {}

        feature_map_info = {}
        for box_node_names in boxes_pattern_nodes:
            conv_output_tensor_name = node_mapping[box_node_names[0]].output[0]
            conv_output_tensor = get_value_info_by_name(
                self.loader.model, conv_output_tensor_name
            )
            if conv_output_tensor is None:
                log_error(f"No shape available for tensor: {conv_output_tensor}.")
                return False, {}
            conv_output_tensor_shape = get_shape_from_value_info_proto(
                conv_output_tensor
            )
            # Assuming NCHW layout for conv output shape.
            if len(conv_output_tensor_shape) != 4:
                log_error(
                    f"Shape for the {conv_output_tensor_shape} shall be of rank 4."
                )
                return False, {}
            feature_map_info[conv_output_tensor_name] = {
                "shape": conv_output_tensor_shape,
                "h": conv_output_tensor_shape[2],
                "w": conv_output_tensor_shape[3],
            }
        return True, feature_map_info

    def get_model_info(self, matched_nodes: List[List[NodeProto]]) -> Tuple[bool, Dict]:
        """
        Fetch model related information from the identified nodes.

        :param List[List[NodeProto]] matched_nodes: List of list of nodes that
            are found by pattern identification.
        :return Tuple[bool, Dict]: Tuple of two results.
            - Boolean status indicating whether the model information is
              identified correctly or not.
            - Dict containing feature map and model's input related information.
        """
        self.loader.utils.native_shape_inference()
        status, image_input_dict = get_image_input_details(self.loader)
        if not status:
            log_error("Failed to get the image input from model.")
            return False, None
        fmap_status, feature_map_info = self.get_fmap_info(matched_nodes)
        if not fmap_status:
            log_error("Failed to extract feature extractor details.")
            return False, None
        model_info = {"fmap_info": feature_map_info, "input_info": image_input_dict}
        return True, model_info

    def add_abp_nms_nodes(
        self, loader: FrameworkModelLoader, model_info: Dict
    ) -> Tuple[bool, FrameworkModelLoader]:
        """
        Prepare ssd based models by adding anchor box processing nodes and NMS
        node in the graph.

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :param Dict model_info: Dict containing useful information related to
            feature maps and model's input.
        :return Tuple[bool, FrameworkModelLoader]: Tuple of two results.
            - First value represents the status of the subgraph generation.
            - Second value represent the updated onnx model loader.
        """
        fmap_info = model_info["fmap_info"]
        self.check_and_update_anchors(self.nms_params, fmap_info)

        def get_tensor_shape(tensor_name: str) -> List[int]:
            """
            Get the shape of the tensor from the tensor name.

            :param str tensor_name: Name of the tensor.
            :return List[int]: Tensor shape.
            """
            tensor = get_value_info_by_name(loader.model, tensor_name)
            tensor_shape = get_shape_from_value_info_proto(tensor)
            return tensor_shape

        def get_boxes_and_scores_tensors(model_outputs: List[str]) -> Tuple[bool, List]:
            """
            Differentiate between boxes and scores tensor from the given model outputs.

            :param List[str] model_outputs: List of model outputs which contains
                boxes tensor and scores tensor.
            :return Tuple[bool, List]: Tuple of two results.
                - boolean status indicating whether boxes and scores tensors
                  can be differentiated successfully or not.
                - List of boxes tensor and scores tensor.
            """
            if len(model_outputs) != 2:
                log_error(
                    "Model should have two output tensors after subgraph "
                    + "extraction. One for box prediction and other for scores."
                )
                return False, []
            tensor_1_name, tensor_2_name = model_outputs
            tensor_1_shape = get_tensor_shape(tensor_1_name)
            tensor_2_shape = get_tensor_shape(tensor_2_name)
            if (len(tensor_1_shape) == 2) and (len(tensor_2_shape) == 2):
                if (tensor_1_shape[1] == 4 * self.nms_params["num_anchors"]) and (
                    tensor_2_shape[1]
                    == self.nms_params["num_classes"] * self.nms_params["num_anchors"]
                ):
                    return True, [tensor_1_name, tensor_2_name]
                elif (tensor_2_shape[1] == 4 * self.nms_params["num_anchors"]) and (
                    tensor_1_shape[1]
                    == self.nms_params["num_classes"] * self.nms_params["num_anchors"]
                ):
                    return True, [tensor_2_name, tensor_1_name]
                else:
                    log_error(
                        "Not able to differentiate between boxes and scores tensors."
                    )
                    return False, []

            elif (len(tensor_1_shape) in [3, 4]) and (len(tensor_2_shape) in [3, 4]):
                if (4 in tensor_1_shape[1:]) and (4 not in tensor_2_shape[1:]):
                    return True, [tensor_1_name, tensor_2_name]
                elif (4 not in tensor_1_shape[1:]) and (4 in tensor_2_shape[1:]):
                    return True, [tensor_2_name, tensor_1_name]
                else:
                    log_error(
                        "Not able to differentiate between boxes and scores tensors."
                    )
                    return False, []
            else:
                log_error("Not able to differentiate between boxes and scores tensors.")
                return False, []

        output_names = loader.get_output_names()
        status, updated_output_names = get_boxes_and_scores_tensors(output_names)
        if not status:
            return False, loader
        boxes_tensor, scores_tensor = updated_output_names
        boxes_shape = get_tensor_shape(boxes_tensor)
        scores_shape = get_tensor_shape(scores_tensor)
        boxes_squeeze_index = None
        boxes_reshape_new_shape = None
        scores_squeeze_index = None
        scores_reshape_new_shape = None
        if len(boxes_shape) == 4:
            if 1 in boxes_shape[1:]:
                boxes_squeeze_index = boxes_shape[1:].index(1) + 1
            else:
                log_error("Incorrect shape found for boxes tensor.")
                return False, loader
        elif len(boxes_shape) == 2:
            if 4 * self.nms_params["num_anchors"] == boxes_shape[1]:
                boxes_reshape_new_shape = [
                    boxes_shape[0],
                    self.nms_params["num_anchors"],
                    4,
                ]
            else:
                log_error("Incorrect shape found for boxes tensor.")
                return False, loader

        if len(scores_shape) == 4:
            if 1 in scores_shape[1:]:
                scores_squeeze_index = scores_shape[1:].index(1) + 1
            else:
                log_error("Incorrect shape found for scores tensor.")
                return False, loader
        elif len(scores_shape) == 2:
            if (
                scores_shape[1]
                == self.nms_params["num_classes"] * self.nms_params["num_anchors"]
            ):
                scores_reshape_new_shape = [
                    boxes_shape[0],
                    self.nms_params["num_anchors"],
                    self.nms_params["num_classes"],
                ]
            else:
                log_error("Incorrect shape found for scores tensor.")
                return False, loader

        if boxes_reshape_new_shape is not None:
            reshape_node, initializers = loader.utils.create_node(
                [boxes_tensor],
                {
                    "op_type": "Reshape",
                    "shape": boxes_reshape_new_shape,
                    "allowzero": 0,
                },
                self.model_opset,
            )
            loader.utils.add_node(reshape_node)
            loader.utils.add_initializers(initializers)
            boxes_tensor = reshape_node.output[0]
            boxes_shape = boxes_reshape_new_shape
        if scores_reshape_new_shape is not None:
            reshape_node, initializers = loader.utils.create_node(
                [scores_tensor],
                {
                    "op_type": "Reshape",
                    "shape": scores_reshape_new_shape,
                    "allowzero": 0,
                },
                self.model_opset,
            )
            loader.utils.add_node(reshape_node)
            loader.utils.add_initializers(initializers)
            scores_tensor = reshape_node.output[0]
            scores_shape = scores_reshape_new_shape

        if 4 in boxes_shape[1:]:
            boxes_coordinate_index = boxes_shape[1:].index(4) + 1
        else:
            log_error("Incorrect shape found for boxes tensor.")
            return False, loader
        if self.nms_params["num_anchors"] in boxes_shape[1:]:
            boxes_anchor_index = (
                boxes_shape[1:].index(self.nms_params["num_anchors"]) + 1
            )
        else:
            log_error("Incorrect shape found for boxes tensor.")
            return False, loader

        if self.nms_params["num_classes"] in scores_shape[1:]:
            scores_classes_index = (
                scores_shape[1:].index(self.nms_params["num_classes"]) + 1
            )
        else:
            log_error("Incorrect shape found for scores tensor.")
            return False, loader
        if self.nms_params["num_anchors"] in scores_shape[1:]:
            scores_anchor_index = (
                scores_shape[1:].index(self.nms_params["num_anchors"]) + 1
            )
        else:
            log_error("Incorrect shape found for scores tensor.")
            return False, loader

        # boxes  : [B x 4 x N] or [B x 4 x 1 x N] or [B x N x 4] or any such combination
        # scores : [B x 81 x N] or [B x N x 81] or [B x 1 x N x 81] or any such combination

        if boxes_squeeze_index is not None:
            squeeze_node, initializers = loader.utils.create_node(
                [boxes_tensor],
                {"op_type": "Squeeze", "axes": [boxes_squeeze_index]},
                self.model_opset,
            )
            loader.utils.add_node(squeeze_node)
            loader.utils.add_initializers(initializers)
            boxes_tensor = squeeze_node.output[0]
            boxes_shape = (
                boxes_shape[:boxes_squeeze_index]
                + boxes_shape[(boxes_squeeze_index + 1) :]
            )
        if boxes_shape[1:] != [self.nms_params["num_anchors"], 4]:
            permute_index = [0, boxes_anchor_index, boxes_coordinate_index]
            transpose_node, initializers = loader.utils.create_node(
                [boxes_tensor],
                {"op_type": "Transpose", "perm": permute_index},
                self.model_opset,
            )
            loader.utils.add_node(transpose_node)
            loader.utils.add_initializers(initializers)
            boxes_tensor = transpose_node.output[0]
            boxes_shape = [
                boxes_shape[i] for i in [0, boxes_anchor_index, boxes_coordinate_index]
            ]

        if scores_squeeze_index is not None:
            squeeze_node, initializers = loader.utils.create_node(
                [scores_tensor],
                {"op_type": "Squeeze", "axes": [scores_squeeze_index]},
                self.model_opset,
            )
            loader.utils.add_node(squeeze_node)
            loader.utils.add_initializers(initializers)
            scores_tensor = squeeze_node.output[0]
            scores_shape = (
                scores_shape[:scores_squeeze_index]
                + scores_shape[(scores_squeeze_index + 1) :]
            )
        if scores_shape[1:] != [
            self.nms_params["num_anchors"],
            self.nms_params["num_classes"],
        ]:
            permute_index = [0, scores_anchor_index, scores_classes_index]
            transpose_node, initializers = loader.utils.create_node(
                [scores_tensor],
                {"op_type": "Transpose", "perm": permute_index},
                self.model_opset,
            )
            loader.utils.add_node(transpose_node)
            loader.utils.add_initializers(initializers)
            scores_tensor = transpose_node.output[0]
            scores_shape = [
                scores_shape[i] for i in [0, scores_anchor_index, scores_classes_index]
            ]

        # boxes : [B x N x 4]
        # scores: [B x N x 81]
        activation_type = self.nms_params["scores_activation"]
        act_dict = {"op_type": activation_type}
        if activation_type == "Softmax":
            act_dict["axis"] = -1
        # FIXME: Add a correct value of axis everywhere instead of -1.

        act_node, initializers = loader.utils.create_node(
            [scores_tensor], act_dict, self.model_opset
        )
        loader.utils.add_node(act_node)
        loader.utils.add_initializers(initializers)
        scores_tensor = act_node.output[0]

        if self.nms_params["arch_type"] not in self.yxhw_prediction_map:
            log_error("No information found for raw detection ordering.")
            return False, loader
        use_yxhw_input = self.yxhw_prediction_map[self.nms_params["arch_type"]]

        # boxes: [1, num_boxes, 4]
        # scores: [1, num_boxes, num_classes]
        final_outputs = []
        if self.nms_params["nms_type"] == NMSType.HOST:
            if use_yxhw_input and (
                self.nms_params["arch_type"] != ModelArchType.EFFICIENTDET
            ):
                split_node, initializers = loader.utils.create_node(
                    [boxes_tensor],
                    {
                        "op_type": "Split",
                        "split": [1, 1, 1, 1],
                        "axis": 2,
                        "num_outputs": 4,
                    },
                    self.model_opset,
                )
                loader.utils.add_node(split_node)
                loader.utils.add_initializers(initializers)
                boxes_y, boxes_x, boxes_h, boxes_w = split_node.output

                concat_node_scores, initializers = loader.utils.create_node(
                    [boxes_x, boxes_y, boxes_w, boxes_h],
                    {"op_type": "Concat", "axis": -1},
                    self.model_opset,
                )
                loader.utils.add_node(concat_node_scores)
                loader.utils.add_initializers(initializers)
                boxes_tensor = concat_node_scores.output[0]

            # Return as the feature maps are now as per Host NMS requirement.
            final_outputs.extend([boxes_tensor, scores_tensor])
        elif self.nms_params["nms_type"] == NMSType.DEVICE:
            if self.nms_params["background_class_idx"] is not None:
                if self.nms_params["background_class_idx"] == 0:
                    slice_node, initializers = loader.utils.create_node(
                        [scores_tensor],
                        {
                            "op_type": "Slice",
                            "starts": [self.nms_params["background_class_idx"] + 1],
                            "ends": [scores_shape[-1]],
                            "axes": [-1],
                            "steps": [1],
                        },
                        self.model_opset,
                    )
                    loader.utils.add_node(slice_node)
                    loader.utils.add_initializers(initializers)
                    scores_tensor = slice_node.output[0]
                elif (
                    self.nms_params["background_class_idx"]
                    == self.nms_params["num_classes"] - 1
                ):
                    slice_node, initializers = loader.utils.create_node(
                        [scores_tensor],
                        {
                            "op_type": "Slice",
                            "starts": [0],
                            "ends": [scores_shape[-1] - 1],
                            "axes": [-1],
                            "steps": [1],
                        },
                        self.model_opset,
                    )
                    loader.utils.add_node(slice_node)
                    loader.utils.add_initializers(initializers)
                    scores_tensor = slice_node.output[0]
                else:
                    slice_node_1, initializers = loader.utils.create_node(
                        [scores_tensor],
                        {
                            "op_type": "Slice",
                            "starts": [0],
                            "ends": [self.nms_params["background_class_idx"]],
                            "axes": [-1],
                            "steps": [1],
                        },
                        self.model_opset,
                    )
                    loader.utils.add_node(slice_node_1)
                    loader.utils.add_initializers(initializers)
                    slice_node_1_output = slice_node_1.output[0]

                    slice_node_2, initializers = loader.utils.create_node(
                        [scores_tensor],
                        {
                            "op_type": "Slice",
                            "starts": [self.nms_params["background_class_idx"] + 1],
                            "ends": [scores_shape[-1]],
                            "axes": [-1],
                            "steps": [1],
                        },
                        self.model_opset,
                    )
                    loader.utils.add_node(slice_node_2)
                    loader.utils.add_initializers(initializers)
                    slice_node_2_output = slice_node_2.output[0]

                    concat_node_scores, initializers = loader.utils.create_node(
                        [slice_node_1_output, slice_node_2_output],
                        {"op_type": "Concat", "axis": -1},
                        self.model_opset,
                    )
                    loader.utils.add_node(concat_node_scores)
                    loader.utils.add_initializers(initializers)
                    scores_tensor = concat_node_scores.output[0]

            split_node, initializers = loader.utils.create_node(
                [boxes_tensor],
                {"op_type": "Split", "split": [2, 2], "axis": 2, "num_outputs": 2},
                self.model_opset,
            )
            loader.utils.add_node(split_node)
            loader.utils.add_initializers(initializers)
            boxes_xy, boxes_wh = split_node.output

            # FIXME: If scale_xy is 1.0 then we dont need this node. Do the same for all the other arithmatic nodes.
            mul_node_boxes, initializers = loader.utils.create_node(
                [boxes_xy],
                {
                    "op_type": "Mul",
                    "B": np.array(self.nms_params["scale_xy"], dtype=np.float32),
                },
                self.model_opset,
            )
            loader.utils.add_node(mul_node_boxes)
            loader.utils.add_initializers(initializers)
            boxes_xy = mul_node_boxes.output[0]

            mul_node_boxes, initializers = loader.utils.create_node(
                [boxes_wh],
                {
                    "op_type": "Mul",
                    "B": np.array(self.nms_params["scale_wh"], dtype=np.float32),
                },
                self.model_opset,
            )
            loader.utils.add_node(mul_node_boxes)
            loader.utils.add_initializers(initializers)
            boxes_wh = mul_node_boxes.output[0]

            anchor_xy = self.nms_params["anchor_data"][:, 0:2]
            anchor_wh = self.nms_params["anchor_data"][:, 2:4]

            anchor_xy = (
                np.concatenate([anchor_xy[:, 1:2], anchor_xy[:, 0:1]], axis=1)
                if use_yxhw_input
                else anchor_xy
            )  # Convert XY into YX
            anchor_wh = (
                np.concatenate([anchor_wh[:, 1:2], anchor_wh[:, 0:1]], axis=1)
                if use_yxhw_input
                else anchor_wh
            )  # Convert WH into HW

            mul_node_boxes, initializers = loader.utils.create_node(
                [boxes_xy], {"op_type": "Mul", "B": anchor_wh}, self.model_opset
            )
            loader.utils.add_node(mul_node_boxes)
            loader.utils.add_initializers(initializers)
            boxes_xy = mul_node_boxes.output[0]

            add_node_boxes, initializers = loader.utils.create_node(
                [boxes_xy], {"op_type": "Add", "B": anchor_xy}, self.model_opset
            )
            loader.utils.add_node(add_node_boxes)
            loader.utils.add_initializers(initializers)
            boxes_xy = add_node_boxes.output[0]

            exp_node, initializers = loader.utils.create_node(
                [boxes_wh], {"op_type": "Exp"}, self.model_opset
            )
            loader.utils.add_node(exp_node)
            loader.utils.add_initializers(initializers)
            boxes_wh = exp_node.output[0]

            mul_node_boxes, initializers = loader.utils.create_node(
                [boxes_wh], {"op_type": "Mul", "B": anchor_wh}, self.model_opset
            )
            loader.utils.add_node(mul_node_boxes)
            loader.utils.add_initializers(initializers)
            boxes_wh = mul_node_boxes.output[0]

            # FIXME: This mul node can be fused with above mul node.
            mul_node_boxes, initializers = loader.utils.create_node(
                [boxes_wh],
                {"op_type": "Mul", "B": np.array([0.5], dtype=np.float32)},
                self.model_opset,
            )
            loader.utils.add_node(mul_node_boxes)
            loader.utils.add_initializers(initializers)
            boxes_wh = mul_node_boxes.output[0]

            sub_node_x1y1, initializers = loader.utils.create_node(
                [boxes_xy, boxes_wh], {"op_type": "Sub"}, self.model_opset
            )
            loader.utils.add_node(sub_node_x1y1)
            loader.utils.add_initializers(initializers)
            boxes_x1y1 = sub_node_x1y1.output[0]

            add_node_x2y2, initializers = loader.utils.create_node(
                [boxes_xy, boxes_wh], {"op_type": "Add"}, self.model_opset
            )
            loader.utils.add_node(add_node_x2y2)
            loader.utils.add_initializers(initializers)
            boxes_x2y2 = add_node_x2y2.output[0]

            if use_yxhw_input:
                # boxes_x1y1 and boxes_x2y2 are in the yx format. We need the boxes in yxyx format. So just concat them.
                concat_node_boxes, initializers = loader.utils.create_node(
                    [boxes_x1y1, boxes_x2y2],
                    {"op_type": "Concat", "axis": -1},
                    self.model_opset,
                )
                loader.utils.add_node(concat_node_boxes)
                loader.utils.add_initializers(initializers)
                boxes_tensor = concat_node_boxes.output[0]
                # boxes_tensor : [B x N x 4]
            else:
                split_node, initializers = loader.utils.create_node(
                    [boxes_x1y1],
                    {"op_type": "Split", "split": [1, 1], "axis": -1, "num_outputs": 2},
                    self.model_opset,
                )
                loader.utils.add_node(split_node)
                loader.utils.add_initializers(initializers)
                boxes_x1, boxes_y1 = split_node.output

                split_node, initializers = loader.utils.create_node(
                    [boxes_x2y2],
                    {"op_type": "Split", "split": [1, 1], "axis": -1, "num_outputs": 2},
                    self.model_opset,
                )
                loader.utils.add_node(split_node)
                loader.utils.add_initializers(initializers)
                boxes_x2, boxes_y2 = split_node.output

                concat_node_boxes, initializers = loader.utils.create_node(
                    [boxes_y1, boxes_x1, boxes_y2, boxes_x2],
                    {"op_type": "Concat", "axis": -1},
                    self.model_opset,
                )
                loader.utils.add_node(concat_node_boxes)
                loader.utils.add_initializers(initializers)
                boxes_tensor = concat_node_boxes.output[0]
                # boxes_tensor : [B x N x 4]

            unsqueeze_node, initializers = loader.utils.create_node(
                [boxes_tensor], {"op_type": "Unsqueeze", "axes": [2]}, self.model_opset
            )
            loader.utils.add_node(unsqueeze_node)
            loader.utils.add_initializers(initializers)
            boxes_tensor = unsqueeze_node.output[0]
            # boxes_tensor : [B x N x 1 x 4]

            if self.nms_params["class_specific_nms"]:
                # 'repeats' input is used for tile opset >= 6
                # 'tiles' and 'axis' inputs are used for tile opset=1
                num_tiles = (
                    self.nms_params["num_classes"]
                    if self.nms_params["background_class_idx"] is None
                    else (self.nms_params["num_classes"] - 1)
                )
                tile_node, initializers = loader.utils.create_node(
                    [boxes_tensor],
                    {
                        "op_type": "Tile",
                        "repeats": [1, 1, num_tiles, 1],
                        "tiles": num_tiles,
                        "axis": 2,
                    },
                    self.model_opset,
                )
                # Total number of tiles are num_classes - 1 because we need to
                # ignore the background class present in ssd model.
                loader.utils.add_node(tile_node)
                loader.utils.add_initializers(initializers)
                boxes_tensor = tile_node.output[0]
                # boxes_tensor : [B x N x C x 4]

            loader.utils.native_shape_inference()
            comb_nms_outputs = self.add_combined_nms_node(
                loader,
                [boxes_tensor, scores_tensor],
                model_info["input_info"]["image_bs"],
            )
            final_outputs = comb_nms_outputs

            # Add the qnms domain into the model's opset so that it will be identified
            # as a custom op by onnx.
            qnms_opset = onnx.helper.make_operatorsetid(
                COMBINEDNMS_DOMAIN, COMBINEDNMS_OPSET
            )
            loader.model.opset_import.append(qnms_opset)

        # No need to infer shapes for Device NMS as we have already inferred
        # the shape and added a custom op and its shape manually. We dont want
        # to lose the shapes of that custom op.
        if self.nms_params["nms_type"] == NMSType.HOST:
            loader.utils.native_shape_inference()

        # Add new outputs
        loader.utils.add_outputs(final_outputs, infer_shape=False)
        # Remove existing outputs
        outputs_to_be_removed = [
            output_name
            for output_name in loader.get_output_names()
            if output_name not in final_outputs
        ]
        loader.utils.remove_outputs(outputs_to_be_removed)

        loader.utils.clean_model().topological_sort()
        loader.utils.native_shape_inference()
        try:
            loader.native_checker()
        except Exception as e:
            log_error(f"Generated onnx model is invalid due to exception : {e} ")
            return False, loader
        return True, loader

    def get_output_info(self, loader: FrameworkModelLoader) -> Tuple[bool, List]:
        """
        Obtain the output info of the SSD model after performing output related
        validations.

        :param FrameworkModelLoader loader: Onnx model loader reference.
        :return Tuple[bool, List]: Tuple of 2 results.
            - boolean value indicating whether the output info is obtained or not.
            - List of below items
              - boxes concat tensor name
              - scores concat tensor name
              - boxes concat tensor layout
              - scores concat tensor layout
        """
        output_info = loader.get_output_info()
        boxes_tensor_name = None
        scores_tensor_name = None
        for tensor_name, tensor_info in output_info.items():
            shape = tensor_info.shape
            # Host NMS supports NCD, NDC, NCHW, NHWC layout only for 3d/4d tensors coming out of SSD models.
            # NCD -> [batch, predictions, num_boxes]
            # NDC -> [batch, num_boxes, predictions]
            # The outputs of SSD models are already made in NDC layout.
            if len(shape) != 3:
                log_error(
                    "For SSD models with Host NMS variant, should produce 3d outputs with NCD or NDC layouts."
                )
                return False, None
            if shape[1] != self.nms_params["total_boxes"]:
                log_error(
                    f"For SSD models with Host NMS variant, should have {self.nms_params['total_boxes']} shape at 2nd axis but got {shape[1]}."
                )
                return False, None
            if (shape[2] != (self.nms_params["num_classes"])) and (shape[2] != 4):
                log_error(
                    f"For SSD models with Host NMS variant, should have {self.nms_params['num_classes']} or 4 shape at 3rd axis but got {shape[2]}."
                )
                return False, None
            if shape[2] == 4:
                boxes_tensor_name = tensor_name
            else:
                scores_tensor_name = tensor_name
        return True, [boxes_tensor_name, scores_tensor_name, "NDC", "NDC"]

    def generate_hostnms_config(
        self, loader: FrameworkModelLoader, model_info: Dict
    ) -> Tuple[bool, str]:
        """
        Generate the host nms config and dump it on the output directory.

        :param FrameworkModelLoader loader: Onnx model loader reference.
        :param Dict model_info: Dict containing useful information related to
            feature maps and model's input.
        :return Tuple[bool, str]: Tuple of two results.
            - First value indicates the status of the operation
            - Second value indicates the path at which the config is saved.
        """
        if self.nms_params["arch_type"] not in self.host_nms_anchor_transforms:
            log_error(
                f"No anchor transform found for given arch_type '{self.nms_params['arch_type']}'"
            )
            return False, None
        host_nms_anchors_transform = self.host_nms_anchor_transforms[
            self.nms_params["arch_type"]
        ]
        self.nms_params["anchor_data"] = host_nms_anchors_transform(
            self.nms_params["anchor_data"]
        )

        status, hostnms_config = self.generate_hostnms_common_config(loader, model_info)
        if not status:
            return False, None

        status, output_info = self.get_output_info(loader)
        if not status:
            return False, None
        [boxes_output, scores_output, boxes_layout, scores_layout] = output_info

        # Remove "/" or "." from output names as these names will change at the
        # time of serialization.
        hostnms_config["bbox-output-list"] = self.cleanup_output_names([boxes_output])
        hostnms_config["score-output-list"] = self.cleanup_output_names([scores_output])

        hostnms_config["layout"] = boxes_layout

        hostnms_config["do-softmax"] = (
            False  # Since we already applied Softmax/Sigmoid activation
        )
        hostnms_config["background-class-idx"] = self.nms_params["background_class_idx"]

        output_names = [scores_output, boxes_output]
        output_shapes = []
        output_info = loader.get_output_info()
        for output_name in output_names:
            if output_name not in output_info:
                log_error(f"Tensor {output_name} shall be the output of the model.")
                return False, None
            output_shapes.append(output_info[output_name].shape)

        hostnms_yaml_path = os.path.join(
            os.path.dirname(self.nms_params["dlc_path"]),
            os.path.splitext(loader.model_wrapper.model_name)[0]
            + "_hostnms_config.yaml",
        )
        dump_yaml(hostnms_config, hostnms_yaml_path)
        log_debug1(f"Host NMS config file dumped at: {hostnms_yaml_path}")
        return True, hostnms_yaml_path
