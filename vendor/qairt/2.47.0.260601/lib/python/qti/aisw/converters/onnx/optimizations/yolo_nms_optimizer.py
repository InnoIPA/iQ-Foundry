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
    get_supported_anchor_free_models,
    get_yolo_fmap_hwa_index,
    make_grid,
)
from qti.aisw.converters.onnx.util import (
    get_node_by_output_name,
    get_shape,
    get_variable_inputs,
)


class OnnxYoloNmsOptimizer(OnnxBaseNMSOptimizer):
    """
    OnnxYoloNmsOptimizer class responsible for applying NMS Optimization to SSD
    kind of models.
    """

    def __init__(self, loader: FrameworkModelLoader, nms_params: Dict):
        """
        Constructor for Yolo NMS Optimizer.

        :param FrameworkModelLoader loader: Onnx model loader reference.
        :param Dict nms_params: Dict containing useful NMS related parameters
            obtained from user via argparser.
        :raises RuntimeError: If the user provided NMS params or anchors are invalid.
        """
        super().__init__(loader, nms_params)
        self.patterns = patterns["yolo"]
        self.abp_decode_fn = {
            ModelArchType.YOLOV2: self.yolov2v3v4_decode,
            ModelArchType.YOLOV3: self.yolov2v3v4_decode,
            ModelArchType.YOLOV4: self.yolov2v3v4_decode,
            ModelArchType.YOLOV5: self.yolov5v7_decode,
            ModelArchType.YOLOV7: self.yolov5v7_decode,
            ModelArchType.YOLOX: self.yolox_decode,
        }
        self.host_nms_anchor_transforms = {
            ModelArchType.YOLOV2: self.multiply_anchors_by_stride,
            ModelArchType.YOLOV3: lambda anchor, _: anchor,
            ModelArchType.YOLOV4: lambda anchor, _: anchor,
            ModelArchType.YOLOV5: lambda anchor, _: anchor,
            ModelArchType.YOLOV7: lambda anchor, _: anchor,
            ModelArchType.YOLOX: lambda anchor, _: anchor,
        }
        self.host_nms_do_softmax = {
            ModelArchType.YOLOV2: True,
            ModelArchType.YOLOV3: False,
            ModelArchType.YOLOV4: False,
            ModelArchType.YOLOV5: False,
            ModelArchType.YOLOV7: False,
            ModelArchType.YOLOX: False,
        }

    def multiply_anchors_by_stride(
        self, anchors: np.ndarray, fmap_info: Dict
    ) -> np.ndarray:
        """
        Update the anchors by strides of respective layers.
        Note: Anchors are assumed to be in increamental order e.g. first pair of
        anchors are to be used for largest size feature map (smaller stride) and
        so on.

        :param np.ndarray anchors: Anchors in the form of numpy array with shape
            as [num_layers x num_anchors_per_layers x 4]
        :param Dict fmap_info: Dict containing useful information related to
            feature maps.
        :raises RuntimeError: If model's feature maps and layers found from
            anchors are not equal.
        :return np.ndarray: Updated anchors in the same shape as before.
        """
        strides = [
            [fmap_dict["stride_w"], fmap_dict["stride_h"]]
            for fmap_dict in fmap_info.values()
        ]
        sorted_strides = sorted(strides, key=lambda stride: stride[0] * stride[1])

        num_layers = anchors.shape[0]
        if len(strides) != num_layers:
            raise RuntimeError(
                f"Number of feature maps and number of layers found from anchors are different for {self.nms_params['arch_type']}."
            )
        num_anchors = anchors.shape[1]
        for l in range(num_layers):
            for a in range(num_anchors):
                anchors[l][a] *= sorted_strides[l]
        return anchors

    def check_and_update_anchors(self, nms_params: Dict, fmap_info: Dict) -> None:
        """
        Update the anchors as per the Yolo model variant.
        Note: This function will update fmap_info with anchors assigned to
        each feature map.

        :param Dict nms_params: Dict containing useful NMS related parameters
            obtained from user via argparser.
        :param Dict fmap_info: Dict containing useful information related to
            feature maps.
        :raises RuntimeError: If anchors are of incorrect shapes.
        """
        if self.nms_params["arch_type"] in get_supported_anchor_free_models():
            return

        num_layers = len(fmap_info)

        fmap_names = list(fmap_info.keys())
        fmap_h_list = np.array(
            [fmap_info[name]["shape"][fmap_info[name]["h_idx"]] for name in fmap_names]
        )
        desc_sorted_fmap_h_idx = np.argsort(-fmap_h_list)
        desc_sorted_fmap_names = [fmap_names[i] for i in desc_sorted_fmap_h_idx]

        num_anchors_per_fmap = nms_params["num_anchors_per_fmap"]
        if nms_params["anchor_data"].size != num_anchors_per_fmap * num_layers * 2:
            raise RuntimeError(
                f"Provided anchors are incompatible. Anchors should be of shape {[num_anchors_per_fmap, num_layers, 2]}."
            )

        nms_params["anchor_data"] = nms_params["anchor_data"].reshape(
            num_layers, num_anchors_per_fmap, 2
        )

        # We can safely assume that the user provided anchors are usually in
        # ascending order of their size. The smallest anchor is used by
        # feature map of largest size and largest anchor is used by feature
        # map of smallest size. Hence we will shuffle the anchors and bind
        # respective anchors to respective layers.
        for i, desc_sorted_fmap_name in enumerate(desc_sorted_fmap_names):
            fmap_anchors = nms_params["anchor_data"][i : i + 1]
            fmap_anchors = fmap_anchors.reshape(1, num_anchors_per_fmap, 1, 1, 2)
            fmap_info[desc_sorted_fmap_name]["anchor"] = fmap_anchors

    def get_fmap_info(
        self, image_h: int, image_w: int, matched_nodes: List[List[NodeProto]]
    ) -> Tuple[bool, Dict]:
        """
        Fetch Yolo model specific feature map related information from the
        identified nodes.

        :param int image_h: Height of the image input of model.
        :param int image_w: Width of the image input of model.
        :param List[List[NodeProto]] matched_nodes: List of list of nodes that
            are found by registered patterns.
        :return Tuple[bool, Dict]: Tuple of two results.
            - Boolean status indicating whether the feature map information is
              identified correctly or not
            - Dict containing feature map related information.
        """
        feature_map_info = {}
        total_feature_extractors = len(matched_nodes)
        if self.nms_params["arch_type"] in get_supported_anchor_free_models():
            num_anchors_per_fmap = 1
        else:
            num_anchors_per_fmap = (
                self.nms_params["num_anchors"] // total_feature_extractors
            )
        self.nms_params["num_anchors_per_fmap"] = num_anchors_per_fmap

        for matched_fmap in matched_nodes:
            start_matched_node = matched_fmap[0]
            end_matched_node = matched_fmap[-1]
            start_tensor_name = get_variable_inputs(
                self.loader.model, start_matched_node
            )
            start_tensor_name = start_tensor_name[0]
            end_tensor_name = end_matched_node.output[0]
            try:
                start_tensor_shape = get_shape(self.loader.model, start_tensor_name)
            except Exception as e:
                log_error(
                    f"Shape not found for tensor '{start_tensor_name}'. Exception: {e}"
                )
                return False, None
            try:
                end_tensor_shape = get_shape(self.loader.model, end_tensor_name)
            except Exception as e:
                log_error(
                    f"Shape not found for tensor '{end_tensor_name}'. Exception: {e}"
                )
                return False, None

            status, index_dict = get_yolo_fmap_hwa_index(
                start_matched_node,
                start_tensor_name,
                start_tensor_shape,
                end_matched_node,
                end_tensor_name,
                end_tensor_shape,
                num_anchors_per_fmap,
                self.nms_params["num_classes"],
            )
            if not status:
                return False, None
            h_idx = index_dict["h_idx"]
            w_idx = index_dict["w_idx"]
            stride_h = image_h // end_tensor_shape[h_idx]
            stride_w = image_w // end_tensor_shape[w_idx]
            feature_map_info[end_tensor_name] = {
                "shape": end_tensor_shape,
                "stride_h": stride_h,
                "stride_w": stride_w,
            }
            feature_map_info[end_tensor_name] = {
                **feature_map_info[end_tensor_name],
                **index_dict,
            }
        return True, feature_map_info

    def get_model_info(self, matched_nodes: List[List[NodeProto]]) -> Tuple[bool, Dict]:
        """
        Fetch feature map related information from the identified nodes.

        :param List[List[NodeProto]] matched_nodes: List of list of nodes that
            are found by registered patterns.
        :return Tuple[bool, Dict]: Tuple of two results.
            - Boolean status indicating whether the feature map information is
              identified correctly or not
            - Dict containing feature map and model's input related information.
        """
        self.loader.utils.native_shape_inference()
        status, image_input_dict = get_image_input_details(self.loader)
        if not status:
            log_error("Failed to get the image input from model.")
            return False, None
        image_h = image_input_dict["image_h"]
        image_w = image_input_dict["image_w"]
        fmap_status, feature_map_info = self.get_fmap_info(
            image_h, image_w, matched_nodes
        )
        if not fmap_status:
            log_error("Failed to extract feature extractor details.")
            return False, None
        model_info = {"fmap_info": feature_map_info, "input_info": image_input_dict}
        return True, model_info

    def yolov2v3v4_decode(
        self,
        loader: FrameworkModelLoader,
        fmap_output_name: str,
        fmap_h: int,
        fmap_w: int,
        fmap_stride: List[int],
        anchors: np.ndarray,
    ) -> List[str]:
        """
        Anchor box decoding for YoloV2, YoloV3 and YoloV4 models based on
        feature map details. Anchor box decoding nodes will be added on the
        bottom of fmap_output_name tensor.
        Note: loader's model will be populated by ABP nodes in this call.

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :param str fmap_output_name: Name of the feature extractor.
        :param int fmap_h: Feature map height value.
        :param int fmap_w: Feature map width value.
        :param List[int] fmap_stride: Feature map stride.
        :param np.ndarray anchors: Anchor associated with the given feature map.
        :return List[str]: List of output tensors' name of anchor box decoding nodes.
        """
        # Assume fmap_output tensor is having shape as [B, 3, H, W, 85]
        # Following operation is performed in this function.
        # xy : [B, 3, H, W, 0:2]
        # wh : [B, 3, H, W, 2:4]
        # scores : [B, 3, H, W, 4:]

        # For yolov4:
        #   xy = sigmoid(xy) * scale_xy - 0.5 * (scale_xy - 1)
        # For yolov2 and yolov3
        #   xy = sigmoid(xy)

        # xy = (xy + grid) * stride
        # wh = exp(wh) * anchor

        # For yolov2:
        #   wh = wh * stride

        # xywh = concat([xy, wh], axis=-1)

        # For yolov2
        #   objness = sigmoid(scores[:, :, :, :, 0:1])
        #   cls_probs = softmax(scores[:, :, :, :, 1:])
        #   scores = objness * cls_probs
        # For yolov3 and yolov4
        #   scores = sigmoid(scores)
        grid = make_grid(fmap_w, fmap_h)

        if self.nms_params["arch_type"] == ModelArchType.YOLOV2:
            split_node, initializers = loader.utils.create_node(
                [fmap_output_name],
                {
                    "op_type": "Split",
                    "split": [2, 2, 1, self.nms_params["num_classes"]],
                    "axis": 4,
                    "num_outputs": 4,
                },
                self.model_opset,
            )
            loader.utils.add_node(split_node)
            loader.utils.add_initializers(initializers)
            [
                split_xy_output,
                split_wh_output,
                split_objness_output,
                split_cls_prob_output,
            ] = split_node.output
        else:
            split_node, initializers = loader.utils.create_node(
                [fmap_output_name],
                {
                    "op_type": "Split",
                    "split": [2, 2, 1 + self.nms_params["num_classes"]],
                    "axis": 4,
                    "num_outputs": 3,
                },
                self.model_opset,
            )
            loader.utils.add_node(split_node)
            loader.utils.add_initializers(initializers)
            [split_xy_output, split_wh_output, split_scores_output] = split_node.output

        sigmoid_node, initializers = loader.utils.create_node(
            [split_xy_output], {"op_type": "Sigmoid"}, self.model_opset
        )
        loader.utils.add_node(sigmoid_node)
        loader.utils.add_initializers(initializers)

        output_name = sigmoid_node.output[0]

        if self.nms_params["arch_type"] == ModelArchType.YOLOV4:
            mul_node, initializers = loader.utils.create_node(
                [output_name],
                {
                    "op_type": "Mul",
                    "B": np.array(self.nms_params["scale_xy"], dtype=np.float32),
                },
                self.model_opset,
            )
            loader.utils.add_node(mul_node)
            loader.utils.add_initializers(initializers)

            add_node, initializers = loader.utils.create_node(
                [mul_node.output[0]],
                {
                    "op_type": "Add",
                    "B": np.array(
                        -0.5 * (self.nms_params["scale_xy"] - 1), dtype=np.float32
                    ),
                },
                self.model_opset,
            )
            loader.utils.add_node(add_node)
            loader.utils.add_initializers(initializers)

            output_name = add_node.output[0]

        add_node, initializers = loader.utils.create_node(
            [output_name],
            {"op_type": "Add", "B": np.array(grid, dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(add_node)
        loader.utils.add_initializers(initializers)

        mul_node_xy, initializers = loader.utils.create_node(
            [add_node.output[0]],
            {"op_type": "Mul", "B": np.array(fmap_stride, dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(mul_node_xy)
        loader.utils.add_initializers(initializers)
        xy_tensor = mul_node_xy.output[0]

        exp_node, initializers = loader.utils.create_node(
            [split_wh_output], {"op_type": "Exp"}, self.model_opset
        )
        loader.utils.add_node(exp_node)
        loader.utils.add_initializers(initializers)

        mul_node_wh, initializers = loader.utils.create_node(
            [exp_node.output[0]],
            {"op_type": "Mul", "B": np.array(anchors, dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(mul_node_wh)
        loader.utils.add_initializers(initializers)
        wh_tensor = mul_node_wh.output[0]

        if self.nms_params["arch_type"] == ModelArchType.YOLOV2:
            mul_node_wh, initializers = loader.utils.create_node(
                [wh_tensor],
                {"op_type": "Mul", "B": np.array(fmap_stride, dtype=np.float32)},
                self.model_opset,
            )
            loader.utils.add_node(mul_node_wh)
            loader.utils.add_initializers(initializers)
            wh_tensor = mul_node_wh.output[0]

        concat_xywh, initializers = loader.utils.create_node(
            [xy_tensor, wh_tensor], {"op_type": "Concat", "axis": 4}, self.model_opset
        )
        loader.utils.add_node(concat_xywh)
        loader.utils.add_initializers(initializers)
        xywh_tensor = concat_xywh.output[0]

        if self.nms_params["arch_type"] == ModelArchType.YOLOV2:
            sigmoid_node_objness, initializers = loader.utils.create_node(
                [split_objness_output], {"op_type": "Sigmoid"}, self.model_opset
            )
            loader.utils.add_node(sigmoid_node_objness)
            loader.utils.add_initializers(initializers)
            objness_score_tensor = sigmoid_node_objness.output[0]

            softmax_node_cls_score, initializers = loader.utils.create_node(
                [split_cls_prob_output],
                {"op_type": "Softmax", "axis": 4},
                self.model_opset,
            )
            loader.utils.add_node(softmax_node_cls_score)
            loader.utils.add_initializers(initializers)
            cls_score_tensor = softmax_node_cls_score.output[0]

            concat_node_score, initializers = loader.utils.create_node(
                [objness_score_tensor, cls_score_tensor],
                {"op_type": "Concat", "axis": 4},
                self.model_opset,
            )
            loader.utils.add_node(concat_node_score)
            loader.utils.add_initializers(initializers)
            scores_tensor = concat_node_score.output[0]
        else:
            sigmoid_node_score, initializers = loader.utils.create_node(
                [split_scores_output], {"op_type": "Sigmoid"}, self.model_opset
            )
            loader.utils.add_node(sigmoid_node_score)
            loader.utils.add_initializers(initializers)
            scores_tensor = sigmoid_node_score.output[0]

        # xywh  : [B x 3 x H x W x 4]
        # scores: [B x 3 x H x W x 81]
        return [xywh_tensor, scores_tensor]

    def yolov5v7_decode(
        self,
        loader: FrameworkModelLoader,
        fmap_output_name: str,
        fmap_h: int,
        fmap_w: int,
        fmap_stride: List[int],
        anchors: np.ndarray,
    ) -> List[str]:
        """
        Anchor box decoding for YoloV5 models based on feature map details.
        Anchor box decoding nodes will be added on the bottom of
        fmap_output_name tensor.
        Note: loader's model will be populated by ABP nodes in this call.

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :param str fmap_output_name: Name of the feature extractor.
        :param int fmap_h: Feature map height value.
        :param int fmap_w: Feature map width value.
        :param List[int] fmap_stride: Feature map stride.
        :param np.ndarray anchors: Anchor associated with the given feature map.
        :return List[str]: List of output tensors' names of anchor box decoding nodes.
        """
        # Assume fmap_output tensor is having shape as [B, 3, H, W, 85]
        # Following operation is performed in this function.
        # xy : [B, 3, H, W, 0:2]
        # wh : [B, 3, H, W, 2:4]
        # scores : [B, 3, H, W, 4:]

        # xy = (xy * 2 - 0.5 + grid) * stride
        # wh = ((wh * 2) ^ 2) * anchor
        # xywh = concat([xy, wh], axis=-1)

        grid = make_grid(fmap_w, fmap_h)

        sigmoid_node, initializers = loader.utils.create_node(
            [fmap_output_name], {"op_type": "Sigmoid"}, self.model_opset
        )
        loader.utils.add_node(sigmoid_node)
        loader.utils.add_initializers(initializers)

        split_node, initializers = loader.utils.create_node(
            [sigmoid_node.output[0]],
            {
                "op_type": "Split",
                "split": [2, 2, 1 + self.nms_params["num_classes"]],
                "axis": 4,
                "num_outputs": 3,
            },
            self.model_opset,
        )
        loader.utils.add_node(split_node)
        loader.utils.add_initializers(initializers)
        [split_xy_output, split_wh_output, split_scores_output] = split_node.output

        mul_node, initializers = loader.utils.create_node(
            [split_xy_output],
            {"op_type": "Mul", "B": np.array([2.0], dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(mul_node)
        loader.utils.add_initializers(initializers)

        add_node, initializers = loader.utils.create_node(
            [mul_node.output[0]],
            {"op_type": "Add", "B": np.array([-0.5], dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(add_node)
        loader.utils.add_initializers(initializers)

        add_node, initializers = loader.utils.create_node(
            [add_node.output[0]],
            {"op_type": "Add", "B": np.array(grid, dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(add_node)
        loader.utils.add_initializers(initializers)

        mul_node_xy, initializers = loader.utils.create_node(
            [add_node.output[0]],
            {"op_type": "Mul", "B": np.array(fmap_stride, dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(mul_node_xy)
        loader.utils.add_initializers(initializers)

        mul_node, initializers = loader.utils.create_node(
            [split_wh_output],
            {"op_type": "Mul", "B": np.array([2.0], dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(mul_node)
        loader.utils.add_initializers(initializers)

        pow_node, initializers = loader.utils.create_node(
            [mul_node.output[0]],
            {"op_type": "Pow", "Y": np.array([2.0], dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(pow_node)
        loader.utils.add_initializers(initializers)

        mul_node_wh, initializers = loader.utils.create_node(
            [pow_node.output[0]],
            {"op_type": "Mul", "B": np.array(anchors, dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(mul_node_wh)
        loader.utils.add_initializers(initializers)

        concat_node, initializers = loader.utils.create_node(
            [mul_node_xy.output[0], mul_node_wh.output[0]],
            {"op_type": "Concat", "axis": 4},
            self.model_opset,
        )
        loader.utils.add_node(concat_node)
        loader.utils.add_initializers(initializers)

        xywh_tensor = concat_node.output[0]  # xywh  : [B x 3 x H x W x 4]
        scores_tensor = split_scores_output  # scores: [B x 3 x H x W x 81]
        return [xywh_tensor, scores_tensor]

    def yolox_decode(
        self,
        loader: FrameworkModelLoader,
        fmap_output_name: str,
        fmap_h: int,
        fmap_w: int,
        fmap_stride: List[int],
        anchors: np.ndarray = None,
    ) -> List[str]:
        """
        Anchor box decoding for YoloX models based on feature map details.
        Anchor box decoding nodes will be added on the bottom of
        fmap_output_name tensor.
        Note: loader's model will be populated by ABP nodes in this call.

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :param str fmap_output_name: Name of the feature extractor.
        :param int fmap_h: Feature map height value.
        :param int fmap_w: Feature map width value.
        :param List[int] fmap_stride: Feature map stride.
        :param np.ndarray anchors: Anchor associated with the given feature map.
            This field is not required as YoloX is anchor free detector.
            Defaults to None
        :return List[str]: List of output tensors' names of anchor box decoding nodes.
        """
        # Assume fmap_output tensor is having shape as [B, 1, H, W, 85]
        # Following operation is performed in this function.
        # xy : [B, 3, H, W, 0:2]
        # wh : [B, 3, H, W, 2:4]
        # scores : [B, 3, H, W, 4:]

        # xy = (xy + grid) * stride
        # wh = exp(wh) * stride
        # xywh = concat([xy, wh], axis=-1)
        # scores = sigmoid(scores)

        grid = make_grid(fmap_w, fmap_h)

        split_node, initializers = loader.utils.create_node(
            [fmap_output_name],
            {
                "op_type": "Split",
                "split": [2, 2, 1 + self.nms_params["num_classes"]],
                "axis": 4,
                "num_outputs": 3,
            },
            self.model_opset,
        )
        loader.utils.add_node(split_node)
        loader.utils.add_initializers(initializers)
        [split_xy_output, split_wh_output, split_scores_output] = split_node.output

        add_node_xy, initializers = loader.utils.create_node(
            [split_xy_output],
            {"op_type": "Add", "B": np.array(grid, dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(add_node_xy)
        loader.utils.add_initializers(initializers)
        xy_tensor = add_node_xy.output[0]

        mul_node_xy, initializers = loader.utils.create_node(
            [xy_tensor],
            {"op_type": "Mul", "B": np.array(fmap_stride, dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(mul_node_xy)
        loader.utils.add_initializers(initializers)
        xy_tensor = mul_node_xy.output[0]

        exp_node, initializers = loader.utils.create_node(
            [split_wh_output], {"op_type": "Exp"}, self.model_opset
        )
        loader.utils.add_node(exp_node)
        loader.utils.add_initializers(initializers)
        wh_tensor = exp_node.output[0]

        mul_node_wh, initializers = loader.utils.create_node(
            [wh_tensor],
            {"op_type": "Mul", "B": np.array(fmap_stride, dtype=np.float32)},
            self.model_opset,
        )
        loader.utils.add_node(mul_node_wh)
        loader.utils.add_initializers(initializers)
        wh_tensor = mul_node_wh.output[0]

        concat_node, initializers = loader.utils.create_node(
            [xy_tensor, wh_tensor], {"op_type": "Concat", "axis": 4}, self.model_opset
        )
        loader.utils.add_node(concat_node)
        loader.utils.add_initializers(initializers)
        xywh_tensor = concat_node.output[0]

        sigmoid_node, initializers = loader.utils.create_node(
            [split_scores_output], {"op_type": "Sigmoid"}, self.model_opset
        )
        loader.utils.add_node(sigmoid_node)
        loader.utils.add_initializers(initializers)
        score_tensor = sigmoid_node.output[0]

        return [xywh_tensor, score_tensor]

    def remove_yolox_sigmoid(
        self, loader: FrameworkModelLoader, tensor_name: str
    ) -> None:
        """
        Remove sigmoid activation found in YoloX model. Sigmoid is applied to
        classification and objectness tensor. This is because Host NMS variant
        doesn't need Sigmoid as a part of model. However, for Device NMS Sigmoid
        will be applied after concating the feature maps.

        # Below is one of the YoloX feature extractor
        # Conv (objectness)     --> Sigmoid --\
        # Conv (classification) --> Sigmoid ---> Concat
        # Conv (boxes)          -->         --/

        # It will be modified as per below nodes.
        # Conv (objectness)     --\
        # Conv (classification) ---> Concat
        # Conv (boxes)          --/

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :param str tensor_name: Name of the tensor from where the Sigmoid is to
            be traced as a parent of the node.
        :raises RuntimeError: No node found with name as tensor_name.
        :raises RuntimeError: Parent node to be checked has more than one parent.
        """
        get_node_by_output = get_node_by_output_name(loader.model)
        if tensor_name not in get_node_by_output:
            raise RuntimeError(f"No node found which generates tensor '{tensor_name}'.")
        node = get_node_by_output[tensor_name]
        parent_nodes = loader.get_parent_nodes(node)
        nodes_to_be_removed = [n for n in parent_nodes if n.op_type == "Sigmoid"]
        for _node in nodes_to_be_removed:
            loader.utils.remove_node(_node)
            _p_nodes = loader.get_parent_nodes(_node)
            if len(_p_nodes) != 1:
                raise RuntimeError(
                    f"Given Sigmoid node '{_node.name}' should have exact 1 parent node."
                )
            _p_output_tensor = _p_nodes[0].output[0]
            existing_output_tensor = _node.output[0]

            # Replace the connections
            for i, ip_tensor in enumerate(node.input):
                if ip_tensor == existing_output_tensor:
                    node.input[i] = _p_output_tensor

    def prepare_fmaps(
        self,
        loader: FrameworkModelLoader,
        fmap_name: str,
        h_idx: int,
        w_idx: int,
        a_idx: int,
        shape: List[int],
    ) -> str:
        """
        Prepare given feature map tensor to make it compatible with Host NMS and
        Device NMS configuration.

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :param str fmap_name: Name of the feature extractor.
        :param int h_idx: Index of feature map height value.
        :param int w_idx: Index of feature map width value.
        :param int a_idx: Index of feature map anchor value.
        :param List[int] shape: Feature map shape.
        :return str: Name of output tensor of feature map modification nodes.
        """
        if self.nms_params["arch_type"] == ModelArchType.YOLOX:
            self.remove_yolox_sigmoid(loader, fmap_name)

        output_name = fmap_name
        if a_idx is None:
            if [h_idx, w_idx] == [1, 2]:
                # NHWC
                num_channels = shape[3]
                num_detections = num_channels // self.nms_params["num_anchors_per_fmap"]
                new_shape = [
                    -1,
                    shape[h_idx],
                    shape[w_idx],
                    self.nms_params["num_anchors_per_fmap"],
                    num_detections,
                ]
                # NHWC to NHWAD
                reshape_node, initializers = loader.utils.create_node(
                    [output_name],
                    {"op_type": "Reshape", "shape": new_shape},
                    self.model_opset,
                )
                # reshape_tensor = transpose_node.output[0]
                loader.utils.add_node(reshape_node)
                loader.utils.add_initializers(initializers)
                # batch_shape = np.prod(shape) / (shape[h_idx] * shape[w_idx] * self.nms_params["num_anchors_per_fmap"] * num_detections)
                # new_shape = [
                #     batch_shape,
                #     shape[h_idx],
                #     shape[w_idx],
                #     self.nms_params["num_anchors_per_fmap"],
                #     num_detections,
                # ]
                # reshape_tensor_val_info = loader.utils.create_value_info(reshape_tensor, np.float32, new_shape)
                # loader.utils.add_value_info(reshape_tensor_val_info)

                # NHWAD to NAHWD
                permute_index = [0, 3, 1, 2, 4]
                transpose_node, initializers = loader.utils.create_node(
                    [reshape_node.output[0]],
                    {"op_type": "Transpose", "perm": permute_index},
                    self.model_opset,
                )
                loader.utils.add_node(transpose_node)
                loader.utils.add_initializers(initializers)
                output_name = transpose_node.output[0]

                # new_shape = [new_shape[i] for i in permute_index]
                # output_tensor_val_info = loader.utils.create_value_info(output_name, np.float32, new_shape)
                # loader.utils.add_value_info(output_tensor_val_info)

            elif [h_idx, w_idx] == [2, 3]:
                # NCHW
                if self.nms_params["arch_type"] == ModelArchType.YOLOX:
                    num_anchors = 1
                else:
                    num_anchors = self.nms_params["num_anchors"]

                num_channels = shape[1]
                num_detections = num_channels // num_anchors
                new_shape = [
                    -1,
                    num_anchors,
                    num_detections,
                    shape[h_idx],
                    shape[w_idx],
                ]

                # NCHW to NADHW
                reshape_node, initializers = loader.utils.create_node(
                    [output_name],
                    {"op_type": "Reshape", "shape": new_shape},
                    self.model_opset,
                )
                # reshape_tensor = reshape_node.output[0]
                loader.utils.add_node(reshape_node)
                loader.utils.add_initializers(initializers)
                # batch_shape = np.prod(shape) / (shape[h_idx] * shape[w_idx] * self.nms_params["num_anchors_per_fmap"] * num_detections)
                # new_shape = [
                #     batch_shape,
                #     self.nms_params["num_anchors"],
                #     num_detections,
                #     shape[h_idx],
                #     shape[w_idx],
                # ]
                # reshape_tensor_val_info = loader.utils.create_value_info(reshape_tensor, np.float32, new_shape)
                # loader.utils.add_value_info(reshape_tensor_val_info)

                # NADHW to NAHWD
                permute_index = [0, 1, 3, 4, 2]
                transpose_node, initializers = loader.utils.create_node(
                    [reshape_node.output[0]],
                    {"op_type": "Transpose", "perm": permute_index},
                    self.model_opset,
                )
                loader.utils.add_node(transpose_node)
                loader.utils.add_initializers(initializers)
                output_name = transpose_node.output[0]

                # new_shape = [new_shape[i] for i in permute_index]
                # output_tensor_val_info = loader.utils.create_value_info(output_name, np.float32, new_shape)
                # loader.utils.add_value_info(output_tensor_val_info)

        else:
            if [h_idx, w_idx, a_idx] == [2, 3, 1]:
                # NAHWD - Do nothing as this is what we want.
                pass
            elif [h_idx, w_idx, a_idx] == [3, 4, 1]:
                # NADHW to NAHWD
                permute_index = [0, 1, 3, 4, 2]
                transpose_node, initializers = loader.utils.create_node(
                    [output_name],
                    {"op_type": "Transpose", "perm": permute_index},
                    self.model_opset,
                )
                loader.utils.add_node(transpose_node)
                loader.utils.add_initializers(initializers)
                output_name = transpose_node.output[0]

                # new_shape = [shape[i] for i in permute_index]
                # output_tensor_val_info = loader.utils.create_value_info(output_name, np.float32, new_shape)
                # loader.utils.add_value_info(output_tensor_val_info)

            elif [h_idx, w_idx, a_idx] == [2, 3, 4]:
                # NDHWA to NAHWD
                permute_index = [0, 4, 2, 3, 1]
                transpose_node, initializers = loader.utils.create_node(
                    [output_name],
                    {"op_type": "Transpose", "perm": permute_index},
                    self.model_opset,
                )
                loader.utils.add_node(transpose_node)
                loader.utils.add_initializers(initializers)
                output_name = transpose_node.output[0]

                # new_shape = [shape[i] for i in permute_index]
                # output_tensor_val_info = loader.utils.create_value_info(output_name, np.float32, new_shape)
                # loader.utils.add_value_info(output_tensor_val_info)

            elif [h_idx, w_idx, a_idx] == [1, 2, 3]:
                # NHWAD to NAHWD
                permute_index = [0, 3, 1, 2, 4]
                transpose_node, initializers = loader.utils.create_node(
                    [output_name],
                    {"op_type": "Transpose", "perm": permute_index},
                    self.model_opset,
                )
                loader.utils.add_node(transpose_node)
                loader.utils.add_initializers(initializers)
                output_name = transpose_node.output[0]

                # new_shape = [shape[i] for i in permute_index]
                # output_tensor_val_info = loader.utils.create_value_info(output_name, np.float32, new_shape)
                # loader.utils.add_value_info(output_tensor_val_info)

        return output_name

    def add_abp_nms_nodes(
        self, loader: FrameworkModelLoader, model_info: Dict
    ) -> Tuple[bool, FrameworkModelLoader]:
        """
        Prepare yolo based models by adding anchor box processing nodes and NMS
        node in the graph.

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :param Dict model_info: Dict containing useful information related to
            feature maps and model's input.
        :return Tuple[bool, FrameworkModelLoader]: Tuple of two objects.
            - Boolean status indicating whether the subgraph generation is
              successful or not.
            - Updated onnx model loader.
        """
        self.check_and_update_anchors(self.nms_params, model_info["fmap_info"])

        final_outputs = []

        all_fmap_boxes = []
        all_fmap_scores = []
        for fmap_name, fmap_info in model_info["fmap_info"].items():
            output_name = fmap_name
            shape = fmap_info["shape"]
            h_idx = fmap_info["h_idx"]
            w_idx = fmap_info["w_idx"]
            a_idx = fmap_info.get("a_idx", None)
            fmap_stride = [fmap_info["stride_w"], fmap_info["stride_h"]]

            output_name = self.prepare_fmaps(
                loader, fmap_name, h_idx, w_idx, a_idx, shape
            )

            # Feature extractor shape : [B, A, H, W, D]
            if self.nms_params["nms_type"] == NMSType.HOST:
                # Return as the feature maps are now as per Host NMS requirement.
                final_outputs.append(output_name)
                continue

            if self.nms_params["arch_type"] not in self.abp_decode_fn:
                log_error(
                    f"No anchor box processing function found for given arch_type: '{self.nms_params['arch_type']}'."
                )
                return False, loader

            decode_fn = self.abp_decode_fn[self.nms_params["arch_type"]]
            [xywh_tensor, scores_tensors] = decode_fn(
                loader,
                output_name,
                shape[h_idx],
                shape[w_idx],
                fmap_stride,
                fmap_info.get("anchor", None),
            )

            # xywh : [B x 3 x h x w x 4] -> [B, 3*h*w, 4]
            new_shape = [
                -1,
                self.nms_params["num_anchors_per_fmap"] * shape[h_idx] * shape[w_idx],
                4,
            ]
            reshape_node, initializers = loader.utils.create_node(
                [xywh_tensor],
                {"op_type": "Reshape", "shape": new_shape, "allowzero": 0},
                self.model_opset,
            )
            loader.utils.add_node(reshape_node)
            loader.utils.add_initializers(initializers)
            xywh_tensor = reshape_node.output[0]

            # scores : [B x 3 x h x w x 81] -> [B, 3*h*w, 81]
            new_shape = [
                -1,
                self.nms_params["num_anchors_per_fmap"] * shape[h_idx] * shape[w_idx],
                self.nms_params["num_classes"] + 1,
            ]
            reshape_node, initializers = loader.utils.create_node(
                [scores_tensors],
                {"op_type": "Reshape", "shape": new_shape, "allowzero": 0},
                self.model_opset,
            )
            loader.utils.add_node(reshape_node)
            loader.utils.add_initializers(initializers)
            scores_tensors = reshape_node.output[0]

            all_fmap_boxes.append(xywh_tensor)
            all_fmap_scores.append(scores_tensors)

        if self.nms_params["nms_type"] == NMSType.DEVICE:
            if len(all_fmap_boxes) > 1:
                concat_node_boxes, initializers = loader.utils.create_node(
                    all_fmap_boxes, {"op_type": "Concat", "axis": 1}, self.model_opset
                )
                loader.utils.add_node(concat_node_boxes)
                loader.utils.add_initializers(initializers)
                concated_boxes_tensor = concat_node_boxes.output[0]
            else:
                concated_boxes_tensor = all_fmap_boxes[0]

            if len(all_fmap_scores) > 1:
                concat_node_scores, initializers = loader.utils.create_node(
                    all_fmap_scores, {"op_type": "Concat", "axis": 1}, self.model_opset
                )
                loader.utils.add_node(concat_node_scores)
                loader.utils.add_initializers(initializers)
                concated_scores_tensor = concat_node_scores.output[0]
            else:
                concated_scores_tensor = all_fmap_scores[0]

            split_node_scores, initializers = loader.utils.create_node(
                [concated_scores_tensor],
                {
                    "op_type": "Split",
                    "split": [1, self.nms_params["num_classes"]],
                    "axis": 2,
                    "num_outputs": 2,
                },
                self.model_opset,
            )
            loader.utils.add_node(split_node_scores)
            loader.utils.add_initializers(initializers)
            [objness_score_tensor, cls_prob_score_tensor] = split_node_scores.output

            mul_node_scores, initializers = loader.utils.create_node(
                [objness_score_tensor, cls_prob_score_tensor],
                {"op_type": "Mul"},
                self.model_opset,
            )
            loader.utils.add_node(mul_node_scores)
            loader.utils.add_initializers(initializers)
            final_scores = mul_node_scores.output[0]

            final_boxes = self.convert_xywh_to_yxyx(loader, concated_boxes_tensor)

            unsqueeze_node, initializers = loader.utils.create_node(
                [final_boxes], {"op_type": "Unsqueeze", "axes": [2]}, self.model_opset
            )
            loader.utils.add_node(unsqueeze_node)
            loader.utils.add_initializers(initializers)
            final_boxes = unsqueeze_node.output[0]

            if self.nms_params["class_specific_nms"]:
                # repeats input is used for tile opset >= 6
                # tiles and axis inputs are used for tile opset=1
                tile_node, initializers = loader.utils.create_node(
                    [final_boxes],
                    {
                        "op_type": "Tile",
                        "repeats": [1, 1, self.nms_params["num_classes"], 1],
                        "tiles": self.nms_params["num_classes"],
                        "axis": 2,
                    },
                    self.model_opset,
                )
                loader.utils.add_node(tile_node)
                loader.utils.add_initializers(initializers)
                final_boxes = tile_node.output[0]

            loader.utils.native_shape_inference()
            comb_nms_outputs = self.add_combined_nms_node(
                loader,
                [final_boxes, final_scores],
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
            log_error(
                "Generated NMS optimized onnx model is invalid due to "
                f"exception : {e} "
            )
            return False, loader
        return True, loader

    def get_output_info(self, loader: FrameworkModelLoader) -> Tuple[bool, List[str]]:
        """
        Get the output info of the model after applying Host NMS modification.

        :param FrameworkModelLoader loader: Onnx model loader instance.
        :return Tuple[bool, List[str]]: Tuple of two objects.
            - Boolean status indicating whether the output info from the loader
              is identified or not.
            - List of tensor names in descending order of their height width.
        """
        output_info = loader.get_output_info()
        output_h = []
        output_names = []
        for tensor_name, tensor_info in output_info.items():
            shape = tensor_info.shape
            # Host NMS supports NAHWC layout only for 5d tensors coming out of Yolo models.
            if len(shape) != 5:
                log_error(
                    "For Yolo models with Host NMS variant should produce 5d outputs with NAHWD layout."
                )
                return False, None
            if shape[1] != self.nms_params["num_anchors_per_fmap"]:
                log_error(
                    f"For Yolo models with Host NMS variant should have {self.nms_params['num_anchors_per_fmap']} shape at 2nd axis but got {shape[1]}."
                )
                return False, None
            if shape[4] != (self.nms_params["num_classes"] + 5):
                log_error(
                    f"For Yolo models with Host NMS variant should have {self.nms_params['num_classes'] + 5} shape at 4th axis but got {shape[4]}."
                )
                return False, None
            output_h.append(shape[2])
            output_names.append(tensor_name)

        desc_sorted_idx = np.argsort(-np.array(output_h))
        desc_sorted_output_names = [output_names[i] for i in desc_sorted_idx]

        return True, desc_sorted_output_names

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
            self.nms_params["anchor_data"], model_info["fmap_info"]
        )

        status, hostnms_config = self.generate_hostnms_common_config(loader, model_info)
        if not status:
            return False, None

        status, sorted_outputs = self.get_output_info(loader)
        if not status:
            return False, None

        # Remove "/" or "." from output names as these names will change at the
        # time of serialization.
        hostnms_config["bbox-output-list"] = self.cleanup_output_names(
            [*sorted_outputs]
        )

        hostnms_config["layout"] = "NAHWC"
        hostnms_config["score-output-list"] = []

        if self.nms_params["arch_type"] not in self.host_nms_do_softmax:
            log_error(
                "No information found for whether to apply softmax on class prediction or not."
            )
            return False, None
        do_softmax = self.host_nms_do_softmax[self.nms_params["arch_type"]]

        hostnms_config["do-softmax"] = do_softmax
        hostnms_config["background-class-idx"] = (
            0  # This is not going to be used for yolo models.
        )

        sorted_output_shapes = []
        output_info = loader.get_output_info()
        for output_name in sorted_outputs:
            if output_name not in output_info:
                log_error(f"Tensor {output_name} shall be the output of the model.")
                return False, None
            sorted_output_shapes.append(output_info[output_name].shape)

        hostnms_yaml_path = os.path.join(
            os.path.dirname(self.nms_params["dlc_path"]),
            os.path.splitext(loader.model_wrapper.model_name)[0]
            + "_hostnms_config.yaml",
        )
        dump_yaml(hostnms_config, hostnms_yaml_path)
        log_debug1(f"Host NMS config file dumped at: {hostnms_yaml_path}")
        return True, hostnms_yaml_path
