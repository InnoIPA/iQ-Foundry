# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from __future__ import annotations

import os
from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
import yaml
from onnx import NodeProto
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils.converter_utils import log_error
from qti.aisw.converters.common.utils.framework_utils import TensorInfo

COMBINEDNMS_DOMAIN = "qti_aisw"
COMBINEDNMS_OPSET = 1


class NMSType(Enum):
    """
    Enum class to represent two different types of NMS models supported.
    """

    HOST = "HOST"
    DEVICE = "DEVICE"

    @classmethod
    def from_value(cls: Enum, value: str) -> Union[None, NMSType]:
        """
        Attempts to create an instance of NMSType from the given value.

        Args:
            value (str): The value to convert to a NMSType instance.

        Returns:
            NMSType: The corresponding NMSType instance if the value is valid.
            None: If the value is not valid.
        """
        try:
            return cls(value)
        except ValueError:
            return None

    @classmethod
    def get_nms_types(cls: Enum) -> List[str]:
        """
        Get the list of supported NMS types.

        :return List[str]: List of NMS types supported.
        """
        return [e.value for e in NMSType]


class ModelArchType(Enum):
    """
    Enum Class to represent the model architectures that are supported.
    """

    YOLOV2 = "YOLOV2"
    YOLOV3 = "YOLOV3"
    YOLOV4 = "YOLOV4"
    YOLOV5 = "YOLOV5"
    # YOLOV6 = "YOLOV6"
    YOLOV7 = "YOLOV7"
    YOLOV7_TINY = "YOLOV7_TINY"
    # YOLOV8 = "YOLOV8"
    YOLOX = "YOLOX"
    SSDVGG = "SSDVGG"
    SSDMOBILENET = "SSDMOBILENET"
    SSDINCEPTION = "SSDINCEPTION"
    SSDRESNET = "SSDRESNET"
    EFFICIENTDET = "EFFICIENTDET"
    RETINANET = "RETINANET"
    # TODO: Also, check for SSDVGG and SSDINCEPTION, and how are these mapping to other ssd. Do we need these many SSD arch types?

    @classmethod
    def from_value(cls: Enum, value: str) -> Union[None, ModelArchType]:
        """
        Attempts to create an instance of ModelArchType from the given value.

        Args:
            value (str): The value to convert to a ModelArchType instance.

        Returns:
            ModelArchType: The corresponding ModelArchType instance if the value is valid.
            None: If the value is not valid.
        """
        try:
            return cls(value)
        except ValueError:
            return None

    @classmethod
    def get_supported_models(cls: Enum) -> List[str]:
        """
        Get the list of supported models.

        :return List[str]: List of supported models.
        """
        return [e.value for e in ModelArchType]


def get_host_nms_arch_type(arch_type: str) -> Tuple[bool, str]:
    """
    Get the equivalent architecture name for host nms execution based on
    given model name

    :param str arch_type: Architecture of model.
    :return Tuple[bool, str]: Tuple of 2 results.
        - boolean value indicating whether the hostnms config is generated
            correctly or not.
        - Equivalent Host nms model architecture type.
    """
    host_nms_mapping = {
        ModelArchType.YOLOV2: "yolov2",
        ModelArchType.YOLOV2: "yolov2",
        ModelArchType.YOLOV3: "yolov3",
        ModelArchType.YOLOV4: "yolov4",
        ModelArchType.YOLOV5: "yolov5",
        # ModelArchType.YOLOV6 : "yolov6",
        ModelArchType.YOLOV7: "yolov7",
        ModelArchType.YOLOV7_TINY: "yolov7_tiny",
        # ModelArchType.YOLOV8 : "yolov8",
        ModelArchType.YOLOX: "yolox",
        ModelArchType.SSDVGG: "resnet18ssd",
        ModelArchType.SSDMOBILENET: "resnet18ssd",
        ModelArchType.SSDINCEPTION: "resnet18ssd",
        ModelArchType.SSDRESNET: "resnet18ssd",
        ModelArchType.EFFICIENTDET: "efficientdet-d0",
        ModelArchType.RETINANET: "retinanet-resnext50",
    }
    if arch_type not in host_nms_mapping:
        log_error(
            f"Provided arch_type {arch_type} is not supported by HostNMS application."
        )
        return False, None
    return True, host_nms_mapping[arch_type]


def get_supported_yolo_models() -> List[str]:
    """
    Get the list of supported Yolo models.

    :return List[str]: List of supported yolo models.
    """
    return [e.value for e in ModelArchType if "YOLO" in e.value]


def get_supported_anchor_free_models() -> List[str]:
    """
    Get the list of supported anchor free models.

    :return List[str]: List of supported anchor free models.
    """
    return [ModelArchType.YOLOX.value]


def get_supported_ssd_models() -> List[str]:
    """
    Get the list of supported SSD models.

    :return List[str]: List of supported ssd models.
    """
    filtered_names = []
    for e in ModelArchType:
        model_name = e.value
        if (
            ("SSD" in model_name)
            or ("EFFICIENTDET" in model_name)
            or ("RETINANET" in model_name)
        ):
            filtered_names.append(model_name)
    return filtered_names


def validate_nms_params(nms_params: Dict) -> bool:
    """
    Validate nms parameters obtained from arg parser.

    :param Dict nms_params: Dict containing useful NMS related parameters
        obtained from user via argparser.
    :return bool: Boolean status indicating whether parameters are correct
        or not.
    """
    ssd_model_except_effdet = get_supported_ssd_models()
    ssd_model_except_effdet.remove(ModelArchType.EFFICIENTDET.value)

    for key, value in nms_params.items():
        if nms_params.get("nms_type") is None:
            log_error(f"Value for flag --{key} is not provided.")
            return False
        if nms_params.get("arch_type") is None:
            log_error(f"Value for flag --{key} is not provided.")
            return False

        if nms_params["nms_type"] == NMSType.HOST:
            if (key in ["map_coco_80_to_90"]) and (value is None):
                log_error(f"Value for flag --{key} is not provided.")
                return False

        if nms_params["arch_type"] in ssd_model_except_effdet:
            if (key == "background_class_idx") and (value is None):
                log_error(f"Value for flag --{key} is not provided.")
                return False
            if nms_params["nms_type"] == NMSType.DEVICE:
                if (key in ["boxes_format", "scale_xy", "scale_wh"]) and (
                    value is None
                ):
                    log_error(f"Value for flag --{key} is not provided.")
                    return False

        if (
            nms_params["arch_type"] in get_supported_anchor_free_models()
        ) and key == "anchor_data":
            continue

        if (
            key
            not in [
                "map_coco_80_to_90",
                "background_class_idx",
                "boxes_format",
                "scale_xy",
                "scale_wh",
            ]
        ) and (value is None):
            log_error(f"Value for flag --{key} is not provided.")
            return False

    if (nms_params["arch_type"] in ssd_model_except_effdet) and (
        nms_params["background_class_idx"] >= nms_params["num_classes"]
    ):
        log_error("Provided background class id is greater than number of classes.")
        return False
    if (nms_params["iou_threshold"] < 0) or (nms_params["iou_threshold"] > 1):
        log_error("Provided iou_threshold value should be between 0 and 1.")
        return False
    if (nms_params["score_threshold"] < 0) or (nms_params["score_threshold"] > 1):
        log_error("Provided score_threshold value should be between 0 and 1.")
        return False

    if (nms_params["arch_type"] not in get_supported_anchor_free_models()) and (
        not os.path.isfile(nms_params["anchor_data"])
    ):
        log_error("No file found at the provided anchor_data path.")
        return False

    return True


def read_anchors(nms_params: Dict) -> bool:
    """
    Read anchor data from the provided anchor path.
    Note: This function will update the nms_params with anchor_data inplace and
          add new key num_anchors in the nms_params dict.

    :param Dict nms_params: Dict containing useful NMS related parameters
        obtained from user via argparser.
    :return bool: Boolean status indicating whether anchors are read correct
        or not.
    """
    if nms_params["arch_type"] in get_supported_anchor_free_models():
        return True

    nms_params["anchor_data"] = np.fromfile(nms_params["anchor_data"], dtype=np.float32)

    if nms_params["arch_type"].value in get_supported_yolo_models():
        if nms_params["anchor_data"].flatten().shape[0] % 2 != 0:
            log_error("Anchor data for yolo category of model shall be of shape N x 2.")
            return False
        nms_params["anchor_data"] = nms_params["anchor_data"].reshape(-1, 2)
    elif nms_params["arch_type"].value in get_supported_ssd_models():
        if nms_params["anchor_data"].flatten().shape[0] % 4 != 0:
            log_error("Anchor data for ssd category of model shall be of shape N x 4.")
            return False
        nms_params["anchor_data"] = nms_params["anchor_data"].reshape(-1, 4)
    else:
        log_error(f"Unknown model architecture provided: {nms_params['arch_type']}")
        return False
    nms_params["num_anchors"] = nms_params["anchor_data"].shape[0]
    return True


def make_grid(fmap_w: int, fmap_h: int) -> np.ndarray:
    """
    Generate anchor grid based on given feature map height width.

    :param int fmap_w: Value of feature map width.
    :param int fmap_h: Value of feature map height.
    :return np.ndarray: Numpy array representing grid for Yolo models.
    """
    xv, yv = np.meshgrid(np.arange(fmap_h), np.arange(fmap_w))
    grid = np.stack((xv, yv), 2).astype(np.float32)
    return grid.reshape(1, 1, fmap_h, fmap_w, 2)


def filter_image_input(input_info: Dict[str, TensorInfo]) -> Tuple[bool, TensorInfo]:
    """
    Find the image tensor info from give input_info dict.
    Note: This function will try to find a tensor with 4d shape only.

    :param Dict[str, TensorInfo] input_info: Input info dict for a given model.
    :return Tuple[bool, TensorInfo]: Tuple of two values.
        - A boolean status indicating whether there is a image tensor in the
          provided input dict or not.
        - TensorInfo of the image tensor.
    """
    image_tensor_info = None
    image_input_found = False
    for _, tensor_info in input_info.items():
        if len(tensor_info.shape) == 4:
            if image_input_found:
                log_error("More than one input found with 4d shape.")
                return False, None
            image_tensor_info = tensor_info
            image_input_found = True
    if not image_input_found:
        log_error("Not able to identify image input with 4d shape in the given model.")
        return False, None
    return True, image_tensor_info


def get_image_hw_layout(tensor_shape: List[int]) -> Tuple:
    """
    Get the layout of image tensor based on its shape.

    :param List[int] tensor_shape: Shape of the image tensor.
    :return Tuple: Tuple of two values.
        - A boolean status indicating whether the layout is identified correctly
          or not
        - List of image height-widht and layout type.
    """
    image_channels = 3
    if tensor_shape[1:].index(image_channels) == 0:
        # NCHW
        return True, [tensor_shape[2:4], "NCHW"]
    elif tensor_shape[1:].index(image_channels) == 2:
        # NHWC
        return True, [tensor_shape[1:3], "NHWC"]
    else:
        log_error(
            "Not able to identify the layout of image input based on "
            f"provided tensor shape '{tensor_shape}'"
        )
        return False, []


def check_image_input_shapes(tensor_name: str, tensor_shape: List[int]) -> bool:
    """
    Check whether the given tensor is actually an image or not based on the its
    shape.

    :param str tensor_name: Name of the image tensor.
    :param List[int] tensor_shape: Shape of the image tensor.
    :return bool: Status whether the tensor is an image input tensor or not.
    """
    if 3 in tensor_shape[1:]:
        if tensor_shape[1:].index(3) == 0:
            return True
        elif tensor_shape[1:].index(3) == 2:
            return True
        else:
            log_error(
                f"Image input '{tensor_name} should have either NCHW or NHWC layout."
            )
            return False
    else:
        log_error(f"Image input '{tensor_name} should have 3 number of channels.")
        return False


def get_image_input_details(loader: FrameworkModelLoader) -> Tuple[bool, Dict]:
    """
    Get the info about image input from the given model loader.

    :param FrameworkModelLoader loader: Onnx loader object.
    :return Tuple[bool, Dict]: Tuple of two values.
        - A boolean status indicating whether the image input info is obtained
          successfully or not
        - Dict containing image input info such as name, batch size,
          height-width and layout of the image input tensor.
    """
    input_info = loader.get_input_info()
    status, image_tensor_info = filter_image_input(input_info)
    if not status:
        return False, None
    status = check_image_input_shapes(image_tensor_info.name, image_tensor_info.shape)
    if not status:
        return False, None
    status, [[image_h, image_w], image_layout] = get_image_hw_layout(
        image_tensor_info.shape
    )
    if not status:
        return False, None
    input_info = {
        "image_input_name": image_tensor_info.name,
        "image_bs": image_tensor_info.shape[0],
        "image_h": image_h,
        "image_w": image_w,
        "image_layout": image_layout,
    }
    return True, input_info


def get_yolo_fmap_hwa_index(
    start_node: NodeProto,
    start_tensor_name: str,
    start_tensor_shape: List[int],
    end_node: NodeProto,
    end_tensor_name: str,
    end_tensor_shape: List[int],
    num_anchors_per_fmap: int,
    num_classes: int,
) -> Tuple[bool, Dict]:
    """
    Get the feature map height, width and anchor related details based on
    the identified pattern's start and end tensors' shapes.

    :param NodeProto start_node: Start node of identified pattern.
    :param str start_tensor_name: Start tensor name in the identified pattern.
    :param List[int] start_tensor_shape: Shape of the start tensor in
        identified pattern.
    :param NodeProto end_node: End node of identified pattern.
    :param str end_tensor_name: End tensor name in the identified pattern.
    :param List[int] end_tensor_shape: Shape of the end tensor in identified
        pattern.
    :param int num_anchors_per_fmap: Number of anchors per feature maps in
        the model.
    :param int num_classes: Number of classes model is predicting.
    :return Tuple[bool, Dict]: Tuple of two results.
        - Boolean status indicating success of the process
        - Dict containing all the feature map related information.
    """
    index_dict = {}
    if len(end_tensor_shape) == 4:
        if end_node.op_type == "Conv":
            # If the end_tensor is generated from a conv node then it must be NCHW.
            index_dict["h_idx"] = 2
            index_dict["w_idx"] = 3
            return True, index_dict
        total_predictions_per_fmap = num_anchors_per_fmap * (5 + num_classes)
        if total_predictions_per_fmap not in end_tensor_shape[1:]:
            log_error(
                f"The {end_tensor_name} should have shape which contains {total_predictions_per_fmap} number of channels."
            )
            return False, None
        channel_idx = end_tensor_shape[1:].index(total_predictions_per_fmap) + 1
        if channel_idx == 1:
            # NCHW layout
            index_dict["h_idx"] = 2
            index_dict["w_idx"] = 3
        elif channel_idx == 3:
            # NHWC layout
            index_dict["h_idx"] = 1
            index_dict["w_idx"] = 2
        else:
            log_error(f"The {end_tensor_name} should have either NCHW or NHWC layout.")
            return False, None
        return True, index_dict
    elif len(end_tensor_shape) == 5:
        # Assuming NCHW layout for onnx models.
        if len(start_tensor_shape) != 4:
            log_error(
                "Can not determine height and width values for tensor '{tensor_name}'."
            )
            return False, None
        if start_node.op_type == "Conv":
            # If the start_tensor is used in a conv node then it must be NCHW.
            h_value = start_tensor_shape[2]
            w_value = start_tensor_shape[3]
        else:
            log_error(
                f"The {start_tensor_name} should be used in a Convolution node which is part of pattern."
            )
            return False, None
        if num_anchors_per_fmap not in end_tensor_shape[1:]:
            log_error(
                f"Tensor '{end_tensor_name}' should have {num_anchors_per_fmap} as one of the axis in its shape."
            )
            return False, None
        anchor_idx = 1 + end_tensor_shape[1:].index(num_anchors_per_fmap)
        h_index = [
            i
            for i, s in enumerate(end_tensor_shape)
            if ((s == h_value) and (i != anchor_idx))
        ]
        if len(h_index) == 0:
            log_error(
                f"Height value can't be determined for tensor '{end_tensor_name}'."
            )
            return False, None
        h_index = h_index[0]
        w_index = [
            i
            for i, s in enumerate(end_tensor_shape)
            if ((s == w_value) and (i != anchor_idx) and (i != h_index))
        ]
        if len(w_index) == 0:
            log_error(
                f"Width value can't be determined for tensor '{end_tensor_name}'."
            )
            return False, None
        w_index = w_index[0]
        if w_index != h_index + 1:
            # Width and height index should be in HW order only.
            log_error(
                f"Width value can't be determined for tensor '{end_tensor_name}'."
            )
            return False, None
        index_dict["h_idx"] = h_index
        index_dict["w_idx"] = w_index
        index_dict["a_idx"] = anchor_idx
        return True, index_dict
    else:
        log_error(
            f"Can not determine h, w and anchor axis shape for tensor '{end_tensor_name}'"
        )
        return False, None


def dump_yaml(data_dict: Dict, yaml_path: str) -> None:
    """
    Create a yaml file based on given dict at the provided path.

    :param Dict data_dict: Data to be dumped in the yaml.
    :param str yaml_path: Yaml file path.
    """
    with open(yaml_path, "w") as f:
        yaml.dump(data_dict, f)
