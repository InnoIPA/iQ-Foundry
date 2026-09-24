##############################################################################
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# Source: https://github.com/fizyr/keras-retinanet
# License: https://github.com/fizyr/keras-retinanet/blob/main/LICENSE
##############################################################################
# MIT License
# Copyright (c) 2020 Ayushman Buragohain
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# Source: https://github.com/benihime91/pytorch_retinanet
# License: https://github.com/benihime91/pytorch_retinanet/blob/master/LICENSE
##############################################################################
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


import math
import os
from typing import Iterable

import numpy as np
from PIL import Image
from qti.aisw.tools.core.utilities.data_processing import ImageRepresentation, PostProcessor
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class MlCommonsRetinaNetPostProcessor(PostProcessor):
    """A post-processor for MlCommons RetinaNet models.

    Args:
        image_dimensions (tuple[int]): Dimensions of the images in the dataset.
        prior_boxes_file_path (os.PathLike): Path to the file containing prior boxes.
        score_threshold (float): Minimum confidence score for a detection to be considered valid.
        nms_threshold (float): IoU threshold for non-maximum suppression.
        max_detections_per_image (int): Maximum number of detections allowed per image.
        num_classes_in_dataset (int): Number of classes in the dataset.
        feature_map_dimensions (tuple[int]): Dimensions of feature maps from FPN.
    """

    def __init__(
        self,
        image_dimensions: tuple[int],
        prior_boxes_file_path: os.PathLike,
        score_threshold: float,
        nms_threshold: float,
        max_detections_per_image: int,
        num_classes_in_dataset: int,
        feature_map_dimensions: tuple[int],
    ):
        """Initializes the post-processor for RetinaNet model.

        Args:
            image_dimensions (tuple[int]): Dimensions of the images in the dataset.
            prior_boxes_file_path (os.PathLike): Path to the file containing prior boxes.
            score_threshold (float): Minimum confidence score for a detection to be considered valid.
            nms_threshold (float): IoU threshold for non-maximum suppression.
            max_detections_per_image (int): Maximum number of detections allowed per image.
            num_classes_in_dataset (int): Number of classes in the dataset.
            feature_map_dimensions (tuple[int]): Dimensions of feature maps from FPN.

        """
        self.image_dimensions = image_dimensions
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections_per_image = max_detections_per_image
        self.num_classes_in_dataset = num_classes_in_dataset
        self.feature_map_dimensions = feature_map_dimensions
        self.prior_boxes_file_path = prior_boxes_file_path
        # Load prior boxes from the file.
        torch = Helper.safe_import_package("torch", "2.4.0")
        self.validate()
        self.anchors = np.fromfile(prior_boxes_file_path, dtype=np.float32).reshape(-1, 4)
        self.anchors = torch.from_numpy(self.anchors)

    def validate(self):
        """Validate the MlCommonsRetinaNetPostProcessor parameters provided.
        Checks that all required parameters are present and have valid values.

        - raises ValueError: If any of the following conditions are met:
            - `prior_boxes_file_path` is not a string or path-like object, or does not exist
            - Any image dimension is invalid (less than 1)
            - `score_threshold` or `nms_threshold` is not a number between 0 and 1
            - `max_detections_per_image` is less than 0
            - `num_classes_in_dataset` is less than 1
        - raises TypeError: If any of the following conditions are met:
            - `score_threshold`, `nms_threshold`, or `max_detections_per_image` are not numbers
            - Any image dimension is not an integer
        """
        if not isinstance(self.prior_boxes_file_path, (str, os.PathLike)) or not os.path.exists(
            self.prior_boxes_file_path
        ):
            raise ValueError("Prior boxes file path is invalid or does not exist.")

        if not isinstance(self.image_dimensions, Iterable) or len(self.image_dimensions) != 2:
            raise ValueError("image dimensions must be provided as (height, width)")

        for dim in self.image_dimensions:
            if not isinstance(dim, int) or dim < 1:
                raise ValueError(f"Invalid image dimension: {dim}")

        if (
            not isinstance(self.score_threshold, (int, float))
            or self.score_threshold < 0
            or self.score_threshold > 1
        ):
            raise ValueError(
                f"Score threshold must be a number between 0 and 1. Given: {self.score_threshold}"
            )

        if (
            not isinstance(self.nms_threshold, (int, float))
            or self.nms_threshold < 0
            or self.nms_threshold > 1
        ):
            raise ValueError(f"NMS threshold must be a number between 0 and 1. Given: {self.nms_threshold}")

        if not isinstance(self.max_detections_per_image, int) or self.max_detections_per_image < 1:
            raise ValueError(
                "Maximum detections per image must be an postive number greater"
                f" than or equal to 1. Given: {self.max_detections_per_image}"
            )

        if not isinstance(self.num_classes_in_dataset, int) or self.num_classes_in_dataset < 1:
            raise ValueError(
                "Number of classes must be an postive number greater than or"
                f" equal to 1. Given: {self.num_classes_in_dataset}"
            )

        if not isinstance(self.feature_map_dimensions, Iterable):
            raise ValueError(
                "Feature map dimensions must be provided as a tuple of numbers, e.g [250, 400, 9000]"
            )
        for dim in self.feature_map_dimensions:
            if not isinstance(dim, int) or dim < 1:
                raise ValueError(f"Invalid feature map dimension: {dim}")

    def execute(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Execute the post-processing on an image representation.

        Args:
            input_sample (ImageRepresentation): Input image representation.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        torchvision = Helper.safe_import_package("torchvision", "0.19.0")
        orig_image = input_sample.metadata["source_paths"][0]
        image_name = os.path.basename(orig_image)
        image_src = Image.open(orig_image)
        image_shape = list(image_src.size)

        # Process boxes and scores for each level of feature pyramid.
        image_boxes = []
        image_scores = []
        image_labels = []
        num_anchors = 0

        for level_idx, feature_dim in enumerate(self.feature_map_dimensions):
            boxes = input_sample.data[level_idx][0]
            scores = input_sample.data[level_idx + 5][0]
            topk_idxs = input_sample.data[level_idx + 10][0]

            # Convert to PyTorch tensors.
            boxes = torch.from_numpy(boxes)
            scores = torch.from_numpy(scores)
            topk_idxs = torch.from_numpy(topk_idxs)

            anchor_idxs = torch.div(topk_idxs, self.num_classes_in_dataset, rounding_mode="floor")
            keep_idxs = scores >= self.score_threshold
            boxes_per_level = boxes[keep_idxs]
            scores_per_level = scores[keep_idxs]
            labels_per_level = topk_idxs[keep_idxs] % self.num_classes_in_dataset
            anchor_idxs_per_level = anchor_idxs[keep_idxs]

            # Get anchors for the current level.
            anchors_per_level = self.anchors[num_anchors : num_anchors + feature_dim]
            num_anchors += feature_dim

            # Transform boxes to image coordinates and clip them if necessary.
            boxes_per_level = self.bbox_transform_inv(
                boxes_per_level, anchors_per_level[anchor_idxs_per_level]
            )
            boxes_per_level = self.clip_boxes(boxes_per_level, self.image_dimensions)
            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)

        # Merge processed data for all levels.
        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        # Perform non-maximum suppression.
        keep_mask = torch.zeros_like(image_scores, dtype=torch.bool)
        for class_id in torch.unique(image_labels):
            curr_indices = torch.where(image_labels == class_id)[0]
            curr_keep_indices = torchvision.ops.nms(
                image_boxes[curr_indices], image_scores[curr_indices], self.nms_threshold
            )
            keep_mask[curr_indices[curr_keep_indices]] = True

        # Get top-scoring detections.
        keep_indices = torch.where(keep_mask)[0]
        keep = keep_indices[image_scores[keep_indices].sort(descending=True)[1]]
        keep = keep_indices[: self.max_detections_per_image]

        # Prepare output data.
        dets = image_boxes[keep]
        scores = image_scores[keep]
        labels = image_labels[keep]
        dets[:, 0:4:2], dets[:, 1:4:2] = (
            (dets[:, 0:4:2] * image_shape[0]) / self.image_dimensions[0],
            (dets[:, 1:4:2] * image_shape[1]) / self.image_dimensions[1],
        )
        final_output = [
            [
                float(labels[ind]),
                float(scores[ind]),
                float(j[0]),
                float(j[1]),
                float(j[2] - j[0]),
                float(j[3] - j[1]),
            ]
            for ind, j in enumerate(dets)
        ]

        # Prepare output string.
        entry = f"{image_name},{len(final_output)}"
        if final_output:
            entry += "," + ",".join(",".join(map(str, i)) for i in final_output)
        entry += "\n"

        # Update input sample data with processed output string.
        input_sample.data = [entry]
        return input_sample

    def bbox_transform_inv(self, detections: "torch.Tensor", anchor_boxes: "torch.Tensor") -> "torch.Tensor":
        """Inverse bounding box transformation from model output to original coordinates.

        Args:
            detections (torch.Tensor): Model output containing predicted bounding box offsets.
            anchor_boxes (torch.Tensor): Anchor boxes for which the prediction is made.

        Returns:
            torch.Tensor: Predicted bounding box coordinates in xywh format.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        bbox_xform_clip = math.log(1000.0 / 16)
        widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
        heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]
        ctr_x = anchor_boxes[:, 0] + 0.5 * widths
        ctr_y = anchor_boxes[:, 1] + 0.5 * heights
        wx, wy, ww, wh = 1, 1, 1, 1
        dx = detections[:, 0::4] / wx
        dy = detections[:, 1::4] / wy
        dw = detections[:, 2::4] / ww
        dh = detections[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=bbox_xform_clip)
        dh = torch.clamp(dh, max=bbox_xform_clip)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_width = torch.exp(dw) * widths[:, None]
        pred_height = torch.exp(dh) * heights[:, None]

        # Distance from center to box's corner.
        c_to_c_h = 0.5 * pred_height
        c_to_c_w = 0.5 * pred_width
        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_x + c_to_c_w
        pred_boxes4 = pred_ctr_y + c_to_c_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes

    def clip_boxes(self, bounding_boxes: "torch.Tensor", image_shape: tuple) -> "torch.Tensor":
        """Clips the bounding boxes which exceed the image boundaries.

        Args:
            bounding_boxes: A tensor of shape [N, 4* num_classes] containing the box coordinates.
            image_shape: A tuple of two integers representing the height and width of the image.

        Returns:
            torch.Tensor: Clipped bounding boxes.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        bounding_boxes[:, 0] = torch.clamp(bounding_boxes[:, 0], min=0)  # x-coordinate
        bounding_boxes[:, 1] = torch.clamp(bounding_boxes[:, 1], min=0)  # y-coordinate
        bounding_boxes[:, 2] = torch.clamp(bounding_boxes[:, 2], max=image_shape[0])  # width
        bounding_boxes[:, 3] = torch.clamp(bounding_boxes[:, 3], max=image_shape[1])  # height

        return bounding_boxes
