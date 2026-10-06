##############################################################################

# MIT License

# Copyright (c) 2019 StarClouds

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Source: https://github.com/Star-Clouds/CenterFace/blob/master/prj-python/
# License: https://github.com/Star-Clouds/CenterFace

##############################################################################
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os

import numpy as np
from PIL import Image
from qti.aisw.tools.core.utilities.data_processing import (
    NDArrayRepresentation,
    PostProcessor,
)


def non_maximum_suppression(detections: np.ndarray, overlap_threshold: float) -> list:
    """Perform Non-Maximum Suppression (NMS) on a list of detections. Greedily
    select boxes with high confidence and overlap with current maximum <=
    thresh, rule out overlap >= thresh.

    Args:
      detections: A 2D NumPy array where each row contains the box coordinates and score in the format
                [x1, y1, x2, y2, score].
      overlap_threshold: The minimum overlap ratio below which boxes are considered to be separate entities.
                Values range from 0.0 (completely non-overlapping) to 1.0 (completely overlapping).

    Returns:
      A list of indices corresponding to the top-scoring boxes that pass the NMS filtering criteria.
    """
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]
    # Calculate areas of all boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Sort boxes by score in descending order
    sorted_indices = scores.argsort()[::-1]

    # Initialize list to store indices of top-scoring boxes that pass NMS filtering
    kept_indices = []
    while sorted_indices.size > 0:
        # Select the highest-scoring box (the first one in the sorted array)
        i = sorted_indices[0]
        # Add this box's index to the list of kept indices
        kept_indices.append(i)
        # Calculate overlaps between the selected box and all other boxes
        xx1 = np.maximum(x1[i], x1[sorted_indices[1:]])
        yy1 = np.maximum(y1[i], y1[sorted_indices[1:]])
        xx2 = np.minimum(x2[i], x2[sorted_indices[1:]])
        yy2 = np.minimum(y2[i], y2[sorted_indices[1:]])
        # Calculate widths and heights of the overlap regions
        w_overlaps = np.maximum(0.0, xx2 - xx1 + 1)
        h_overlaps = np.maximum(0.0, yy2 - yy1 + 1)

        # Calculate areas of the overlap regions
        overlaps_areas = w_overlaps * h_overlaps

        # Calculate overlap ratios
        overlaps_ratios = overlaps_areas / (areas[i] + areas[sorted_indices[1:]] - overlaps_areas)

        # Find indices of boxes with overlap ratio greater than threshold
        selected_indices = np.where(overlaps_ratios <= overlap_threshold)[0]
        # Remove these indices from the sorted array
        sorted_indices = sorted_indices[selected_indices + 1]
    return kept_indices


class CenterFacePostProcessor(PostProcessor):
    """Postprocessor for centerface model.

    Attributes:
        image_dimensions (tuple[int, int]): Dimensions of the input image in pixels.
        heatmap_threshold (float): Minimum confidence score to consider a detection as valid.
        nms_threshold (float): Non-maximum suppression threshold for detecting multiple detections per object.
    """

    def __init__(
        self, image_dimensions: tuple[int, int], heatmap_threshold: float = 0.05, nms_threshold: float = 0.3
    ) -> None:
        """Initializes the postprocessor.

        Args:
            image_dimensions (tuple[int, int]): Dimensions of the input image in pixels.
            heatmap_threshold (float): Minimum confidence score to consider a detection as valid.
            nms_threshold (float): Non-maximum suppression threshold for detecting
                                multiple detections per object.
        """
        self.image_dimensions = image_dimensions
        self.heatmap_threshold = heatmap_threshold
        self.nms_threshold = nms_threshold
        self.validate()

    def validate(self):
        """Validate the input parameters."""
        if len(self.image_dimensions) < 2:
            raise ValueError("image_dimensions must be a tuple of at least two positive integers")
        elif not all((isinstance(dim, int) and dim > 0) for dim in self.image_dimensions):
            raise ValueError("All dimensions in image_dimensions must be positive integers")
        if not isinstance(self.heatmap_threshold, float) or self.heatmap_threshold < 0:
            raise ValueError("heatmap threshold must be a float >= 0")
        if not isinstance(self.nms_threshold, float) and not (1 < self.nms_threshold <= 0):
            raise ValueError("NMS threshold must be in the range (0, 1)")

    def execute(self, input_sample: NDArrayRepresentation) -> NDArrayRepresentation:
        """Processing the inference outputs.

        Args:
            input_sample: The input data from the model output.

        Returns:
            The processed output with bounding boxes and keypoints.
        """
        # Get the input data
        input_data = input_sample.data

        # Reshape all model outputs
        heatmap = input_data[0]
        scale = input_data[1]
        offset = input_data[2]
        landmark_map = input_data[3]

        # Reshape the heatmap to a single value
        heatmap = np.squeeze(heatmap)

        # Get the scales and offsets for x and y coordinates
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        # Get the indices where the heatmap value is greater than the threshold
        x_indices, y_indices = np.where(heatmap > self.heatmap_threshold)

        # Get the original image metadata
        orig_image = input_sample.metadata["source_paths"][0]
        image_name = os.path.basename(orig_image)
        image_src = Image.open(orig_image)
        image_shape = list(image_src.size)

        # Get input dimensions
        size = [
            int(self.image_dimensions[1]),
            int(self.image_dimensions[0]),
        ]  # height, width

        # Extract Pad-size, pad_h or pad_w
        # Larger dimension will have 0 padding
        # Other Axis will have size difference / 2
        # In case size is odd, extra pad line is added on the right and at the bottom
        # Pad-size calculated based on padding algo in image_operations.py
        img_scale = min(size[0] / image_shape[0], size[1] / image_shape[1])
        pad = [
            round(((size[0] - int(image_shape[0] * img_scale)) / 2) - 0.1),
            round(((size[1] - int(image_shape[1] * img_scale)) / 2) - 0.1),
        ]
        # Calculate pad width and height
        pad_w = pad[0]
        pad_h = pad[1]

        landmark = landmark_map
        boxes, landmark_map = [], []
        if len(x_indices) > 0:
            for i in range(len(x_indices)):
                s0, s1 = (
                    np.exp(scale0[x_indices[i], y_indices[i]]) * 4,
                    np.exp(scale1[x_indices[i], y_indices[i]]) * 4,
                )
                o0, o1 = (
                    offset0[x_indices[i], y_indices[i]],
                    offset1[x_indices[i], y_indices[i]],
                )
                s = heatmap[x_indices[i], y_indices[i]]
                x1, y1 = (
                    max(0, (y_indices[i] + o1 + 0.5) * 4 - s1 / 2),
                    max(0, (x_indices[i] + o0 + 0.5) * 4 - s0 / 2),
                )
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                # Create box
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])

                lm = []
                for j in range(5):
                    lm.append(landmark[0, j * 2 + 1, x_indices[i], y_indices[i]] * s1 + x1)
                    lm.append(landmark[0, j * 2, x_indices[i], y_indices[i]] * s0 + y1)
                # Create landmark map
                landmark_map.append(lm)

            boxes = np.asarray(boxes, dtype=np.float32)
            # Perform non-maximum suppression
            keep = non_maximum_suppression(boxes, self.nms_threshold)
            # Filter boxes based on NMS results
            boxes = boxes[keep, :]
            landmark_map = np.asarray(landmark_map, dtype=np.float32)
            landmark_map = landmark_map[keep, :]

        dets = boxes
        if len(dets) > 0:
            # Adjust box coordinates

            dets[:, 0:4:2] = (dets[:, 0:4:2] - pad_w) / img_scale
            dets[:, 1:4:2] = (dets[:, 1:4:2] - pad_h) / img_scale
            landmark_map[:, 0:10:2] = (landmark_map[:, 0:10:2] - pad_w) / img_scale
            landmark_map[:, 1:10:2] = (landmark_map[:, 1:10:2] - pad_h) / img_scale
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            landmark_map = np.empty(shape=[0, 10], dtype=np.float32)

        # Convert boxes to yolo format
        bboxes = [[int(i[0]), int(i[1]), int(i[2]) - int(i[0]), int(i[3]) - int(i[1]), i[4]] for i in boxes]

        name = orig_image[(orig_image[0 : len(orig_image) - len(image_name) - 1]).rfind("/") + 1 :]
        processed_output = []
        processed_output.append(name.split("/")[0])
        processed_output.append(name.split("/")[1].split(".")[0])
        processed_output.append(np.array(bboxes))
        # Update input sample with processed output
        input_sample.data = processed_output
        return input_sample
