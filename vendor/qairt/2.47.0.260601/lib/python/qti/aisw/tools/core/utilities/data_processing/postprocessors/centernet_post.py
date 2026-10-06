##############################################################################
# MIT License

# Copyright (c) 2019 Xingyi Zhou
# All rights reserved.

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

# Source: https://github.com/xingyizhou/CenterNet#object-detection-on-coco-validation
# License: https://github.com/xingyizhou/CenterNet/blob/master/LICENSE

##############################################################################
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
from typing import Any, Optional

import cv2
import numpy as np
from qti.aisw.tools.core.utilities.data_processing import (
    ImageRepresentation,
    PostProcessor,
)
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class CenterNetPostProcessor(PostProcessor):
    """Postprocessing class for CenterNet Model"""

    def __init__(
        self,
        output_dimensions: (int, int),
        top_k: int = 100,
        num_classes: int = 1,
        score_threshold: float = 0.1,
    ):
        """Initialize the CenterNetPostProcessor.

        Args:
            output_dimensions (tuple[int, int]): Output dimensions of the model.
            top_k (int, optional): Top K values to consider. Defaults to 100.
            num_classes (int, optional): Number of classes. Defaults to 1.
            score_threshold (float, optional): Score threshold for detections. Defaults to 0.1.
        """
        self.top_k = top_k
        self.output_dimensions = output_dimensions
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.validate()

    def validate(self):
        """Validate the parameters."""
        if len(self.output_dimensions) < 2 or not all([int(i) > 0 for i in self.output_dimensions]):
            raise ValueError(
                "Invalid output dimensions. Expected a list of two non negative"
                f" integers, got: {self.output_dimensions}"
            )
        if not isinstance(self.top_k, int):
            raise TypeError(f"Invalid top k value. Expected an integer, got {type(self.top_k)}")
        if self.num_classes < 1 or not isinstance(self.num_classes, int):
            raise ValueError(
                f"Invalid number of classes. Expected a positive integer, got: {self.num_classes}"
            )
        if not (0 <= self.score_threshold and self.score_threshold <= 1):
            raise ValueError(
                f"Invalid score_threshold value. Expected a float between 0 & 1, got: {self.score_threshold}"
            )

    def validate_input(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Validate the input sample to ensure it contains the required metadata.

        Args:
            input_sample (ImageRepresentation): The input sample to be validated.

        Returns:
            ImageRepresentation: The validated input sample.

        Raises:
            ValueError: If the input sample is invalid.
        """
        if "source_paths" not in input_sample.metadata:
            raise ValueError("Invalid input sample. Missing original image path required for processing")
        return input_sample

    @PostProcessor.validate_input_output
    def execute(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Execute the post-processing for the given input sample.

        Args:
            input_sample (ImageRepresentation): Input sample to process.

        Returns:
            ImageRepresentation: Processed image representation.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        # Get the output from the model
        input_data = input_sample.data
        out_height, out_width = self.output_dimensions
        orig_image_path = input_sample.metadata["source_paths"][0]
        image_name = os.path.basename(orig_image_path)
        orig_image = cv2.imread(orig_image_path)

        # Get the height and width of the original image
        height, width = orig_image.shape[0:2]

        # Resize the output dimensions to match the original image size
        new_height = int(height)
        new_width = int(width)

        # Compute the center coordinates (c) and scale factor (s) for resizing
        c = np.array([new_width / 2.0, new_height / 2.0], dtype=np.float32)
        s = max(height, width) * 1.0
        scale = 1.0

        output = {}
        node_names = ["hm", "wh", "hps", "reg", "hm_hp", "hp_offset"]

        # Load the model outputs into a dictionary
        for idx, node_name in enumerate(node_names):
            output[node_name] = torch.from_numpy(input_data[idx])

        # Apply sigmoid to hm and hm_hp outputs
        output["hm"] = output["hm"].sigmoid_()
        output["hm_hp"] = output["hm_hp"].sigmoid_()

        # Get the regression (reg), heatmap (hm_hp), and head point offset (hp_offset) outputs
        reg = output["reg"]
        hm_hp = output["hm_hp"]
        hp_offset = output["hp_offset"]

        # Perform multi-pose decoding on the model outputs
        detections = self.multi_pose_decode(
            heat=output["hm"],
            wh=output["wh"],
            keypoints=output["hps"],
            reg_offset=reg,
            head_heatmap=hm_hp,
            head_offset=hp_offset,
            K=self.top_k,
        )

        # Convert the detections to a numpy array and reshape it
        detections = detections.detach().cpu().numpy().reshape(1, -1, detections.shape[2])

        # Perform multi-pose post-processing on the detections
        detections = self.multi_pose_post_process(detections.copy(), c, s, out_height, out_width)

        # Iterate over each class and convert the detections to float32 format
        for j in range(1, self.num_classes + 1):
            detections[0][j] = np.array(detections[0][j], dtype=np.float32).reshape(-1, 39)
            detections[0][j][:, :4] /= scale
            detections[0][j][:, 5:] /= scale

        # Select the category id and extract the detections for that class
        category_id = 1
        selected_detections = detections[0][category_id]

        # Filter out detections with low scores and store them in a list
        filtered_detections = []
        for det in selected_detections:
            if det[4] > self.score_threshold:
                filtered_detections.append(det)

        # Create the output string
        entry = image_name.split(".")[0]
        entry += "," + str(len(filtered_detections))

        if len(filtered_detections) > 0:
            for det in filtered_detections:
                bbox = det[:4]
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]

                # Create the detection string
                det_str = f"{category_id},{det[4]},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                entry += "," + det_str

                # Append additional detection data
                for i in range(len(det[5 : len(det)])):
                    entry += "," + str(det[i + 5])

        # Update the input sample with the output string and return it
        input_sample.data = [entry]
        return input_sample

    def multi_pose_decode(
        self,
        heat: "torch.Tensor",  # noqa: F821
        wh: "torch.Tensor",  # noqa: F821
        keypoints: "torch.Tensor",  # noqa: F821
        reg_offset: "torch.Tensor" = None,  # noqa: F821
        head_heatmap: "torch.Tensor" = None,  # noqa: F821
        head_offset: "torch.Tensor" = None,  # noqa: F821
        K: int = 100,
    ) -> "torch.Tensor":  # noqa: F821
        """Decode the output of the CenterNet model.

        Args:
            heat (torch.Tensor): The heatmap output from the model.
            wh (torch.Tensor): The wh output from the  model.
            keypoints (torch.Tensor): The keypoints output from the model.
            reg_offset (torch.Tensor, optional): The regression offset output
                                     from the model. Defaults to None.
            head_heatmap (torch.Tensor, optional): The heatmap for head keypoints output
                                     from the model. Defaults to None.
            head_offset (torch.Tensor, optional): The offset for head keypoints output
                                     from the model. Defaults to None.
            K (int, optional): The number of top detections to return. Defaults to 100.

        Returns:
            torch.Tensor: The decoded detection tensor.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        batch_size = heat.size()[0]

        num_keypoints = keypoints.shape[1] // 2

        # perform nms on heatmap
        heat = self.nms(heat)

        scores, inds, classes, y_coords, x_coords = self.topk(heat, k=K)

        keypoints = self.transpose_and_gather_feature(keypoints, inds)
        keypoints = keypoints.view(batch_size, K, num_keypoints * 2)
        keypoints[..., ::2] += x_coords.view(batch_size, K, 1).expand(batch_size, K, num_keypoints)
        keypoints[..., 1::2] += y_coords.view(batch_size, K, 1).expand(batch_size, K, num_keypoints)

        if reg_offset is not None:
            reg_offset = self.transpose_and_gather_feature(reg_offset, inds)
            reg_offset = reg_offset.view(batch_size, K, 2)
            x_coords = x_coords.view(batch_size, K, 1) + reg_offset[:, :, 0:1]
            y_coords = y_coords.view(batch_size, K, 1) + reg_offset[:, :, 1:2]
        else:
            x_coords = x_coords.view(batch_size, K, 1) + 0.5
            y_coords = y_coords.view(batch_size, K, 1) + 0.5

        wh = self.transpose_and_gather_feature(wh, inds)
        wh = wh.view(batch_size, K, 2)

        classes = classes.view(batch_size, K, 1).float()
        scores = scores.view(batch_size, K, 1)

        bounding_boxes = torch.cat(
            [
                x_coords - wh[..., 0:1] / 2,
                y_coords - wh[..., 1:2] / 2,
                x_coords + wh[..., 0:1] / 2,
                y_coords + wh[..., 1:2] / 2,
            ],
            dim=2,
        )
        if head_heatmap is not None:
            head_heatmap = self.nms(head_heatmap)
            head_threshold = 0.1
            keypoints = keypoints.view(batch_size, K, num_keypoints, 2).permute(0, 2, 1, 3).contiguous()
            reg_keypoints = keypoints.unsqueeze(3).expand(batch_size, num_keypoints, K, K, 2)

            head_scores, head_inds, head_y_coords, head_x_coords = self.topk_channel(head_heatmap, K=K)

            if head_offset is not None:
                head_offset = self.transpose_and_gather_feature(head_offset, head_inds.view(batch_size, -1))
                head_offset = head_offset.view(batch_size, num_keypoints, K, 2)
                head_x_coords = head_x_coords + head_offset[:, :, :, 0]
                head_y_coords = head_y_coords + head_offset[:, :, :, 1]
            else:
                head_x_coords = head_x_coords + 0.5
                head_y_coords = head_y_coords + 0.5

            head_mask = (head_scores > head_threshold).float()
            head_scores = (1 - head_mask) * -1 + head_mask * head_scores
            head_y_coords = (1 - head_mask) * (-10000) + head_mask * head_y_coords
            head_x_coords = (1 - head_mask) * (-10000) + head_mask * head_x_coords

            head_keypoints = (
                torch.stack([head_x_coords, head_y_coords], dim=-1)
                .unsqueeze(2)
                .expand(batch_size, num_keypoints, K, K, 2)
            )

            distance = ((reg_keypoints - head_keypoints) ** 2).sum(dim=4) ** 0.5
            min_distance, min_ind = distance.min(dim=3)  # b x J x K

            head_scores = head_scores.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
            min_distance = min_distance.unsqueeze(-1)

            min_ind = min_ind.view(batch_size, num_keypoints, K, 1, 1).expand(
                batch_size, num_keypoints, K, 1, 2
            )
            head_keypoints = head_keypoints.gather(3, min_ind)
            head_keypoints = head_keypoints.view(batch_size, num_keypoints, K, 2)

            left = bounding_boxes[:, :, 0].view(batch_size, 1, K, 1).expand(batch_size, num_keypoints, K, 1)
            top = bounding_boxes[:, :, 1].view(batch_size, 1, K, 1).expand(batch_size, num_keypoints, K, 1)
            right = bounding_boxes[:, :, 2].view(batch_size, 1, K, 1).expand(batch_size, num_keypoints, K, 1)
            bottom = bounding_boxes[:, :, 3].view(batch_size, 1, K, 1).expand(batch_size, num_keypoints, K, 1)

            head_mask = (
                (head_keypoints[..., 0:1] < left)
                + (head_keypoints[..., 0:1] > right)
                + (head_keypoints[..., 1:2] < top)
                + (head_keypoints[..., 1:2] > bottom)
                + (head_scores < head_threshold)
                + (min_distance > torch.max(bottom - top, right - left) * 0.3)
            )

            head_mask = (head_mask > 0).float().expand(batch_size, num_keypoints, K, 2)
            keypoints = (1 - head_mask) * head_keypoints + head_mask * keypoints
            keypoints = keypoints.permute(0, 2, 1, 3).contiguous().view(batch_size, K, num_keypoints * 2)
        detections = torch.cat([bounding_boxes, scores, keypoints, classes], dim=2)
        return detections

    def multi_pose_post_process(
        self,
        detection_results: "torch.Tensor",  # noqa: F821
        image_center: tuple[float, float],
        scale_factor: float,
        image_height: int,
        image_width: int,
    ) -> list[dict[int, Any]]:
        """Post-process the detection output from the CenterNet model.

        This function takes in a tensor of detection results and returns a list of
        dictionaries containing the post-processed detection results.

        Args:
            detection_results (torch.Tensor): The detection tensor. Shape should be batch x max_dets x 40
            image_center (tuple[float, float]): A tuple containing the center coordinates of the image.
            scale_factor (float): The scale factor of the image.
            image_height (int): The height of the image.
            image_width (int): The width of the image.

        Returns:
            list[dict[int, Any]]: A list of dictionaries containing the post-processed detection results.
        """
        # Detection tensor shape should be batch x max_detections x 40
        predictions = []
        for i in range(detection_results.shape[0]):
            bounding_boxes = self.transform_preds(
                detection_results[i, :, :4].reshape(-1, 2),
                image_center,
                scale_factor,
                (image_width, image_height),
            )

            keypoints = self.transform_preds(
                detection_results[i, :, 5:39].reshape(-1, 2),
                image_center,
                scale_factor,
                (image_width, image_height),
            )

            # Get top predictions by concatenating bounding box coordinates and other attributes
            top_predictions = (
                np.concatenate(
                    [
                        bounding_boxes.reshape(-1, 4),
                        detection_results[i, :, 4:5],
                        keypoints.reshape(-1, 34),
                    ],
                    axis=1,
                )
                .astype(np.float32)
                .tolist()
            )
            # Convert top predictions to a list of dictionaries with key 1 and mapped to top predictions
            predictions.append({np.ones(1, dtype=np.int32)[0]: top_predictions})
        return predictions

    def transform_preds(
        self,
        coords: "torch.Tensor",  # noqa: F821
        center: tuple[float, float],
        scale: float,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        """Transform the coordinates of a point from image space to the desired
        output size.

        Args:
            coords (torch.Tensor): The coordinates of the point.
            center (tuple[float, float]): The center coordinates of the image.
            scale (float): The scale factor of the image.
            output_size (tuple[int, int]): The desired output size.

        Returns:
            np.ndarray: The transformed coordinates.
        """
        target_coords = np.zeros(coords.shape)
        trans = self.get_affine_transform(
            center, scale, rotation_angle_degrees=0, output_size=output_size, inverse_transform=1
        )
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = self.affine_transform(coords[p, 0:2], trans)
        return target_coords

    def affine_transform(self, pt: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Apply an affine transformation to a point.

        Args:
            pt (np.ndarray): The original coordinates of the point.
            t (np.ndarray): The affine transformation matrix.

        Returns:
            np.ndarray: The transformed coordinates of the point.
        """
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def get_affine_transform(
        self,
        image_center: tuple[float, float],
        scale_factor: float,
        rotation_angle_degrees: float,
        output_size: tuple[int, int],
        shift_vector: np.ndarray = np.array([0, 0], dtype=np.float32),
        inverse_transform: bool = False,
    ) -> np.ndarray:
        """Compute the affine transformation matrix.

        Args:
            image_center (tuple[float, float]): The center coordinates of the image.
            scale_factor (float): The scale factor of the image.
            rotation_angle_degrees (float): The rotation angle in degrees.
            output_size (tuple[int, int]): The desired output size.
            shift_vector (np.ndarray, optional): The shift vector.
                         Defaults to np.array([0, 0], dtype=np.float32).
            inverse_transform (bool, optional): Whether to compute the inverse
                         transformation matrix. Defaults to False.

        Returns:
            np.ndarray: The affine transformation matrix.
        """
        if not isinstance(scale_factor, np.ndarray) and not isinstance(scale_factor, list):
            scale_factor = np.array([scale_factor, scale_factor], dtype=np.float32)
        # Scale factor for width and height
        scale_tmp = scale_factor
        # Source image width
        src_width = scale_tmp[0]

        # Destination image width and height
        dst_width = output_size[0]
        dst_height = output_size[1]

        # Convert rotation angle to radians
        rot_rad = np.pi * rotation_angle_degrees / 180
        # Source and Destination direction vector
        src_dir = self.get_dir([0, src_width * -0.5], rot_rad)
        dst_dir = np.array([0, dst_width * -0.5], np.float32)

        # Initialize source and destination points arrays
        src_points = np.zeros((3, 2), dtype=np.float32)
        dst_points = np.zeros((3, 2), dtype=np.float32)
        # Set the center point as a source point
        src_points[0, :] = image_center + scale_tmp * shift_vector
        # Calculate two additional points on the edge of the image
        # (top-left and top-right corners)
        src_points[1, :] = image_center + src_dir + scale_tmp * shift_vector
        dst_points[0, :] = [dst_width / 2.0, dst_height / 2.0]
        # Add destination direction vector to the corner points
        dst_points[1, :] = np.array([dst_width * 0.5, dst_height * 0.5], np.float32) + dst_dir

        # Calculate third point on the line between the first two points
        # for both source and destination images
        src_points[2:, :] = self.get_3rd_point(src_points[0, :], src_points[1, :])
        dst_points[2:, :] = self.get_3rd_point(dst_points[0, :], dst_points[1, :])

        if inverse_transform:
            # Get the affine transformation matrix for the inverse transform
            trans = cv2.getAffineTransform(np.float32(dst_points), np.float32(src_points))
        else:
            # Get the affine transformation matrix for the forward transform
            trans = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))

        return trans

    def get_dir(self, src_point: list[float], rot_rad: float) -> list[float]:
        """Compute the direction vector.

        Args:
            src_point (list[float]): The source point coordinates.
            rot_rad (float): The rotation angle in radians.

        Returns:
            list[float]: The direction vector.
        """
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        # Calculate the new x and y coordinates after shifting by rotation
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return src_result

    def get_3rd_point(self, a: list[float], b: list[float]) -> np.ndarray:
        """Compute the third point of an affine transformation.

        Args:
            a (list[float]): The first two points.
            b (list[float]): The second two points.

        Returns:
            np.ndarray: The third point coordinates.
        """
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def nms(self, heat_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Perform non-maximum suppression on a heatmap.

        Args:
            heat_map (np.ndarray): The heatmap.
            kernel_size (int, optional): The size of the kernel. Defaults to 3.

        Returns:
            np.ndarray: The filtered heatmap.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        pad = (kernel_size - 1) // 2
        hmax = torch.nn.functional.max_pool2d(heat_map, (kernel_size, kernel_size), stride=1, padding=pad)
        keep_mask = (hmax == heat_map).float()
        return heat_map * keep_mask

    def topk(self, scores: "torch.Tensor", k: int = 40) -> tuple["torch.Tensor"]:  # noqa: F821
        """Get the top K scores and corresponding indices.

        Args:
            scores (nn.Tensor): The input scores.
            k (int): The number of top scores to retrieve. Defaults to 40.

        Returns:
            tuple: A tuple containing the top K scores, indices, class labels,
                   y-coordinates, and x-coordinates.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        batch, cat, height, width = scores.size()

        # Get top K scores and corresponding indices in each grid cell
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k)

        # Calculate y-coordinates and x-coordinates for top-k indices
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # Get top K scores, class labels, y-coordinates, and x-coordinates
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
        topk_classes = (topk_ind / k).int()
        topk_inds = self.gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
        topk_ys = self.gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
        topk_xs = self.gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)

        return topk_score, topk_inds, topk_classes, topk_ys, topk_xs

    def gather_feature(
        self,
        feature_map: "torch.Tensor",  # noqa: F821
        indices: "torch.Tensor",  # noqa: F821
        mask: Optional["torch.Tensor"] = None,  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        """Gather features from the feature map based on the given indices.

        Args:
            feature_map (torch.Tensor): The feature map tensor.
            indices (torch.Tensor): The indices to gather features at.
            mask (Optional[torch.Tensor], optional): A binary mask to apply
                                        when gathering features. Defaults to None.

        Returns:
            torch.Tensor: The gathered feature tensor.
        """
        num_dimensions = feature_map.size(2)
        indices_tensor = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), num_dimensions)
        feature_map_gathered = feature_map.gather(1, indices_tensor)
        if mask is not None:
            mask_tensor = mask.unsqueeze(2).expand_as(feature_map_gathered)
            feature_map_gathered = feature_map_gathered[mask_tensor]
            feature_map_gathered = feature_map_gathered.view(-1, num_dimensions)
        return feature_map_gathered

    def transpose_and_gather_feature(
        self,
        feature_map: "torch.Tensor",  # noqa: F821
        indices: "torch.Tensor",  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        """Transpose and gather features from the feature map based on the
        given indices.

        Args:
            feature_map (torch.Tensor): The feature map tensor.
            indices (torch.Tensor): The indices to gather features at.

        Returns:
            torch.Tensor: The transposed and gathered feature tensor.
        """
        feature_transposed = feature_map.permute(0, 2, 3, 1).contiguous()
        feature_viewed = feature_transposed.view(feature_transposed.size(0), -1, feature_transposed.size(3))
        gathered_feature = self.gather_feature(feature_viewed, indices)
        return gathered_feature

    def topk_channel(self, scores: "torch.Tensor", K: int = 40) -> tuple["torch.Tensor"]:  # noqa: F821
        """Perform top-K channel operation on the given scores tensor.

        Args:
            scores (torch.Tensor): The input scores tensor.
            K (int, optional): The number of top scores to keep. Defaults to 40.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            the top-K scores,  their corresponding indices, and the y/x coordinates of these indices.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        # Get batch size, category count, height, and width from the input tensor
        batch, cat, height, width = scores.size()

        # Use top-K operation to get the top scores and their indices
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        # Extract y/x coordinates from the indices
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs
