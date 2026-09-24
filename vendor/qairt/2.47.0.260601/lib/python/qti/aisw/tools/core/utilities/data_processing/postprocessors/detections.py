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
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image
from qti.aisw.tools.core.utilities.data_processing import (
    ImageRepresentation,
    PostProcessor,
)
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class ObjectDetectionPostProcessor(PostProcessor):
    """Post Processing class for  Object-detection models"""

    def __init__(
        self,
        image_dimensions: tuple[int, int],
        type: Optional[Literal["letterbox", "stretch", "aspect_ratio", "orgimage"]] = None,
        mask_dims: str = None,
        label_offset: int = 0,
        score_threshold: float = 0.001,
        skip_padding: bool = False,
        scale: str = "1",
        mask: bool = False,
    ):
        """Initialize the ObjectDetectionPostProcessor.

        Args:
            image_dimensions (tuple[int, int]): Output dimensions of the model.
            type (Optional[Literal['letterbox', 'stretch', 'aspect_ratio', 'orgimage']]): The type of
                post-processing to apply. Defaults to None.
            mask_dims (str): The dimensions for the mask. Defaults to None.
            label_offset (int): The offset to apply to the label indices. Defaults to 0.
            score_threshold (float): Score threshold for detections. Defaults to 0.001.
            skip_padding (bool): Whether to skip padding during postprocessing. Default is False.
            scale (str): The scaling factor. Default is "1".
            mask (bool): Whether to include mask processing. Default is False.
        """
        self.image_dimensions = image_dimensions
        self.type = type
        self.mask_dims = mask_dims
        self.label_offset = label_offset
        self.score_threshold = score_threshold
        self.skip_padding = skip_padding
        self.scale = scale
        self.mask = mask
        if self.mask_dims:
            self.mdims = mask_dims.split(",")
        self.validate()

    def validate(self):
        """Validate the parameters."""
        if len(self.image_dimensions) < 2:
            raise ValueError("image_dimensions must be a tuple of at least two positive integers")
        if not isinstance(self.type, str):
            raise TypeError("Invalid type value. Expected a string.")
        if self.type not in ["letterbox", "stretch", "aspect_ratio", "orgimage"]:
            raise ValueError(
                "type not supported. type must be one of 'letterbox', \
             'stretch', 'aspect_ratio', 'orgimage'"
            )
        if self.mask_dims is not None and not isinstance(self.mask_dims, str):
            raise ValueError("mask_dims does not exist")
        if not (0 <= self.score_threshold and self.score_threshold <= 1):
            raise ValueError("Invalid score_threshold value. Expected a float between 0 & 1")
        if not isinstance(self.skip_padding, bool):
            raise ValueError("skip_padding does not exist")
        if not isinstance(self.scale, str):
            raise TypeError("Invalid type value. Expected a string")
        if not isinstance(self.mask, bool):
            raise ValueError("mask does not exist")

    def execute(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Processing the inference outputs.

        Function expects the outputs in the
        order: bboxes, scores, labels and mask. Order to be controlled via config.yaml
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        pycocotools = Helper.safe_import_package("pycocotools", "2.0.7")  # noqa: F841
        input_data = input_sample.data
        bboxes = input_data[0]
        scores = input_data[1]
        labels = input_data[2]

        if len(input_data) == 3:
            bboxes = bboxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

        else:
            counts = input_data[3][0]
            if self.mask:
                mask_output = input_data[4]  # input_data[3] corresponds to count
                mask_output = mask_output.reshape(
                    1, int(self.mdims[0]), int(self.mdims[1]), int(self.mdims[2]), int(self.mdims[3])
                )
                mask_indices = np.arange(mask_output.shape[1])
                mask_prob = mask_output[:, mask_indices, labels][:, :, None]
                mask_prob = mask_prob.reshape(-1, 1, int(self.mdims[2]), int(self.mdims[3]))

            bboxes = bboxes.reshape(-1, 4)[:counts]
            scores = scores.reshape(-1)[:counts]
            labels = labels.reshape(-1)[:counts]

        height = int(self.image_dimensions[0])
        width = int(self.image_dimensions[1])

        try:
            orig_image_path = input_sample.metadata["source_paths"][0]
            image_name = os.path.basename(orig_image_path)
            image_src = Image.open(orig_image_path)
        except KeyError as e:
            raise ValueError(f"Invalid input sample. Missing original image. {e}")
        w, h = image_src.size

        labels -= self.label_offset

        if self.type == "letterbox":
            bboxes = torch.from_numpy(np.array(bboxes))
            bboxes = self.postprocess_letterbox(
                input_shape=(height, width), coords=bboxes, image_shape=(h, w), skip_padding=self.skip_padding
            )
        elif self.type == "aspect_ratio":
            bboxes = torch.from_numpy(np.array(bboxes))
            bboxes = self.postprocess_letterbox(
                input_shape=(height, width), coords=bboxes, image_shape=(h, w), skip_padding=True
            )
        elif self.type == "stretch":
            bboxes = self.postprocess_stretch(image_shape=(h, w), scale=self.scale, coords=bboxes)
        elif self.type == "orgimage":
            bboxes = torch.from_numpy(np.array(bboxes))
            bboxes = self.postprocess_orgimage(input_shape=(height, width), coords=bboxes, image_shape=(h, w))
        out_data = []
        entry = self.det_result(
            boxes=bboxes,
            confs=scores,
            labels=labels,
            image_name=image_name,
            score_threshold=self.score_threshold,
        )
        out_data.append(entry)

        if self.mask:
            pred_masks = self.resize_mask(mask_prob, bboxes, (h, w))
            # stores binary masks in RLE format
            # for more info link: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
            mask_util = Helper.safe_import_package("pycocotools.mask")
            mask_rles = [
                mask_util.encode(np.array(pmask[:, :, None], order="F", dtype="uint8"))[0]
                for pmask in pred_masks
            ]
            out_data.append(mask_rles)
        input_sample.data = out_data
        return input_sample

    def det_result(
        self,
        boxes: List[np.ndarray],
        confs: List[float],
        labels: List[int],
        image_name: str,
        score_threshold: float,
    ) -> str:
        """Generates a formatted string of detection results for an image

        Returns:
            str: A formatted string.
        """
        prev_box_entry = ""
        num_entry = 0
        temp_line = ""
        for i, box in enumerate(boxes):
            if confs[i] < score_threshold:
                continue
            x1, y1, x2, y2 = map(np.float32, box)
            box_entry = ""
            box_entry += "," + str(int(labels[i]))
            box_entry += "," + str(confs[i])
            box_entry += "," + str(x1.round(3)) + "," + str(y1.round(3))
            box_entry += "," + str((x2 - x1).round(3)) + "," + str((y2 - y1).round(3))
            if box_entry != prev_box_entry:
                temp_line += box_entry
                num_entry += 1
                prev_box_entry = box_entry

        curr_line = "{},{}{} \n".format(image_name, num_entry, temp_line)
        return curr_line

    def postprocess_letterbox(
        self,
        input_shape: Tuple[int, int],
        coords: Any,
        image_shape: Tuple[int, int],
        skip_padding: bool = False,
    ) -> Any:
        """Post-processes bounding box coordinates using the letterboxing technique.

        Args:
            input_shape: The shape of the input image.
            coords: A tensor containing the bounding box coordinates.
            image_shape: The shape of the original image.
            skip_padding: Whether to skip padding when applying the technique. Defaults to False.

        Returns:
            The post-processed bounding box coordinates.
        """
        # Rescale coords (xyxy) from input_shape to image_shape
        scale = min(input_shape[0] / image_shape[0], input_shape[1] / image_shape[1])
        if not skip_padding:
            pad = [
                (input_shape[1] - int(image_shape[1] * scale)) / 2,
                (input_shape[0] - int(image_shape[0] * scale)) / 2,
            ]
            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= scale

        # Clip bounding xyxy bounding boxes to image shape (height, width)
        coords[:, 0].clamp_(0, image_shape[1])  # x1
        coords[:, 1].clamp_(0, image_shape[0])  # y1
        coords[:, 2].clamp_(0, image_shape[1])  # x2
        coords[:, 3].clamp_(0, image_shape[0])  # y2
        return coords

    def postprocess_stretch(self, image_shape: Tuple[int, int], scale: Any, coords: np.ndarray) -> np.ndarray:
        """Post-processes bounding box coordinates using the stretching technique.

        Args:
            image_shape: The shape of the original image.
            scale: A tuple or scalar representing the scaling factor.
            coords: A tensor containing the bounding box coordinates.

        Returns:
            The post-processed bounding box coordinates.
        """
        # Rescale coords based on image_shape (h, w)
        scale_x = scale_y = int(scale[0])
        if len(scale) == 2:
            scale_y = int(scale[1])
        coords[:, [0, 2]] *= image_shape[1]
        coords[:, [1, 3]] *= image_shape[0]
        coords[:, [0, 2]] /= scale_x
        coords[:, [1, 3]] /= scale_y

        return coords

    def postprocess_orgimage(
        self, input_shape: Tuple[int, int], coords: Any, image_shape: Tuple[int, int]
    ) -> Any:
        """Post-processes bounding box coordinates using the original image technique.

        Args:
            input_shape: The shape of the input image.
            coords: A tensor containing the bounding box coordinates.
            image_shape: The shape of the original image.

        Returns:
            The post-processed bounding box coordinates.
        """
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        coords[:, 0].clamp_(0, input_shape[1])  # x1
        coords[:, 1].clamp_(0, input_shape[0])  # y1
        coords[:, 2].clamp_(0, input_shape[1])  # x2
        coords[:, 3].clamp_(0, input_shape[0])  # y2

        coords[:, 0:4:2], coords[:, 1:4:2] = (
            (coords[:, 0:4:2] * image_shape[1]) / input_shape[1],
            (coords[:, 1:4:2] * image_shape[0]) / input_shape[0],
        )

        return coords

    def resize_mask(
        self, mask_prob: np.ndarray, bboxes: np.ndarray, image_size: Tuple[int, int], threshold: float = 0.5
    ) -> np.ndarray:
        """Resizes the predicted masks to the specified image size.

        Args:
            mask_prob: A tensor containing the predicted mask probabilities.
            bboxes: A tensor containing the bounding box coordinates.
            image_size: The shape of the target image.
            threshold: The confidence threshold for mask prediction. Defaults to 0.5.

        Returns:
            A tensor containing the resized masks.
        """
        # resize predicted masks to image_size(h,w)
        torch = Helper.safe_import_package("torch", "2.4.0")
        nc = bboxes.shape[0]
        image_mask = torch.zeros(
            (nc, image_size[0], image_size[1]), dtype=torch.bool if threshold >= 0 else torch.uint8
        )
        for mid in range(nc):
            masks_patch, sptl_inds = ObjectDetectionPostProcessor.paste_masks_in_image(
                mask_prob[mid, None, :, :], bboxes[mid, None, :], image_size
            )
            if threshold >= 0:
                masks_patch = (masks_patch >= threshold).to(dtype=torch.bool)
            else:
                masks_patch = (masks_patch * 255).to(dtype=torch.uint8)
            image_mask[(mid,) + sptl_inds] = masks_patch

        return image_mask

    @staticmethod
    def paste_masks_in_image(masks: np.ndarray, bboxes: np.ndarray, image_size: Tuple[int, int]) -> Any:
        """Paste mask of a fixed resolution (e.g., 28 x 28) into an image.

        The location, height, and width for pasting each mask is
        determined by their corresponding bounding boxes in bboxes
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        bboxes = torch.from_numpy(np.array(bboxes))
        masks = torch.from_numpy(masks)
        # Add a channel dimension to masks
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # Shape becomes [N, 1, H_in, W_in]
        box_x0, box_y0, box_x1, box_y1 = torch.split(bboxes, 1, dim=1)
        coord_y = torch.arange(0, image_size[0], device="cpu", dtype=torch.float32) + 0.5
        coord_x = torch.arange(0, image_size[1], device="cpu", dtype=torch.float32) + 0.5
        coord_y = (coord_y - box_y0) / (
            box_y1 - box_y0
        ) * 2 - 1  # normalize coordinates and shift it into [-1 , 1], shape (N, y)
        coord_x = (coord_x - box_x0) / (
            box_x1 - box_x0
        ) * 2 - 1  # normalize coordinates and shift it into [-1 , 1], shape (N, x)

        gx = coord_x[:, None, :].expand(masks.shape[0], coord_y.size(1), coord_x.size(1))  # shape (N, y, w)
        gy = coord_y[:, :, None].expand(masks.shape[0], coord_y.size(1), coord_x.size(1))  # shape (N, y, w)
        grid_xy = torch.stack([gx, gy], dim=3)  # grid of xy coordinates
        image_mask = torch.nn.functional.grid_sample(
            masks, grid_xy.to(masks.dtype), align_corners=False
        )  # resize mask to image shape

        return image_mask[:, 0], (slice(0, image_size[0]), slice(0, image_size[1]))
