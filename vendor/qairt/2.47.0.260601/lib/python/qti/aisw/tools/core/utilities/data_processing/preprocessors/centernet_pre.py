# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Optional

import cv2
import numpy as np
from PIL import Image
from qti.aisw.tools.core.utilities.data_processing import ImageRepresentation, PreProcessor


class CenternetPreprocessor(PreProcessor):
    """A pre-processing class for images used with CenterNet.
    This class scales and pads an input image to a fixed resolution.

    Args:
        output_dimensions (tuple[int, int]): The desired width and height of the output image.
        scale (int): A scaling factor to be applied to the input images. Default is 1.
    """

    def __init__(self, output_dimensions: (int, int), scale: int = 1):
        """Initializes the pre-processor with desired dimensions, scale factor, and resolution.

        Args:
        output_dimensions (tuple[int, int]): The desired width and height of the output image.
        scale (int): A scaling factor to be applied to the input images. Default is 1.
        """
        self.output_dimensions = output_dimensions
        self.scale = scale
        self.validate()

    def validate(self):
        """Validates that the input parameters are valid.

        Raises:
            ValueError: If the scale is not an integer, fix_resolution is not a boolean, or
            dimensions are not integers.
        """
        if not isinstance(self.scale, int) or self.scale <= 0:
            raise ValueError("scale must be a non-negative integer")
        if not all((isinstance(dim, int) and dim > 0) for dim in self.output_dimensions):
            raise ValueError(
                "Invalid image dimensions provided. Both width and height should be integers greater than 0."
            )

    def validate_input(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Validate and preprocess the input sample for Centernet model.
        This method checks if the input sample contains only one image item.
        If multiple items are present, it raises a RuntimeError. It also converts
        the Image object to a numpy array.

        Args:
            input_sample (ImageRepresentation): The input sample to be validated and preprocessed.

        Returns:
            ImageRepresentation: The validated and preprocessed input sample.

        Raises:
            RuntimeError: If the input sample contains multiple items.
        """
        if len(input_sample.data) > 1:
            raise RuntimeError(f"{self.__class__.__name__} takes only one image input item")
        image_data = input_sample.data[0]
        if isinstance(image_data, Image.Image):
            image_data = np.asarray(image_data)
        input_sample.data[0] = image_data
        return input_sample

    @PreProcessor.validate_input_output
    def execute(self, input_data: ImageRepresentation) -> ImageRepresentation:
        """Scales and pads the input image to the desired resolution.

        Args:
            input_data (ImageRepresentation): The input image to process.

        Returns:
            ImageRepresentation: The processed image.
        """
        image = input_data.data[0]
        input_h, input_w = int(self.output_dimensions[0]), int(self.output_dimensions[1])  # height, width

        height, width = image.shape[0:2]
        new_height = int(height * self.scale)
        new_width = int(width * self.scale)

        inp_height, inp_width = input_h, input_w
        center = np.array([new_width / 2.0, new_height / 2.0], dtype=np.float32)
        scale = max(height, width) * 1.0

        trans_input = CenternetPreprocessor.get_affine_transform(
            center, scale, rot=0, output_size=[inp_width, inp_height]
        )
        resized_image = cv2.resize(image, (new_width, new_height))
        img = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

        input_data.data = [img]
        return input_data

    @staticmethod
    def get_affine_transform(
        center,
        scale: int | np.ndarray,
        rot: float,
        output_size: (int, int),
        shift: Optional[np.ndarray] = None,
        invert: bool = False,
    ):
        """Calculates the affine transform to map an image from its original size and position
         to a new size and position.

        Args:
            center (np.ndarray): The coordinates of the center point of the original image.
            scale (int or np.ndarray): The scaling factor(s) to apply to the original image.
             If it's a list, it should contain two values for height and width.
            rot (float): The rotation angle in degrees.
            output_size (tuple): The desired size of the output image.
            shift (np.ndarray): An optional shift vector to add to the transformed coordinates.
                             Default is None.
            invert (bool): Whether to invert the transform so that it maps from output_size to
                    center, scale, rot instead. Default is False.

        Returns:
            np.ndarray: The affine transform matrix.
        """
        if shift is None:
            shift = np.array([0, 0], dtype=np.float32)

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        src_w = scale[0]
        dst_w, dst_h = output_size
        rot_rad = np.pi * rot / 180
        src_dir = CenternetPreprocessor.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = CenternetPreprocessor.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = CenternetPreprocessor.get_3rd_point(dst[0, :], dst[1, :])

        if invert:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    @staticmethod
    def get_dir(src_point: list[float], rot_rad: float) -> list[float]:
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

    @staticmethod
    def get_3rd_point(a: list[float], b: list[float]) -> np.ndarray:
        """Compute the third point of an affine transformation.

        Args:
            a (list[float]): The first two points.
            b (list[float]): The second two points.

        Returns:
            np.ndarray: The third point coordinates.
        """
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
