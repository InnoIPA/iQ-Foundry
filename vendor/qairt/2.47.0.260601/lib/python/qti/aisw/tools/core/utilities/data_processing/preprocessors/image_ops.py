# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from typing import Literal, Optional, Union

import cv2
import numpy as np
from PIL import Image
from qti.aisw.tools.core.utilities.data_processing import ImageRepresentation, PreProcessor
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class CropImage(PreProcessor):
    """Takes a image data as numpy array, crops it with the given configuration
    and returns the cropped image.

    Args:
        image_dimensions (tuple[int, int]): The height and width of the output image.
        library (Optional[Literal['numpy', 'torchvision']]) = 'numpy': The library used to perform cropping.
        typecasting_required (Optional[bool]): Whether the function should attempt to convert the image
            to a NumPy array before cropping. Defaults to True.
    """

    def __init__(
        self,
        image_dimensions: tuple[int, int],
        library: Optional[Literal["numpy", "torchvision"]] = "numpy",
        typecasting_required: Optional[bool] = True,
    ):
        """Initializes an CropImage instance.

        Args:
            image_dimensions (tuple[int, int]): The height and width of the output image.
            library (Optional[Literal['numpy', 'torchvision']]): The library used to perform cropping.
                Defaults to 'numpy'.
            typecasting_required (Optional[bool]): Whether the function should attempt to convert the image
                to a NumPy array before cropping. Defaults to True.
        """
        self.image_dimensions = image_dimensions
        self.library = library
        self.typecasting_required = typecasting_required
        self.validate()
        self.output_height, self.output_width = self.image_dimensions

    def validate(self):
        """Validate whether the parameters have been set correctly.

        Raises:
            ValueError: If library provided is not in 'numpy' or 'torchvision'.
            ValueError: If typecasting required is not a boolean.
            ValueError: If image dimensions are not all positive integers.
        """
        if self.library not in ("numpy", "torchvision"):
            raise ValueError("Invalid library provided. Library must be one of 'numpy' or 'torchvision'")
        if not all((isinstance(dim, int) and dim > 0) for dim in self.image_dimensions):
            raise ValueError(
                "Invalid image dimensions provided. Both width and height should be"
                " postive integers greater than 0."
            )

    def validate_input(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Validates the input data to ensure it conforms to the expected format.

        Args:
            input_sample (ImageRepresentation): The input image sample.

        Returns:
            (ImageRepresentation): The validated input sample.
        """
        # Check if each item in the input data is an array or PIL Image instance
        for i, item in enumerate(input_sample.data):
            # Convert to numpy array if using numpy library
            if self.library == "numpy":
                # If using numpy, ensure each item is an array. If not, convert it to a numpy array.
                if not isinstance(item, np.ndarray):
                    input_sample.data[i] = np.asarray(item)

            # Ensure PIL Image instance if using torchvision library
            elif self.library == "torchvision":
                # If using torchvision, ensure each item is an instance of PIL Image. If not,
                # raise a RuntimeError with the incorrect type.
                if not isinstance(item, Image.Image):
                    raise RuntimeError(
                        "When library is 'torchvision', input data must be an instance of PIL"
                        f" Image, Got {type(item)}"
                    )

        return input_sample

    @PreProcessor.validate_input_output
    def execute(self, input_data: ImageRepresentation) -> ImageRepresentation:
        """Execute the preprocessor on a list of images.

        Args:
            input_data (ImageRepresentation): A list of image data.

        Returns:
            ImageRepresentation: The output image data after execution.
        """
        cropped_images = []
        for image_arr in input_data.data:
            if self.library == "numpy":
                cropped_image = self.crop_numpy(image_arr, self.output_height, self.output_width)
                cropped_images.append(cropped_image)
            elif self.library == "torchvision":
                cropped_image = self.crop_tv(
                    image_arr,
                    self.output_height,
                    self.output_width,
                    typecasting_required=self.typecasting_required,
                )
                cropped_images.append(cropped_image)
        input_data.data = cropped_images
        return input_data

    @staticmethod
    def crop_numpy(image: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
        """Crop a numpy array to the specified dimensions.

        Args:
            image (np.ndarray): The input image data.
            output_height (int): The desired height of the cropped image.
            output_width (int): The desired width of the cropped image.

        Returns:
            np.ndarray: The cropped image data as a numpy array.
        """
        # Ensure the input is a 3D numpy array
        if image.ndim != 3:
            raise ValueError("Input dimension for crop image must be 3")
        # Ensure the image dimensions are sufficient to crop
        if image.shape[0] < output_height or image.shape[1] < output_width:
            raise ValueError("Image dimensions are smaller than desired cropping dimensions")

        # Calculate the left, right, top and bottom indices for cropping
        x_left = int(round((image.shape[1] - output_width) / 2))
        x_right = x_left + output_width
        y_top = int(round((image.shape[0] - output_height) / 2))
        y_bottom = y_top + output_height
        # Crop the image to the desired dimensions
        cropped_image = image[y_top:y_bottom, x_left:x_right]

        return cropped_image

    @staticmethod
    def crop_tv(
        image: Image.Image, output_height: int, output_width: int, typecasting_required: bool = True
    ) -> Union[np.ndarray, Image.Image]:
        """Crop a PIL image to the specified dimensions using torchvision.

        Args:
            image (Image.Image): The input image data.
            output_height (int): The desired height of the cropped image.
            output_width (int): The desired width of the cropped image.
            typecasting_required (bool): Whether the function should attempt to convert the image
                to a NumPy array before cropping. Defaults to True.


        Returns:
            Union[np.ndarray, Image.Image]: The cropped image data as a numpy array or PIL image object.
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Torchvision supports only valid PIL images as input")
        if output_height == output_width:
            crop_size = output_height
        else:
            crop_size = (output_height, output_width)

        torchvision = Helper.safe_import_package("torchvision", "0.19.0")
        # Apply center cropping using torchvision
        cropped_image = torchvision.transforms.functional.center_crop(image, crop_size)

        # Typecast the image to float32 if required
        if typecasting_required:
            cropped_image = np.asarray(cropped_image, dtype=np.float32)

        return cropped_image


class ExpandDimensions(PreProcessor):
    """Plugin to add N dimension to the input data.

    This plugin expands a specified axis of the input data.
    For example, it can convert HWC (Height, Width, Channel) to NHWC.

    Args:
        axis: The index of the axis to expand. Defaults to 0.

    Attributes:
        expanded_axis: The axis to be expanded in the input data.
    """

    def __init__(self, axis=0):
        """Initializes an instance of ExpandDimensions.

        Args:
            axis (int): The index of the axis to expand. Defaults to 0.
        """
        self.expanded_axis = axis
        self.validate()

    def validate(self):
        """Validate the ExpandDimensions parameters value."""
        if not isinstance(self.expanded_axis, int) or self.expanded_axis < 0:
            raise ValueError("Invalid axis value. Axis must be an non-negative integer.")

    def execute(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Expands the specified dimension in the input data.

        Args:
            input_sample (ImageRepresentation): The input image data.

        Returns:
            ImageRepresentation: The modified input sample with expanded dimensions.
        """
        output_data = []
        for data in input_sample.data:
            inp = np.expand_dims(data, axis=self.expanded_axis)
            output_data.append(inp)
        input_sample.data = output_data
        return input_sample


class ConvertNCHW(PreProcessor):
    """Preprocessor to swap the input from NHWC to NCHW.

    Args:
        expand_dims (bool, optional): Whether to add a new axis. Defaults to False.
    """

    def __init__(self, expand_dims: bool = False) -> None:
        """Initializes an instance of ConvertNCHW.

        Args:
            expand_dims (bool): Whether to add an extra dimension to the output.
        """
        self.expand_dims = expand_dims
        self.validate()

    def validate(self):
        """Validates the ConvertNCHW Parameters"""
        if not isinstance(self.expand_dims, bool):
            raise ValueError("expand_dims must be boolean. i.e. Either True or False")

    def execute(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Execute the preprocessing operation on the input image.

        Args:
            input_sample (ImageRepresentation): The input image representation.

        Returns:
            ImageRepresentation: The preprocessed image representation.
        """
        output_data = []
        for data in input_sample.data:
            if len(data.shape) < 3:
                raise ValueError("Input data must have at least 3 axes, but got {}".format(len(data.shape)))
            # Swap axes from NHWC to NCHW
            input_array = data.transpose([2, 0, 1])
            if self.expand_dims:
                input_array = np.expand_dims(input_array, axis=0)
            output_data.append(input_array)
        input_sample.data = output_data
        return input_sample


class FlipImage(PreProcessor):
    """Processor to flip the image.

    This processor flips the input image horizontally or vertically.

    Attributes:
        axis (int): The axis along which the image is flipped.
            Default: 3, indicating a horizontal flip for RGB images.
    """

    def __init__(self, axis: int = 3) -> None:
        """Initializes the FlipImage processor with the specified axis.

        Args:
            axis (int): The axis along which the image is flipped. Defaults to 3.
        """
        self.axis = axis
        self.validate()

    def validate(self):
        """Validates the FlipImage Parameters"""
        if not isinstance(self.axis, int):
            raise TypeError("Axis must be an postive integer.")
        if not (0 <= self.axis < 4):
            raise ValueError("Axis must be between 0 and 3.")

    def execute(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Executes the processing on the input sample.


        Flips the input image data horizontally or vertically based on the specified axis.

        Args:
            input_sample (ImageRepresentation): The input image data.

        Returns:
            ImageRepresentation: The processed image data.
        """
        flipped_data = []
        for img_data in input_sample.data:
            # Use np.flip with the correct axis to flip the image
            fliped_inp = np.flip(img_data, self.axis)
            flipped_data.append(fliped_inp)
        input_sample.data = flipped_data
        return input_sample


class PadImage(PreProcessor):
    """Image padding with constant size or based on target dimensions.

    Attributes:
        target_dimensions (tuple[int, int]): Target height and width for 'target_dims' pad type.
        color (int): Padding value for all planes when using 'constant' pad type.
        pad_type (Literal['constant', 'target_dims']): Type of padding to apply.
    """

    def __init__(
        self,
        target_dimensions: Optional[tuple[int, int]] = None,
        pad_type: Optional[Literal["constant", "target_dims"]] = "constant",
        constant_pad_size: Optional[int] = None,
        image_position: Optional[Literal["corner", "center"]] = "center",
        color_value: Optional[int] = 114,
    ):
        """Initialize the PadImage instance.

        Args:
            target_dimensions (tuple[int, int]): Target height and width for 'target_dims' pad type.
            pad_type (Optional[Literal['constant', 'target_dims']], optional): Type of padding to apply.
                             Defaults to "constant".
            constant_pad_size (Optional[int], optional): Padding size when using 'constant' pad type.
                             Defaults to None.
            image_position (Optional[Literal["corner", "center"]], optional): Position of the image within the
                              padded region. Defaults to "center".
            color_value (int, optional): Padding value for all planes when using 'constant' pad type.
                              Defaults to 114.
        """
        self.pad_type = pad_type
        self.constant_pad_size = constant_pad_size
        self.image_position = image_position
        self.color = color_value
        if target_dimensions:
            self.target_height, self.target_width = target_dimensions
        else:
            self.target_height = self.target_width = None
        self.validate()

    def validate(self):
        """Validate the PadImage parameters"""
        if self.pad_type not in ["constant", "target_dims"]:
            raise ValueError("Pad type not supported. Pad type must be one of 'constant', 'target_dims'")
        if self.constant_pad_size and not isinstance(self.constant_pad_size, int):
            raise ValueError("Constant pad size must be an integer.")
        if self.image_position not in ["corner", "center"]:
            raise ValueError("Image position must be one of 'corner', 'center'")
        if not isinstance(self.color, int):
            raise ValueError("Color value must be an integer.")
        if self.pad_type == "constant" and self.constant_pad_size is None:
            raise ValueError("constant_pad_size must be provided when pad_type is set to 'constant'")
        if self.pad_type == "target_dims":
            if self.target_height is None or self.target_width is None:
                raise ValueError("target_dimensions must be provided when pad type is set to 'target_dims'")
            if (
                not (isinstance(self.target_height, int) and isinstance(self.target_width, int))
                and self.target_height < 0
                and self.target_width < 0
            ):
                raise ValueError("target_dimensions provided must be a pair of positive integer.")

    def execute(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Apply padding to the input image based on the configured pad type.

        Args:
            input_sample (ImageRepresentation): Input image representation.

        Returns:
            ImageRepresentation: Padded image representation.
        """
        input_data = input_sample.data
        for data in input_data:
            if self.pad_type == "constant":
                img = self.create_border_constant_pad(
                    data, constant_pad_size=self.constant_pad_size, color_value=self.color
                )
            elif self.pad_type == "target_dims":
                color_dim1 = int(self.color)  # Padding value for all planes is same.
                color_value = (color_dim1, color_dim1, color_dim1)
                img = self.create_border_target_dims(
                    data,
                    target_height=self.target_height,
                    target_width=self.target_width,
                    color_value=color_value,
                    image_position=self.image_position,
                )

        input_sample.data = [img]
        return input_sample

    @staticmethod
    def create_border_constant_pad(image: np.ndarray, constant_pad_size: int, color_value: int = 114):
        """Pads the image with a specified value.

        Args:
            image (np.ndarray): The input image.
            constant_pad_size (int): The size of padding on each side.
            color_value (int, optional): The value to fill the padded region. Defaults to 114.

        Returns:
            np.ndarray: The padded image.
        """
        top = bottom = left = right = constant_pad_size

        img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_value)
        return img

    @staticmethod
    def create_border_target_dims(
        image: np.ndarray,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        color_value: tuple[int, int, int] = (114, 114, 114),
        library_name: Optional[str] = None,
        mode: Optional[str] = None,
        image_position: str = "center",
    ):
        """Resizes the image to fit within a target size with specified padding.

        Args:
            image (np.ndarray): The input image.
            target_height (Optional[int], optional): The target height. Defaults to None.
            target_width (Optional[int], optional): The target width. Defaults to None.
            color_value (tuple[int, int, int], optional): The value to fill the padded region.
             Defaults to (114, 114, 114).
            library_name (Optional[str], optional): The name of the library. Defaults to None.
            mode (Optional[str], optional): The mode of operation. Defaults to None.
            image_position (str, optional): The position of the image within the target size.
             Defaults to 'center'.

        Returns:
            np.ndarray: The resized and padded image.

        Raises:
            ValueError: If `image_position` is not one of 'center' or 'corner'.
        """
        if isinstance(image, Image.Image):
            orig_height, orig_width = image.size[:2]
        else:
            orig_height, orig_width = image.shape[:2]
        if image_position == "center":
            # Calculate padding values to center the image
            pad_w, pad_h = (target_width - orig_width) / 2, (target_height - orig_height) / 2
            top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
            left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        elif image_position == "corner":
            # Calculate padding values to place the image in the top-left corner
            pad_w, pad_h = (target_width - orig_width), (target_height - orig_height)
            top, bottom = 0, pad_h
            left, right = 0, pad_w
        img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_value)
        return img


class NormalizeImage(PreProcessor):
    """Normalize the given image data with the given configuration and returns
    the normalized image.
    """

    supported_libraries = ["numpy", "torchvision"]

    def __init__(
        self,
        library: Literal["numpy", "torchvision"] = "numpy",
        means: dict[str, float] = {"R": 0, "G": 0, "B": 0},
        std: dict[str, float] = {"R": 1, "G": 1, "B": 1},
        channel_order: Literal["RGB", "BGR"] = "RGB",
        norm: float = 255.0,
        normalize_first: bool = True,
        typecasting_required: bool = True,
    ):
        """Initializes the NormalizeImage instance.

        Args:
        library (str, optional): The library to use for normalization. Defaults to "numpy".
            Must be one of {"numpy", "torchvision"}.
        means (dict[str, float], optional): A dictionary containing mean values for each channel.
            Defaults to {"R": 0, "G": 0, "B": 0}.
            The keys should be one of {"R", "G", "B"} and the values are floats.
        std (dict[str, float], optional): A dictionary containing standard deviation values for each channel.
            Defaults to {"R": 1, "G": 1, "B": 1}.
            The keys should be one of {"R", "G", "B"} and the values are floats.
        channel_order (str, optional): The order of channels in the image. Defaults to "RGB".
            Must be one of {"RGB", "BGR"}.
        norm (float, optional): The normalization value. Defaults to 255.0.
            This value is used for both mean and standard deviation calculations.
        normalize_first (bool, optional): Whether to normalize first or not. Defaults to True.
        typecasting_required (bool, optional): Whether type casting is required. Defaults to True.
        """
        self.library = library
        self.means = means
        self.std = std
        self.channel_order = channel_order
        self.norm = norm
        self.normalize_first = normalize_first
        self.typecasting_required = typecasting_required
        self.validate()
        # Prepare the mean and std values based on channel order
        if self.channel_order == "BGR":
            self.mean_values = [self.means["B"], self.means["G"], self.means["R"]]
            self.std_values = [self.std["B"], self.std["G"], self.std["R"]]
        else:
            self.mean_values = [self.means["R"], self.means["G"], self.means["B"]]
            self.std_values = [self.std["R"], self.std["G"], self.std["B"]]

    def validate(self):
        """Validates the NormalizeImage parameters provided during initialization."""
        if self.library not in self.supported_libraries:
            raise ValueError(
                f"normalize plugin does not support library {self.library},"
                f" only supports: {', '.join(self.supported_libraries)}"
            )
        if self.channel_order not in ["RGB", "BGR"]:
            raise ValueError("Channel order must be either 'RGB' or 'BGR'")
        if not isinstance(self.norm, (int, float)) or self.norm <= 0:
            raise ValueError("norm value must be a positive number")
        if not isinstance(self.normalize_first, bool):
            raise ValueError("normalize_first must be a boolean")
        if not isinstance(self.typecasting_required, bool):
            raise ValueError("typecasting_required must be a boolean")
        required_keys = ["R", "G", "B"]
        if not isinstance(self.means, dict):
            raise ValueError("means value provided should be a dictionary")
        if not isinstance(self.std, dict):
            raise ValueError("std value provided should be a dictionary")
        for key in required_keys:
            if key not in self.means:
                raise ValueError(
                    f"Required keys missing from means dictionary. Required keys are {required_keys}"
                )
            if not isinstance(self.means[key], (int, float)):
                raise ValueError(
                    f"Value of {key} in means dictionary should be an integer or floating point number"
                )
            if key not in self.std:
                raise ValueError(
                    f"Required keys missing from std dictionary. Required keys are {required_keys}"
                )
            if not isinstance(self.std[key], (int, float)):
                raise ValueError(
                    f"Value of {key} in std dictionary should be an integer or floating point number"
                )

    def validate_input(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Validates the input data to ensure it conforms to the expected format.

        Args:
            input_sample (ImageRepresentation): The input image sample.

        Returns:
            (ImageRepresentation): The validated input sample.
        """
        # Check if each item in the input data is an array or PIL Image instance
        for i, item in enumerate(input_sample.data):
            # Convert to numpy array if using numpy library
            if self.library == "numpy":
                """
                If using numpy, ensure each item is an array. If not, convert it to a numpy array.
                """
                if not isinstance(item, np.ndarray):
                    input_sample.data[i] = np.asarray(item)
                # Check if the input shape has three channels
                if not (list(np.shape(item))[-1] == 3):
                    raise ValueError("Normalization must be applied with the data in NHWC format.")

            # Ensure PIL Image instance if using torchvision library
            elif self.library == "torchvision":
                """
                If using torchvision, ensure each item is an instance of PIL Image. If not,
                raise a RuntimeError with the incorrect type.
                """
                if not isinstance(item, Image.Image):
                    raise RuntimeError(
                        "When library is 'torchvision', input data must be an instance of PIL"
                        f" Image, Got {type(item)}"
                    )
        return input_sample

    @PreProcessor.validate_input_output
    def execute(
        self,
        input_sample: ImageRepresentation,
    ) -> ImageRepresentation:
        """Normalizes the given image data.

        Args:
            input_sample (ImageRepresentation): The input sample containing image data.

        Returns:
            ImageRepresentation: The normalized image data.
        """
        out_data = []
        for input_data in input_sample.data:
            if self.library == "numpy":
                normalized_image = self.norm_numpy(
                    input_data, self.mean_values, self.std_values, self.norm, self.normalize_first
                )
            elif self.library == "torchvision":
                normalized_image = self.norm_tv(
                    input_data,
                    self.mean_values,
                    self.std_values,
                    typecasting_required=self.typecasting_required,
                )
            out_data.append(normalized_image.astype(np.float32))
        input_sample.data = out_data
        return input_sample

    @staticmethod
    def norm_numpy(
        inp: np.ndarray,
        means: list[float],
        std: list[float],
        norm: float = 255.0,
        normalize_first: bool = True,
    ) -> np.ndarray:
        """Normalize the given image data using numpy.

        Parameters:
            inp (np.ndarray): The input image data.
            means (list): A list of mean values for each channel (R, G, B).
            std (list): A list of standard deviation values for each channel (R, G, B).
            norm (float): The normalization value. Default is 255.0.
            normalize_first (bool): Whether to normalize first or not. Default is True.

        Returns:
            np.ndarray: The normalized image data.
        """
        # Convert mean and standard deviation values to numpy arrays
        means = np.array(means, dtype=np.float32)
        std = np.array(std, dtype=np.float32)

        # Apply normalization logic based on whether we normalize first or not
        if normalize_first:
            # Divide by normalization value before subtracting means and scaling by std
            inp = np.true_divide(inp, norm)

        # Subtract means and Scale by STD
        inp = (inp - means) / std
        if not normalize_first:
            # Divide by normalization value after subtracting means and scaling by std
            inp = np.true_divide(inp, norm)
        return inp

    @staticmethod
    def norm_tv(
        inp: Image.Image,
        means: list[float],
        std: list[float],
        typecasting_required: bool = True,
    ) -> np.ndarray:
        """Normalizes the given image data using torchvision library.

        Args:
            inp (Image.Image): The input image data.
            means (list[float]): A list of mean values for each channel (R, G, B).
            std (list[float]): A list of standard deviation values for each channel (R, G, B).
            typecasting_required (bool): Whether type casting is required. Defaults to True.

        Returns:
            np.ndarray: The normalized image data.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        torchvision = Helper.safe_import_package("torchvision", "0.19.0")
        # Convert PIL input to tensor if required
        inp = torchvision.transforms.functional.to_tensor(inp)

        # Normalize the input using torchvision's normalize function
        normalized_inp = torchvision.transforms.functional.normalize(
            inp, mean=torch.tensor(means), std=torch.tensor(std)
        )
        # Perform type casting if required
        if typecasting_required:
            normalized_inp = normalized_inp.numpy()
        return normalized_inp


class ResizeImage(PreProcessor):
    """A class used to resize images.

    Attributes:
        image_dimensions (tuple[int, int]): The new dimensions to resize the image.
        library (str): The library used for image processing. Defaults to "opencv".
        interpolation_method (Optional[str], optional): The method to use for resizing.
         Defaults to None.
        typecasting_required (bool, optional): Whether type casting is required after resizing.
         Defaults to False.
        resize_type (Optional[str], optional): The type of resize operation to perform.
         Defaults to None.
        channel_order (Optional[str], optional): The order of the color channels in the image.
         Defaults to "RGB".
        resize_before_typecast (bool, optional): Whether to resize before performing type casting.
         Defaults to True.
        mean (dict[str, float]): The mean values for each color channel.
         Defaults to {"R": 0.0, "G": 0.0, "B": 0.0}.
        std (dict[str, float]): The standard deviation of each color channel.
         Defaults to {"R": 1.0, "G": 1.0, "B": 1.0}.
        norm (float): The factor used for normalizing the image. Defaults to 255.0.
        normalize_first (bool, optional): Whether to normalize the image before resizing.
         Defaults to True.
        normalize_before_resize (bool, optional): Whether to normalize the image before
         resizing it. Defaults to False.

    Raises:
        ValueError: If the chosen library is not supported or if any of the dimensions
         are not positive integers.
        RuntimeError: If type casting is required and resize before type cast flag
         is False for PIL Image input.

    """

    supported_libraries = ["opencv", "torchvision", "pillow", "tensorflow"]
    valid_interpolation_flags = {
        "opencv": ["bilinear", "nearest", "area"],
        "torchvision": ["bilinear", "nearest", "bicubic"],
        "pillow": ["bicubic", "bilinear", "nearest", "box", "hamming", "lanczos"],
        "tensorflow": [
            "bicubic",
            "bilinear",
            "nearest",
            "area",
            "gaussian",
            "lanczos3",
            "lanczos5",
            "mitchellcubic",
        ],
    }

    def __init__(
        self,
        image_dimensions: tuple[int, int],
        library: Literal["opencv", "torchvision", "pillow", "tensorflow"] = "opencv",
        interpolation_method: Optional[str] = None,
        typecasting_required: bool = False,
        resize_type: Optional[str] = None,
        channel_order: Optional[str] = None,
        resize_before_typecast: Optional[bool] = True,
        mean: dict[str, float] = {"R": 0.0, "G": 0.0, "B": 0.0},
        std: dict[str, float] = {"R": 1.0, "G": 1.0, "B": 1.0},
        norm: float = 255.0,
        normalize_first: bool = True,
        normalize_before_resize: bool = False,
    ) -> None:
        """Initialize the ResizeImage class.

        Args:
            image_dimensions (tuple[int, int]): The new dimensions to resize the image.
            library (str): The library used for image processing. Defaults to "opencv".
            interpolation_method (Optional[str], optional): The method to use for resizing.
             Defaults to None.
            typecasting_required (bool, optional): Whether type casting is required after resizing.
             Defaults to False.
            resize_type (Optional[str], optional): The type of resize operation to perform.
             Defaults to None.
            channel_order (Optional[str], optional): The order of the color channels in the image.
            Defaults to "RGB".
            resize_before_typecast (bool, optional): Whether to resize before performing type casting.
             Defaults to True.
            mean (dict[str, float]): The mean values for each color channel. Defaults to
            {"R": 0.0, "G": 0.0, "B": 0.0}.
            std (dict[str, float]): The standard deviation of each color channel. Defaults to
            {"R": 1.0, "G": 1.0, "B": 1.0}.
            norm (float): The factor used for normalizing the image. Defaults to 255.0.
            normalize_first (bool, optional): Whether to normalize the image before resizing.
             Defaults to True.
            normalize_before_resize (bool, optional): Whether to normalize the image before resizing it.
             Defaults to False.

        Raises:
            ValueError: If the chosen library is not supported or if any of the dimensions
             are not positive integers.
            RuntimeError: If type casting is required and resize before type cast flag
             is False for PIL Image input.
        """
        self.image_dimensions = image_dimensions
        self.library = library
        self.interpolation_method = interpolation_method
        self.typecasting_required = typecasting_required
        self.resize_type = resize_type
        self.channel_order = channel_order
        self.resize_before_typecast = resize_before_typecast
        self.mean = mean
        self.std = std
        self.norm = norm
        self.normalize_first = normalize_first
        self.normalize_before_resize = normalize_before_resize
        self.validate()
        if self.library == "pillow":
            self.typecasting_required = True
            self.interpolation_method = self.interpolation_method if self.interpolation_method else "bicubic"
        else:
            self.interpolation_method = self.interpolation_method if self.interpolation_method else "bilinear"
        if self.channel_order is None:
            self.channel_order = "RGB"

    def validate(self):
        """Validate the input parameters.

        Raises:
            ValueError: If the chosen library is not supported or if any of the dimensions
             are not positive integers.
            RuntimeError: If type casting is required and resize before type cast flag is
             False for PIL Image input.

        """
        if self.library not in self.supported_libraries:
            raise ValueError(
                f"'{self.library}' is an unsupported image library. "
                f"Please choose from the following options: {', '.join(self.supported_libraries)}"
            )
        if any([not isinstance(d, int) for d in self.image_dimensions]):
            raise RuntimeError("Dimensions must be positive integers")
        if self.typecasting_required and not isinstance(self.typecasting_required, bool):
            raise ValueError("typecasting_required must be a boolean value")
        if self.resize_before_typecast and not isinstance(self.resize_before_typecast, bool):
            raise ValueError("resize_before_typecast must be a boolean value")

        if (
            self.interpolation_method is not None
            and self.interpolation_method not in self.valid_interpolation_flags[self.library]
        ):
            raise ValueError(
                f"'{self.interpolation_method}' is an unsupported interpolation method. "
                "Please choose from the "
                f"following options: {self.valid_interpolation_flags[self.library]}"
            )
        if self.library in ["pillow", "tensorflow", "torchvision"] and not self.resize_before_typecast:
            raise RuntimeError(
                "typecasting before resizing is not supported for PIL Image "
                f"input and chosen library: {self.library}"
            )

    def validate_input(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Validates the input by converting it to a standard format based on the specified library.

        Args:
            input_sample (ImageRepresentation): The input image sample

        Returns:
            ImageRepresentation: The validated input image sample
        """
        for idx, item in enumerate(input_sample.data):
            if self.library == "opencv" and isinstance(item, Image.Image):
                input_sample.data[idx] = np.asarray(item)
            elif (
                self.library == "pillow" or self.library == "torchvision" or self.library == "tensorflow"
            ) and isinstance(item, np.ndarray):
                input_sample.data[idx] = Image.fromarray(item)
        return input_sample

    @PreProcessor.validate_input_output
    def execute(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """This method executes image operations on an input sample.

        Args:
            input_sample (ImageRepresentation): The input data containing images or tensors.

        Returns:
            ImageRepresentation: The modified input sample after applying the specified image operations.
        """
        # Extract height and width from dimensions
        height, width = self.image_dimensions[:2]
        given_dimensions = [int(height), int(width)]
        out_data = []
        for idx, img in enumerate(input_sample.data):
            # Get image dimensions
            image_dimensions = self.get_image_dimensions(img, self.library)

            # Calculate resize dimensions and letterbox scale
            resize_dims, letterbox_scale = ResizeImage.get_resize_dims(
                given_dimensions, image_dimensions, self.resize_type
            )
            resize_height, resize_width = resize_dims

            # Apply resizing operations
            if self.library == "torchvision":
                img = self.resize_tv(
                    image=img,
                    desired_height=resize_height,
                    desired_width=resize_width,
                    interpolation_mode=self.interpolation_method,
                    resize_before_typecast=self.resize_before_typecast,
                    channel_order=self.channel_order,
                    typecast_required=self.typecasting_required,
                )
            elif self.library == "opencv":
                img = self.resize_cv(
                    image=img,
                    desired_height=resize_height,
                    desired_width=resize_width,
                    interpolation_mode=self.interpolation_method,
                    resize_before_typecast=self.resize_before_typecast,
                    channel_order=self.channel_order,
                    image_dimensions=image_dimensions,
                    given_dimensions=given_dimensions,
                    resize_type=self.resize_type,
                    letterbox_scale=letterbox_scale,
                )
            elif self.library == "pillow":
                img = self.resize_pil(
                    image=img,
                    desired_height=resize_height,
                    desired_width=resize_width,
                    resize_before_typecast=self.resize_before_typecast,
                    channel_order=self.channel_order,
                    resize_dims=resize_dims,
                    given_dimensions=given_dimensions,
                    resize_type=self.resize_type,
                    interpolation_mode=self.interpolation_method,
                )
            elif self.library == "tensorflow":
                img = self.resize_tf(
                    image=img,
                    desired_height=resize_height,
                    desired_width=resize_width,
                    interpolation_mode=self.interpolation_method,
                    resize_before_typecast=self.resize_before_typecast,
                    channel_order=self.channel_order,
                    given_dimensions=given_dimensions,
                    mean=self.mean,
                    std=self.std,
                    normalize_before_resize=self.normalize_before_resize,
                    norm_factor=self.norm,
                    normalize_first=self.normalize_first,
                )

            out_data.append(img)

        input_sample.data = out_data
        return input_sample

    @staticmethod
    def resize_tv(
        image: Image.Image,
        desired_height: int,
        desired_width: int,
        resize_before_typecast: bool,
        interpolation_mode: str = "bilinear",
        channel_order: str = "RGB",
        typecast_required: bool = True,
    ) -> np.ndarray:
        """Resizes an image to the specified height and width using tochvision library.

        Args:
            image (Image.Image): The input image.
            desired_height (int): The desired height of the output image.
            desired_width (int): The desired width of the output image.
            resize_before_typecast (bool): Flag indicating whether to resize first or not.
            interpolation_mode (str, optional): Interpolation mode. Defaults to "bilinear".
                Possible choices: bilinear, nearest and bicubic
            channel_order (str, optional): Channel order. Defaults to "RGB".
            typecast_required (bool, optional): Whether to typecast the output. Defaults to True.

        Returns:
            np.ndarray: The resized image as a numpy array.
        """
        # Define interpolation modes and their corresponding flags
        interp_flags = {"bilinear": Image.BILINEAR, "nearest": Image.NEAREST, "bicubic": Image.BICUBIC}
        # Get the interpolation mode flag from the dictionary or set to None if not found
        interpolation_mode_tv = interp_flags.get(interpolation_mode, None)
        # Convert the image to RGB channel order if necessary
        if channel_order == "RGB":
            image = image.convert("RGB")

        # Determine the resize size based on whether height and width are equal
        if desired_height == desired_width:
            resize_size = desired_height
        else:
            resize_size = (desired_height, desired_width)

        torchvision = Helper.safe_import_package("torchvision", "0.19.0")
        image = torchvision.transforms.functional.resize(image, resize_size, interpolation_mode_tv)
        # Typecast the output if required and based on whether resizing was done first
        if typecast_required and resize_before_typecast:
            # TODO : Need to find a way to handle this scenario
            image = np.asarray(image, dtype=np.float32)
        return image

    @staticmethod
    def resize_cv(
        image: np.ndarray,
        desired_height: int,
        desired_width: int,
        resize_before_typecast: bool,
        image_dimensions: tuple[int, int],
        given_dimensions: tuple[int, int],
        letterbox_scale: float,
        resize_type: str,
        channel_order: str = "RGB",
        interpolation_mode: Optional[str] = "bilinear",
    ) -> np.ndarray:
        """Resizes an input image using OpenCV.

        Args:
            image (np.ndarray): Input image to be resized.
            desired_height (int): Desired height of output image.
            desired_width (int): Desired width of output image.
            resize_before_typecast (bool): Flag to determine if resizing should be done first or after
             conversion to float32.
            image_dimensions (tuple[int, int]): Original dimensions of input image.
            given_dimensions (tuple[int, int]): Target dimensions for the output image.
            letterbox_scale (float): Scale factor for letterbox padding. If None, no padding is applied.
            resize_type (str, optional): Type of resizing to perform.
            channel_order (str, optional): Channel order for input image. Defaults to "RGB".
            interpolation_mode (Optional[str], optional): Interpolation mode for resizing.
             Defaults to "bilinear". Possible choices: bilinear, nearest, area

        Returns:
            np.ndarray: Resized output image.
        """
        interp_flags = {"bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST, "area": cv2.INTER_AREA}
        input_height, input_width = given_dimensions
        interpolation_mode_cv = interp_flags.get(interpolation_mode, None)

        if channel_order == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if letterbox_scale and image_dimensions != (desired_height, desired_width):
            interpolation_mode_cv = cv2.INTER_AREA if letterbox_scale < 1 else cv2.INTER_LINEAR

        # Type-cast before resizing if necessary
        if not resize_before_typecast:
            image = image.astype(np.float32)

        # Perform the actual resizing
        image = cv2.resize(image, (desired_width, desired_height), interpolation=interpolation_mode_cv)

        # Apply letterbox padding if requested
        if resize_type == "letterbox":
            image = PadImage.create_border_target_dims(
                image, target_height=input_height, target_width=input_width
            )

        # Type-cast after resizing if necessary
        if resize_before_typecast:
            image = image.astype(np.float32)
        return image

    @staticmethod
    def resize_pil(
        image: Image.Image,
        desired_height: int,
        desired_width: int,
        resize_before_typecast: bool,
        channel_order: str,
        resize_dims: tuple,  # (h, w)
        given_dimensions: tuple,
        resize_type: Optional[str] = None,
        interpolation_mode: str = "bicubic",
    ) -> np.ndarray:
        """Resize an image using Pillow.

        Args:
            image (PIL.Image): The input image.
            desired_height (int): The desired height of the output image.
            desired_width (int): The desired width of the output image.
            resize_before_typecast (bool): Flag to indicate whether resizing should be done first.
            channel_order (str): The order of color channels in the output image ('RGB', 'BGR' or None).
            resize_dims (tuple): The dimensions of the resized image ((h, w)).
            given_dimensions (tuple): The original dimensions of the input image ((h, w)).
            resize_type (str, optional): Type of resizing to perform ('letterbox' or None).
            interpolation_mode (str): Interpolation mode used for resizing ('bicubic', 'bilinear',
            'nearest', 'box', 'hamming', etc.). Defaults to 'bicubic'.

        Returns:
            np.ndarray: The resized image.
        """
        interp_flags = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "box": Image.BOX,
            "hamming": Image.HAMMING,
            "lanczos": Image.LANCZOS,
        }
        interpolation_mode_pil = interp_flags.get(interpolation_mode, None)
        input_height, input_width = given_dimensions
        if channel_order == "BGR" or channel_order == "bgr":
            image = np.asarray(image)
            image = image[:, :, ::-1]
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image = image.resize((desired_width, desired_height), interpolation_mode_pil)

        if resize_type == "letterbox":
            image = PadImage.create_border_target_dims(
                image,
                target_height=input_height,
                target_width=input_width,
                library_name="pillow",
                mode=channel_order,
            )
        # Type-cast after resize
        if resize_before_typecast:
            image = np.asarray(image, dtype=np.float32)
        return image

    @staticmethod
    def resize_tf(
        image: Image,
        desired_height: int,
        desired_width: int,
        given_dimensions: tuple,
        mean: dict[str, float],
        std: dict[str, float],
        resize_before_typecast: bool = True,
        channel_order: Optional[str] = "RGB",
        interpolation_mode: Optional[str] = "bilinear",
        normalize_before_resize: bool = False,
        norm_factor: float = 255.0,
        normalize_first: bool = True,
    ) -> Image:
        """Resize the input image using TensorFlow's resize function.

        Args:
            image (Image): Input image to be resized.
            desired_height (int): Desired height of the output image.
            desired_width (int): Desired width of the output image.
            given_dimensions (tuple): Original dimensions of the input image.
            mean (dict[str, float]): Dictionary containing mean values for each channel.
            std (dict[str, float]): Dictionary containing standard deviation values for each channel.
            resize_before_typecast (bool, optional): Flag to determine whether resizing should be done first.
             Defaults to True.
            channel_order (Optional[str], optional): Channel order of the input image ('RGB' or 'BGR').
             Defaults to "RGB".
            interpolation_mode (Optional[str], optional): Interpolation mode for resizing ('bicubic',
             'bilinear', 'nearest', 'area', 'gaussian', 'lanczos3', 'lanczos5', etc.). Defaults to "bilinear".
            normalize_before_resize (bool, optional): Flag to determine whether normalization should be done
             before resizing. Defaults to False.
            norm_factor (float, optional): Normalization factor. Defaults to 255.0.
            normalize_first (bool, optional): Flag to determine whether normalization should be done first.
                Defaults to True.

        Returns:
            Image: Resized image.
        """
        tf = Helper.safe_import_package("tensorflow", "2.10.1")
        interp_flags = {
            "bicubic": tf.image.ResizeMethod.BICUBIC,
            "bilinear": tf.image.ResizeMethod.BILINEAR,
            "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            "area": tf.image.ResizeMethod.AREA,
            "gaussian": tf.image.ResizeMethod.GAUSSIAN,
            "lanczos3": tf.image.ResizeMethod.LANCZOS3,
            "lanczos5": tf.image.ResizeMethod.LANCZOS5,
            "mitchellcubic": tf.image.ResizeMethod.MITCHELLCUBIC,
        }
        interpolation_mode_tf = interp_flags.get(interpolation_mode, None)
        input_height, input_width = given_dimensions

        # Convert the image to RGB if necessary

        # Swap channels if needed
        if channel_order.lower() == "bgr":
            np_img = np.asarray(image)
            np_img = np_img[:, :, ::-1]
            image = Image.fromarray(np_img)

        image = image.convert("RGB")
        # Convert the image to a numpy array
        np_img = np.asarray(image)

        # Normalize the image if necessary
        if normalize_before_resize:
            # Ensure mean and std values are properly formatted
            if channel_order.lower() == "bgr":
                mean_values = [mean["B"], mean["G"], mean["R"]]
                std_values = [std["B"], std["G"], std["R"]]
            else:
                mean_values = [mean["R"], mean["G"], mean["B"]]
                std_values = [std["R"], std["G"], std["B"]]
            np_img = NormalizeImage.norm_numpy(np_img, mean_values, std_values, norm_factor, normalize_first)

        # resize
        tf_image = tf.convert_to_tensor(np_img, dtype=tf.float32)
        tf_image_r = tf.image.resize(tf_image, [desired_height, desired_width], method=interpolation_mode_tf)
        scaled_image = tf_image_r[0:input_height, 0:input_width, :]
        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, input_height, input_width)
        # Type-cast after resize
        if resize_before_typecast:
            image = output_image.numpy()
        return image

    @staticmethod
    def get_resize_dims(
        given_dimensions: tuple[int, int],  # renamed from given_dims
        image_dimensions: tuple[int, int],  # renamed from img_dims
        resize_type: str,
    ) -> tuple[tuple[int, int], Optional[float]]:
        """Get the new dimensions after resizing an image based on the specified type.

        Args:
            given_dimensions: The desired height and width of the resized image.
            image_dimensions: The original height and width of the image.
            resize_type (str): The type of resizing to perform. Can be "letterbox", "imagenet",
                or None.

        Returns:
            A tuple containing the new dimensions as a tuple of two integers, and the
                letterbox scale (if applicable).
        """
        desired_height, desired_width = given_dimensions
        original_height, original_width = image_dimensions

        # Determine if letterboxing is required
        letterbox_scale = None
        # Resize based on type.
        if resize_type == "letterbox":
            new_width, new_height, letterbox_scale = ResizeImage.letterbox_resize(
                desired_height,
                desired_width,
                original_height,
                original_width,
            )
        elif resize_type == "imagenet":
            new_height, new_width = ResizeImage.imagenet_resize(
                desired_height, desired_width, original_height, original_width
            )
        elif resize_type == "aspect_ratio":
            ratio = min(desired_height / original_height, desired_width / original_width)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
        else:
            new_height, new_width = desired_height, desired_width

        return (new_height, new_width), letterbox_scale

    @staticmethod
    def imagenet_resize(
        desired_height: int,
        desired_width: int,
        original_height: int,
        original_width: int,
    ) -> tuple[int, int]:
        """Resize the image to fit within a square bounding box of size `desired_width` x `desired_height`,
        while maintaining its aspect ratio.

        Args:
            desired_height (int): The desired height of the resized image.
            desired_width (int): The desired width of the resized image.
            original_height (int): The original height of the image.
            original_width (int): The original width of the image.

        Returns:
            A tuple containing the new height and width of the resized image.
        """
        if original_height > original_width:
            new_width = desired_width
            new_height = int(desired_height * original_height / original_width)
        else:
            new_height = desired_height
            new_width = int(desired_width * original_width / original_height)

        return new_height, new_width

    @staticmethod
    def letterbox_resize(
        desired_height: int,
        desired_width: int,
        original_height: int,
        original_width: int,
    ) -> tuple[int, int, float]:
        """Resize the image while maintaining its aspect ratio.

        Args:
            desired_height (int): The desired height of the resized image.
            desired_width (int): The desired width of the resized image.
            original_height (int): The original height of the image.
            original_width (int): The original width of the image.

        Returns:
            A tuple containing the new width, new height, and scaling factor (if applicable).
        """
        # Scale ratio (new / old)
        scale = min(desired_height / original_height, desired_width / original_width)
        new_width, new_height = int(original_width * scale), int(original_height * scale)

        return new_width, new_height, scale

    @staticmethod
    def get_image_dimensions(
        image: Image.Image | np.ndarray,
        library_name: Literal["opencv", "pillow", "torchvision", "tensorflow"],
    ) -> tuple[int, int]:
        """Get the dimensions of an image.

        Args:
            image (str): The image to extract dimensions from.
                Can be either a PIL Image or numpy array
            library_name (str): The library used to load the image. Supported libraries are 'opencv',
                'pillow', 'torchvision', and 'tensorflow'.

        Returns:
            tuple[int, int]: A tuple containing the height and width of the image/

        Raises:
            ValueError if library is invalid.
        """
        if library_name == "opencv":
            # Assume image to numpy array
            orig_height, orig_width = image.shape[:2]
        elif library_name in ["pillow", "torchvision", "tensorflow"]:
            # Assume image is PIL image
            orig_width, orig_height = image.size
        else:
            raise ValueError(f"invalid_library provide. Supported library: {ResizeImage.supported_libraries}")
        return (orig_height, orig_width)
