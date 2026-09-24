# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from PIL import Image
from qti.aisw.tools.core.utilities.data_processing import ImageRepresentation, PreProcessor
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class MlCommonsRetinaNetPreprocessor(PreProcessor):
    """A preprocessor for the RetinaNet model.

    This class normalizes images to have mean and standard deviation as specified by
    the ImageNet dataset.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (800, 800),
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        """Initializes the preprocessor.

        Args:
            image_size (tuple[int, int]): The size to which images should be resized.
            mean (list[float]): The mean values for normalization.
            std (list[float]): The standard deviation values for normalization.
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        torch = Helper.safe_import_package("torch", "2.4.0")  # noqa: F841
        torchvision = Helper.safe_import_package("torchvision", "0.19.0")  # noqa: F841
        self.validate()

    def validate(self):
        """Validate the parameters supplied to MLCommonRetinaNetPreprocessor"""
        if len(self.mean) != 3:
            raise ValueError(f"Mean values must be of length 3, but got {len(self.mean)}")
        if len(self.std) != 3:
            raise ValueError(f"Standard deviation values must be of length 3, but got {len(self.std)}")
        if len(self.image_size) != 2:
            raise ValueError(f"Image size must be of length 2, but got {len(self.image_size)}")
        for dim in self.image_size:
            if not isinstance(dim, int) or dim < 1:
                raise ValueError(f"Image dimensions must be positive integers, but got {type(dim)} {dim}")

    def validate_input(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Validate and preprocess the input sample for MlCommonsRetinaNet model.
        This method checks if the input sample contains only one image item.
        If multiple items are present, it raises a RuntimeError.

        Args:
            input_sample (ImageRepresentation): The input sample to be validated and preprocessed.

        Returns:
            ImageRepresentation: The validated and preprocessed input sample.

        Raises:
            RuntimeError: If the input sample contains multiple items.
        """
        if len(input_sample.data) != 1:
            raise RuntimeError(f"{self.__class__.__name__} takes only a single image.")
        return input_sample

    @PreProcessor.validate_input_output
    def execute(self, input_data: ImageRepresentation) -> ImageRepresentation:
        """Normalizes an image using the RetinaNet preprocessor.

        Args:
            input_data (ImageRepresentation): The input data to be processed.

        Returns:
            ImageRepresentation: The normalized image.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        torchvision = Helper.safe_import_package("torchvision", "0.19.0")
        # Normalize the image
        image = input_data.data[0]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = torchvision.transforms.functional.to_tensor(image)
        else:
            image = torch.from_numpy(image)

        # Normalize the image
        image = self.normalize(image)

        # Resize the image
        image = torch.nn.functional.interpolate(
            image[None], size=self.image_size, mode="bilinear", align_corners=False
        )[0]
        input_data.data = [image.numpy()]
        return input_data

    def normalize(
        self,
        image: "torch.Tensor",  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        """Normalizes the given image to have mean and standard deviation as specified by
        the ImageNet dataset.

        Args:
            image (torch.Tensor): The image to be normalized.

        Returns:
            torch.Tensor: The normalized image.
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        # Check if the input image is a floating-point type, which is expected for
        # this function
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )

        # Convert the mean and standard deviation values to torch tensors for easy manipulation
        mean = torch.as_tensor(self.mean)
        std = torch.as_tensor(self.std)

        # Normalize the image by subtracting the mean and then dividing by the standard deviation
        normalized_image = (image - mean[:, None, None]) / std[:, None, None]

        return normalized_image
