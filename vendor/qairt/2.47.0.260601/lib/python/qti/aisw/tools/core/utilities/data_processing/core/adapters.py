# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from abc import ABC, abstractmethod

import numpy as np
from qti.aisw.tools.core.utilities.data_processing.core.representations import Representation
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class OutputAdapter(ABC):
    """An abstract base class for output adapters.

    Output adapters are used to transform model outputs into a
    consistent representation.
    """

    def __init__(self, **kwargs):
        """Initialize the adapters."""
        # Initialize the adapter with any additional keyword arguments
        pass

    @abstractmethod
    def transform(self, input_data: Representation) -> Representation:
        """Transform the input data according to the output adapter's
        requirements.

        This method should filter out any model outputs that are not required,
        and return a new representation containing only the necessary outputs.

        Args:
            input_data (Representation): The input data to be transformed.

        Returns:
            Representation: A new representation with filtered outputs.
        """
        # filter other model outputs that are not required for further processing
        # return a representation containing only required model outputs
        return input_data


class ClassificationOutputAdapter(OutputAdapter):
    """A class to adapt output of a classification model.

    This adapter transforms the output of a classification model into a
    single value, assuming the model outputs a list of probabilities.
    """

    def __init__(self, softmax_index: int = 0):
        """Initialize the ClassificationOutputAdapter.

        Args:
            softmax_index (int): The index of the softmax output.
                Defaults to 0.
        """
        self.softmax_index = softmax_index
        self.validate()

    def validate(self):
        """Validate the input parameters.

        Raises:
            TypeError: If softmax_index is not an integer.
        """
        if not isinstance(self.softmax_index, int):
            raise TypeError("The softmax index must be an integer")

    def transform(self, input_data: Representation) -> Representation:
        """Transforms the given input data by extracting a single value from
        its output list.

        Args:
            input_data (Representation): The input representation object.

        Returns:
            Representation: The transformed representation object.
        """
        if not (isinstance(input_data, Representation) and len(input_data.data) > 0):
            raise ValueError("The input data must be a representation object")
        if self.softmax_index >= len(input_data.data) or self.softmax_index < -1 * len(input_data.data):
            raise ValueError("Invalid index for the softmax output.")
        input_data.data = [input_data.data[self.softmax_index]]
        return input_data


class BoundingBoxOutputAdapter(OutputAdapter):
    """A class to adapt BoundingBox outputs of a object detection model.

    This adapter transforms the bbox output of a object detection model based on user's inputs.
    It allows for conversion from (x, y, w, h) format to (x1, y1, x2, y2) format and swapping
         of X and Y coordinates.
    """

    def __init__(self, xywh_to_xyxy: bool = False, xy_swap: bool = False):
        """Initialize the BoundingBoxOutputAdapter.

        Args:
            xywh_to_xyxy (bool): Whether to convert output from (x, y, w, h) format to
                 (x1, y1, x2, y2) format. Defaults to False.
            xy_swap (bool): Whether to swap X and Y coordinates of bounding boxes. Defaults to False.
        """
        self.xywh_to_xyxy = xywh_to_xyxy
        self.xy_swap = xy_swap
        self.validate()

    def validate(self):
        """Validate the input parameters."""
        if not isinstance(self.xywh_to_xyxy, bool):
            raise TypeError("The xywh_to_xyxy should be bool")

        if not isinstance(self.xy_swap, bool):
            raise TypeError("The xy_swap should be bool")

    def transform(self, input_data: Representation) -> Representation:
        """Transforms the given input data by extracting a single value from
        its output list.

        Args:
            input_data (Representation): The input representation object.

        Returns:
            Representation: The transformed representation object.
        """
        if not (isinstance(input_data, Representation)):
            raise ValueError("The input data must be a representation object")

        bboxes = input_data.data[0]

        # Swap XY coordinates of bbox
        if self.xy_swap:
            bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]

        # Convert bbox format from (x,y,width,height) to (x1,y1,x2,y2)
        if self.xywh_to_xyxy:
            bboxes = self.xywh_to_xyxy_func(bbox=bboxes)

        input_data.data[0] = bboxes
        return input_data

    def xywh_to_xyxy_func(self, bbox: np.ndarray) -> "torch.Tensor":  # noqa: F821
        """Convert nx4 boxes format from [x,y,w,h] to [x1,y1,x2,y2] where
        xy1=top-left, xy2=bottom-right.

        Args:
            bbox (np.ndarray): Bounding box coordinates in format [x, y, w, h].

        Returns:
            torch.Tensor: Converted bounding box coordinates in format [x1, y1, x2, y2].
        """
        torch = Helper.safe_import_package("torch", "2.4.0")
        # Convert numpy array to tensor if needed
        y = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
        # Calculate top-left and bottom-right coordinates from center and width/height
        y[:, 0] = bbox[:, 0] - bbox[:, 2] / 2  # top left x
        y[:, 1] = bbox[:, 1] - bbox[:, 3] / 2  # top left y
        y[:, 2] = bbox[:, 0] + bbox[:, 2] / 2  # bottom right x
        y[:, 3] = bbox[:, 1] + bbox[:, 3] / 2  # bottom right y
        return y
