# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import DirectoryPath
from qti.aisw.tools.core.utilities.data_processing import (
    Annotation,
    AnnotationType,
    ImageRepresentation,
    NDArrayRepresentation,
    Representation,
)
from qti.aisw.tools.core.utilities.data_processing.datasets.base import IndexableDataset
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class LegacyDataset(IndexableDataset):
    """A dataset class that handles legacy data formats.

    This class provides a way to work with datasets in a legacy format,
    where the input data is stored in a text file, and the annotations
    are optional.

    Attributes:
        inputlist_path (Optional[os.PathLike| str]): The path to the input list file.
        annotation_path (Optional[os.PathLike| str | DirectoryPath]]): The path to the annotation file.
                                             If None, no annotations will be used.
        calibration_path (Optional[os.PathLike | str]): The path to the calibration file.
                                             If None, no calibration data will be used.
        calibration_indices (Optional[list[int]]): A list of indices into the input list that
                                             correspond to calibration data.
        use_calibration (Optional[bool]): Whether to use calibration data or not. Defaults to False.
    """

    def __init__(
        self,
        inputlist_path: Optional[os.PathLike | str] = None,
        annotation_path: Optional[os.PathLike | str | DirectoryPath] = None,
        calibration_path: Optional[os.PathLike | str] = None,
        calibration_indices: Optional[list[int] | os.PathLike] = None,
        max_samples: Optional[int] = None,
        use_calibration: Optional[bool] = False,
    ):
        """Initialize the LegacyDataset object.

        Args:
            inputlist_path (Optional[os.PathLike| str]): The path to the input list file.
            annotation_path (Optional[os.PathLike| str | DirectoryPath]]): The path to the annotation file.
                                                 If None, no annotations will be used.
            calibration_path (Optional[os.PathLike | str]): The path to the calibration file.
                                                 If None, no calibration data will be used.
            calibration_indices (Optional[list[int]| str| os.PathLike]): A list of indices into the input 
                                list that correspond to calibration data.
            use_calibration (Optional[bool]): Whether to use calibration data or not. Defaults to False.
            max_samples: Optional[int]: Maximum number of samples to load from the dataset

        """
        self.inputlist_path = inputlist_path
        self.annotation_path = annotation_path
        self.calibration_path = calibration_path
        self.calibration_indices = calibration_indices
        self.use_calibration = use_calibration
        self.max_samples = max_samples
        if isinstance(self.calibration_indices, (str, os.PathLike)):
            calibration_indices_str = open(self.calibration_indices).read().strip()
            self.calibration_indices = [int(idx) for idx in calibration_indices_str.split(",")]
        # Validate the input parameters
        self.validate()
        self._setup()

    def validate(self):
        """Validate the input parameters.

        Raises:
            ValueError: If the maximum number of samples is set to 0.
            ValueError: If calibration data is requested but not provided.
        """
        if not os.path.exists(self.inputlist_path):
            raise FileNotFoundError(f"Input list file does not exist. provided: {self.inputlist_path}")
        if self.annotation_path and not os.path.exists(self.annotation_path):
            raise FileNotFoundError(
                f"Annotation file provided does not exist. provided: {self.annotation_path}"
            )
        if self.calibration_path and not os.path.exists(self.calibration_path):
            raise FileNotFoundError(
                f"Calibration file provided does not exist. provided: {self.calibration_path}"
            )
        if self.max_samples == 0:
            raise ValueError("max_samples cannot be set to 0")
        if self.use_calibration and (self.calibration_path is None and self.calibration_indices is None):
            raise ValueError(
                "When use_calibration is set to True, user must provide either calibration_path"
                " or calibration_indices"
            )

        if self.calibration_path and self.calibration_indices:
            raise ValueError(
                "'calibration_path' and 'calibration_index' are mutually exclusive options. "
                "Only one of them must be specified."
            )

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.data)

    def limit_samples(self):
        """Limit the number of samples to use from the dataset.

        If max_samples is set, truncate the data and annotation lists to
        that length.
        """
        if self.max_samples <= len(self.data):
            self.data = self.data[: self.max_samples]
            if isinstance(self.annotation, list):
                self.annotation = self.annotation[: self.max_samples]
        else:
            logging.warning(
                "Using the entire configured data as max_samples provided is more than \
                input samples."
            )

    def _setup(self):
        """Setup the dataset.

        Read in the input list, prepare the data and annotation, and
        limit the number of samples if necessary.
        """
        self.data = self.prepare_data()
        if self.annotation_path:
            self.annotation = self.prepare_annotation()
        else:
            self.annotation = [None] * len(self.data)

        # if max_samples is None or -1, set max_samples to the total number of samples.
        if self.max_samples not in [None, -1]:
            self.limit_samples()
        else:
            self.max_samples = len(self.data)

    def prepare_data(self) -> list[os.PathLike | Path]:
        """Prepare the input data.

        Read in the input list file and convert it into a list of paths.
        If calibration data is requested, use only the specified indices.

        Returns:
            list: The prepared input data.
        """
        # Determine where to read the input list from
        base_path = os.path.dirname(self.inputlist_path)
        if self.use_calibration and self.calibration_path is not None:
            # Read in calibration file instead of input list
            input_per_line = open(self.calibration_path).readlines()
            base_path = os.path.dirname(self.calibration_path)
        elif self.use_calibration and self.calibration_indices:
            # Use specified indices from the input list
            with open(self.inputlist_path, "r") as f:
                lines = f.readlines()
            max_calib_index = max(self.calibration_indices)
            if max_calib_index > len(lines):
                raise ValueError(
                    f"No Calibration Data available for index={max_calib_index} provided \
                    in the inputlist "
                )
            input_per_line = [lines[idx] for idx in self.calibration_indices]
        else:
            # Read in entire input list
            input_per_line = open(self.inputlist_path).readlines()

        # Convert each line into a path and add to data list
        data = []
        for line in input_per_line:
            input_paths = []
            for path in line.strip().split(","):
                if len(path) > 0:
                    input_paths.append(os.path.join(base_path, path))
            data.append(input_paths)

        return data

    def prepare_annotation(self) -> list[Annotation]:
        """Prepare the annotation data.

        This method is optional and should be implemented by subclasses to
        provide ground truth annotations for the dataset.

        Returns:
            tuple: A tuple containing the type of annotation and the annotation data.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Representation:
        """Get a representation of the data at the specified index.

        Args:
            index (int): The index into the dataset to retrieve.

        Returns:
            Representation: A representation object containing the data and annotation
            for the given index.
        """
        return NotImplementedError("__getitem__ method must be implemented in a subclass.")


class LegacyImageDataset(LegacyDataset):
    """A dataset class for legacy images, allowing the choice of image processing backend.

    Args:
        *args: Arguments passed to the parent class.
        image_backend (str): The image processing library to use. Defaults to "opencv".
        **kwargs: Additional keyword arguments passed to the parent class.

    Attributes:
        image_backend (str): The chosen image processing library.
    """

    def __init__(self, *args, image_backend: str = "opencv", **kwargs):
        """Args:
        image_backend (str): Backend for loading images. Defaults to "opencv".
        """
        super().__init__(*args, **kwargs)
        self.image_backend = image_backend


class ImagenetDataset(LegacyImageDataset):
    """A class representing the ImageNet dataset."""

    def prepare_annotation(self) -> dict[str, str]:
        """Prepare the annotation dictionary from the annotation file.

        Returns:
            dict[str, str]: A dictionary containing the ground truth annotations.

        Notes:
            This method assumes that each line in the annotation file contains an image
             ID followed by a space and then the label. The image IDs are
             assumed to be case-sensitive.
        """
        # Optional: Required when the user wants to compute metric.
        # Must return a list containing ground truth
        annotation_contents = open(self.annotation_path).readlines()
        # Create an empty dictionary to store annotations
        annotation = {}
        for line in annotation_contents:
            # Split the line into image ID and label
            img_id, label = line.split(" ")
            # Map image IDs to lowercase for consistency
            annotation[img_id.lower().strip()] = int(label.strip().lower())
        return annotation

    def __getitem__(self, index: int) -> ImageRepresentation:
        """Get a image representation of the data at the specified index.

        Args:
            index (int): The index into the dataset to retrieve.

        Returns:
            ImageRepresentation: A ImageRepresentation object containing the data and annotation
            for the given index.
        """
        # Get the image path from the data list
        img_path = self.data[index][0]
        # Extract the image ID from the file name
        if self.annotation_path is not None:
            img_id = os.path.basename(img_path).lower()
        else:
            img_id = index

        return ImageRepresentation.from_file(
            [img_path],
            annotation=Annotation(self.annotation[img_id]),
            idx=index,
            image_backend=self.image_backend,
        )


class WIDERFaceDataset(LegacyImageDataset):
    """A class representing the WIDERFace dataset."""

    def __init__(self, *args, annotation_path: Optional[DirectoryPath] = None, **kwargs):
        """Initialize a WIDERFace related parameters and annotation info."""
        self._annotation_mat_files = [
            "wider_easy_val.mat",
            "wider_face_val.mat",
            "wider_hard_val.mat",
            "wider_medium_val.mat",
        ]
        super().__init__(*args, annotation_path=annotation_path, **kwargs)

    def _setup(self):
        """Setup the dataset.

        Read in the input list, prepare the data and annotation, and
        limit the number of samples if necessary.
        """
        self.data = self.prepare_data()
        # Setup the annotation; Can be dataset specific
        if self.annotation_path:
            self.annotation = self.prepare_annotation()
        else:
            self.annotation = [None] * len(self.data)

        if self.max_samples is not None:
            self.limit_samples()
        else:
            self.max_samples = len(self.data)

    def validate(self):
        """Validate the dataset parameters."""
        super().validate()
        if self.annotation_path:
            if not os.path.isdir(self.annotation_path):
                raise ValueError(
                    f"Annotation path is invalid. Annotation path provided \
                        must be a directory containing. {self._annotation_mat_files}"
                )
            else:
                for mat_file in self._annotation_mat_files:
                    if not os.path.exists(os.path.join(self.annotation_path, mat_file)):
                        raise ValueError(
                            f"Annotation path does not contain the required \
                            annotation file :{mat_file}.\n Annotation path \
                            provided must be a directory containing. {self._annotation_mat_files}"
                        )

    def prepare_annotation(self) -> Annotation:
        """Prepare the WIDERFACE dataset annotation.

        Returns:
            Annotation: Annotation object containing the ground truth information.
        """
        return Annotation(self.annotation_path, AnnotationType.DATASET)

    def __getitem__(self, index: int) -> ImageRepresentation:
        """Get a image representation of the data at the specified index.

        Args:
            index (int): The index into the dataset to retrieve.

        Returns:
            ImageRepresentation: A ImageRepresentation object containing the data and annotation
            for the given index.
        """
        img_path = self.data[index]
        if not self.annotation_path:
            annotation = Annotation(None, AnnotationType.DATASET)
        else:
            annotation = self.annotation
        return ImageRepresentation.from_file(
            img_path, annotation=annotation, idx=index, image_backend=self.image_backend
        )


class WMT20Dataset(LegacyDataset):
    """A dataset class for the WMT20 task"""

    def prepare_data(self) -> list[str]:
        """Reads and returns the contents of the input list file.

        This method reads each line from the specified input list file, which should contain
        one item per line. It does not perform any filtering or processing on this data;
        it simply returns all lines as a list.

        Returns:
            list[str]: A list of strings, where each string represents a single item in the dataset.
        """
        data = open(self.inputlist_path).readlines()
        if self.use_calibration and self.calibration_indices:
            data = [data[idx] for idx in self.calibration_indices]
        return data

    def prepare_annotation(self) -> list[Annotation]:
        """Reads and returns the contents of the annotation file.

        This method reads each line from the specified annotation file,
        which should contain one annotation per line. It parses each line into an Annotation object,
         which is returned as a list. This allows for further processing or filtering to be applied
         to the annotations later on in the pipeline.

        Returns:
            list[Annotation]: A list of Annotation objects, where each object represents a single
          annotation associated with the data.
        """
        annotation_contents = open(self.annotation_path).readlines()
        annotation_data = [
            Annotation(data=annotation_data.strip()) for annotation_data in annotation_contents
        ]
        if self.use_calibration and self.calibration_indices:
            annotation_data = [annotation_data[idx] for idx in self.calibration_indices]
        return annotation_data

    def __getitem__(self, index: int) -> NDArrayRepresentation:
        """Get a ndarray representation of the data at the specified index.

        Args:
            index (int): The index into the dataset to retrieve.

        Returns:
            NDArrayRepresentation: A NDArrayRepresentation object containing the data and annotation
            for the given index.
        """
        data_str_array = np.array(self.data[index])
        if not self.annotation_path:
            annotation = Annotation(None)
        else:
            annotation = self.annotation[index]
        return NDArrayRepresentation([data_str_array], annotation=annotation, idx=index)


class SYN_CHINESE_LP_Dataset(LegacyImageDataset):
    """A class representing the Synthetic Chinese License Plates dataset."""

    def prepare_annotation(self) -> dict[str, str]:
        """Prepare the annotation dictionary from the annotation file.

        Returns:
            dict[str, str]: A dictionary containing the ground truth annotations.

        Notes:
            This method assumes that each line in the annotation file contains an image
             ID followed by a space and then the label. The image IDs are
             assumed to be case-sensitive.
        """
        # Optional: Required when the user wants to compute metric.
        # Must return a list containing ground truth
        annotation_contents = open(self.annotation_path).readlines()
        # Create an empty dictionary to store annotations
        annotation = {}
        for line in annotation_contents:
            # Split the line into image ID and label
            img_id, label = line.split(" ")
            # Map image IDs to lowercase for consistency
            annotation[img_id.lower().strip()] = label.strip()
        return annotation

    def __getitem__(self, index: int) -> ImageRepresentation:
        """Get a image representation of the data at the specified index.

        Args:
            index (int): The index into the dataset to retrieve.

        Returns:
            ImageRepresentation: A ImageRepresentation object containing the data and annotation
            for the given index.
        """
        # Get the image path from the data list
        img_path = self.data[index][0]
        # Extract the image ID from the file name
        if self.annotation_path is not None:
            img_id = os.path.basename(img_path).lower()
        else:
            img_id = index

        return ImageRepresentation.from_file(
            [img_path],
            annotation=Annotation(self.annotation[img_id]),
            idx=index,
            image_backend=self.image_backend,
        )


class COCO2017Dataset(LegacyImageDataset):
    """A class representing the COCO2017 dataset.

    This class provides a way to work with COCO datasets, where the input data is stored in a JSON file.
    """

    def __getitem__(self, index: int) -> ImageRepresentation:
        """Get a representation of the data at the specified index.

        Args:
            index (int): The index into the dataset to retrieve.

        Returns:
            ImageRepresentation: A representation object containing the data and annotation
            for the given index.
        """
        img_path = self.data[index]
        annotation = Annotation(self.annotation_path, AnnotationType.DATASET)
        return ImageRepresentation.from_file(img_path, annotation=annotation,
                 image_backend=self.image_backend, idx=index)
