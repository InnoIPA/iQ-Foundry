# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import DirectoryPath
from qti.aisw.tools.core.utilities.data_processing import (
    Annotation,
    AnnotationType,
    NDArrayRepresentation,
    Representation,
)


class IndexableDataset(ABC):
    """Base class for creating map style/indexable datasets.

        Subclasses should implement the following abstract methods:
            - __len__(): Should return the number of items in the dataset.
            - __getitem__(index): Should return an item at a given index in the dataset.

    The IndexableDataset class is specifically designed to support datasets that can be
    entirely loaded into memory or saved on disk, offering the flexibility of random
    access to the data. This approach is not well-suited for handling large datasets
    or for managing data that is streamed from external sources.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Parse the parameters required for the dataset
        # Add logic to setup data and labels
        """Initializes the Dataset Base class with an optional arguments."""
        pass  # Implement specific dataset initialization logic here

    @abstractmethod
    def __len__(self) -> int:
        """Abstract method to be implemented by subclasses. Should return the
        number of items in the dataset.

        Returns:
            int: The length of the dataset.
        """
        raise NotImplementedError("Each dataset must implement the length method.")

    @abstractmethod
    def __getitem__(self, index: int) -> Representation:
        """Abstract method to be implemented by subclasses. Should return an
        item at a given index in the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Representation: A data object containing the data and label from
            the dataset corresponding to the provided index.
        """
        raise NotImplementedError("Each dataset must implement the get item method.")


class RawInputListDataset(IndexableDataset):
    """A dataset class that handles input lists containing path to raw numpy
    arrays.

    This class provides a way to work with datasets where path to the input
    data is stored as a list in a text file, and annotations are optional.

    Attributes:
        inputlist_file (os.PathLike | str): The path to the input list file.
        dtypes (Optional[list[str]]]): A list of data types for each input array.
                                     Required if files provided are '.raw' files.
        shapes (Optional[list[tuple[int]]]): A list of shapes for each input array.
                                     Required if files provided are '.raw' files.
        annotation_file (Optional[str | os.PathLike | DirectoryPath ]): The path to the annotation file.
                                     If None, no annotations will be used for creating representations.
        annotation_type (Optional[AnnotationType]): The type of annotations in the annotation file.
                                     If None, no annotations will default to AnnotationType.SINGLE.

        absolute_path_list (bool): Whether or not to use absolute paths in the input list file.
    """

    def __init__(
        self,
        inputlist_file: os.PathLike | str,
        dtypes: Optional[list[str]] = None,
        shapes: Optional[list[tuple[int]]] = None,
        annotation_file: Optional[str | os.PathLike | DirectoryPath] = None,
        annotation_type: AnnotationType = AnnotationType.SINGLE,
        absolute_path_list: bool = False
    ):
        """Initialize the InputListDataset object.

        Args:
            inputlist_file (os.PathLike | str): The path to the input list file.
            dtypes (Optional[list[str]]]): A list of data types for each input array.
                                 Required if files provided are '.raw' files.
            shapes (Optional[list[tuple[int]]]): A list of shapes for each input array.
                                 Required if files provided are '.raw' files.
            annotation_file (Optional[str | os.PathLike | DirectoryPath]): The path to the annotation file.
                                 If None, no annotations will be used.
            annotation_type (Optional[AnnotationType]): The type of annotations in the annotation file.
                                 If None, no annotations will default to AnnotationType.SINGLE.
            absolute_path_list (bool): Whether or not to use absolute paths in the input list file.
        """
        self.inputlist_file = inputlist_file
        self.annotation_file = annotation_file
        self.annotation_type = annotation_type
        self.dtypes = dtypes
        self.shapes = shapes
        self.absolute_path_list = absolute_path_list
        # Validate the input parameters
        self.validate()
        self._setup()

    def validate(self):
        """Validate the input parameters."""
        # check the input list file exists and is a text file
        if not os.path.exists(self.inputlist_file):
            raise FileNotFoundError(f"Input list file {self.inputlist_file} provided doesn't exist")
        # check the annotation file exists and is a text file
        if self.annotation_file and (not os.path.exists(self.annotation_file)):
            raise FileNotFoundError(f"Annotation file {self.annotation_file} provided doesn't exist")

    def _setup(self):
        """Setup the dataset.

        Read in the input list, prepare the data and annotation
        information.
        """
        self.data = self.prepare_data()
        # Setup the annotation; Can be dataset specific
        if self.annotation_file:
            self.annotation = self.prepare_annotation()
        else:
            self.annotation = [None] * len(self.data)

    def prepare_data(self) -> list[os.PathLike | Path]:
        """Prepare the input data.

        Read in the input list file and convert it into a list of paths.
        If calibration data is requested, use only the specified indices.

        Returns:
            list[os.PathLike | Path]: list containing paths to the prepared input data files.
        """
        # Determine where to read the input list from
        base_path = os.path.dirname(self.inputlist_file)
        input_per_line = open(self.inputlist_file).readlines()

        # Convert each line into a path and add to data list
        input_paths = [line.strip().split(",") for line in input_per_line]
        if self.absolute_path_list:
            data = [[os.path.abspath(path) for path in paths if len(path) > 0] for paths in input_paths]
        else:
            data = [[os.path.join(base_path, path) for path in paths if len(path) > 0]
                        for paths in input_paths]
        return data

    def prepare_annotation(self) -> list[Annotation]:
        """Prepare the annotation data.

        Returns:
            list[Annotation]: A list containing Annotation object which has
                         type of annotation and the annotation data.
        """
        with open(self.annotation_file) as f:
            annotation_contents = f.readlines()

        if self.annotation_type == AnnotationType.SINGLE:
            if len(self.data) != len(annotation_contents):
                raise ValueError("The number of input data and ground truth annotations do not match.")
            return [Annotation(data.strip()) for data in annotation_contents]
        elif self.annotation_type == AnnotationType.DATASET:
            return [Annotation(self.annotation_file, AnnotationType.DATASET) for _ in range(len(self.data))]
        else:
            raise ValueError(
                f"Invalid annotation type. Annotation type must be one of {AnnotationType.values()}"
            )

    def __getitem__(self, index: int) -> Representation:
        """Get a representation of the data at the specified index.

        Args:
            index (int): The index into the dataset to retrieve.

        Returns:
            Representation: A representation object containing the data and annotation for the given index.
        """
        return NDArrayRepresentation.from_file(
            filepaths=self.data[index],
            dtypes=self.dtypes,
            shapes=self.shapes,
            annotation=self.annotation[index],
            idx=index,
        )

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)


class CalibrationListDataset(IndexableDataset):
    """A dataset class that handles calibration lists.

    This class provides a way to work with datasets where the calibration data is stored in a text file.
    If indices need to be filtered out and returned, users must provide them as an input list.

    Attributes:
        calibrationlist_file (str): The path to the calibration list file.
        calibration_indices (Optional[list[int]]): A list of indices into the input list that
                                     correspond to calibration data.
        dtypes (Optional[list[str]]]): A list of data types for each input array.
                                     Required if files provided are '.raw' files.
        shapes (Optional[list[tuple[int]]]): A list of shapes for each input array
    """

    def __init__(
        self,
        calibrationlist_file: os.PathLike | str,
        dtypes: Optional[list[str]] = None,
        shapes: Optional[list[tuple[int]]] = None,
        calibration_indices: Optional[list[int]] = None,
    ):
        """Initialize the CalibrationListDataset object.

        Args:
            calibrationlist_file (os.PathLike | str): The path to the calibration list file.
            dtypes (Optional[list[str]]]): A list of data types for each input array.
                                     Required if files provided are '.raw' files.
            shapes (Optional[list[tuple[int]]]): A list of shapes for each input array.
                                     Required if files provided are '.raw' files.
            calibration_indices (Optional[list[int]]): A list of indices into the input list
                                     that correspond to calibration data.
        """
        self.calibrationlist_file = calibrationlist_file
        self.calibration_indices = calibration_indices
        self.dtypes = dtypes
        self.shapes = shapes

        # Validate the input parameters
        self.validate()
        self._setup()

    def validate(self):
        """Validate the input parameters."""
        # check the calibration list file exists and is a text file
        if not os.path.exists(self.calibrationlist_file):
            raise FileNotFoundError(f"Calibration list file {self.calibrationlist_file} doesn't exist")
        calibrations_paths = open(self.calibrationlist_file).readlines()
        calibration_count = len(calibrations_paths)
        # check the indices provided are valid
        if isinstance(self.calibration_indices, list) and not len(self.calibration_indices) > 0:
            raise ValueError(
                "Invalid calibration indices were provided. Please provide at least one index as a list."
            )
        if self.calibration_indices and max(self.calibration_indices) >= calibration_count:
            raise ValueError(
                f"The maximum index provided is {max(self.calibration_indices)}, but the number"
                " of calibrations samples in the provided list file is only {calibration_count}"
            )

    def _setup(self):
        """Setup the dataset.

        Read in the calibration list, prepare the data and filter out
        indices if necessary.
        """
        self.calibration_data = self.prepare_data()

    def prepare_data(self) -> list[os.PathLike | Path]:
        """Prepare the calibration data.

        Read in the calibration list file and convert it into a list of paths.
        If filtering is requested, use only the specified indices.

        Returns:
            list[os.PathLike | Path]: list containing paths to the prepared calibration data files.
        """
        # Determine where to read the input list from
        base_path = os.path.dirname(self.calibrationlist_file)
        with open(self.calibrationlist_file) as calibration_file:
            calibration_file_lines = calibration_file.readlines()
            # Filter out indices if necessary
            if self.calibration_indices:
                calibration_file_lines = [calibration_file_lines[i] for i in self.calibration_indices]
        # Convert each line into a path and add to data list
        calibration_data = []
        for line in calibration_file_lines:
            line = line.strip()
            paths = [base_path / Path(p.strip()) for p in line.split(",")]
            calibration_data.append(paths)
        return calibration_data

    def __getitem__(self, index: int) -> Representation:
        """Get a representation of the data at the specified index.

        Args:
            index (int): The index into the dataset to retrieve.

        Returns:
            Representation: A representation object containing the data and annotation for the given index.
        """
        # Get the file name from the list of files.
        # Read the file and return the data.
        return NDArrayRepresentation.from_file(
            filepaths=self.calibration_data[index], dtypes=self.dtypes, shapes=self.shapes, idx=index
        )

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.calibration_data)
