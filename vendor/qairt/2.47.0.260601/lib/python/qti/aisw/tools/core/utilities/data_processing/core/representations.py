# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np
from PIL import Image
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class CustomEnum(Enum):
    """A custom enum class that allows for easy retrieval of all enum values.

    Methods:
        values() -> list: Returns a list containing all enum value strings.
    """

    @classmethod
    def values(cls):
        """A class method that returns a list of all enum values as strings.

        Returns:
            list[str]: A list containing all enum value strings.
        """
        return [item.value for item in list(cls)]


class ImageBackends(CustomEnum):
    """Enum for supported image backends."""

    OPENCV = "opencv"
    PILLOW = "pillow"


class SupportedDtypes(CustomEnum):
    """Enum for supported data types."""

    FLOAT32 = "float32"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    UINT8 = "uint8"


class AnnotationType(CustomEnum):
    """Enum for supported annotation types."""

    SINGLE = "single"
    DATASET = "dataset"  # This type provides reference to annotation for entire dataset


IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]  # Supported file extensions


def is_list_of_ints(shape: Iterable) -> bool:
    """Checks if a given data structure (list or tuple) contains only integers.

    Args:
        shape: The input list or tuple to be checked. e.g., [1,2] or (3,)

    Returns:
        bool: True if the input is a list of lists of integers, False otherwise.
    """
    # Check if input is an iterable
    if not isinstance(shape, (list, tuple)):
        return False

    for dim_value in shape:
        if not isinstance(dim_value, int):
            return False
    return True


def get_file_extension(file_path: str | Path) -> str:
    """Returns the file extension of a given file path.

    Args:
        file_path (str | Path): The file path to extract the extension from.

    Returns:
        str: The file extension in lowercase, without leading dot.

    Raises:
        TypeError: If `file_path` is neither a string nor a Path object.
    """
    if not isinstance(file_path, (str, Path)):
        raise TypeError(f"Input must be a string or a Path object. Given type: {type(file_path)}")

    file_path = str(file_path).strip()
    if not file_path:
        return ""

    file_extension = os.path.splitext(file_path)[1].lstrip(".").lower()
    return file_extension


class Annotation:
    """Class representing an annotation.

    Attributes:
        data: The actual data being annotated.
        annotation_type (AnnotationType): Type of the annotation. Defaults to AnnotationType.SINGLE.
    """

    def __init__(self, data: Any, annotation_type: AnnotationType = AnnotationType.SINGLE):
        """Initialize an Annotation object with the given data and type."""
        self.data = data
        self.type = annotation_type

    def __eq__(self, other):
        """Check if two annotations are equal.

        Args:
            other (Annotation): The other annotation to compare with.

        Returns:
            bool: True if the annotations are equal, False otherwise.
        """
        return isinstance(other, Annotation) and self.data == other.data and self.type == other.type


class Representation(ABC):
    """Base class for all representations.

    This class provides a basic structure for representing data, including metadata and
    annotations. It serves as an interface for subclasses to implement specific loading,
    saving, and data retrieval methods.
    """

    def __init__(
        self,
        data: list[np.ndarray],
        metadata: Optional[dict] = {},
        annotation: Optional[Annotation] = None,
        idx: Optional[int] = -1,
        **kwargs,
    ):
        """Initializes the Representation object.

        Args:
            data (list[np.ndarray]): The data to be represented, should be a list of numpy arrays.
            metadata (Optional[dict]): Additional information about the input sample. Defaults to {}.
            annotation (Optional[Annotation]): Label or ground truth information. Defaults to None.
            idx (Optional[int]): Identifier for differentiating and identifying input samples. Defaults to -1.
            kwargs (Optional[dict]): Additional arguments that may be required by the representation class.
        """
        self._data = data
        self._metadata = metadata if metadata is not None else {}
        self._annotation = annotation
        self._idx = idx  # Identifier to be used by Tools for differentiating and identifying input samples.

    def save(self, path):
        """Save self.data to path supplied by the user on the disk.

        This method is intended to be overridden by subclasses. The default
        implementation raises a NotImplementedError.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @property
    def data(self) -> list[np.ndarray]:
        """Retrieves the data associated with this representation.

        Returns:
            list[np.ndarray]: A list of NumPy arrays representing the data.
        """
        return self._data

    @data.setter
    def data(self, data: list[np.ndarray]):
        """Sets the data field of this Representation object.

        Args:
            data (list[np.ndarray]): The new data to be represented.
        """
        self._data = data
        self._infer_meta_data()

    @property
    def annotation(self) -> Annotation:
        """Get the annotation stored in this Representation object.

        Returns:
            Annotation: The label or ground truth information.
        """
        return self._annotation

    @annotation.setter
    def annotation(self, annotation: Annotation):
        """Set the annotation field of this Representation object.

        Args:
            annotation (Annotation): The new label or ground truth information.
        """
        self._annotation = annotation

    @property
    def metadata(self, key: Optional[str] = None) -> Any:
        """Get a metadata value by its key.

        Args:
            key: The key to look up in the metadata dictionary.
            if key is not provided, return all of the metadata values.

        Returns:
            Any: The value associated with the key, or None if it does not exist.
        """
        if key is None:
            return self._metadata
        else:
            return self._metadata[key]

    @metadata.setter
    def metadata(self, value, key=None):
        """Set a metadata value by its key.

        Args:
            key: The key to associate with the new value.
            value: The new value to be stored in the metadata dictionary.
        """
        if key is None:
            self._metadata[key] = value
        self._metadata = value

    @property
    def idx(self) -> int:
        """Get the index identifier of this Representation object.

        Returns:
            int: The identifier.
        """
        return self._idx

    def _infer_meta_data(self) -> None:
        """Infer metadata for numpy array data.

        If self.data is a list of numpy array, infer datatype and shapes of each data
        item in the list if it is not already set
        """
        shapes = []
        dtypes = []
        for item in self.data:
            if isinstance(item, np.ndarray):
                shapes.append(item.shape)
                dtypes.append(str(item.dtype))
            elif isinstance(item, Image.Image):
                shapes.append(item.size)
                dtypes.append(type(item))
        self.shapes = shapes
        self.dtypes = dtypes


class NDArrayRepresentation(Representation):
    """A representation class for numpy arrays."""

    def __init__(self, data: list[np.ndarray], **kwargs) -> None:
        """Initialize the NDArrayRepresentation.

        Args:
          data: A list of numpy arrays.
          kwargs: Additional keyword arguments.
        """
        super().__init__(data=data, **kwargs)
        self.validate()
        self._infer_meta_data()

    @staticmethod
    def _validate_file_paths(file_paths: list[str | os.PathLike]) -> None:
        """Validate a list of file paths.

        Args:
          file_paths: A list of file paths.
        """
        if not file_paths or len(file_paths) == 0:
            raise ValueError("No files were provided.")
        for path in file_paths:
            if not os.path.exists(str(path).strip()):
                raise ValueError(f"File {path} does not exist.")

    @classmethod
    def from_file(
        cls, filepaths: list[os.PathLike | str], dtypes: list[str], shapes: list[list[int]], **kwargs
    ) -> "NDArrayRepresentation":
        """Initialize the NDArrayRepresentation from a list of files.

        Args:
          filepaths (list[os.PathLike | str]): A list of file paths to read data from.
          dtypes (list[str]): A list of dtypes for each file. Required when filepaths are ".raw" files.
          shapes (list[list[int]]): A list of shapes for each file. Required when filepaths are ".raw" files.
          kwargs: Additional keyword arguments.

        Returns:
            NDArrayRepresentation: A NDArrayRepresentation object with the data loaded from file paths.

        Raises:
            ValueError: If the number of files, shapes, or data types do not match.
            ValueError: If any file path does not have '.raw' extension.
            ValueError: If invalid data type is provided.
            ValueError: If invalid shape is provided.

        Usage Example:
            >>> from qti.aisw.tools.core.utilities.data_processing import NDArrayRepresentation
            >>> nd_arr_repr = NDArrayRepresentation.from_file(
                filepaths=['path/to/file1.raw', 'path/to/file2.raw'],
                dtypes=["float32", "int8"],
                shapes=[[10, 10], [20, 20]])
        """
        cls._validate_file_paths(filepaths)
        if not (isinstance(dtypes, list) and all(dtype in SupportedDtypes.values() for dtype in dtypes)):
            raise ValueError(
                f"Invalid data type provided. Data type must be one of {SupportedDtypes.values()}"
            )
        if not (
            isinstance(shapes, list)
            and all(isinstance(shape, (list, tuple)) and is_list_of_ints(shape) for shape in shapes)
        ):
            raise ValueError("Invalid shape provided. Shape must be a list of integers")
        if not (len(filepaths) == len(shapes) == len(dtypes)):
            raise ValueError("Number of files must match number of shapes and data types provided.")

        # Check if all filepaths have '.raw' extension
        is_raw_file_paths = all(get_file_extension(path) == "raw" for path in filepaths)
        if not is_raw_file_paths:
            raise ValueError("NDArrayRepresentation must be supplied only with files having '.raw' extension")
        data = []
        for idx in range(len(filepaths)):
            data.append(np.fromfile(str(filepaths[idx]).strip(), dtype=dtypes[idx]).reshape(shapes[idx]))
        return cls(data, **kwargs)

    def validate(self):
        """Validate the data."""
        if not all(isinstance(d, np.ndarray) for d in self.data):
            raise ValueError(
                f"Data must be a list of numpy arrays. Use {self.__class__.__name__}.from_file()"
                " method to load  raw files"
            )

    def save(self, paths: list[str | os.PathLike]) -> None:
        """Save the numpy array data to disk.

        Args:
            paths (list[str | os.PathLike]): The list of file paths to write to
        Raises:
            ValueError: If the number of files paths does not match with the number of items in data.
        """
        if len(paths) != len(self.data):
            raise ValueError(
                f"Number of files paths ({len(paths)}) provided does not match with the number of items in"
                "data ({len(self.data)})."
            )
        for idx, path in enumerate(paths):
            self.data[idx].tofile(path)

    def __repr__(self) -> str:
        """Returns a string representation of the object."""
        class_name = self.__class__.__name__
        fmt_string = ""
        for k, v in self.__dict__.items():
            if "__" not in k:
                fmt_string += f"{k}={v}, "
        return f"{class_name}({fmt_string.rstrip(', ')})"


class TextRepresentation(NDArrayRepresentation):
    """Text representation of data."""

    def __init__(self, data: list[np.ndarray], **kwargs):
        """Initializes the TextRepresentation object.

        Args:
            data (list[np.ndarray]): The list of tokenized numpy arrays.
            **kwargs: Additional keyword arguments to be passed to the TextRepresentation constructor.
        """
        super().__init__(data=data, **kwargs)

    @classmethod
    def from_string(cls, data: str, tokenizer: str | Callable[[str], dict], **kwargs) -> "TextRepresentation":
        """Creates a TextRepresentation object from a string input.

        Args:
            data (str): The input string to be tokenized.
            tokenizer (str | Callable[[str], dict]): A string representing the name of the transformer-based
             tokenizer, or a callable function that takes a string and returns a dictionary with
             tokenized outputs.
            **kwargs: Additional keyword arguments to be passed to the TextRepresentation constructor.

        Returns:
            TextRepresentation: A TextRepresentation object with the input string, metadata, and tokenized
            outputs.

        Usage example:
            >>> from qti.aisw.tools.core.utilities.data_processing import TextRepresentation
            >>> text_data = "Hello world"
            >>> tokenizer_name = "bert-base-uncased"
            >>> tokenizer = TextRepresentation.from_string(data=text_data, tokenizer=tokenizer_name)
        """
        transformers = Helper.safe_import_package("transformers", "4.44.0")
        metadata = {}
        if not isinstance(data, str):
            raise TypeError(f"The input data must be a string but got {type(data)}.")
        # If tokenizer is a string, assume it's the name of a pre-trained model
        if isinstance(tokenizer, str):
            metadata["tokenizer_name"] = tokenizer
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
            except Exception:
                raise ValueError(
                    "Failed to create tokenizer with supplied tokenizer name is using transformers"
                    "AutoTokenizer API"
                )
        if not callable(tokenizer):
            raise TypeError(
                "The supplied tokenizer must be a callable function that takes a string and returns"
                " a dictionary with tokenized outputs."
            )
        # Get the tokenized outputs from the transformer-based tokenizer
        tokenized_outputs = tokenizer(data).data
        if not isinstance(tokenized_outputs, dict):
            raise ValueError(
                "The output of the supplied "
                f"tokenizer must be a dictionary but got {type(tokenized_outputs)}."
            )
        # Store the input string and tokenizer for later use
        metadata["source_string"] = data
        metadata["tokenizer"] = tokenizer
        # Convert the tokenized outputs to numpy arrays and store them in a list
        data = [np.array(val) for val in tokenized_outputs.values()]
        return cls(data, metadata=metadata, **kwargs)


class ImageRepresentation(NDArrayRepresentation):
    """Representation class for image data."""

    def __init__(self, data: list[np.ndarray | Image.Image], **kwargs):
        """Initializes the ImageRepresentation instance.

        Args:
            data (list[np.ndarray]): A list of NumPy arrays or PIL Images containing the image data.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(data=data, **kwargs)
        self.__dict__.update(kwargs)

    def validate(self):
        """Validate the data."""
        if not all(isinstance(d, (np.ndarray, Image.Image)) for d in self.data):
            raise ValueError(
                f"Data must be a list of numpy arrays or PIL Images. Use {self.__class__.__name__}.from_file("
                ") method to load raw files"
            )

    @classmethod
    def from_file(
        cls,
        filepaths: list[str | os.PathLike],
        image_backend: Optional[ImageBackends] = "pillow",
        dtypes: Optional[list[str]] = None,
        shapes: Optional[list[list[int]]] = None,
        **kwargs,
    ) -> "ImageRepresentation":
        """Creates an ImageRepresentation instance from a list of file
        paths.

        Args:
            filepaths (list[str | os.PathLike]): A list of file paths to the images.
            image_backend (Optional['pillow', 'opencv']): The backend to use for loading images.
            dtypes (list[str]): A list of dtypes for each file. Required when filepaths are ".raw" files.
            shapes (list[list[int]]): A list of shapes for each file. Required when filepaths are ".raw" files.
            **kwargs: Additional keyword arguments.

        Returns:
            ImageRepresentation: An instance of the ImageRepresentation class.
        Usage Example:
            >>> from qti.aisw.tools.core.utilities.data_processing import ImageRepresentation
            >>> image_repr = ImageRepresentation.from_file(file_paths=["image1.jpg", "image2.png"])
        """  # noqa: E501
        cls._validate_file_paths(filepaths)
        is_raw_file_paths = all(get_file_extension(path) == "raw" for path in filepaths)
        if is_raw_file_paths:
            return super().from_file(
                filepaths, image_backend=image_backend, dtypes=dtypes, shapes=shapes, **kwargs
            )

        # check if the filepaths have valid extensions. If not, raise an error.
        for filepath in filepaths:
            if get_file_extension(filepath).lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"The file extension of {filepath} is not supported.")
        # check if valid image_backend is provided
        if image_backend not in ImageBackends.values():
            raise ValueError(
                f"{image_backend} is not a valid image backend."
                f" Please choose one from {ImageBackends.values()}"
            )

        metadata = {"image_backend": image_backend, "source_paths": filepaths}
        data = []
        file_extensions = []
        image_modes = []
        for path in filepaths:
            if image_backend == "opencv":
                data.append(cv2.imread(str(path)))
            else:
                loaded_image = Image.open(str(path))
                data.append(loaded_image)
                file_extensions.append(loaded_image.format)
                image_modes.append(loaded_image.mode)
        metadata["file_extension"] = file_extensions
        metadata["image_mode"] = image_modes
        return cls(data=data, metadata=metadata, **kwargs)
