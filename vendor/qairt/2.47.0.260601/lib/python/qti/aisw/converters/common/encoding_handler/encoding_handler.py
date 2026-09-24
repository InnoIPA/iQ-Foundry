# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import copy
import numpy as np
import json
import jsonschema
import os
from pathlib import Path
import numbers

# Schema cache to avoid repeated slow file reads
_SCHEMA_CACHE = {}

def load_encoding_schema(version: str) -> Dict[str, Any]:
    """
    Load JSON schema for the specified encoding version.

    Args:
        version: Encoding version string (e.g., "0.6.1", "1.0.0", "2.0.0")

    Returns:
        Dictionary containing the JSON schema

    Raises:
        ValueError: If schema file is not found or invalid
        FileNotFoundError: If schema file doesn't exist
    """
    if version in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[version]

    current_dir = Path(__file__).parent
    schema_file = current_dir / "schemas" / f"encoding_schema_v_{version.replace('.', '_')}.json"

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    try:
        with open(schema_file, 'r') as f:
            schema = json.load(f)

        _SCHEMA_CACHE[version] = schema
        return schema

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema file {schema_file}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading schema file {schema_file}: {e}")

class EncodingVersion(Enum):
    ENCODING_VERSION_V_0_6_1 = "0.6.1"
    ENCODING_VERSION_V_1_0_0 = "1.0.0"
    ENCODING_VERSION_V_2_0_0 = "2.0.0"

class EncodingType(Enum):
    PER_TENSOR = "PER_TENSOR"
    PER_CHANNEL = "PER_CHANNEL"
    PER_BLOCK = "PER_BLOCK"
    LPBQ = "LPBQ"
    VECTOR = "VECTOR"

class TensorType(Enum):
    PARAM = "PARAM"
    ACTIVATION = "ACTIVATION"
    UNKNOWN = "UNKNOWN"

class DType(Enum):
    INT = "INT"
    UINT = "UINT"
    FLOAT = "FLOAT"

@dataclass
class EncodingInfo:
    """
    Data class representing encoding information for a tensor.

    This provides a unified representation of encoding parameters across
    all encoding versions, with scale and zero_point always as lists
    to handle both per-tensor and per-channel quantization uniformly.
    """
    name: str
    scale: Optional[List[float]] = None  # Always a list, [0.1] for per-tensor, [0.1, 0.2, 0.15] for per-channel
    zero_point: Optional[List[Union[int, float]]] = None  # Always a list, [128] for per-tensor, [128, 127, 130] for per-channel
    tensor_type: TensorType = TensorType.UNKNOWN  # TensorType enum value
    is_symmetric: Optional[bool] = None
    block_size: Optional[List[int]] = None
    per_block_int_scale: Optional[List] = None
    per_channel_float_scale: Optional[List] = None
    encoding_type: Optional[EncodingType] = None
    compressed_bw: Optional[int] = None
    axis: Optional[int] = None
    bitwidth: Optional[int] = None
    dtype: Optional[DType] = None
    min: float = None
    max: float = None
    zero_point_shift: Optional[List[float]] = None

    def __post_init__(self):
        """Validate all member types and raise TypeError if invalid."""

        # Validate required string field
        if not isinstance(self.name, str):
            raise TypeError(f"EncodingInfo.name must be str, got {type(self.name).__name__}: {self.name}")

        # Validate types
        self._validate_type('compressed_bw', int)
        self._validate_type('axis', int)
        self._validate_type('bitwidth', int)
        self._validate_type('min', (int, float, list))
        self._validate_type('max', (int, float, list))
        self._validate_type('tensor_type', TensorType)
        self._validate_type('encoding_type', EncodingType)
        self._validate_type('dtype', DType)

        # Validate list element types
        self._validate_list_type('scale', numbers.Number)
        self._validate_list_type('per_channel_float_scale', numbers.Number)
        self._validate_list_type('block_size', int)
        self._validate_list_type('per_block_int_scale', int)
        self._validate_list_type('zero_point_shift', numbers.Number)
        self._validate_zero_point()

    def _validate_type(self, field_name: str, expected_type):
        """Helper method to validate types that can be None or the expected type."""
        value = getattr(self, field_name)
        if value is not None and not isinstance(value, expected_type):
            expected_type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
            raise TypeError(f"EncodingInfo.{field_name} must be {expected_type_name} or None, "
                          f"got {type(value).__name__}: {value}")

    def _validate_list_type(self, field_name: str, expected_type):
        value = copy.deepcopy(getattr(self, field_name))
        if value is not None and not isinstance(value, list):
            expected_type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
            raise TypeError(f"EncodingInfo.{field_name} must contain {expected_type_name} or None, "
                        f"got {type(value).__name__}: {value}")

        while value:
            item = value.pop()
            if isinstance(item, list):
                value.extend(item)
            elif not isinstance(item, expected_type):
                expected_type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
                raise TypeError(f"EncodingInfo.{field_name} must contain {expected_type_name} or None, "
                          f"got {type(value).__name__}: {value}")

    def _validate_zero_point(self):
        zero_point = copy.deepcopy(self.zero_point)
        if zero_point is not None and not isinstance(zero_point, list):
            raise TypeError(f"EncodingInfo.zero_point must be a list or None, "
                        f"got {type(zero_point)}.")

        while zero_point:
            item = zero_point.pop()
            if isinstance(item, list):
                zero_point.extend(item)
            elif not isinstance(item, numbers.Number):
                raise TypeError(f"EncodingInfo.zero_point must contain numbers, "
                          f"got {type(item)}")

    def get_block_size(self):
        block_size = getattr(self, "block_size")
        if isinstance(block_size, list) and len(block_size) == 1:
            block_size = block_size[0]
        return block_size

    def infer_symmetricity(self) -> bool:
        """
        Infer symmetricity of the encoding.

        Returns is_symmetric if it's not None, otherwise infers symmetry
        using the logic from C++ checkSymmetricity function.

        Returns:
            bool: True if symmetric, False if asymmetric

        Raises:
            ValueError: If required fields are missing for symmetricity inference
        """
        if self.is_symmetric is not None:
            return self.is_symmetric

        if self.dtype is None:
            raise ValueError("Cannot infer symmetricity: dtype is required")

        if self.dtype == DType.FLOAT:
            raise ValueError("Cannot infer symmetricity: FLOAT encodings do not have symmetricity information")

        zero_point_array = np.array(self.zero_point) if self.zero_point is not None else np.array([0])
        scale_array = np.array(self.scale) if self.scale is not None else None

        # Signed Symmetric Check
        if self.dtype == DType.INT:
            return np.all(zero_point_array == 0)

        # Unsigned Symmetric Check
        elif self.dtype == DType.UINT:
            if self.bitwidth is None:
                raise ValueError("Cannot infer symmetricity for unsigned encoding: bitwidth is required")

            expected_symmetric_offset = 2**(self.bitwidth - 1)

            # ALL zero_points must match the expected symmetric offset
            if not np.all(zero_point_array == expected_symmetric_offset):
                return False

            # Calculate effective min values for all encodings
            effective_min_array = None

            # Use provided min if available and non-zero
            if self.min is not None and self.min != 0:
                effective_min_array = np.full(len(zero_point_array), self.min)
            # Calculate min from scale and offset if scale is available
            elif scale_array is not None:
                # offset = -zero_point in C++ terms
                offset_array = -zero_point_array
                effective_min_array = scale_array * offset_array
            else:
                raise ValueError("Cannot infer symmetricity for unsigned encoding: either min or scale is required")

            # For unsigned symmetric: ALL effective mins must be < 0.0
            return np.all(effective_min_array < 0.0)
        else:
            raise ValueError(f"Cannot infer symmetricity for encoding with {self.dtype.value} dtype.")

class EncodingHandler(ABC):
    """
    Abstract base class for handling different encoding versions.

    This class provides a unified interface for converting version-specific
    encoding formats to a common schema, eliminating the need for conditional
    version checks throughout the codebase.
    """

    def __init__(self, encodings: Dict[str, Any]):
        """
        Initialize handler with encoding dictionary and convert to unified format.

        Args:
            encodings: Version-specific encoding dictionary
        """
        self.encodings = self._convert_to_unified_encoding(encodings)

    def get_encoding_info(self, tensor_name: str) -> Optional[EncodingInfo]:
        """
        Get encoding information for a specific tensor.

        Args:
            tensor_name: Name of the tensor to get encoding for

        Returns:
            EncodingInfo object if found, None otherwise
        """
        return self.encodings.get(tensor_name)

    def has_encoding(self, tensor_name: str) -> bool:
        """
        Check if tensor has encoding.

        Args:
            tensor_name: Name of the tensor

        Returns:
            True if EncodingInfo object is found, False otherwise
        """
        return tensor_name in self.encodings

    @abstractmethod
    def get_version(self) -> EncodingVersion:
        """
        Get the encoding version this handler supports.

        Returns:
            EncodingVersion enum value
        """
        pass

    def validate_encodings(self, encodings: Dict[str, Any]) -> None:
        """
        Validate the raw encoding dictionary format using JSON schema.

        Args:
            encodings: Version-specific encoding dictionary to validate

        Raises:
            ValueError: If encodings are invalid or malformed
        """
        try:
            schema = load_encoding_schema(self.get_version().value)
            jsonschema.validate(encodings, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid encoding format for version {self.get_version().value}: {e.message}")

    @abstractmethod
    def _convert_to_unified_encoding(self, encodings: Dict[str, Any]) -> Dict[str, EncodingInfo]:
        """
        Convert version-specific encoding format to unified format.

        Args:
            encodings: Version-specific encoding dictionary

        Returns:
            Dictionary mapping tensor names to EncodingInfo objects:
            {
                "tensor_name_1": EncodingInfo(...),
                "tensor_name_2": EncodingInfo(...),
                ...
            }
        """
        pass

    def set_encoding(self, tensor_name: str, encoding_info: EncodingInfo) -> None:
        """
        Set a tensor encoding in the handler (adds new or updates existing).

        Args:
            tensor_name: Name of the tensor
            encoding_info: EncodingInfo object to set
        """
        self.encodings[tensor_name] = encoding_info

    def remove_encoding(self, tensor_name: str) -> None:
        """
        Remove a tensor encoding from the handler.

        Args:
            tensor_name: Name of the tensor to remove
        """
        if tensor_name in self.encodings:
            del self.encodings[tensor_name]

    def copy_encoding(self, source_tensor: str, dest_tensor: str) -> None:
        """
        Copy encoding from one tensor to another.

        Args:
            source_tensor: Name of the source tensor
            dest_tensor: Name of the destination tensor
        """
        if source_tensor in self.encodings:
            # Create a copy of the encoding info with the new name
            source_encoding = self.encodings[source_tensor]
            dest_encoding = copy.deepcopy(source_encoding)
            dest_encoding.name = dest_tensor
            self.encodings[dest_tensor] = dest_encoding

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Get encodings in the handler's native format.

        Returns:
            Dictionary in the version-specific format
        """
        pass

    @abstractmethod
    def get_tensor_encoding(self, tensor_name):
        """
        Get tensor encoding in the handler's native format.

        Returns:
            Encoding in the version-specific format
        """
        pass

    @abstractmethod
    def convert_encoding_to_info(self, tensor_name: str, encoding: Any, tensor_type: TensorType) -> EncodingInfo:
        """
        Convert version-specific default encoding to EncodingInfo object.

        Args:
            tensor_name: Name of the tensor
            encoding: Version-specific encoding dictionary
            tensor_type: TensorType enum value

        Returns:
            EncodingInfo object in unified format
        """
        pass

    def get_v0_6_1_tensor_encoding(self, tensor_name):
        def get_per_channel_property(encoding_info, property, channel_index):
            property_value = getattr(encoding_info, property)
            if isinstance(property_value, list):
                return property_value[channel_index]
            else:
                return property_value

        def get_is_symmetric(encoding_info):
            is_symmetric = encoding_info.is_symmetric
            if isinstance(is_symmetric, str):
                if is_symmetric.lower() == "true":
                    return "True"
                elif is_symmetric.lower() == "false":
                    return "False"
                else:
                    raise ValueError(f"Unrecognized value, {is_symmetric}, for is_symmetric.")
            return is_symmetric

        tensor_encoding = list()

        encoding_info = self.get_encoding_info(tensor_name)

        per_channel_properties = ['scale', 'zero_point', 'min', 'max']
        num_channels = max([len(getattr(encoding_info, prop)) if isinstance(getattr(encoding_info, prop), list) else 1 for prop in per_channel_properties])

        for i in range(num_channels):
            encoding_dict = dict()

            if encoding_info.bitwidth is not None:
                encoding_dict['bitwidth'] = encoding_info.bitwidth
            if encoding_info.scale is not None:
                encoding_dict['scale'] = get_per_channel_property(encoding_info, 'scale', i)
            if encoding_info.is_symmetric is not None:
                encoding_dict['is_symmetric'] = get_is_symmetric(encoding_info)
            if encoding_info.dtype is not None:
                encoding_dict['dtype'] = encoding_info.dtype.value.lower()
            if encoding_info.block_size is not None:
                encoding_dict['block_size'] = encoding_info.get_block_size()
            if encoding_info.per_block_int_scale is not None:
                encoding_dict['per_block_int_scale'] = encoding_info.per_block_int_scale
            if encoding_info.zero_point is not None:
                encoding_dict['offset'] = -get_per_channel_property(encoding_info, 'zero_point', i)
            if encoding_info.encoding_type is not None:
                encoding_dict['enc_type'] = encoding_info.encoding_type.value
            if encoding_info.compressed_bw is not None:
                encoding_dict['compressed_bw'] = encoding_info.compressed_bw
            if encoding_info.min is not None:
                encoding_dict['min'] = get_per_channel_property(encoding_info, 'min', i)
            if encoding_info.max is not None:
                encoding_dict['max'] = get_per_channel_property(encoding_info, 'max', i)

            tensor_encoding.append(encoding_dict)
        return tensor_encoding


    def get_v0_6_1_encodings(self) -> Dict[str, Any]:
        """
        Convert unified encodings to v0.6.1 format.

        Returns:
            Dictionary in v0.6.1 encoding format with param_encodings and activation_encodings

        Raises:
            ValueError: If this handler was initialized with v2.0.0 encodings (tensor_type unknown)
        """
        # Check if this handler was initialized with v2.0.0 encodings
        if self.get_version() == EncodingVersion.ENCODING_VERSION_V_2_0_0:
            raise ValueError(
                "Cannot convert v2.0.0 encodings to v0.6.1 format: tensor_type is unknown. "
                "v0.6.1 requires distinction between param and activation tensors."
            )

        param_encodings = dict()
        activation_encodings = dict()

        for tensor_name, encoding_info in self.encodings.items():
            tensor_encoding = self.get_v0_6_1_tensor_encoding(tensor_name)

            # Add to appropriate category
            if encoding_info.tensor_type == TensorType.PARAM:
                param_encodings[tensor_name] = tensor_encoding
            elif encoding_info.tensor_type == TensorType.ACTIVATION:
                activation_encodings[tensor_name] = tensor_encoding
            else:
                raise ValueError(
                    f"tensor_type of EncodingInfo for {tensor_name} is not {TensorType.PARAM} "
                    f"or {TensorType.ACTIVATION} and cannot be converted to 1.0.0 version encoding."
                )

        return {
            "version": "0.6.1",
            "param_encodings": param_encodings,
            "activation_encodings": activation_encodings,
            "excluded_layers": [],
            "quantizer_args": {}
        }

    def get_v1_0_0_tensor_encoding(self, tensor_name):
        encoding_info = self.get_encoding_info(tensor_name)
        encoding_dict = {
                "name": encoding_info.name,
            }
        if encoding_info.bitwidth is not None:
            encoding_dict['bw'] = encoding_info.bitwidth
        if encoding_info.scale is not None:
            encoding_dict['scale'] = encoding_info.scale
        if encoding_info.is_symmetric is not None:
            encoding_dict['is_sym'] = encoding_info.is_symmetric
        if encoding_info.dtype is not None:
            encoding_dict['dtype'] = encoding_info.dtype.value.upper()
        if encoding_info.block_size is not None:
            encoding_dict['block_size'] = encoding_info.get_block_size()
        if encoding_info.per_block_int_scale is not None:
            encoding_dict['per_block_int_scale'] = encoding_info.per_block_int_scale
        if encoding_info.zero_point is not None:
            if isinstance(encoding_info.zero_point, list):
                encoding_dict['offset'] = [-val for val in encoding_info.zero_point]
            else:
                encoding_dict['offset'] = -encoding_info.zero_point

        # Preserve zero_point_shift if present
        if hasattr(encoding_info, 'zero_point_shift') and encoding_info.zero_point_shift is not None:
            encoding_dict['zero_point_shift'] = encoding_info.zero_point_shift

        if encoding_info.encoding_type is not None:
            encoding_dict['enc_type'] = encoding_info.encoding_type.value
        if encoding_info.compressed_bw is not None:
            encoding_dict['compressed_bw'] = encoding_info.compressed_bw
        if encoding_info.min is not None:
            encoding_dict['min'] = encoding_info.min
        if encoding_info.max is not None:
            encoding_dict['max'] = encoding_info.max
        return encoding_dict


    def get_v1_0_0_encodings(self) -> Dict[str, Any]:
        """
        Convert unified encodings to v1.0.0 format.

        Returns:
            Dictionary in v1.0.0 encoding format with param_encodings and activation_encodings as arrays

        Raises:
            ValueError: If this handler was initialized with v2.0.0 encodings (tensor_type unknown)
        """
        # Check if this handler was initialized with v2.0.0 encodings
        if self.get_version() == EncodingVersion.ENCODING_VERSION_V_2_0_0:
            raise ValueError(
                "Cannot convert v2.0.0 encodings to v1.0.0 format: tensor_type is unknown. "
                "v1.0.0 requires distinction between param and activation tensors."
            )

        param_encodings = []
        activation_encodings = []

        for tensor_name, encoding_info in self.encodings.items():
            encoding_dict = self.get_v1_0_0_tensor_encoding(tensor_name)

            # Add to appropriate category
            if encoding_info.tensor_type == TensorType.PARAM:
                param_encodings.append(encoding_dict)
            elif encoding_info.tensor_type == TensorType.ACTIVATION:
                activation_encodings.append(encoding_dict)
            else:
                raise ValueError(
                    f"tensor_type of EncodingInfo for {tensor_name} is not {TensorType.PARAM} "
                    f"or {TensorType.ACTIVATION} and cannot be converted to 1.0.0 version encoding."
                )

        return {
            "version": "1.0.0",
            "param_encodings": param_encodings,
            "activation_encodings": activation_encodings,
            "excluded_layers": [],
            "quantizer_args": {}
        }

    def get_v2_0_0_tensor_encoding(self, tensor_name):
        def get_attribute(encoding_info, attribute):
            value = getattr(encoding_info, attribute)
            # if the zero point is per-tensor value, extract the single value
            if isinstance(value, list) and len(value) == 1 and not isinstance(value[0], list):
                return value[0]
            else:
                return value

        def set_scales(encoding_info, encoding_dict):
            if encoding_info.per_channel_float_scale is not None:
                encoding_dict['per_channel_float_scale'] = encoding_info.per_channel_float_scale
            elif encoding_info.encoding_type == EncodingType.LPBQ:
                encoding_dict['per_channel_float_scale'] = get_attribute(encoding_info, 'scale')
            elif encoding_info.scale is not None:
                 encoding_dict['y_scale'] = get_attribute(encoding_info, 'scale')
            else:
                raise ValueError(f"Missing scale encoding information for {encoding_info.name}")
            return encoding_dict

        encoding_info = self.get_encoding_info(tensor_name)
        encoding_dict = {
            "name": encoding_info.name
        }

        if encoding_info.dtype is not None and encoding_info.bitwidth is not None:
            encoding_dict['output_dtype'] = encoding_info.dtype.value.lower() + str(encoding_info.bitwidth)
        if encoding_info.zero_point is not None and not np.all(np.array(encoding_info.zero_point) == 0):
            encoding_dict['y_zero_point'] = get_attribute(encoding_info, 'zero_point')
        if encoding_info.axis is not None:
            encoding_dict['axis'] = encoding_info.axis
        if encoding_info.block_size is not None:
            encoding_dict['block_size'] = encoding_info.get_block_size()
        if encoding_info.per_block_int_scale is not None:
            encoding_dict['per_block_int_scale'] = encoding_info.per_block_int_scale

        encoding_dict = set_scales(encoding_info, encoding_dict)
        return encoding_dict


    def get_v2_0_0_encodings(self) -> Dict[str, Any]:
        """
        Convert unified encodings to v2.0.0 format.

        Returns:
            Dictionary in v2.0.0 encoding format with unified encodings array

        Note:
            This conversion works regardless of the original version since v2.0.0 doesn't distinguish
            between param and activation tensors.
        """
        encodings_list = []

        for tensor_name in self.encodings:
            encoding_dict = self.get_v2_0_0_tensor_encoding(tensor_name)
            encodings_list.append(encoding_dict)

        return {
            "version": "2.0.0",
            "encodings": encodings_list
        }
