# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Dict, List, Any
from .encoding_handler import EncodingHandler, EncodingInfo, EncodingVersion, EncodingType, TensorType, DType


class EncodingHandlerV_1_0_0(EncodingHandler):
    """
    Encoding handler for version 1.0.0.

    Version 1.0.0 uses a list-based structure where:
    - param_encodings and activation_encodings are lists of dictionaries
    - Each dictionary has a "name" field identifying the tensor
    - Uses "bw" instead of "bitwidth" and "offset" instead of "zero_point"
    """

    def get_version(self) -> EncodingVersion:
        """Get the encoding version this handler supports."""
        return EncodingVersion.ENCODING_VERSION_V_1_0_0

    def _convert_to_unified_encoding(self, encodings: Dict[str, Any]) -> Dict[str, EncodingInfo]:
        """
        Convert version 1.0.0 encoding format to unified schema.

        Args:
            encodings: Version 1.0.0 encoding dictionary

        Returns:
            Dictionary mapping tensor names to EncodingInfo objects
        """

        unified_encodings = {}

        # Convert param_encodings - v1.0.0 uses list of dicts with "name" field
        for param_enc in encodings.get("param_encodings", []):
            tensor_name = param_enc["name"]
            encoding_info = self.convert_encoding_to_info(tensor_name, param_enc, TensorType.PARAM)
            unified_encodings[tensor_name] = encoding_info

        # Convert activation_encodings - v1.0.0 uses list of dicts with "name" field
        for act_enc in encodings.get("activation_encodings", []):
            tensor_name = act_enc["name"]
            encoding_info = self.convert_encoding_to_info(tensor_name, act_enc, TensorType.ACTIVATION)
            unified_encodings[tensor_name] = encoding_info

        return unified_encodings

    def to_dict(self) -> Dict[str, Any]:
        """
        Get encodings in v1.0.0 native format.

        Returns:
            Dictionary in v1.0.0 encoding format
        """
        return self.get_v1_0_0_encodings()

    def get_tensor_encoding(self, tensor_name) -> Dict:
        return self.get_v1_0_0_tensor_encoding(tensor_name)

    def convert_encoding_to_info(self, tensor_name: str, encoding: Any, tensor_type: TensorType) -> EncodingInfo:
        """
        Convert v1.0.0 encoding to EncodingInfo object.

        Args:
            tensor_name: Name of the tensor
            encoding: Dict from get_default_*_encoding methods
            tensor_type: TensorType enum value

        Returns:
            EncodingInfo object in unified format
        """
        def get_scale(encoding):
            scale = encoding.get('scale', None)
            if scale is not None and not isinstance(scale, list):
                scale = [scale]
            return scale

        def get_dtype(encoding):
            dtype = encoding.get('dtype', None)
            if dtype is not None:
                for enum_val in DType:
                    if enum_val.value == dtype.upper():
                        return enum_val
                raise ValueError(f"dtype {dtype} for {encoding['name']} is not recognized.")

        def get_zero_point(encoding):
            zero_point = encoding.get('offset', None)
            if zero_point is not None:
                if not isinstance(zero_point, list):
                    zero_point = [-zero_point]
                else:
                    zero_point = [-val for val in zero_point]
            return zero_point

        def get_enc_type(encoding):
            enc_type = encoding.get('enc_type', None)
            if enc_type is not None:
                for enum_val in EncodingType:
                    if enum_val.value == enc_type.upper():
                        return enum_val
                raise ValueError(f"enc_type {enc_type} for {encoding['name']} is not recognized.")
            return enc_type

        def get_block_size(encoding):
            block_size = encoding.get('block_size', None)
            if block_size is not None:
                if not isinstance(block_size, list):
                    block_size = [block_size]
            return block_size

        encoding_params = {
            'name': tensor_name,
            'bitwidth': encoding.get("bw", None),
            'scale': get_scale(encoding),
            'is_symmetric': encoding.get('is_sym', None),
            'dtype': get_dtype(encoding),
            'block_size': get_block_size(encoding),
            'per_block_int_scale': encoding.get('per_block_int_scale', None),
            'zero_point': get_zero_point(encoding),
            'encoding_type': get_enc_type(encoding),
            'compressed_bw': encoding.get('compressed_bw', None),
            'min': encoding.get('min', None),
            'max': encoding.get('max', None),
            'zero_point_shift': encoding.get('zero_point_shift', None),
            'tensor_type': tensor_type
        }

        return EncodingInfo(**encoding_params)

