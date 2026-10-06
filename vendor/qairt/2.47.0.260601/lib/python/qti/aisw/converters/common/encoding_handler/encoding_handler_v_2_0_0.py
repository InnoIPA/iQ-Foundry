# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Dict, List, Any
from .encoding_handler import EncodingHandler, EncodingInfo, EncodingVersion, TensorType, DType
import numpy as np


class EncodingHandlerV_2_0_0(EncodingHandler):
    """
    Encoding handler for version 2.0.0.

    Version 2.0.0 uses a unified structure where:
    - All encodings are in a single "encodings" array
    - Uses "output_dtype" and "y_scale"/"y_zero_point" fields
    - Does not distinguish between param and activation tensors
    """

    def get_version(self) -> EncodingVersion:
        """Get the encoding version this handler supports."""
        return EncodingVersion.ENCODING_VERSION_V_2_0_0

    def _convert_to_unified_encoding(self, encodings: Dict[str, Any]) -> Dict[str, EncodingInfo]:
        """
        Convert version 2.0.0 encoding format to unified schema.

        Args:
            encodings: Version 2.0.0 encoding dictionary

        Returns:
            Dictionary mapping tensor names to EncodingInfo objects
        """
        unified_encodings = {}

        for enc in encodings.get("encodings", []):
            tensor_name = enc["name"]
            encoding_info = self.convert_encoding_to_info(tensor_name, enc, TensorType.UNKNOWN)
            unified_encodings[tensor_name] = encoding_info

        return unified_encodings

    def to_dict(self) -> Dict[str, Any]:
        """
        Get encodings in v2.0.0 native format.

        Returns:
            Dictionary in v2.0.0 encoding format
        """
        return self.get_v2_0_0_encodings()

    def get_tensor_encoding(self, tensor_name) -> Dict:
        return self.get_v2_0_0_tensor_encoding(tensor_name)

    def convert_encoding_to_info(self, tensor_name: str, encoding: Any, tensor_type: TensorType) -> EncodingInfo:
        """
        Convert v2.0.0 encoding to EncodingInfo object.

        Args:
            tensor_name: Name of the tensor
            encoding: encoding dictionary
            tensor_type: TensorType enum value

        Returns:
            EncodingInfo object in unified format
        """
        def get_bitwidth_and_dtype(output_dtype):
            for enum_val in DType:
                if output_dtype.upper().startswith(enum_val.value):
                    bitwidth = int(''.join(filter(str.isdigit, output_dtype)))
                    dtype = enum_val
                    return bitwidth, dtype
            raise ValueError(f"output_dtype {output_dtype} for {encoding['name']} is not recognized.")

        def get_scale(encoding):
            scale = encoding.get('y_scale', None)
            if scale is not None and not isinstance(scale, list):
                scale = [scale]
            return scale

        def get_zero_point(encoding):
            def get_default_zero_point(encoding):
                scale = get_scale(encoding)
                if scale:
                    shape = np.array(scale).shape
                    zero_point = np.zeros(shape).tolist()
                else:
                    zero_point = [0]
                return zero_point

            zero_point = encoding.get('y_zero_point', get_default_zero_point(encoding))
            if zero_point is not None and not isinstance(zero_point, list):
                zero_point = [zero_point]
            return zero_point

        def get_block_size(encoding):
            block_size = encoding.get('block_size', None)
            if block_size is not None:
                if not isinstance(block_size, list):
                    block_size = [block_size]
            return block_size

        bitwidth, dtype = get_bitwidth_and_dtype(encoding.get('output_dtype'))

        return EncodingInfo(
            name=tensor_name,
            bitwidth=bitwidth,
            dtype=dtype,
            scale=get_scale(encoding),
            zero_point=get_zero_point(encoding),
            tensor_type=TensorType.UNKNOWN,
            axis=encoding.get('axis', None),
            block_size=get_block_size(encoding),
            per_channel_float_scale=encoding.get('per_channel_float_scale', None),
            per_block_int_scale=encoding.get('per_block_int_scale', None)
        )
