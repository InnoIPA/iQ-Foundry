# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Dict, List, Any
from .encoding_handler import EncodingHandler, EncodingInfo, EncodingVersion, TensorType, DType


class EncodingHandlerV_0_6_1(EncodingHandler):
    """
    Encoding handler for version 0.6.1.

    Version 0.6.1 uses a dictionary-based structure where:
    - param_encodings and activation_encodings are dictionaries
    - Tensor names are keys in these dictionaries
    - Values can be single dictionaries or lists of dictionaries for per-channel quantization
    """

    def get_version(self) -> EncodingVersion:
        """Get the encoding version this handler supports."""
        return EncodingVersion.ENCODING_VERSION_V_0_6_1

    def _convert_to_unified_encoding(self, encodings: Dict[str, Any]) -> Dict[str, EncodingInfo]:
        """
        Convert version 0.6.1 encoding format to unified schema.

        Args:
            encodings: Version 0.6.1 encoding dictionary

        Returns:
            Dictionary mapping tensor names to EncodingInfo objects
        """
        unified_encodings = {}

        param_encodings = encodings.get("param_encodings", {})
        for tensor_name, param_enc in param_encodings.items():
            encoding_info = self.convert_encoding_to_info(tensor_name, param_enc, TensorType.PARAM)
            unified_encodings[tensor_name] = encoding_info

        activation_encodings = encodings.get("activation_encodings", {})
        for tensor_name, act_enc in activation_encodings.items():
            encoding_info = self.convert_encoding_to_info(tensor_name, act_enc, TensorType.ACTIVATION)
            unified_encodings[tensor_name] = encoding_info

        return unified_encodings

    def to_dict(self) -> Dict[str, Any]:
        """
        Get encodings in v0.6.1 native format.

        Returns:
            Dictionary in v0.6.1 encoding format
        """
        return self.get_v0_6_1_encodings()

    def get_tensor_encoding(self, tensor_name) -> List[Dict]:
        return self.get_v0_6_1_tensor_encoding(tensor_name)

    def convert_encoding_to_info(self, tensor_name: str, encoding: Any, tensor_type: TensorType) -> EncodingInfo:
        """
        Convert v0.6.1 encoding to EncodingInfo object.

        Args:
            tensor_name: Name of the tensor
            encoding: List of encoding dicts
            tensor_type: TensorType enum value

        Returns:
            EncodingInfo object in unified format
        """
        def get_offset(encoding):
            offset = get_per_channel_property(encoding, 'offset')
            if offset is not None:
                offset = [-val for val in offset]
            return offset

        def get_per_channel_property(encoding, property):
            val = None
            if property in encoding[0]:
                val = [enc[property] for enc in encoding]
            return val

        def get_per_tensor_property(encoding, property):
            val = None
            if property in encoding[0]:
                val = encoding[0][property]

            if isinstance(val, (str, bool)):
                val = str(val).lower()
            return val

        def get_dtype(encoding):
            dtype_str = get_per_tensor_property(encoding, 'dtype')
            if dtype_str is not None:
                for enum_val in DType:
                    if enum_val.value == dtype_str.upper():
                        return enum_val
                raise ValueError(f"dtype {dtype_str} for {encoding['name']} is not recognized.")

        def get_block_size(encoding):
            block_size = get_per_tensor_property(encoding, 'block_size')
            if block_size is not None:
                if not isinstance(block_size, list):
                    block_size = [block_size]
            return block_size

        return EncodingInfo(
            name=tensor_name,
            bitwidth=get_per_tensor_property(encoding, 'bitwidth'),
            scale=get_per_channel_property(encoding, 'scale'),
            is_symmetric=get_per_tensor_property(encoding, 'is_symmetric'),
            dtype=get_dtype(encoding),
            block_size=get_block_size(encoding),
            zero_point=get_offset(encoding),
            encoding_type=get_per_tensor_property(encoding, 'enc_type'),
            compressed_bw=get_per_tensor_property(encoding, 'compressed_bw'),
            min=get_per_channel_property(encoding, 'min'),
            max=get_per_channel_property(encoding, 'max'),
            tensor_type=tensor_type
        )
