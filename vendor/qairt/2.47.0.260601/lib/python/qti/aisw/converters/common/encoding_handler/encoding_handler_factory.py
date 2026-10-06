# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Dict, Any
import jsonschema
from .encoding_handler import EncodingHandler, EncodingVersion
from .encoding_handler_v_0_6_1 import EncodingHandlerV_0_6_1
from .encoding_handler_v_1_0_0 import EncodingHandlerV_1_0_0
from .encoding_handler_v_2_0_0 import EncodingHandlerV_2_0_0


class EncodingHandlerFactory:
    """
    Factory class for creating appropriate encoding handlers based on version.

    This factory eliminates the need for conditional version checks throughout
    the codebase by automatically selecting and instantiating the correct
    handler for a given encoding version.
    """

    _handlers = {
        EncodingVersion.ENCODING_VERSION_V_0_6_1: EncodingHandlerV_0_6_1,
        EncodingVersion.ENCODING_VERSION_V_1_0_0: EncodingHandlerV_1_0_0,
        EncodingVersion.ENCODING_VERSION_V_2_0_0: EncodingHandlerV_2_0_0
    }

    @classmethod
    def _validate_version(cls, encodings: Dict[str, Any]) -> None:
        """
        Validate that the encodings dictionary contains a supported version.

        Args:
            encodings: Encoding dictionary to validate

        Raises:
            ValueError: If the version is missing or not supported
        """
        # Create schema dynamically based on supported versions
        supported_version_strings = [version.value for version in cls._handlers.keys()]
        schema = {
            "type": "object",
            "properties": {
                "version": {
                    "type": "string",
                    "enum": supported_version_strings
                }
            },
            "required": ["version"],
            "additionalProperties": True
        }

        try:
            jsonschema.validate(encodings, schema)
        except jsonschema.ValidationError as e:
            if "version" not in encodings:
                raise ValueError("Encoding dictionary must contain a 'version' field")
            else:
                raise ValueError(
                    f"Unsupported encoding version: {encodings['version']}. "
                    f"Supported versions are: {supported_version_strings}"
                )

    @classmethod
    def create_handler(cls, encodings: Dict[str, Any]) -> EncodingHandler:
        """
        Create appropriate encoding handler based on the encoding version.

        Args:
            encodings: Encoding dictionary containing version information

        Returns:
            EncodingHandler instance for the detected version

        Raises:
            ValueError: If the encoding version is missing or not supported
        """
        cls._validate_version(encodings)
        version_string = encodings["version"]

        for enum_version, handler_class in cls._handlers.items():
            if enum_version.value == version_string:
                return handler_class(encodings)

        raise ValueError(f"No handler found for version: {version_string}")

    @classmethod
    def get_supported_versions(cls) -> list:
        """
        Get list of supported encoding versions.

        Returns:
            List of supported version strings
        """
        return [version.value for version in cls._handlers.keys()]
