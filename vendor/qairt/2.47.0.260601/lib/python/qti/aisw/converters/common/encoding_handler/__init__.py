# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
Encoding Handler Package

This package provides a unified interface for handling different encoding versions
(0.6.1, 1.0.0, 2.0.0) through a strategy pattern implementation.

Usage:
    from qti.aisw.converters.common.encoding_handler import EncodingHandlerFactory

    handler = EncodingHandlerFactory.create_handler(encodings_dict)
    unified_encodings = handler.encodings
"""

from .encoding_handler import EncodingHandler, EncodingInfo
from .encoding_handler_factory import EncodingHandlerFactory
from .encoding_handler_v_0_6_1 import EncodingHandlerV_0_6_1
from .encoding_handler_v_1_0_0 import EncodingHandlerV_1_0_0
from .encoding_handler_v_2_0_0 import EncodingHandlerV_2_0_0

__all__ = [
    'EncodingHandler',
    'EncodingInfo',
    'EncodingHandlerFactory',
    'EncodingHandlerV_0_6_1',
    'EncodingHandlerV_1_0_0',
    'EncodingHandlerV_2_0_0'
]
