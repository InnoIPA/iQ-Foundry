# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from .base import CalibrationListDataset, IndexableDataset, RawInputListDataset
from .legacy import (
    COCO2017Dataset,
    ImagenetDataset,
    LegacyDataset,
    WIDERFaceDataset,
    WMT20Dataset,
)
from .squad import SQUADDataset
from .wikitext import WikiTextDataset
