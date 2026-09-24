# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
from enum import Enum


class Engine(Enum):
    SNPE = "SNPE"
    QNN = "QNN"
    QAIRT = "QAIRT"
