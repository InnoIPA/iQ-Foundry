# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Collections of layout inferer modules."""

import inspect
import itertools

from . import layout_inferer_base
from . import agnostic_layout_inferers
from . import dummy_layout_inferers
from . import heavily_layout_inferers
from . import untrackable_layout_inferers

# Instantiate here for singleton usage.
LayoutInferers = layout_inferer_base.LayoutInfererBank()


# Automatically register layout inferers into bank during import of this module.
for attr in itertools.chain(
    agnostic_layout_inferers.__dict__.values(),
    dummy_layout_inferers.__dict__.values(),
    heavily_layout_inferers.__dict__.values(),
    untrackable_layout_inferers.__dict__.values(),
):
    if not inspect.isclass(attr) or not hasattr(attr, "op_type"):
        continue

    inferer = attr()
    if isinstance(inferer, layout_inferer_base.LayoutInfererBase):
        LayoutInferers.register_layout_inferer(inferer, inferer.op_type)
