# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define base class for layout manager."""

from abc import ABC, abstractmethod

from qti.aisw.converters.common.passes.layout_transform.layout_inferer import (
    LayoutInferers,
)
from qti.aisw.converters.common.passes.layout_transform.layout_recorder import (
    LayoutRecorder,
)
from qti.aisw.converters.common.utils import converter_utils


def _parse_custom_io(user_custom_io):
    """Parse user-specified custom IO configuarion."""
    custom_layout_table = {}
    for entry in user_custom_io:
        buffer_name = entry['IOName']

        if 'Layout' in entry:
            model_layout = entry['Layout']['Model']
            custom_layout = entry['Layout']['Custom']

            # Validate model layout can be transformed to custom one.
            converter_utils.log_assert(
                set(model_layout) == set(custom_layout),
                f'Unable to transform model layout {model_layout} to custom one {custom_layout}.'
            )
            custom_layout_table[buffer_name] = {
                'Source': model_layout, 'Desired': custom_layout
            }

    return custom_layout_table


class LayoutManagerBase(ABC):
    """Layout manager base for transforming layout.

    Layout manager is the entry point to perform layout transform. It organizes all the components,
    including layout inferer, layout recorder, and layout mutator to transform layouts. Since
    the manager is IR-specific, inherited classes must implement abstracted methods for detailed
    transformation process.

    Attributes:
        layout_inferer_bank: An instance of LayoutInfererBank.
        layout_mutator: An instance of LayoutMutatorBase.
        layout_recorder: An instance of LayoutRecorder.

    Methods:
        get_layout_inferer: Get layout inferer by op type.
        apply_transform: Transform layout.
    """

    def __init__(
        self,
        src_layout_table,
        desired_layout_table,
        preferred_perm_seq_table,
        layout_mutator,
        user_custom_io=None
    ):
        """Initialize manager.

        Args:
            src_layout_table: Refer to layout recorder.
            desired_layout_table: Refer to layout recorder.
            preferred_perm_seq_table: Refer to layout recorder.
            layout_mutator: Refer to attributes.
            user_custom_io: A list of dict specifying custom IO information from user. Defaults to
                None indicating no custom IO specified.
        """
        self.layout_inferer_bank = LayoutInferers
        self.layout_recorder = LayoutRecorder(
            src_layout_table,
            desired_layout_table,
            preferred_perm_seq_table,
            _parse_custom_io(user_custom_io) if user_custom_io is not None else {},
        )
        self.layout_mutator = layout_mutator

    def get_layout_inferer(self, op_type):
        """Get layout inferer by op type.

        Args:
            op_type: A str specifying target op type for layout inferer.

        Returns:
            A layout inferer for target op type.
        """
        return self.layout_inferer_bank.get_layout_inferer(op_type)

    @abstractmethod
    def apply_transform(self):
        """Transform layout."""
        raise NotImplementedError(
            "apply_transform for {} not implemented ".format(
                str(self.__class__.__name__)
            )
        )
