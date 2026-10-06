# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define base class for layout inferer.

Classes:
    LayoutInfererBase: A layout inferer to infer target layouts and attributes.
    LayoutInfererBank: An inferer bank registered with layout inferers for various ops.
"""


from qti.aisw.converters.common.passes.layout_transform.layout_recorder import (
    LayoutRecorder,
)

class LayoutInfererBase:
    """A layout inferer to infer target layouts and attributes.

    Note that layout inferer is designed to be independent to any frontend. It simply takes source
    layout and layout memo into consideration and infers target layout according to each op type.

    This base class provides basic call sequence to infer target layouts and update attributes, and
    inherited classes must implement each abstracted method for detailed algorithms. For inferring
    target input permute sequences, reducing the number of inserted transpose is the primiary goal,
    and developer must be aware of it during designing the algorithm. Once target input permute
    sequences are determined, corresponding target output permute sequences and attributes may be
    inferred accordingly.

    Methods:
        infer_target_layouts_and_attrs: Infer input/output permute sequences and update attributes.
    """

    def infer_target_input_perm_seqs(
        self, input_buffer_names: list, input_shapes: list, layout_recorder: LayoutRecorder
    ):
        """Infer target input permute sequences."""
        raise NotImplementedError(
            f"infer_target_input_perm_seqs for {str(self.__class__.__name__)} not implemented"
        )

    def infer_target_output_perm_seqs(self, target_input_perm_seqs: list, output_shapes: list):
        """Infer target output permute sequences."""
        raise NotImplementedError(
            f"infer_target_output_perm_seqs for {str(self.__class__.__name__)} not implemented"
        )

    def update_attr_with_target_input_perm_seqs(self, target_input_perm_seqs: list, attr):
        """Update attribute."""
        raise NotImplementedError(
            f"update_attr_with_target_input_perm_seqs for {str(self.__class__.__name__)} not implemented"
        )

    def infer_target_layouts_and_attrs(
        self,
        input_buffer_names: list,
        input_shapes: list,
        output_shapes: list,
        src_attrs,
        layout_recorder: LayoutRecorder,
    ):
        """Infer input/output permute sequences and update attributes.

        Args:
            input_buffer_names: A list of strs specifying input names.
            input_shapes: A list of list of ints specifying input shapes.
            output_shapes: A list of list of ints specifying input shapes.
            src_attrs: A dict specifying source attributes.
            layout_recorder: A LayoutRecorder instance.

        Returns:
            target_input_perm_seqs: A list of tuple of ints specifying permute sequences for each
                input.
            target_output_perm_seqs: A list of tuple of ints specifying permute sequences for each
                output.
            new_attrs: A dict specifying updated attributes after layout transform.
        """
        target_input_perm_seqs = self.infer_target_input_perm_seqs(
            input_buffer_names, input_shapes, layout_recorder
        )
        new_attrs = self.update_attr_with_target_input_perm_seqs(
            target_input_perm_seqs, src_attrs
        )
        target_output_perm_seqs = self.infer_target_output_perm_seqs(
            target_input_perm_seqs, output_shapes
        )
        return target_input_perm_seqs, target_output_perm_seqs, new_attrs


class LayoutInfererBank:
    """An inferer bank registered with layout inferers for various ops.

    Attributes
        layout_ifnerers: A dict collecting registered layout inferers, mapping each op type to
            corresponding layout inferer instance.

    Methods:
        get_layout_inferer: Get layout inferer by given op type.
        register_layout_inferer: Register layout inferer by given op type.
    """

    def __init__(self):
        """Initialize bank."""
        self.layout_inferers = {}

    def get_layout_inferer(self, op_type):
        """Get layout inferer by given op type.

        Args:
            op_type: A str specifying target op type.

        Returns:
            A corresponding layout inferer instance.

        Raises:
            KeyError: Given op type is not registered.
        """
        if op_type not in self.layout_inferers:
            raise KeyError(f"No layout inferer registered for op type: {op_type}")
        return self.layout_inferers[op_type]

    def register_layout_inferer(self, inferer, op_type):
        """Register layout inferer by given op type.

        Args:
            inferer: An instance of LayoutInfererBase.
            op_type: A str specifying registery key.

        Raises:
            KeyError: Given op type is already registered.
        """
        if op_type in self.layout_inferers:
            raise KeyError(
                f"A layout inferer is already registered for op type: {op_type}"
            )
        self.layout_inferers[op_type] = inferer
