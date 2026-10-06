# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


class Operator(object):
    def __init__(self, op_type, attributes=None, input_shapes=None, output_shapes=None):
        """
        Initialize a Operator object representing an ONNX Operator.

        Args:
            op_type (str): The type of the ONNX Operator (e.g., 'Conv', 'Relu', etc.)
            attributes (dict, optional): Dictionary of Operator attributes. Defaults to None.
            input_shapes (list, optional): List of input tensor shapes. Each shape is a list/tuple of dimensions.
            output_shapes (list, optional): List of output tensor shapes. Each shape is a list/tuple of dimensions.
        """
        if not isinstance(op_type, str) or not op_type:
            raise ValueError("op_type must be a non-empty string")

        self.op_type = op_type

        # Validate attributes
        if attributes is not None:
            if not isinstance(attributes, dict):
                raise ValueError("attributes must be a dictionary")
            # Validate attribute values (basic type checking)
            for key, value in attributes.items():
                if not isinstance(key, str):
                    raise ValueError(f"Attribute key must be a string, got {type(key)}")

        self.attributes = attributes if attributes is not None else {}

        # Validate and set input shapes
        if input_shapes is not None:
            if not isinstance(input_shapes, list):
                raise ValueError("input_shapes must be a list")
            for shape in input_shapes:
                if not isinstance(shape, (list, tuple)):
                    raise ValueError("Each input shape must be a list or tuple")
        self.input_shapes = input_shapes if input_shapes is not None else []

        # Validate and set output shapes
        if output_shapes is not None:
            if not isinstance(output_shapes, list):
                raise ValueError("output_shapes must be a list")
            for shape in output_shapes:
                if not isinstance(shape, (list, tuple)):
                    raise ValueError("Each output shape must be a list or tuple")
        self.output_shapes = output_shapes if output_shapes is not None else []

    def add_attribute(self, name, value):
        """Add or update an Operator attribute."""
        self.attributes[name] = value

    def set_input_shapes(self, shapes):
        """Set the input shapes for the Operator."""
        self.input_shapes = shapes

    def set_output_shapes(self, shapes):
        """Set the output shapes for the Operator."""
        self.output_shapes = shapes

    def get_attribute(self, name, default=None):
        """Get an attribute value by name."""
        return self.attributes.get(name, default)

    def __str__(self):
        """String representation of the Operator."""
        return f"Operator(type={self.op_type}, attributes={list(self.attributes.keys())}, " \
               f"input_shapes={self.input_shapes}, output_shapes={self.output_shapes})"
