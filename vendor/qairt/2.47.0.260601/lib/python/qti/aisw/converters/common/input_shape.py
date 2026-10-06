# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from dataclasses import dataclass, field

@dataclass
class InputShapeInfo:
    input_names: list = field(default_factory=list)
    input_dims: list = field(default_factory=list)
    input_dynamic_axes: dict = field(default_factory=dict)
    has_dynamic_shapes: bool = False

    def __bool__(self):
        return bool(self.input_names)

class InputShapeArgParser:
    def __init__(self, arg_input_shapes):
        self.input_shape_info = InputShapeInfo()
        self.dynamic_marker = "*"
        self._parse_shape_args(arg_input_shapes)

    def _parse_shape(self, input_name, input_shape):
        dim_list_str = "" # e.g. "1,224,224,3"
        dy_axes = set()
        dims = input_shape.split(',')
        for axis in range(len(dims)):
            dim = dims[axis]
            # check if this axis is dynamic
            find_dyn_marker = dim.find(self.dynamic_marker)
            if find_dyn_marker!=-1:
                # dynamic-marker has to be a suffix .e.g. 2,334*,4
                if find_dyn_marker!= len(dim)-1:
                    raise RuntimeError("Got incorrect input shape " + input_shape + " for input name " + input_name + ", dynamic shape marker " + self.dynamic_marker + " should be a suffix. E.g. [1,3,224*,224*]" )
                dy_axes.add(axis)
                dim = dim[:find_dyn_marker]
            try:
                dim_value = int(dim)
            except Exception as error:
                raise RuntimeError("Got incorrect input shape " + input_shape + " for input name " + input_name + ", shape can be integers or integers followed by " + self.dynamic_marker + " to indicate dynamic shape.") from None
            dim_list_str +=  str(dim_value) + ","

        # remove the trailing ","
        dim_list_str = dim_list_str[:-1]
        return dim_list_str, dy_axes


    def _parse_shape_args(self, arg_input_shapes):
        if arg_input_shapes is None:
            return

        for input_name, input_shape in arg_input_shapes:
            self.input_shape_info.input_names.append(input_name)
            dim_list, dy_axes = self._parse_shape(input_name, input_shape)
            if dy_axes:
                self.input_shape_info.has_dynamic_shapes = True
            self.input_shape_info.input_dims.append(dim_list)
            self.input_shape_info.input_dynamic_axes[input_name] = dy_axes