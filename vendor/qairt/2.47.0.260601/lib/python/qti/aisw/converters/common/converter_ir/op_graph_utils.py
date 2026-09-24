# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numbers
import numpy as np

def _apply_scaling_recursively(value, scale_factor, tensor_name):
    """
    Applies a scaling factor based on whether the value is a single number,
    a list of numbers, or a list nested with lists/numbers (recursive).
    If the value type is unexpected, it raises a ValueError.
    """
    if isinstance(value, list):
        scaled_list = []
        for item in value:
            # Recursively call for nested lists
            scaled_list.append(_apply_scaling_recursively(item, scale_factor, tensor_name))
        return scaled_list
    elif isinstance(value, numbers.Number):
        return value * scale_factor
    else:
        raise ValueError(f"Unexpected type '{type(value)}' for encoding value in tensor '{tensor_name}'. Value: {value}.")

def apply_scaling_to_encodings(encodings, scale_factor, tensor_name):
    """
    Applies scaling to the 'max', 'min', 'scale', 'offset', and other encoding parameters
    within the provided encodings dictionary. Handles min/max swap after scaling.
    """
    # Apply scaling to 'max', 'min', 'scale', 'offset'
    if 'max' in encodings:
        encodings['max'] = _apply_scaling_recursively(encodings['max'], scale_factor, tensor_name)
    if 'min' in encodings:
        encodings['min'] = _apply_scaling_recursively(encodings['min'], scale_factor, tensor_name)

    # Handle potential min/max swap *after* scaling
    if 'max' in encodings and 'min' in encodings:
        max_val = encodings['max']
        min_val = encodings['min']
        if isinstance(max_val, numbers.Number) and isinstance(min_val, numbers.Number):
            if max_val < min_val:
                encodings['max'], encodings['min'] = min_val, max_val
        elif isinstance(max_val, list) and isinstance(min_val, list):
            if len(max_val) != len(min_val):
                raise ValueError(f"Encoding of tensor {tensor_name} 'max' and 'min' lists have different lengths ({len(max_val)} vs {len(min_val)}). ")
            else:
                new_max_list = list(max_val)
                new_min_list = list(min_val)
                for i in range(len(new_max_list)):
                    if isinstance(new_max_list[i], numbers.Number) and isinstance(new_min_list[i], numbers.Number):
                        if new_max_list[i] < new_min_list[i]:
                            new_max_list[i], new_min_list[i] = new_min_list[i], new_max_list[i]
                    else:
                        raise ValueError(f"Non-numeric items found at index {i} in 'max' or 'min' lists for encoding of tensor {tensor_name}.")
                encodings['max'] = new_max_list
                encodings['min'] = new_min_list
        else:
            raise ValueError(f"Mismatched types for 'max' and 'min' in encoding of tensor '{tensor_name}'. "
                            f"Expected both numbers or both lists for comparison, but got max type: {type(max_val)}, min type: {type(min_val)}.")
    if 'scale' in encodings:
        encodings['scale'] = _apply_scaling_recursively(encodings['scale'], scale_factor, tensor_name)
    # Encodings 2.0.0.
    if 'y_scale' in encodings:
        encodings['y_scale'] = _apply_scaling_recursively(encodings['y_scale'], scale_factor, tensor_name)
    # Encodings 2.0.0. Low power blockwise quantization (LPBQ)
    if 'per_channel_float_scale' in encodings:
        encodings['per_channel_float_scale'] = _apply_scaling_recursively(encodings['per_channel_float_scale'], scale_factor, tensor_name)
    if 'per_block_int_scale' in encodings:
        encodings['per_block_int_scale'] = _apply_scaling_recursively(encodings['per_block_int_scale'], scale_factor, tensor_name)

def apply_scaling_to_constant(graph, constant_op, scale_factor):
    """Apply scaling factor to constant tensor and update encodings."""
    # 1. Validate scale_factor
    if not np.isfinite(scale_factor):
        raise ValueError(f"Invalid scale_factor {scale_factor}: must be a finite number")
    if scale_factor == 0:
        raise ValueError("scale_factor cannot be zero, as it would zero out all values.")

    # 2. Scale the tensor itself
    constant_op.tensor = constant_op.tensor * scale_factor

    # 3. Update associated encodings if they exist
    override_encodings_list = graph.get_overridden_encoding(constant_op.name)
    if override_encodings_list:
        encodings = override_encodings_list[0]
        apply_scaling_to_encodings(encodings, scale_factor, constant_op.name)