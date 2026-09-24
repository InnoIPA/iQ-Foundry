# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
from pathlib import Path
from typing import Dict

import numpy as np
from pydantic import DirectoryPath, FilePath


def permute_tensor_data_axis_order(target_tensor: np.array, layout_info: dict) -> np.array:
    """This method permutes axis of given target tensor inplace based on given permute_order_to_src

    Args:
        target_tensor: Target output as numpy array
        layout_info: dictionary containing:
                    {
                        'dims': <target dimensions>,
                        'permute_order_to_src': <permute order>
                    }
    """
    # permute order will be a empty list when both source and target outputs are of same layout
    if layout_info["permute_order_to_src"] == []:
        return target_tensor

    # First reshape target tensor to it's original shape and then transpose to align with golden tensor layout
    target_tensor = np.reshape(target_tensor, tuple(layout_info["dims"]))
    target_tensor = np.transpose(target_tensor, layout_info["permute_order_to_src"])

    return target_tensor.flatten()


def get_tensor_paths(tensors_path: DirectoryPath) -> Dict[str, FilePath]:
    """This method returns a dict containing tensor paths where key is tensor name and value is tensor
    raw file path
    """
    tensors = {}
    for dir_path, sub_dirs, files in os.walk(tensors_path):
        for file in files:
            if file.endswith(".raw"):
                tensor_path = os.path.join(dir_path, file)
                # tensor name is part of it's path
                path = os.path.relpath(tensor_path, tensors_path)

                # remove .raw extension
                tensor_name = str(Path(path).with_suffix(""))
                tensors[tensor_name] = tensor_path
    return tensors
