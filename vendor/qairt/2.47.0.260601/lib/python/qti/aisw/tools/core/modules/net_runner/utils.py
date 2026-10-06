# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import re
from os import PathLike
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from qti.aisw.core.model_level_api.utils.subprocess_executor import get_name_input_pairs_from_input_list


InputListInput = Union[PathLike, str]
NamedTensorMapping = Dict[str, np.ndarray]
NetRunnerInputData = Union[
    InputListInput, np.ndarray, List[np.ndarray], NamedTensorMapping, List[NamedTensorMapping]
]


def is_required_tensor_data_provided(input_tensors: List[str], input_data: NetRunnerInputData) -> bool:
    """Checks if all required tensor data is provided in the input data.

    Args:
        input_tensors (List[str]): A list of tensor names that are required.
        input_data (NetRunnerInputData): The input data of type NetRunnerInputData

    Returns:
        bool: True if all required tensor names are found in the input data, False otherwise.
    """
    # If there is only one tensor, then we can't validate if the data is provided for this particular
    # tensor as users don't provide the input_name in case of single input. So, we return True in this case.
    if input_tensors and len(input_tensors) <= 1:
        return True

    str_list: List[str] = []
    if isinstance(input_data, list):
        for entry in input_data:
            if isinstance(entry, dict):
                str_list.extend(entry.keys())
    elif isinstance(input_data, (str, Path)):
        input_list_path = Path(input_data)
        str_list = get_name_input_pairs_from_input_list(input_list_path)

        str_list = [
            sublist[0][0] if isinstance(sublist[0], list) and len(sublist[0]) > 0 else sublist[0]
            for sublist in str_list
        ]

    for name in input_tensors:
        sanitized_name = re.sub(r"\W+", "_", name)
        if sanitized_name not in str_list:
            return False

    return True
