#  @@-COPYRIGHT-START-@@
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os


def get_path_for_target_config(target_config: str) -> str:
    """
    Returns path for target config such as htp, aic, cpu and lpai

    :return: path for target config file
    """
    if target_config not in {'htp', 'aic', 'cpu', 'lpai'}:
        raise ValueError(f"Backend {target_config} does not have backend aware config.")

    target_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      f'{target_config}.json')

    if not os.path.isfile(target_config_path):
        raise FileNotFoundError(f"Could not find target config: {target_config_path}")

    return target_config_path
