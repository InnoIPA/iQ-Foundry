# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from os import getenv, path, walk
from pathlib import Path
from typing import List

def qnn_sdk_root() -> str:
    sdk_root = getenv('QNN_SDK_ROOT')
    if sdk_root is None:
        sdk_root = path.dirname(path.abspath(__file__ + '/../../../../../../../'))
    return sdk_root

def get_oe_linux_gcc_versions() -> List[str]:
    lib_path = Path(qnn_sdk_root(), 'lib')
    gcc_versions = []
    for _, dirs, _ in walk(lib_path):
        for folder in dirs:
            if folder.startswith("aarch64-oe-linux-gcc"):
                gcc_versions.append(folder)
    if len(gcc_versions) == 0:
        raise ValueError(f'The SDK was not compiled for OELinux Toolchains')
    return gcc_versions

def get_newest_oe_linux_gcc_version() -> str:
    max_version = 0.0
    gcc_versions = get_oe_linux_gcc_versions()
    for folder in gcc_versions:
        curr_version = float(folder[len("aarch64-oe-linux-gcc"):])
        if curr_version > max_version:
            max_version = curr_version
    latest_version = "aarch64-oe-linux-gcc" + str(max_version)
    return latest_version