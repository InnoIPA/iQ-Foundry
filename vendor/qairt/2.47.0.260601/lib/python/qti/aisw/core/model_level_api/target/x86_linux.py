#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from typing import Literal
from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.core.model_level_api.executor.native_executor import NativeExecutor
from qti.aisw.tools.core.utilities.devices.x86_64_linux.x86_64_linux_device import X86LinuxDevice
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType


class X86LinuxTarget(Target):
    """Target for X86 Linux device"""
    def __init__(self):
        super().__init__()
        self._device = X86LinuxDevice()

    @property
    def target_name(self) -> Literal['x86_64-linux-clang']:
        return 'x86_64-linux-clang'

    @property
    def target_platform_type(self) -> DevicePlatformType:
        return DevicePlatformType.X86_64_LINUX

    def get_default_executor_cls(self):
        return NativeExecutor
