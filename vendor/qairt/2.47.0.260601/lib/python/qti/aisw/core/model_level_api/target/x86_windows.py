#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Literal

from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.core.model_level_api.executor.native_executor import NativeExecutor



class X86WindowsTarget(Target):
    """ Target for X86 Windows device """

    file_sep = '\\'
    def __init__(self) -> None:
        """
        Initializes the X86WindowsTarget class.
        """
        super().__init__()
        # TODO: AISW-131890: Add support for X86 windows device
        self._device = None

    @property
    def target_name(self) -> Literal['x86_64-windows-msvc']:
        """
        Returns the target name for X86 Windows.
        """
        return 'x86_64-windows-msvc'

    @property
    def target_platform_type(self) -> DevicePlatformType:
        """
        Returns the target platform type for X86 Windows.
        """
        return DevicePlatformType.X86_64_WINDOWS_MSVC

    def get_default_executor_cls(self):
        """
        Returns the default executor class for X86 Windows.
        """
        return NativeExecutor
