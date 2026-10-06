#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Literal
from qti.aisw.core.model_level_api.executor.native_executor import NativeExecutor
from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.tools.core.utilities.devices.windows.wos_device import WOSDevice

from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType

class WOSTarget(Target):
    """ Target for ARM Windows (WOS) device """

    file_sep = '\\'

    def __init__(self) -> None:
        """
        Initializes the WOSTarget class.
        """
        super().__init__()
        self._device = WOSDevice()

    @property
    def target_name(self) -> Literal['arm64x-windows-msvc']:
        """
        Returns the target name for ARM Windows.
        """
        return 'arm64x-windows-msvc'

    @property
    def target_platform_type(self) -> DevicePlatformType:
        """
        Returns the target platform type for ARM Windows.
        """
        return DevicePlatformType.WOS

    def get_default_executor_cls(self):
        """
        Returns the default executor class for ARM Windows.
        """
        return NativeExecutor
