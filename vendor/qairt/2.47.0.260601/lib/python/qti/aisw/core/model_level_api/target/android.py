#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.core.model_level_api.executor.android_subprocess_executor import (
    AndroidSubprocessExecutor,
)
from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.tools.core.utilities.devices.android.android_device import AndroidDevice
from qti.aisw.tools.core.utilities.devices.api.device_interface import *


class AndroidTarget(Target):
    """Target for an Android device"""
    def __init__(self,
                 device_id: Optional[str] = None,
                 hostname: Optional[str] = None,
                 adb_server_port: Optional[int] = None,
                 soc_model: Optional[str] = None):
        super().__init__()
        device_identifier = RemoteDeviceIdentifier(serial_id=device_id,
                                                   hostname=hostname,
                                                   port=adb_server_port,
                                                   soc_model=soc_model)
        device_info = RemoteDeviceInfo(platform_type=DevicePlatformType.ANDROID,
                                       identifier=device_identifier)
        self._device = AndroidDevice(device_info=device_info)

    @property
    def target_name(self) -> Literal['aarch64-android']:
        return 'aarch64-android'

    @property
    def target_platform_type(self) -> DevicePlatformType:
        return DevicePlatformType.ANDROID

    @staticmethod
    def get_available_device_ids(hostname: Optional[str] = None,
                                 port: Optional[int] = None) -> Optional[List[str]]:
        """
        Returns a list of all detected android devices

        Args:
            hostname(str): The hostname of the device to connect to. Defaults to localhost
            port: The port number of the device to connect to. Defaults to ADB default

        Returns:
            Optional[List[str]]: A list of device ids or None if none were found
        """
        return AndroidDevice.get_available_devices(hostname=hostname, port=port)

    def get_default_executor_cls(self):
        return AndroidSubprocessExecutor

    def _is_file(self, filename):
        device_return = self._device.execute('test -f ' + filename)
        return device_return.returncode == 0
