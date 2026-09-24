#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.core.model_level_api.executor.qnx_subprocess_executor import \
    QNXSubprocessExecutor
from qti.aisw.tools.core.utilities.devices.qnx.qnx_device import QNXDevice
from qti.aisw.tools.core.utilities.devices.api.device_interface import *


class QNXTarget(Target):
    """Target for a QNX device"""
    def __init__(self,
                 hostname: str,
                 username: str,
                 port: Optional[PositiveInt] = None,
                 password: Optional[str] = None,
                 soc_model: Optional[str] = None):
        super().__init__()
        device_identifier = RemoteDeviceIdentifier(hostname=hostname,
                                                   port=port,
                                                   soc_model=soc_model)
        device_credentials = DeviceCredentials(username=username,
                                                   password=password)
        device_info = RemoteDeviceInfo(platform_type=DevicePlatformType.QNX,
                                       identifier=device_identifier,
                                       credentials=device_credentials)
        self._device = QNXDevice(device_info=device_info)

    @property
    def target_name(self) -> Literal['aarch64-qnx']:
        return 'aarch64-qnx'

    @property
    def target_platform_type(self) -> DevicePlatformType:
        return DevicePlatformType.QNX

    @staticmethod
    def get_available_device_ids(hostname: str,
                                 username: str,
                                 port: Optional[int] = None,
                                 password: Optional[str] = None) -> Optional[List[str]]:
        """
        Returns a list of all detected android devices

        Args:
            hostname(str): The hostname of the device to connect to. Defaults to localhost
            port: The port number of the device to connect to. Defaults to ADB default

        Returns:
            Optional[List[str]]: A list of device ids or None if none were found
        """
        device_credentials = DeviceCredentials(username=username,
                                                password=password)
        return QNXDevice.get_available_devices(hostname=hostname, port=port, device_credentials=device_credentials)

    def get_default_executor_cls(self):
        return QNXSubprocessExecutor

    def _is_file(self, filename):
        device_return = self._device.execute('test -f ' + filename)
        return device_return.returncode == 0
