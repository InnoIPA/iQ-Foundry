#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.core.model_level_api.executor.native_executor import NativeExecutor
from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.core.model_level_api.executor.oelinux_subprocess_executor import OELinuxSubprocessExecutor
from qti.aisw.tools.core.utilities.devices.linux_embedded.linux_embedded_device import LinuxEmbeddedDevice
from qti.aisw.tools.core.utilities.devices.api.device_interface import *
from qti.aisw.core.model_level_api.utils.qnn_sdk import get_oe_linux_gcc_versions, get_newest_oe_linux_gcc_version


class OELinuxTarget(Target):
    """Target for an OE Linux IOT device"""
    _gcc_versions_supported = []
    _toolchain_name_prefix = "aarch64-oe-linux-gcc"
    def __init__(self,
                 device_id: Optional[str] = None,
                 hostname: Optional[str] = None,
                 adb_server_port: Optional[int] = None,
                 gcc_version: Optional[str] = None,
                 soc_model: Optional[str] = None):
        super().__init__()
        self._gcc_version = None
        is_remote = any([device_id, hostname, adb_server_port, soc_model])
        if is_remote:
            device_identifier = RemoteDeviceIdentifier(
                serial_id=device_id,
                hostname=hostname,
                port=adb_server_port,
                soc_model=soc_model,
            )
            device_info = RemoteDeviceInfo(
                platform_type=DevicePlatformType.LINUX_EMBEDDED,
                identifier=device_identifier,
            )
        else:
            device_info = DeviceInfo(platform_type=DevicePlatformType.LINUX_EMBEDDED)
        self._device = LinuxEmbeddedDevice(device_info=device_info)

        if len(self._gcc_versions_supported) == 0:
            self._gcc_versions_supported = get_oe_linux_gcc_versions()

        if gcc_version is not None:
            if gcc_version in self._gcc_versions_supported:
                self._gcc_version = self._toolchain_name_prefix + gcc_version
            else:
                raise ValueError(
                    f"GCC Version is not supported. Versions supported are {self._gcc_versions_supported}"
                )
        else:
            self._gcc_version = get_newest_oe_linux_gcc_version()

    @property
    def target_name(self) -> str:
        return self._gcc_version

    @property
    def target_platform_type(self) -> DevicePlatformType:
        return DevicePlatformType.LINUX_EMBEDDED

    @staticmethod
    def get_available_device_ids(hostname: Optional[str] = None,
                                 port: Optional[int] = None) -> Optional[List[str]]:
        """
        Returns a list of all detected android devices

        Args:
            hostname(str): The hostname of the device to connect to. Defaults to localhost
            port_number: The port number of the device to connect to. Defaults to ADB default

        Returns:
            Optional[List[str]]: A list of device ids or None if none were found
        """
        return LinuxEmbeddedDevice.get_available_devices(hostname=hostname, port=port)

    def get_default_executor_cls(self):
        if isinstance(self._device.device_info, RemoteDeviceInfo):
            return OELinuxSubprocessExecutor
        else:
            return NativeExecutor

    def _is_file(self, filename):
        device_return = self._device.execute('test -f ' + filename)
        return device_return.returncode == 0
