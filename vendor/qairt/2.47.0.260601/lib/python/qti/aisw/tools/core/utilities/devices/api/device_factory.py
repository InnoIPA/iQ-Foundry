# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from typing import Type, TypeVar

# Device Concrete Class Imports
from qti.aisw.tools.core.utilities.devices.android.android_device import AndroidDevice
from qti.aisw.tools.core.utilities.devices.api.device_definitions import SocDetails
from qti.aisw.tools.core.utilities.devices.api.device_interface import *
from qti.aisw.tools.core.utilities.devices.linux_embedded.linux_embedded_device import LinuxEmbeddedDevice
from qti.aisw.tools.core.utilities.devices.qnx.qnx_device import QNXDevice
from qti.aisw.tools.core.utilities.devices.utils.device_utils import NoInitFactory
from qti.aisw.tools.core.utilities.devices.x86_64_linux.x86_64_linux_device import X86LinuxDevice
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class DeviceFactory(metaclass=NoInitFactory):
    """
    This class enables creation of device instances for different platform types e.x Android.
    """

    _logger = QAIRTLogger.get_logger(name=__name__, level="INFO")

    @staticmethod
    def create_device(device_info: DeviceInfo) -> DeviceInterface:
        """
        Creates a device object given a device info and an optional device status.

        Args:
            device_info: Information about the device

        Returns:
            An instance of the appropriate device class
        """
        if device_info.platform_type == DevicePlatformType.ANDROID:
            return AndroidDevice(device_info)
        elif device_info.platform_type == DevicePlatformType.X86_64_LINUX:
            return X86LinuxDevice(device_info)
        elif device_info.platform_type == DevicePlatformType.LINUX_EMBEDDED:
            return LinuxEmbeddedDevice(device_info)
        elif device_info.platform_type == DevicePlatformType.WOS:
            # TODO: Remove when this feature is added in SDK
            # PLaced here to avoid import errors
            from qti.aisw.tools.core.utilities.devices.windows.wos_device import WOSDevice

            return WOSDevice(device_info)
        else:
            raise DevicePlatformError(f"Device could not be created from type:{device_info.platform_type}")

    @staticmethod
    def create_device_info(
        platform_type: DevicePlatformType,
        connection_type: ConnectionType,
        username: str = "",
        password: str = "",
        ip_addr: Optional[str] = None,
        hostname: Optional[str] = None,
        port: Optional[int] = None,
    ) -> DeviceInfo:
        """
        Creates a device info using user-supplied information. The device info contains properties
        about a device,
        and should be used to create a Device object.

        Args:
            platform_type: Platform type of the device
            connection_type: Whether to connect to the device locally or remotely
            username: Username to use when connecting to the device remotely
            password: Password to use when connecting to the device
            ip_addr: IP address to use when connecting to the device. Either hostname or ip_addr should be provided
            for remote devices.
            hostname: Hostname to use when connecting to the device. Either hostname or ip_addr should be provided
            for remote devices.
            port: Port number to use when connecting to the device remotely

        Returns:
            A device info containing the specified properties
        """
        if connection_type == connection_type.LOCAL:
            return DeviceInfo(platform_type=platform_type, connection_type=connection_type)
        elif connection_type == connection_type.REMOTE:
            credentials = DeviceCredentials(username=username, password=password)

            if not (ip_addr or hostname):
                raise ValueError("IP Address or Hostname must be set when connecting to a remote device.")

            return RemoteDeviceInfo(
                platform_type=platform_type,
                connection_type=connection_type,
                credentials=credentials,
                identifier=RemoteDeviceIdentifier(ip_addr=ip_addr, hostname=hostname, port=port),
            )
        else:
            raise DeviceError(f"Unknown connection type: {connection_type!r}")

    @staticmethod
    def get_device_by_platform_type(
        device_platform_type: DevicePlatformType,
    ) -> Optional[Type[DeviceInterface]]:
        """
        Returns a device class based on the device platform type.

        Args:
            device_platform_type: The platform type of the device

        Returns:
            The device class corresponding to the device platform type, or None if no class exists

        Raises:
            DevicePlatformError: If the device platform type is unknown
        """
        if device_platform_type == DevicePlatformType.ANDROID:
            return AndroidDevice
        elif device_platform_type == DevicePlatformType.X86_64_LINUX:
            return X86LinuxDevice
        elif device_platform_type == DevicePlatformType.LINUX_EMBEDDED:
            return LinuxEmbeddedDevice
        elif device_platform_type == DevicePlatformType.QNX:
            return QNXDevice
        elif device_platform_type == DevicePlatformType.WOS:
            # TODO: Remove when this feature is added in SDK
            # PLaced here to avoid import errors
            try:
                from qti.aisw.tools.core.utilities.devices.windows.wos_device import WOSDevice

                return WOSDevice
            except ImportError:
                return None
        elif device_platform_type in DevicePlatformType:
            return None
        else:
            raise DevicePlatformError(f"Unknown device platform type: {device_platform_type}")

    @classmethod
    def get_available_devices(
        cls,
        connection_type: ConnectionType,
        username: str = "",
        password: str = "",
        hostname: Optional[str] = None,
        ip_addr: Optional[str] = None,
        port: Optional[int] = None,
    ) -> Dict[DevicePlatformType, List[DeviceInfo]]:
        """
        Returns a dictionary of available devices. Device availability is determined by the protocol
        helpers associated with each device interface

        Args:
            connection_type: Whether to connect to the device locally or remotely
            username: The username to use when connecting to the device remotely
            password: The password to use when connecting to the device remotely
            hostname: The hostname to use when connecting to the device remotely
            ip_addr: The IP address to use when connecting to the device remotely
            port: The port number to use when connecting to the device remotely

        Returns:
            A dictionary mapping device platform types to lists of available device infos
        """
        available_devices = {}
        device_infos = []

        for device_platform_type in DevicePlatformType:
            if device_interface := cls.get_device_by_platform_type(device_platform_type):
                if connection_type == ConnectionType.REMOTE:
                    device_infos = device_interface.get_available_devices(
                        connection_type,
                        device_credentials=DeviceCredentials(username=username, password=password),
                        hostname=hostname,
                        ip_addr=ip_addr,
                        port=port,
                    )
                    cls._logger.debug(
                        f" Discovered {len(device_infos)} {connection_type.value.lower()} device for "
                        f"platform: "
                        f"{device_platform_type!s}"
                    )
                else:
                    if isinstance(device_interface, QNXDevice):
                        cls._logger.debug("QNX does not support local connections")
                        continue

                    device_infos = device_interface.get_available_devices()
                    cls._logger.debug(
                        f" Discovered {len(device_infos)} {connection_type.value.lower()} device for "
                        f"platform: "
                        f"{device_platform_type!s}"
                    )

                if device_infos:
                    cls._logger.info(
                        f" Number of available devices for {device_platform_type!s}:{len(device_infos)}"
                    )
                    available_devices[device_platform_type] = device_infos

        return available_devices

    # TODO: Move this to modules so we can decouple device utility from QAIRT
    @staticmethod
    def get_device_soc_details(backend: str, soc_model: str) -> Optional[SocDetails]:
        """Retrieves SOC details for the specified backend and soc_model.

        Args:
            backend: The backend type ex: HTP, CPU, GPU
            soc_model: The SOC model name ex: SA8295

        Returns:
            A instance of SocDetails class containing SOC details (chipset, model, dsp_arch,
            vtcm_size_in_mb, num_of_hvx_threads & supports_fp16) if successful, else None
        """
        if not soc_model or not backend:
            raise Exception("Please provide backend & soc_model")

        # Check whether backend is supported
        from qti.aisw.converters.common import backend_info

        if not backend_info.is_backend_supported(backend):
            raise Exception("Backend {} is not supported".format(backend))

        # Check whether soc_model is supported
        # TODO: For some reason this check fails even for supported soc_models.
        # Commenting it out for now.
        # if soc_model and not backend_info.is_soc_model_supported(soc_model):
        #     DeviceFactory._logger.info(f"SOC model {soc_model} is not supported")

        backend_info_obj = backend_info.PyBackendInfo(backend, soc_model)

        # Check whether backend is supported by given soc
        if not backend_info_obj.is_backend_supported_in_soc():
            raise Exception("Backend {} is not supported by the SOC Model {}.".format(backend, soc_model))

        soc_details = backend_info_obj.get_soc_info_subset()
        if not soc_details:
            return None

        return SocDetails(
            chipset=soc_details.socName,
            model=soc_details.socModel,
            dsp_arch=soc_details.socDspArch,
            vtcm_size_in_mb=soc_details.vtcmSizeinMB,
            num_of_hvx_threads=soc_details.numOfHvxThreads,
            supports_fp16=soc_details.supportsFp16,
        )
