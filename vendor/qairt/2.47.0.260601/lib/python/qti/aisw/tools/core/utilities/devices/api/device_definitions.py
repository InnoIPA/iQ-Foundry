# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from enum import Enum
from ipaddress import ip_address
from os import PathLike
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, PositiveInt, model_validator

from qti.aisw.tools.core.utilities.devices.utils.device_code import InvalidIPAddressError


class DevicePlatformType(Enum):
    """
    Enum representing known device platforms.
    """

    ANDROID = "aarch64-android"
    LINUX_EMBEDDED = "linux-embedded"
    QNX = "qnx"
    WOS = "wos"
    X86_64_LINUX = "x86_64-linux-clang"
    X86_64_WINDOWS_MSVC = "x86_64-windows-msvc"


class ConnectionType(Enum):
    """
    Enum representing available device connection types.
    """

    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


class DeviceCredentials(BaseModel):
    """
    Contains sign-on information for a device


    Attributes:
        username (str): The username for the device.
        password (Optional[str]): The password for the device. Defaults to empty string.
    """

    username: str
    password: str = ""

    class Config:
        extra = "forbid"
        validate_assignment = True


class DeviceEnvironmentContext(BaseModel):
    """
    Captures environment specific fields for a device. This class should only be
    extended with fields that are general to all devices.

    Attributes:
        environment_variables (Optional[Dict[str, str]]): Specifies environment variables. Defaults to None.
        cwd(str | PathLike): Specifies a current working directory. Defaults to ".".
        shell (bool): If set to True then a new shell is spun up. Defaults to False.
    """

    environment_variables: Dict[str, str] = {}
    cwd: str | PathLike = "."
    shell: bool = False

    class Config:
        extra = "forbid"
        validate_assignment = True


class DeviceIdentifier(BaseModel):
    pass


class RemoteDeviceIdentifier(DeviceIdentifier):
    """
    Network identifiers for a remote device.

    Attributes:
        hostname (str): Hostname of the remote device. Either hostname or ip_address should be specified.
        ip_addr (str): A string representing the IP address of the remote device. Either hostname or ip_addr should be
        specified.
        port (Optional[int]): Port number of the remote device. Defaults to None.
        serial_id (Optional[str]): Human-readable name of the device, if available. Defaults to None. e.x android
                                   serial id, linux embedded serial id, etc

    """

    class Config:
        extra = "forbid"
        validate_assignment = True

    # Network related identifier
    hostname: Optional[str] = None
    ip_addr: Optional[str] = None
    port: Optional[PositiveInt] = None

    # Device specific identifier
    serial_id: Optional[str] = None
    soc_model: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _check_hostname_and_ip(cls, values: Any) -> Any:
        """
        Ensure both hostname and ip_addr cannot be specified at the same time.

        Args:
            values (dict): A dictionary containing the values of the attributes of this class.

        Returns:
            dict: The same dictionary with unchanged values.

        Raises:
            ValueError: If both hostname and ip_addr are provided.
        """
        if isinstance(values, Dict) and (values.get("hostname", None) and values.get("ip_addr", None)):
            raise ValueError("Both hostname and ip address cannot be specified at the same time")
        else:
            return values

    @model_validator(mode="after")
    def validate_connection_type(self) -> "RemoteDeviceIdentifier":
        """
        Validates ip_addr value.

        Args:
            self (RemoteDeviceIdentifier): The instance of RemoteDeviceIdentifier.

        Returns:
            RemoteDeviceIdentifier: The instance of RemoteDeviceInfo if the IP address is valid.

        Raises:
            ValueError: If the IP address is not valid.
        """
        try:
            if self.ip_addr:
                _ = ip_address(self.ip_addr)
            return self
        except ValueError as exc:
            raise InvalidIPAddressError(f"IP Address: {self.ip_addr} is not valid") from exc


class DeviceInfo(BaseModel):
    """
    This is a description of a device that should be used to determine the type of device object
    to instantiate.

    Attributes:
        connection_type ('ConnectionType.LOCAL'): The connection type of the device. It can only be set to local.
        platform_type (DevicePlatformType): The platform type of the device.
        identifier (Optional[str]): The identifier of the device. Defaults to None.
    """

    class Config:
        extra = "forbid"
        validate_assignment = True

    connection_type: ConnectionType = ConnectionType.LOCAL
    platform_type: DevicePlatformType
    identifier: Optional[str | Any] = None


class RemoteDeviceInfo(DeviceInfo):
    """
    Description of a remote device.

    Attributes:
        connection_type (Literal[ConnectionType.REMOTE]): The connection type of the device. It can only be set to
        remote.
        credentials (Optional[DeviceCredentials]): The credentials for the device. Defaults to None.
        identifier (RemoteDeviceIdentifier): A set of identifying properties for the device on the network.
        Defaults to None.
    """

    connection_type: Literal[ConnectionType.REMOTE] = ConnectionType.REMOTE
    credentials: Optional[DeviceCredentials] = None
    identifier: RemoteDeviceIdentifier


class SocDetails(BaseModel):
    """This class represents the SOC details of a device.

    Attributes:
        chipset (str): The SOC name.
        model (str): The SOC model number.
        dsp_arch (int): The SOC DSP architecture.
        vtcm_size_in_mb (int): The VTCM size in MB.
        num_of_hvx_threads (int): The number of HVX threads.
        supports_fp16 (bool): Indicates whether FP16 is supported or not.
        num_cores (Optional[int]): The number of HTP cores to use. Defaults to None.
    """

    class Config:
        extra = "forbid"
        validate_assignment = True

    chipset: str
    model: str
    dsp_arch: int
    vtcm_size_in_mb: int
    num_of_hvx_threads: int
    supports_fp16: bool
    num_cores: Optional[int] = None
    # TODO: num_cores cannot be populated by the device factory yet.
    # It must be set explicitly via the cores: spec string entry.
