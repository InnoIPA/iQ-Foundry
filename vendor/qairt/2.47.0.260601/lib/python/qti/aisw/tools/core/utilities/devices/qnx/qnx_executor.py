# ==============================================================================
#
#  Copyright (c)   Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from os import PathLike
from typing import List, Optional

from qti.aisw.tools.core.utilities.devices.api.executor import Executor
from qti.aisw.tools.core.utilities.devices.protocol_helpers.ftp import FTPProtocolHelper
from qti.aisw.tools.core.utilities.devices.protocol_helpers.telnet import TelnetProtocolHelper
from qti.aisw.tools.core.utilities.devices.utils.device_code import (
    DeviceCompletedProcess,
    DeviceFailedProcess,
    DevicePlatformError,
    DeviceReturn,
)
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class QNXExecutor(Executor):
    """
    Class for interacting with a QNX device via FTP and Telnet
    """

    _logger = QAIRTLogger.get_logger(name="QNXExecutor", level="INFO")

    def __init__(
        self,
        hostname: str,
        username: str = "",
        password: str = "",
        *,
        port: int = 0,
        timeout: Optional[int] = None,
        debug_level: int = 0,
        exclude_ftp_connection: bool = False,
        exclude_telnet_connection: bool = False,
    ):
        """

        Args:
            hostname (str): The hostname or IP address of the remote device.
            username (str): The username for authentication. Defaults to empty string
            password (str): The password for authentication. Defaults to empty string
            port (int): The port number for communication. Defaults to 0
            timeout (Optional[int]): The timeout value to wait for a connection. Defaults to
            None.
            debug_level (int): The debug level for logging. Defaults to 0.
            exclude_ftp_connection (bool): Flag indicating whether to exclude the FTP connection
                                            when setting up the executor. Defaults to False.
            exclude_telnet_connection (bool): Flag indicating whether to exclude the Telnet
                                              connection when setting up the executor. Defaults
                                              to False.
        """
        super().__init__()
        self._telnet_protocol_helper = TelnetProtocolHelper
        self._ftp_protocol_helper = FTPProtocolHelper
        self.hostname = hostname.strip()
        self.username = username
        self.password = password
        self.telnet_connection = None
        self.ftp_connection = None

        if not exclude_ftp_connection:
            if conn := self._ftp_protocol_helper.setup_ftp_connection(
                self.hostname,
                self.username,
                self.password,
                port=port,
                timeout=timeout,
                debug_level=debug_level,
            ):
                self.ftp_connection = conn
            else:
                raise DevicePlatformError(f"QNX: Could not setup an FTP connection for: {self.hostname}")
        else:
            self._logger.warning(
                "FTP connection was not established because it was excluded. FTP "
                "connection may be assumed in certain method calls"
            )

        if not exclude_telnet_connection:
            telnet_conn = self._telnet_protocol_helper.setup_telnet_connection(
                self.hostname,
                self.username,
                self.password,
                port=port,
                timeout=timeout,
                debug_level=debug_level,
            )
            if telnet_conn:
                self.telnet_connection = telnet_conn
            else:
                raise DevicePlatformError(f"QNX: Could not setup a telnet connection for: {self.hostname}")
        else:
            self._logger.warning(
                "Telnet connection was not established because it was excluded. "
                "Telnet connection may be assumed in certain method calls"
            )

        self.protocol_helper = self._telnet_protocol_helper

    def push(self, src: str | PathLike, dst: str | PathLike):
        assert self.ftp_connection
        return self._ftp_protocol_helper.push(src, dst, connection=self.ftp_connection)

    def pull(self, src: str | PathLike, dst: str | PathLike) -> DeviceReturn:
        assert self.ftp_connection
        return self._ftp_protocol_helper.pull(
            src,
            dst,
            connection=self.ftp_connection,
        )

    def make_directory(self, dir_name: str | PathLike) -> Optional[DeviceReturn]:
        """
        Method to make a directory on an QNX device.

        Args:
            dir_name (str | PathLike): The name of the directory to create.

        Returns:
            DeviceReturn (DeviceCompletedProcess | DeviceFailedProcess): The device status after
            making the directory or None if the directory exists
        """
        assert self.ftp_connection
        return self._ftp_protocol_helper.make_directory(str(dir_name), connection=self.ftp_connection)

    def close(self) -> DeviceReturn:
        """
        Method to close any open connections that were created by this class instance

        Returns:
            DeviceReturn (DeviceCompletedProcess | DeviceFailedProcess): The device status after
            closing the connection
        """
        if self.ftp_connection:
            ftp_conn_return = self._ftp_protocol_helper.close(self.ftp_connection)

            if isinstance(ftp_conn_return, DeviceFailedProcess):
                return ftp_conn_return

        if self.telnet_connection:
            telnet_device_return = self._telnet_protocol_helper.close(self.telnet_connection)
            if isinstance(telnet_device_return, DeviceFailedProcess):
                return telnet_device_return

        return DeviceCompletedProcess(args=["close"], stdout=None)

    def remove(self, target_path: str | PathLike) -> Optional[DeviceReturn]:
        """
        Method to remove a file or directory from the device.

        Args:
            target_path (str | PathLike): The path of the file or directory to remove.

        Returns:
            DeviceReturn (DeviceCompletedProcess | DeviceFailedProcess): The device status after
            removing the file or directory.
        """
        assert self.ftp_connection
        return self._ftp_protocol_helper.remove(str(target_path), connection=self.ftp_connection)

    def execute(self, command: str, args: Optional[List[str]] = None, **kwargs) -> DeviceReturn:
        """
        Method to execute a command on the QNX device.

        Args:
            command (str): The command to execute on the device.
            args (Optional[List[str]]): The arguments for the command. Defaults to None.
            **kwargs: Additional arguments to pass to the command
                close_connection (bool): Closes the connection after command execution. Defaults to False
        """

        if args is None:
            args = []
        command_args_str = f"{command} {''.join(args)}"
        close_connection = kwargs.get("close_connection", False)
        assert self.telnet_connection
        return self._telnet_protocol_helper.execute(
            command_args_str, connection=self.telnet_connection, close_connection=close_connection
        )

    @classmethod
    def get_available_devices(cls, destination: str, *_destinations) -> List[str]:
        """
        Returns a list of available devices based on one or more destination addresses

        Args:
            destination (str): The address of the host machine
            _destinations (str): Additional destinations to search through

        Returns:
            List[str]: A list of available devices
        """
        available_devices = set(
            FTPProtocolHelper.get_available_devices(destination, *_destinations)
            + TelnetProtocolHelper.get_available_devices(destination, *_destinations)
        )
        return list(available_devices)
