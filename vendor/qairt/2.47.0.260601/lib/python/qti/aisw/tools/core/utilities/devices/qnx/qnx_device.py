# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
from os import PathLike
from typing import Any, List, Literal, Optional

from qti.aisw.tools.core.utilities.devices.api.device_interface import (
    ConnectionType,
    DeviceCredentials,
    DeviceEnvironmentContext,
    DeviceInfo,
    DeviceInterface,
    DevicePlatformType,
    RemoteDeviceIdentifier,
    RemoteDeviceInfo,
)
from qti.aisw.tools.core.utilities.devices.qnx.qnx_executor import QNXExecutor
from qti.aisw.tools.core.utilities.devices.utils.device_code import (
    DeviceCompletedProcess,
    DeviceFailedProcess,
    DeviceReturn,
)
from qti.aisw.tools.core.utilities.devices.utils.device_utils import filter_log_by_string
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class QNXDevice(DeviceInterface):
    def __init__(self, device_info: Optional[RemoteDeviceInfo] = None, logger: Optional[Any] = None):
        super().__init__(device_info, logger)
        self._hostname = None
        if self.device_info:
            assert isinstance(self.device_info, RemoteDeviceInfo)

            remote_device_identifier = self.device_info.identifier
            if remote_device_identifier.hostname:
                self._hostname = remote_device_identifier.hostname
            elif remote_device_identifier.ip_addr:
                self._hostname = remote_device_identifier.ip_addr
            else:
                raise ValueError("QNXDevice requires a hostname or ip_addr")
            self._executor = QNXExecutor(
                hostname=self._hostname,
                port=self.device_info.identifier.port or 0,
                username=self.device_info.credentials.username if self.device_info.credentials else "",
                password=self.device_info.credentials.password
                if self.device_info.credentials and self.device_info.credentials.password is not None
                else "",
            )
            if not self._logger:
                self.set_logger(
                    QAIRTLogger.get_logger(name=f"{self.__class__.__name__}: {self._hostname}", level="INFO")
                )

    @property
    def hostname(self) -> Optional[str]:
        return self._hostname

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        return self._device_info

    @device_info.setter
    def device_info(self, device_info: DeviceInfo):
        """
        Sets the device information if previously unset.

        Raises:
            ValueError: If device info is already set and new value is different from the current.
        """
        if self._device_info is None:
            self._device_info = device_info
        elif self._device_info != device_info:
            raise ValueError("Device info cannot be reset\nCreate a new device instance instead")

    @property
    def executors(self):
        return [self.executor]

    @property
    def executor(self):
        return self._executor

    def execute(
        self, commands: List[str], device_env_context: Optional[DeviceEnvironmentContext] = None
    ) -> DeviceReturn:
        """
         Executes a command on the device using the QNX executor.

         Args:
             commands (list): A list of commands to execute.
             device_env_context (DeviceEnvironmentContext): The environment context of the device.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess]): Execution result
        """

        if device_env_context is None:
            device_env_context = DeviceEnvironmentContext()

        env_vars = []
        cwd = device_env_context.cwd

        if device_env_context.environment_variables:
            env_vars = [
                f"export {env_var}={value}"
                for env_var, value in device_env_context.environment_variables.items()
            ]

        qnx_commands = [f"cd {cwd}"] + env_vars + commands
        qnx_command = " && ".join(qnx_commands)
        device_return = self.executor.execute(qnx_command)

        if isinstance(device_return, DeviceCompletedProcess):
            self._logger.info(
                f"Executed command: {qnx_commands} with return code: {device_return.returncode}"
            )

        return device_return

    def pull(self, src_path: str | PathLike, dst_path: str | PathLike) -> DeviceReturn:
        """
        Method to pull files from the device.

        Args:
            src_path (str | PathLike): The source path on the device.
            dst_path (str | PathLike): The destination path on the local machine.

        Returns
            DeviceReturn (DeviceCompletedProcess | DeviceFailedProcess): Result of executing the
                                                                         pull command
        """
        return self.executor.pull(str(src_path), str(dst_path))

    def push(self, src_path: str | PathLike, dst_path: str | PathLike) -> DeviceReturn:
        """
        Method to push files or directories to the device.

        Args:
            src_path (str | PathLike): The source path on the local machine.
            dst_path (str | PathLike): The destination path on the device.

        Returns:
            DeviceReturn (DeviceCompletedProcess | DeviceFailedProcess): Result of executing
                                                                         the push command
        """

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Path: {src_path} does not exist")
        elif not os.path.isabs(src_path):
            src_path = os.path.abspath(src_path)

        return self.executor.push(str(src_path), str(dst_path))

    def make_directory(self, dir_name: str | PathLike) -> DeviceReturn:
        """
        Method to make a directory on an QNX device.

        Args:
            dir_name (str | PathLike): The name of the directory to create.

        Returns:
            DeviceReturn (DeviceCompletedProcess | DeviceFailedProcess): The result of creating the
                                                                          directory.
        """
        return self.executor.make_directory(dir_name)

    def remove(self, target_path: str | PathLike) -> DeviceReturn:
        """
        Method to remove a file or directory from the device.

        Args:
            target_path (str | PathLike): The path of the file or directory to remove.

        Returns:
            DeviceReturn: The device result after removing the file or directory.
        """
        return self.executor.remove(target_path)

    def close(self) -> DeviceReturn:
        """
        Closes the communication session with the device.

        Returns:
            DeviceReturn (DeviceCompletedProcess | DeviceFailedProcess): Result of closing
                                                                        communication.
        """
        return self.executor.close()

    def get_device_log(self, **kwargs) -> bytes | str | None:
        """
        Get the entire device log or the log following a user defined string expression

        Args:
            kwargs (dict): Additional keyword arguments.
            1. log_filter (Optional[str]): A string expression to filter logs (passed to re.compile).

        Returns:
            bytes | None: The device log content, None if retrieval fails or a custom log content
                          indicating that no log was found

        """
        log_filter = kwargs.get("log_filter", None)
        device_return = self.executor.execute("slog2info", close_connection=True)

        if isinstance(device_return, DeviceFailedProcess):
            self._logger.error("Failed to retrieve device log")
            return None
        elif not device_return.stdout:
            return b"No log was found for device"

        device_log = device_return.stdout
        # Coerce non-bytes/str stdout (e.g. dict) to str for filtering
        if not isinstance(device_log, (bytes, str)):
            device_log = str(device_log)
        # Ensure device_log is bytes before filtering
        if device_log and not isinstance(device_log, bytes):
            device_log = bytes(device_log, encoding="utf-8")

        if log_filter:
            device_log = filter_log_by_string(device_log, log_filter, self._logger)
            if not device_log:
                self._logger.debug(f"No log was found with filter: {log_filter}")

        return device_log

    @staticmethod
    def get_available_devices(  # type: ignore
        connection_type: Literal[ConnectionType.REMOTE] = ConnectionType.REMOTE,
        hostname: Optional[str] = None,
        ip_addr: Optional[str] = None,
        device_credentials: Optional[DeviceCredentials] = None,
        **kwargs,
    ) -> Optional[List[DeviceInfo]]:
        """
        Static method to get the available devices.

        Args:
            connection_type (ConnectionType): The type of connection. Only Remote connections are
                                              possible.
            hostname (Optional[str]): The hostname of the QNX device. Defaults to None
            ip_addr (Optional[str]): The ip address of the QNX device. Defaults to None. ip_addr
                                     will only be used if hostname is not provided.
            device_credentials (Optional[DeviceCredentials]): Credentials containing the username
                                                              and password. Defaults to None.

        Returns:
            Optional[List[DeviceInfo]]: The available devices that have been discovered.
        """
        available_device_info: List[DeviceInfo] = list()
        destination = hostname if hostname else ip_addr
        if connection_type == connection_type.LOCAL or not destination:
            # QNX does not support local connections
            return available_device_info
        else:
            # available devices returns the same destination if it has been successfully pinged
            if QNXExecutor.get_available_devices(destination):
                remote_identifier = RemoteDeviceIdentifier(
                    hostname=hostname, ip_addr=ip_addr, port=kwargs.get("port", None)
                )
                available_device_info.append(
                    RemoteDeviceInfo(
                        platform_type=DevicePlatformType.QNX,
                        identifier=remote_identifier,
                        credentials=device_credentials,
                    )
                )

        return available_device_info
