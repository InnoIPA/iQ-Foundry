# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.tools.core.utilities.devices.api.device_interface import *
from qti.aisw.tools.core.utilities.devices.utils.device_utils import convert_to_win_path
from qti.aisw.tools.core.utilities.devices.windows.windows_native_executor import WindowsNativeExecutor
from qti.aisw.tools.core.utilities.devices.windows.windows_remote_executor import WindowsRemoteExecutor
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class WOSDevice(DeviceInterface):
    def __init__(self, device_info: Optional[DeviceInfo] = None, logger: Optional[Any] = None):
        super().__init__(device_info, logger)
        self.hostname = "localhost"
        if self.device_info:
            self._initialize_executor()

        if not self._logger:
            self.set_logger(
                QAIRTLogger.get_logger(name=f"{self.__class__.__name__}: {self.hostname}", level="INFO")
            )

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        return self._device_info

    @device_info.setter
    def device_info(self, device_info: DeviceInfo):
        """Sets the device information if previously unset.

        Raises:
            ValueError: If device info is already set and new value is different from the current.
        """
        if self._device_info is None:
            self._device_info = device_info
            self._initialize_executor()
        elif isinstance(device_info, RemoteDeviceInfo) and self._device_info != device_info:
            raise ValueError("Device info cannot be reset\nCreate a new device instance instead")

    @property
    def executors(self):
        return [self.executor]

    @property
    def executor(self):
        return self._executor

    def _initialize_executor(self):
        """Initializes the executor if a device info is set"""
        self._executor: Executor
        if isinstance(self.device_info, RemoteDeviceInfo):
            if self.device_info.identifier.hostname:
                self.hostname = self.device_info.identifier.hostname
            elif self.device_info.identifier.ip_addr:
                self.hostname = self.device_info.identifier.ip_addr
            else:
                raise ValueError("Hostname or IP address must be specified for remote device")
            self._executor = WindowsRemoteExecutor(
                hostname=self.hostname,
                port=self.device_info.identifier.port or 0,
                username=self.device_info.credentials.username if self.device_info.credentials else None,
                password=self.device_info.credentials.password if self.device_info.credentials else None,
            )
        else:
            self._executor = WindowsNativeExecutor()

    def execute(
        self, commands: List[str], device_env_context: Optional[DeviceEnvironmentContext] = None
    ) -> DeviceCompletedProcess:
        """Executes a command on the device using the WOS executor.

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
            for env_var, value in device_env_context.environment_variables.items():
                win_path = convert_to_win_path(value)  # ensure paths are windows compatible
                env_cmd = f'$env:{env_var} += "{os.pathsep}{win_path}"'
                env_vars.append(f" if($?) {{ {env_cmd} }} ;")

        wos_commands = [f"cd {cwd}"] + env_vars + commands
        separator = "&" if isinstance(self.device_info, RemoteDeviceInfo) else ";"
        wos_command = f" {separator} ".join(wos_commands).strip()
        device_return = self.executor.execute(wos_command)

        if isinstance(device_return, DeviceCompletedProcess):
            self._logger.debug(
                f"Executed command: {wos_command} with return code: {device_return.returncode}"
            )

        return device_return

    def pull(self, src_path: str | PathLike, dst_path: str | PathLike) -> DeviceReturn:
        """Method to pull files from the device.

        Args:
            src_path (str | PathLike): The source path on the device.
            dst_path (str | PathLike): The destination path on the local machine.
        """
        return self.executor.pull(str(src_path), str(dst_path))

    def push(self, src_path: str | PathLike, dst_path: str | PathLike) -> DeviceReturn:
        """Method to push files or directories to the device.

        Args:
            src_path (str | PathLike): The source path on the local machine.
            dst_path (str | PathLike): The destination path on the device.
        """
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Path: {src_path} does not exist")

        return self.executor.push(src_path, dst_path)

    def make_directory(self, dir_name: str | PathLike):
        """Method to make a directory on an WOS device.

        Args:
            dir_name (str | PathLike): The name of the directory to create.
        """
        return self.executor.make_directory(dir_name)

    def remove(self, target_path: str | PathLike) -> DeviceReturn:
        """Method to remove a file or directory from the device.

        Args:
            target_path (str | PathLike): The path of the file or directory to remove.

        Returns:
            DeviceReturn: The device code after removing the file or directory.
        """
        return self.executor.remove(target_path)

    def close(self) -> DeviceReturn:
        return self.executor.close()

    @staticmethod
    def get_available_devices(
        connection_type: ConnectionType = ConnectionType.LOCAL,
        **kwargs,
    ) -> List[DeviceInfo]:
        """Static method to get the available devices.

        Args:
            connection_type (ConnectionType): The type of connection.
                                              Defaults to ConnectionType.LOCAL.
            kwargs: Additional keyword arguments.
                hostname (Optional[str]): The hostname of the WOS device. Defaults to None
                ip_addr (Optional[str]): The ip address of the WOS device. Defaults to None. ip_addr
                                         will only be used if hostname is not provided.
                device_credentials (Optional[DeviceCredentials]): Credentials containing the username
                                                                  and password. Defaults to None.

        Returns:
            Optional[List[DeviceInfo]]: The available devices.
        """
        hostname = kwargs.get("hostname", None)
        ip_addr = kwargs.get("ip_addr", None)
        device_credentials = kwargs.get("device_credentials", None)
        available_device_info: List[DeviceInfo] = list()
        destination = hostname if hostname else ip_addr
        if connection_type == connection_type.LOCAL:
            return [DeviceInfo(platform_type=DevicePlatformType.WOS)]
        else:
            # available devices returns the same destination if it has been successfully pinged
            if destination and WindowsRemoteExecutor.get_available_devices(destination):
                remote_identifier = RemoteDeviceIdentifier(
                    hostname=hostname, ip_addr=ip_addr, port=kwargs.get("port", None)
                )
                available_device_info.append(
                    RemoteDeviceInfo(
                        platform_type=DevicePlatformType.WOS,
                        identifier=remote_identifier,
                        credentials=device_credentials,
                    )
                )

        return available_device_info
