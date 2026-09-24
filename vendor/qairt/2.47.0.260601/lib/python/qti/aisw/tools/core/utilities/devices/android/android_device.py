# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from datetime import datetime
from pathlib import Path

from qti.aisw.tools.core.utilities.devices.android.android_device_constants import *
from qti.aisw.tools.core.utilities.devices.android.android_executor import *
from qti.aisw.tools.core.utilities.devices.api.device_interface import *
from qti.aisw.tools.core.utilities.devices.utils.device_utils import (
    DateRange,
    filter_log_by_date_range,
    filter_log_by_string,
    format_device_log,
    format_output_as_list,
)
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class AndroidDevice(DeviceInterface):
    def __init__(self, device_info: DeviceInfo, logger: Optional[Any] = None):
        super().__init__(device_info, logger)
        self._executor = AndroidExecutor()

        if self._logger is None:
            self.set_logger(
                QAIRTLogger.get_logger(name=f"{self.__class__.__name__}: {self.id}", level="INFO")
            )

        self._set_device_dependencies()

    def _set_device_dependencies(self):
        """
        Sets dependencies that are needed by the device object such as the device id, logger
        and the hostname.
        """

        self._executor.selected_device_id = self.id

        if isinstance(self.device_info, RemoteDeviceInfo) and self.device_info.identifier:
            self._executor.hostname = self.device_info.identifier.hostname
            self._executor.port = self.device_info.identifier.port

        if self.device_info and not self.device_info.identifier:
            self._logger.warning(
                "Device identifier was not specified. Note the device specified "
                "via $ANDROID_SERIAL will be used if the environment variable is "
                "set. Otherwise, the first available device will be used."
            )

    @property
    def id(self) -> Optional[str]:
        if self.device_info:
            if isinstance(self.device_info, RemoteDeviceInfo):
                return self.device_info.identifier.serial_id
            else:
                return self.device_info.identifier
        return None

    def get_soc_name(self) -> Union[str, Literal["UNKNOWN"]]:
        """
        This function returns the soc name of the device using the internal device id.

        Returns:
            str: The soc name of the device if parsed, else it returns `UNKNOWN_SOC_NAME`
        """
        soc_id = self.executor.query_soc_id()

        soc_name = UNKNOWN_SOC_NAME

        if soc_id:
            soc_id = soc_id.split()[0]
            if soc_id in SOC_ID_TO_SOC_NAME:
                soc_name = SOC_ID_TO_SOC_NAME[soc_id]
                self._logger.debug(f"Retrieved soc name: {soc_name} for soc id: {soc_id}")
            else:
                self._logger.error(f"Could not determine the soc name for Android device: {self.id}.")

        return soc_name

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
            self._set_device_dependencies()
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
    ) -> DeviceCompletedProcess:
        """
         Executes a command on the device using the android executor.

         Args:
             commands (list): A list of commands to execute.
             device_env_context (DeviceEnvironmentContext): The environment context of the device.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess]): Execution result
        """

        if device_env_context is None:
            device_env_context = DeviceEnvironmentContext(shell=True)

        env_vars = []
        cwd = device_env_context.cwd

        if device_env_context.environment_variables:
            env_vars = [
                f'export {env_var}="{value}"'
                for env_var, value in device_env_context.environment_variables.items()
            ]

        android_shell_commands = [f"cd {cwd}"] + env_vars + commands
        android_shell_command = " && ".join(android_shell_commands)
        device_return = self.executor.execute(android_shell_command, shell=device_env_context.shell)

        if isinstance(device_return, DeviceCompletedProcess):
            self._logger.debug(
                f"Executed command: {android_shell_commands} with return code: {device_return.returncode}"
            )

        return device_return

    def pull(self, src_path: Union[str, PathLike], dst_path: Union[str, PathLike]) -> DeviceReturn:
        """
        Method to pull files from the device.

        Args:
            src_path (Union[str, PathLike]): The source path on the device.
            dst_path (Union[str, PathLike]): The destination path on the local machine.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess]): Result of executing
                                                                               the pull command
        """

        # before pulling, we must resolve paths to be created on host
        path_to_create = self._resolve_host_path_creation(dst_path, src_path)

        path_to_create.resolve().mkdir(exist_ok=True, parents=True)

        return self.executor.pull(str(src_path), str(dst_path))

    def _resolve_host_path_creation(
        self, dst_path: Union[str, PathLike], src_path: Union[str, PathLike]
    ) -> Path:
        """
        Method to resolve the path where the file would be pulled to on the host machine.

        Args:
            dst_path (Union[str, PathLike]): The destination path on the local machine.
            src_path (Union[str, PathLike]): The source path on the device.

        Returns:
            Path: The path where the file would be pulled to on the host machine.
        """
        # assume dst is a directory and its entire path should be created
        create_path = Path(dst_path)

        # unless dst is determined to be a file, in which case only its parent directories should be
        # created.
        if self._check_file_exists_on_device(src_path):
            # check if file path is an existing file or
            # can be assumed to be a file given file extension
            dst_path_is_file = Path(dst_path).suffix or Path(dst_path).is_file()

            # If path names match, then dst_path is a file given src_path is a file
            path_names_match = Path(src_path).name == Path(dst_path).name

            if dst_path_is_file or path_names_match:
                create_path = Path(dst_path).parent

        return create_path

    def push(self, src_path: Union[str, PathLike], dst_path: Union[str, PathLike]) -> DeviceReturn:
        """
        Method to push files or directories to the device.

        Args:
            src_path (Union[str, PathLike]): The source path on the local machine.
            dst_path (Union[str, PathLike]): The destination path on the device.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess]): Result of executing
                                                                               the push command
        """

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Path: {src_path} does not exist")
        elif not os.path.isabs(src_path):
            src_path = os.path.abspath(src_path)

        return self.executor.push(str(src_path), str(dst_path))

    def make_directory(self, dir_name: Union[str, PathLike]) -> DeviceReturn:
        """
        Method to make a directory on an android device.

        Args:
            dir_name (Union[str, PathLike]): The name of the directory to create.

        Returns:
            DeviceReturn:
        """
        mk_dir_command = "mkdir"
        return self.executor.execute(mk_dir_command, args=["-p", str(dir_name)], shell=True)

    def remove(self, target_path: Union[str, PathLike]) -> DeviceReturn:
        """
        Method to remove a file or directory from the device.

        Args:
            target_path (Union[str, PathLike]): The path of the file or directory to remove.

        Returns:
            DeviceReturn: The device code after removing the file or directory.
        """
        rm_rf_command = "rm"
        return self.executor.execute(rm_rf_command, args=["-rf", str(target_path)], shell=True)

    def close(self):
        raise NotImplementedError("The method: close has not been implemented for this class")

    def get_device_log(self, **kwargs) -> Union[bytes, str, None]:
        """
        Get the entire device log, optionally filter by date range and/or a string/regex expression.

        Args:
            **kwargs: Keyword arguments.
                date_range (Optional[Dict|DateRange]): Dict with 'start'/'end' datetimes or a DateRange.
                log_filter (Optional[str]): Regex pattern; any line matching is retained.
                max_lines (Optional[int]): Truncate formatted output to last N lines (after filtering).
                use_device_time_hint (Optional[bool]): If True, attempt to use '-t' optimization. Default False.

        Returns:
            bytes | None: Formatted device log content (UTF-8 bytes) or None on failure.
        """
        date_range = kwargs.get("date_range", None)
        log_filter = kwargs.get("log_filter", None)
        max_lines = kwargs.get("max_lines", None)
        use_device_time_hint = kwargs.get("use_device_time_hint", False)

        command = "logcat"
        args = ["-d", "*:D", "-v", "year"]
        applied_time_hint = False

        # Normalize date_range input
        if date_range is not None:
            if isinstance(date_range, DateRange):
                pass
            elif isinstance(date_range, dict):
                date_range = DateRange.from_dict(date_range)
            else:
                self._logger.error("date_range must be a dict or DateRange instance")
                return None
            if use_device_time_hint:
                try:
                    start_dt = date_range.start
                    start_hint = start_dt.strftime("%m-%d %H:%M:%S.%f")[:-3]
                    args_with_hint = args + ["-t", start_hint]
                    device_return_hint = self.executor.execute(command, args_with_hint)
                    if isinstance(device_return_hint, DeviceCompletedProcess) and device_return_hint.stdout:
                        args = args_with_hint
                        self._logger.debug(
                            f"Applied device time hint '-t {start_hint}' for log retrieval (bytes={len(device_return_hint.stdout)})"
                        )
                    else:
                        self._logger.debug("Time hint produced no output; falling back to full log dump.")
                except Exception as e:  # pragma: no cover
                    self._logger.debug(f"Skipping time hint due to error: {e}")

        device_return = self.executor.execute(command, args)

        if isinstance(device_return, DeviceFailedProcess):
            self._logger.error("Failed to retrieve device log")
            return None
        elif not device_return.stdout:
            return format_device_log(b"No log was found for device", max_lines=max_lines)

        adb_stdout = (
            device_return.stdout
            if isinstance(device_return.stdout, bytes)
            else bytes(device_return.stdout, encoding="utf-8")
        )

        if date_range:
            filtered_before_len = len(adb_stdout)
            adb_stdout = filter_log_by_date_range(adb_stdout, date_range, self._logger)
            if not adb_stdout:
                self._logger.debug(
                    f"No log was found in date range start={date_range.start.isoformat()} end={date_range.end.isoformat()}"
                )

        if log_filter and adb_stdout:
            filtered_before_len = len(adb_stdout)
            adb_stdout = filter_log_by_string(adb_stdout, log_filter, self._logger)
            if not adb_stdout:
                self._logger.debug(f"No log was found with filter: {log_filter}")
            elif self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    f"String filter '{log_filter}' reduced log from "
                    f" {filtered_before_len} -> {len(adb_stdout)} bytes"
                )

        return format_device_log(adb_stdout, max_lines=max_lines)

    @staticmethod
    def _get_available_devices_for_platform(
        connection_type: ConnectionType = ConnectionType.LOCAL,
        hostname: Optional[str] = None,
        ip_addr: Optional[str] = None,
        port: Optional[int] = None,
        platform_type: Optional[
            Union[Literal[DevicePlatformType.ANDROID], Literal[DevicePlatformType.LINUX_EMBEDDED]]
        ] = None,
    ) -> List[DeviceInfo | RemoteDeviceInfo]:
        """
        Static method to get the available devices for a given android platform

        Args:
            connection_type (ConnectionType): The type of connection. Defaults to ConnectionType.LOCAL.
            hostname (Optional[str]): The hostname of the machine connected to the device. Defaults to None.
            ip_addr (Optional[str]): The IP address of the machine connected to the device. Defaults to None
            port (Optional[int]): The port number for the connection. Defaults to None

        Returns:
            List[DeviceInfo]: The available devices.
        """
        available_device_info: List[DeviceInfo | RemoteDeviceInfo] = list()
        platform_type = platform_type if platform_type else DevicePlatformType.ANDROID
        if connection_type == connection_type.LOCAL:
            device_ids = AndroidExecutor().get_available_devices()
            for device_id in device_ids:
                available_device_info.append(
                    DeviceInfo(
                        identifier=device_id, connection_type=connection_type, platform_type=platform_type
                    )
                )
        else:
            if not hostname and ip_addr:
                hostname = ip_addr  # can use either for adb commands
            device_ids = AndroidExecutor().get_available_devices(hostname=hostname, port=port)
            if device_ids is None:
                return available_device_info
            for device_id in device_ids:
                available_device_info.append(
                    RemoteDeviceInfo(
                        platform_type=platform_type,
                        identifier=RemoteDeviceIdentifier(
                            hostname=hostname, ip_addr=ip_addr, port=port, serial_id=device_id
                        ),
                    )
                )

        return available_device_info

    @staticmethod
    def get_available_devices(
        connection_type: ConnectionType = ConnectionType.LOCAL,
        hostname: Optional[str] = None,
        ip_addr: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs,
    ) -> List[DeviceInfo]:
        """
        Static method to get the available devices.

        Args:
            connection_type (ConnectionType): The type of connection. Defaults to ConnectionType.LOCAL.
            hostname (Optional[str]): The hostname of the machine connected to the device. Defaults to None.
            ip_addr (Optional[str]): The IP address of the machine connected to the device. Defaults to None
            port (Optional[int]): The port number for the connection. Defaults to None

        Returns:
            List[DeviceInfo]: The available devices.
        """
        devices = []
        devices = AndroidDevice._get_available_devices_for_platform(
            connection_type, hostname, ip_addr, port, platform_type=DevicePlatformType.ANDROID
        )

        return devices

    def _check_file_exists_on_device(self, file_name: Union[str, PathLike]) -> bool:
        """
        Check whether a file exists on the device.

        Args:
            file_name (Union[str, PathLike]): The name of the file to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return_process = self.executor.execute("test", args=["-f", str(file_name)], shell=True)
        return return_process.returncode == 0
