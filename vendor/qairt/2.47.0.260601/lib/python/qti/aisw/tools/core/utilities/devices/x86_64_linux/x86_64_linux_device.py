# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from os import PathLike
from typing import List, Optional, Union

from pydantic import validate_call

from qti.aisw.tools.core.utilities.devices.api.device_interface import *
from qti.aisw.tools.core.utilities.devices.x86_64_linux.x86_64_linux_executor import *
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class X86LinuxDevice(DeviceInterface):
    def __init__(self, device_info: Optional[DeviceInfo] = None, logger: Optional[Any] = None):
        """Initializes the device interface for X86 Linux

        Args:
            device_info (Optional[DeviceInfo]): The device info object. Defaults to None.
            logger (Optional[Any]): The logger for the device interface. Defaults to None.
        """
        super().__init__(device_info=device_info, logger=logger)
        self._executor = X86LinuxExecutor()
        if not self._logger:
            self.set_logger(QAIRTLogger.get_logger(name="X86LinuxDevice", level="INFO"))

    @property
    def executor(self):
        return self._executor

    @property
    def executors(self):
        return [self.executor]

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        return self.device_info

    @device_info.setter
    def device_info(self, device_info: DeviceInfo):
        self.device_info = device_info

    def execute(
        self, commands: List[str], device_env_context: Optional[DeviceEnvironmentContext] = None
    ) -> DeviceReturn:
        """Method to execute commands on the device.

        Args:
            commands (List[str]): The commands to execute.
            device_env_context (Optional[DeviceEnvironmentContext]): The device environment context. Defaults to None.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        if device_env_context is None:
            device_env_context = DeviceEnvironmentContext()

        env_vars = []
        cwd = device_env_context.cwd

        if device_env_context.environment_variables:
            env_vars = [
                f'export {env_var}="{value}"'
                for env_var, value in device_env_context.environment_variables.items()
            ]

        x86_shell_commands = [f"cd {cwd}"] + env_vars + commands
        x86_shell_command = " && ".join(x86_shell_commands)
        return self.executor.execute(
            x86_shell_command, shell=device_env_context.shell, cwd=device_env_context.cwd
        )

    @validate_call
    def pull(self, src_path: Union[str, PathLike], dst_path: Union[str, PathLike]) -> DeviceReturn:
        """Method to pull files from the device. Pull is analogous to push in that the
        device is the local host.

        Args:
            src_path (Union[str, PathLike]): The source path on the device.
            dst_path (Union[str, PathLike]): The destination path on the local machine.
        """
        # TODO: Implement support for remote device in which case pull != push
        return self.executor.pull(src_path, dst_path)

    @validate_call
    def push(self, src_path: Union[str, PathLike], dst_path: Union[str, PathLike]) -> DeviceReturn:
        """Method to push files to the device.

        Args:
            src_path (Union[str, PathLike]): The source path on the local machine.
            dst_path (Union[str, PathLike]): The destination path on the device.
        """
        if not os.path.exists(src_path):
            raise FileNotFoundError("PathLike: {src_path} does not exist")

        return self.executor.push(src_path, dst_path)

    @validate_call
    def make_directory(self, dir_name: Union[str, PathLike]) -> DeviceReturn:
        """Method to make a directory on the device.

        Args:
            dir_name (Union[str, PathLike]): The name of the directory to create.
        """
        return self.executor.make_directory(dir_name)

    @validate_call
    def remove(self, target_path: Union[str, PathLike]) -> DeviceReturn:
        """Method to remove a file or directory from the device.

        Args:
            target_path (Union[str, PathLike]): The path of the file or directory to remove.

        Returns:
            DeviceCompletedProcess after removing the file or directory.
        """
        return self.executor.remove(target_path)

    def close(self) -> DeviceReturn:
        raise NotImplementedError("The method: close is not applicable to this class")

    @staticmethod
    def get_available_devices(
        connection_type: ConnectionType = ConnectionType.LOCAL, **kwargs
    ) -> List[DeviceInfo]:
        """This static method returns available devices on the system

        Args:
            connection_type (ConnectionType): The type of connection. Defaults to ConnectionType.LOCAL

        Returns:
            list: A list of DeviceInfo objects representing the available devices on the system
        """
        if connection_type == ConnectionType.REMOTE:
            return []
        return [DeviceInfo(platform_type=DevicePlatformType.X86_64_LINUX)]
