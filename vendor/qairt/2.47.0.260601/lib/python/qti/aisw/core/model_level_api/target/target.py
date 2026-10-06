#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Target API definition"""

from os import PathLike
from abc import ABC, abstractmethod
import platform
from typing import Union, Dict, Optional, List, Tuple, AnyStr
from qti.aisw.tools.core.utilities.devices.api.device_interface import (DeviceInterface,
                                                                        DeviceEnvironmentContext)
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.tools.core.utilities.devices.utils.device_code import DeviceFailedProcess


class Target(ABC):
    """ Target API that should be sub-classed for each platform e.x Android"""
    file_sep = '/'
    def __init__(self):
        # A device instance is required to use this class
        self._device: Union[None, DeviceInterface] = None

    @property
    @abstractmethod
    def target_name(self) -> str:
        pass

    @property
    @abstractmethod
    def target_platform_type(self) -> DevicePlatformType:
        pass

    @property
    def device(self) -> DeviceInterface:
        if self._device is None:
            raise RuntimeError('The device property of this object must be set before use')
        return self._device

    def pull(self, src: Union[str, PathLike], dst: Union[str, PathLike]) -> None:
        """
        Method to pull artifacts (files and directories) from the device.

        Args:
            src (Union[str, PathLike]): The source path on the device.
            dst (Union[str, PathLike]): The destination path on the local machine.

        Raises:
            RuntimeError: If any error occurs during the pull operation.
        """
        device_return = self.device.pull(src, dst)
        if isinstance(device_return, DeviceFailedProcess):
            raise RuntimeError(device_return.stderr)

    def push(self, src: Union[str, PathLike], dst: Union[str, PathLike]) -> None:
        """
        Method to push artifacts (files and directories)to the device.

        Args:
            src (Union[str, PathLike]): The source path on the local machine.
            dst (Union[str, PathLike]): The destination path on the device.

        Raises:
            RuntimeError: If any error occurs during the push operation.
        """
        device_return = self.device.push(src, dst)
        if isinstance(device_return, DeviceFailedProcess):
            raise RuntimeError(device_return.stderr)

    def run_commands(self, commands: List[str], cwd: Union[str, PathLike] = '.',
                     env: Optional[Dict[str, str]] = None) -> \
            Tuple[int, Optional[AnyStr], Optional[AnyStr]]:
        """
        Method to execute a list of commands on the device.

        Args:
            commands (List[str]): A list of strings representing the commands to be executed.
            cwd (Union[str, PathLike]): The working directory for the commands.
                                                   Defaults to '.'
            env: Optional[Dict[str, str]]: A dictionary containing the environment variables to be
                                           used for the commands. Defaults to None

        Returns:
            Tuple[int, Optional[AnyStr], Optional[AnyStr]]: Tuple containing the status code,
            stdout, and stderr of the commands which where run
        """
        if env is None:
            env = {}

        env_context = DeviceEnvironmentContext(cwd=cwd, environment_variables=env, shell=True)
        device_return = self.device.execute(commands, device_env_context=env_context)
        return device_return.returncode, device_return.stdout, device_return.stderr

    def run_command(self, command: str, cwd: Union[str, PathLike] = '.',
                    env: Optional[Dict[str, str]] = None) -> \
            Tuple[int, Optional[AnyStr], Optional[AnyStr]]:
        """
        Method to execute a command on the device.

        Args:
            command (str): A string representing the command to be executed.
            cwd: Union[str, PathLike]: The working directory for the commands.
                                      Defaults to '.'
            env: Optional[Dict[str, str]]: A dictionary containing the environment variables to be
                                           used for the commands. Defaults to None

        Returns:
            Tuple[int, Optional[AnyStr], Optional[AnyStr]]: Tuple containing the status code,
            stdout, and stderr of the command which was run
        """
        return self.run_commands([command], cwd, env)

    def make_directory(self, directory: Union[str, PathLike]) -> None:
        """
        Method to make a directory on the device.

        Args:
          directory (Union[str, PathLike]): The name or path of the directory to create.

        Raises:
            RuntimeError: If any error occurs during the pull operation.
        """
        device_return = self.device.make_directory(directory)
        if isinstance(device_return, DeviceFailedProcess):
            raise RuntimeError(device_return.stderr)

    def remove(self, path: Union[str, PathLike]) -> None:
        """
        Method to remove a file or directory from the device.

        Args:
           path (Union[str, PathLike]): The path of the file or directory to remove.

        Raises:
            RuntimeError: If any error occurs during the pull operation.
        """
        device_return = self.device.remove(path)
        if isinstance(device_return, DeviceFailedProcess):
            raise RuntimeError(device_return.stderr)

    @abstractmethod
    def get_default_executor_cls(self):
        pass


    @staticmethod
    def create_host_target() -> 'Target':
        """
        Factory method to create the appropriate Target subclass based on the host machine's platform and architecture.

        Returns:
            Target: An instance of the appropriate Target subclass for the host machine.

        Raises:
            RuntimeError: If the platform or architecture is unsupported.
        """

        system = platform.system()
        arch = platform.machine()
        processor = platform.processor()

        if system == "Linux":
            if arch == "x86_64":
                from qti.aisw.core.model_level_api.target.x86_linux import X86LinuxTarget
                return X86LinuxTarget()
            elif arch == "aarch64":
                from qti.aisw.core.model_level_api.target.oelinux import OELinuxTarget
                return OELinuxTarget()
            else:
                raise RuntimeError(f"Unsupported platform: {system} {arch}")
        elif system == "Windows":
            if "AMD64" in processor or "Intel64" in processor:
                from qti.aisw.core.model_level_api.target.x86_windows import X86WindowsTarget
                return X86WindowsTarget()
            elif "ARMv8" in processor:
                from qti.aisw.core.model_level_api.target.arm_windows import WOSTarget
                return WOSTarget()
            else:
                raise RuntimeError(f"Unsupported platform: {system} {arch} {processor}")
        else:
            raise RuntimeError(f"Unsupported platform: {system} {arch} {processor}")
