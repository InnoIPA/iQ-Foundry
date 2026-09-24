# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import shutil
from typing import List, Optional

from qti.aisw.tools.core.utilities.devices.protocol_helpers.protocol_helper import ProtocolHelper
from qti.aisw.tools.core.utilities.devices.utils import subprocess_helper
from qti.aisw.tools.core.utilities.devices.utils.device_code import *
from qti.aisw.tools.core.utilities.devices.utils.device_utils import format_output_as_list
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class AdbProtocolHelper(ProtocolHelper):
    """
    Helper class to enable Android device interaction using ADB

    Attributes:
        _EXECUTABLE: The path to the executable for ADB. Defaults to a path
                    that is presumed to be added to the local environment.
        _ADB_DEFAULT_TIMEOUT: Determines how long an adb call should await a result
                              before returning
        _logger: The logger to use
    """

    _EXECUTABLE = "adb"
    _ADB_DEFAULT_TIMEOUT = 300
    _logger = QAIRTLogger.get_logger(name=__name__, level="INFO")

    def __init__(self, adb_executable_path: Optional[str] = None):
        """
        Initialize an instance of ADB protocol helper.

        Args:
            adb_executable_path (Optional[str]): A path to an adb executable.
        """
        if adb_executable_path:
            # check that the provided path exists
            if os.path.exists(adb_executable_path):
                self._EXECUTABLE = os.path.abspath(adb_executable_path)
            else:
                raise FileNotFoundError(
                    f"The provided adb executable path does not exist: {adb_executable_path}"
                )
        else:
            # check default path is retrievable
            self._check_adb_exists()

        self._ADB_DEFAULT_TIMEOUT = int(os.getenv("ADB_DEFAULT_TIMEOUT", self._ADB_DEFAULT_TIMEOUT))

        self._logger.debug("AdbProtocolHelper initialized with executable: {}".format(self._EXECUTABLE))
        self._logger.debug("Default ADB timeout set to: {}".format(self._ADB_DEFAULT_TIMEOUT))

    @classmethod
    def _check_adb_exists(cls) -> bool:
        """Check if adb executable is present"""
        adb_executable_found = shutil.which(cls._EXECUTABLE)
        if not adb_executable_found:
            cls._logger.error(
                "No Adb executable was found at: {}, please set the path "
                "manually or ensure <adb_path> is discoverable from <PATH>.".format(os.environ.get("PATH"))
            )
            return False
        return True

    def execute(
        self,
        command: str,
        args: Optional[List[str]] = None,
        *,
        device_id: Optional[str] = None,
        hostname: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> DeviceReturn:
        """
        Executes a command on the device.

        Args:
           command (str): The command to execute.
           args (Optional[List[str]]): A list of arguments for the command.
           device_id (Optional[str]): The ID of the device.
           hostname (Optional[str]): The hostname that the device is connected to.
           port (Optional[int]): The port number to connect to
           timeout (Optional[int]): Specifies time to wait before a result is returned

        Returns:
           DeviceReturn: The result of the command execution.
        """

        if args is None:
            args = []

        if device_id is None:
            adb_command_args = [command] + args
        else:
            adb_command_args = ["-s", device_id, command] + args

        if hostname:
            adb_command_args = ["-H", hostname] + adb_command_args

        if port:
            adb_command_args = ["-P", str(port)] + adb_command_args

        # Check env again if not provided; coerce to int safely
        if timeout is None:
            env_val = os.getenv("ADB_DEFAULT_TIMEOUT")
            try:
                timeout = int(env_val) if env_val is not None else self._ADB_DEFAULT_TIMEOUT
            except ValueError:
                self._logger.warning(
                    f"Invalid ADB_DEFAULT_TIMEOUT value '{env_val}', falling back to {self._ADB_DEFAULT_TIMEOUT}"
                )
                timeout = self._ADB_DEFAULT_TIMEOUT

        device_return = subprocess_helper.execute(
            self._EXECUTABLE,
            adb_command_args,
            timeout=timeout,
            logger=self._logger,
        )

        return device_return

    def shell(
        self,
        command: str,
        args: Optional[List[str]] = None,
        *,
        device_id: Optional[str] = None,
        hostname: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> DeviceReturn:
        """
        Executes a shell command on the device.

        Args:
           command (str): The command to execute.
           args (Optional[List[str]]): A list of arguments for the command.
           device_id (Optional[str]): The ID of the device.
           hostname (Optional[str]): The hostname of the device.
           port(Optional[int]): The port number to connect to
           timeout (Optional[int]): Specifies time to wait before a result is returned

        Returns:
            DeviceReturn: The result of the command execution.
        """
        if args is None:
            args = []

        # print the return code of the command run on device on stdout, so it can be parsed on
        # the host. The command: 'adb shell' will return a return code = 0 even if the command
        # fails on device TODO: This needs to be re-verified after refactor, seems like a bug in
        #  adb. May be able to remove this check completely
        adb_command_args = [f"{command} {' '.join(args)}; echo '\n'$?"]
        device_return = self.execute(
            "shell", adb_command_args, device_id=device_id, hostname=hostname, timeout=timeout, port=port
        )
        stdout_lines = format_output_as_list(device_return.stdout)

        if device_return.returncode == 0:
            if len(stdout_lines) > 0:
                try:
                    # retrieve return code from stdout
                    device_return.returncode = int(stdout_lines[-1])
                    # strip return code, new lines and spacing from stdout
                    device_return.stdout = "".join(
                        stdout_lines[:-1] if len(stdout_lines) > 1 else stdout_lines
                    )
                except ValueError as ex:
                    self._logger.error(str(ex))
                    device_return.returncode = -1
                    device_return.stdout += str(ex)
            else:
                device_return.returncode = -1

        if device_return.returncode == -1:
            self._logger.error(
                f"Shell command:{device_return.args} failed with error: {device_return.returncode}"
            )
            device_return = DeviceFailedProcess(
                args=device_return.args,
                returncode=device_return.returncode,
                stdout=device_return.stdout,
                stderr=device_return.stderr,
                timeout=device_return.timeout,
            )

        return device_return

    @classmethod
    def get_available_devices(cls, hostname: Optional[str] = None, port: Optional[int] = None) -> List[str]:
        """
        Gets the list of available devices.

        Args:
            hostname (Optional[str]): The hostname to query for available devices. Defaults to
                                      localhost
            port (int): The port number to use when connecting to the adb server

        Returns:
            List[str]: The list of available devices, or empty list if none are available.
        """

        timeout_value = 30
        device_return = AdbProtocolHelper().execute(
            command="devices", timeout=timeout_value, hostname=hostname, port=port
        )
        devices = []

        if isinstance(device_return, DeviceFailedProcess) or not device_return.stdout:
            cls._logger.warning(
                f"Could not retrieve list of connected adb devices, "
                f"stdout: {device_return.stdout}, stderr: {device_return.stderr}"
            )
        else:
            lines = device_return.stdout.splitlines()

            # Skip the first line, which is a header row
            if len(lines) > 1:
                devices = [line.split("\t")[0] for line in lines[1:] if line]

        return devices
