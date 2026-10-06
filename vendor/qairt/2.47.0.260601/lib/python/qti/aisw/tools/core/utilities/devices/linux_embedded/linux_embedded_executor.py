# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os

from qti.aisw.tools.core.utilities.devices.api.executor import *
from qti.aisw.tools.core.utilities.devices.api.native_executor import NativeExecutor
from qti.aisw.tools.core.utilities.devices.utils import subprocess_helper
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class LinuxEmbeddedExecutor(NativeExecutor):
    """Class for interacting with Linux Embedded device locally (native execution)"""

    def __init__(self, logger=None):
        if logger is None:
            self._logger = QAIRTLogger.get_logger(name="LinuxEmbeddedExecutor", level="INFO")
        else:
            self._logger = logger
        super().__init__(logger=self._logger)

    def execute(self, command: str, args: Optional[List[str]] = None, **kwargs) -> DeviceReturn:
        """Execute the command on Linux Embedded device natively

        Args:
            command (str): The command to run.
            args (Optional[List[str]], optional): List of arguments for the command. Defaults to None.
            kwargs: Additional keyword arguments.
                shell (bool, optional): Whether to use shell. Defaults to True.
                cwd (str): Current working directory. Defaults to os.getcwd()

        Returns:
            DeviceReturn Union[DeviceCompletedProcess, DeviceFailedProcess]
        """
        shell = kwargs.get("shell", True)
        cwd = kwargs.get("cwd", os.getcwd())
        return subprocess_helper.execute(command, args, cwd=cwd, shell=shell)
