# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import shutil
from typing import List, Optional, Union

from pydantic import validate_call

from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    ConnectionType,
    DeviceEnvironmentContext,
    DeviceInfo,
)
from qti.aisw.tools.core.utilities.devices.api.executor import Executor
from qti.aisw.tools.core.utilities.devices.utils.device_code import (
    DeviceCode,
    DeviceCompletedProcess,
    DeviceFailedProcess,
    DeviceReturn,
)
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class NativeExecutor(Executor):
    """Base class for interacting with Native device"""

    def __init__(self, logger=None):
        super().__init__()
        self.log_area = LogAreas.register_log_area("NativeExecutor")
        self._logger = (
            logger if logger else QAIRTLogger.register_area_logger(area=self.log_area, level="INFO")
        )

    def execute(self, command: str, args: Optional[List[str]] = None, **kwargs) -> DeviceReturn:
        raise NotImplementedError("Execute method is not implemented for this class")

    @validate_call
    def pull(self, src_path: Union[str, os.PathLike], dst_path: Union[str, os.PathLike]) -> DeviceReturn:
        """Method to pull files from the device. Pull is analogous to push in that the
        device is the local host.

        Args:
            src_path (Union[str, Path]): The source path on the device.
            dst_path (Union[str, Path]): The destination path on the local machine.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        # TODO: Implement support for remote device in which case pull != push
        return self.push(src_path, dst_path)

    @validate_call
    def push(self, src_path: Union[str, os.PathLike], dst_path: Union[str, os.PathLike]) -> DeviceReturn:
        """Method to push files to the device.

        Args:
            src_path (Union[str, Path]): The source path on the local machine.
            dst_path (Union[str, Path]): The destination path on the device.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        if os.path.isdir(src_path):
            return self._copy_directory(src_path, dst_path)
        return self._copy_file(src_path, dst_path)

    @validate_call
    def make_directory(self, dir_name: Union[str, os.PathLike]) -> DeviceReturn:
        """Method to make a directory on the device.

        Args:
            dir_name (Union[str, Path]): The name of the directory to create.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        try:
            os.makedirs(dir_name, exist_ok=True)
            self._logger.debug(f"Created directory: {dir_name}")
        except OSError as err:
            self._logger.error(f"Error creating directory: {dir_name!r}\n str({err})")
            return DeviceFailedProcess(
                args=[str(dir_name)],
                returncode=DeviceCode.DEVICE_UNKNOWN_ERROR,
                stderr=str(err),
                orig_error=err,
            )

        return DeviceCompletedProcess(args=[str(dir_name)], returncode=DeviceCode.DEVICE_SUCCESS)

    @validate_call
    def remove(self, target_path: Union[str, os.PathLike]) -> DeviceReturn:
        """Method to remove a file or directory from the device.

        Args:
            target_path (Union[str, Path]): The path of the file or directory to remove.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        if os.path.isfile(target_path):
            os.remove(target_path)
        elif os.path.isdir(target_path):
            shutil.rmtree(target_path)

        return DeviceCompletedProcess(args=[str(target_path)], returncode=DeviceCode.DEVICE_SUCCESS)

    def _copy_directory(
        self, src_path: Union[str, os.PathLike], dst_path: Union[str, os.PathLike]
    ) -> DeviceReturn:
        """This function copies the content of src_path into dst_path

        Args:
            src_path (Union[str, os.PathLike]): The source path to copy from.
            dst_path (Union[str, os.PathLike]): The destination path to copy to.

        Return:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        if not os.path.exists(src_path):
            self._logger.error(f"Source path: {src_path!r} not found for Device")
            return DeviceFailedProcess(
                args=[str(src_path), str(dst_path)], returncode=DeviceCode.DEVICE_UNKNOWN_ERROR
            )
        try:
            # Ensure src_path does not have a trailing slash
            src_path = os.path.abspath(src_path)
            dir_name = os.path.basename(src_path)

            if not dir_name:  # copy only contents of src_path
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                # copy the entire src_path
                shutil.copytree(src_path, os.path.join(dst_path, dir_name), dirs_exist_ok=True)
        except shutil.Error as err:
            self._logger.error(f"File: {src_path!r} could not be copied  to location: {dst_path!r}")
            return DeviceFailedProcess(
                args=[str(src_path), str(dst_path)],
                returncode=DeviceCode.DEVICE_UNKNOWN_ERROR,
                orig_error=err,
                stderr=str(err),
            )

        return DeviceCompletedProcess(args=[str(src_path), str(dst_path)])

    def _copy_file(
        self, src_path: Union[str, os.PathLike], dst_path: Union[str, os.PathLike]
    ) -> DeviceReturn:
        """This method copies the file from src_path to dst_path

        Args:
            src_path (Union[str, os.PathLike]): Source path of the file to be copied.
            dst_path (Union[str, os.PathLike]): Destination path of the file to be copied.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        if not os.path.exists(src_path):
            self._logger.error(f"Source path: {src_path} not found for Device")

        if os.path.isfile(src_path):
            if not os.path.exists(os.path.dirname(dst_path)):
                self.make_directory(os.path.dirname(dst_path))
            shutil.copy(src_path, dst_path)

            return DeviceCompletedProcess(args=[str(src_path), str(dst_path)])

        return DeviceFailedProcess(
            args=[str(src_path), str(dst_path)], returncode=DeviceCode.DEVICE_UNKNOWN_ERROR
        )

    @staticmethod
    def close(self) -> DeviceReturn:
        raise NotImplementedError("Close is not valid for class: {self.__name__}")

    @staticmethod
    def get_available_devices(
        connection_type: ConnectionType = ConnectionType.LOCAL, **kwargs
    ) -> Optional[List[DeviceInfo]]:
        raise NotImplementedError("get available devices is not implemented for this class")
