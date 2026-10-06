# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import ftplib
import os
from typing import ClassVar, Dict, List, Optional

from qti.aisw.tools.core.utilities.devices.protocol_helpers.protocol_helper import ProtocolHelper
from qti.aisw.tools.core.utilities.devices.utils import ping
from qti.aisw.tools.core.utilities.devices.utils.device_code import (
    DeviceCompletedProcess,
    DeviceFailedProcess,
    DeviceReturn,
)
from qti.aisw.tools.core.utilities.devices.utils.device_utils import SingletonABC
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


def _is_file(connection, target_path):
    try:
        _ = list(connection.mlsd(target_path))  # evaluate generated object to get error
        return False
    except ftplib.all_errors as e:
        if "Not a directory" in str(e):
            return True
        raise e


class FTPProtocolHelper(ProtocolHelper, metaclass=SingletonABC):
    _logger = QAIRTLogger.get_logger(name=__name__, level="INFO")
    _FTP_DEFAULT_TIMEOUT = 10
    _FTP_DEFAULT_PORT = 21
    _CONNECTED_INSTANCE_CACHE: ClassVar[Dict[str, ftplib.FTP]] = {}

    @classmethod
    def setup_ftp_connection(
        cls,
        host: str,
        username: str = "",
        password: str = "",
        *,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
        debug_level: int = 0,
    ) -> Optional[ftplib.FTP]:
        """
        This method sets up an FTP connection to the specified host and returns the
        connection object

        Args:
            host (str): The host or IP address of the target connection.
            username (str): The username used to authenticate with the target connection.
                      Defaults to empty string.
            password (str): The password used to authenticate with the target connection.
                      Defaults to empty string.
            port (Optional[int]): The port number to connect to.
                                  Defaults to the default FTP port (21).
            timeout (int): The socket timeout value in seconds. Defaults to _FTP_DEFAULT_TIMEOUT
            debug_level (int): The debug level for the FTP connection instance. Defaults to 0
                               which means no debug info.

        Returns:
            Optional[ftplib.FTP]: An FTP connection object if successful, otherwise None.
        """
        if host in cls._CONNECTED_INSTANCE_CACHE:
            cls._logger.info(f"Connection has already been established for host: {host}")
            return cls._CONNECTED_INSTANCE_CACHE[host]

        connection = ftplib.FTP()
        port = port if port else cls._FTP_DEFAULT_PORT
        timeout = timeout if timeout else cls._FTP_DEFAULT_TIMEOUT
        try:
            connection.connect(host, port, timeout)
            connection.login(username, password)
            connection.set_debuglevel(debug_level)
            cls._logger.debug(f"FTP connection successful. {connection.welcome}")
            cls._CONNECTED_INSTANCE_CACHE[host] = connection
        except ftplib.all_errors as e:
            cls._logger.error(f"Failed to establish FTP connection for {host} with error: {e!s}")
            return None
        return connection

    @classmethod
    def is_connected(cls, connection: ftplib.FTP) -> bool:
        """This method checks whether the FTP instance is connected or not.

        Args:
            connection (ftplib.FTP): The FTP connection object to check.

        Returns:
            bool: True if the connection is active, False otherwise.
        """
        try:
            connection.voidcmd("NOOP")
            return True
        except ftplib.all_errors:
            return False

    @classmethod
    def push(
        cls,
        src_path: str | os.PathLike,
        dst_path: str | os.PathLike,
        connection: ftplib.FTP,
        *,
        cwd: Optional[str] = None,
    ) -> DeviceReturn:
        """
        This method pushes files from the specified source path to the specified destination path
        on the FTP server. If the destination path does not exist, it will be created automatically.

        Args:
            src_path (str | os.PathLike): The source path on the local system to copy files from.
            dst_path (str | os.PathLike): The destination path on the FTP server to copy files to.
            connection (ftplib.FTP): An FTP connection object.
            cwd (str | os.PathLike): The current working directory on the FTP server.
                                     Defaults to None.

        Returns:
            DeviceReturn: A DeviceCompletedProcess object if successful,
                          otherwise a DeviceFailedProcess object.
        """

        def push_file(local_file_path_, remote_file_path_):
            with open(local_file_path_, "rb") as local_file:
                connection.storbinary(f"STOR {remote_file_path_}", local_file)

        def push_dir(local_dir_path_, remote_dir_path_):
            cls.make_directory(remote_dir_path_, connection=connection)
            paths = os.listdir(local_dir_path_)

            if not paths:
                return

            while paths:
                path_ = paths.pop()
                full_path = os.path.join(local_dir_path_, path_)
                if os.path.isfile(full_path):
                    push_file(full_path, os.path.join(remote_dir_path_, path_))
                elif os.path.isdir(full_path):
                    push_dir(full_path, os.path.join(remote_dir_path_, path_))
                else:
                    raise TypeError(f"{full_path} is not a file or directory")

        try:
            if cwd:
                connection.cwd(cwd)

            # Ensure the source path exists
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Error: Source path '{src_path}' does not exist.")

            # Determine if source is a file or directory
            if os.path.isfile(src_path):
                # upload the file
                with open(src_path, "rb") as _:
                    if os.path.basename(src_path) != os.path.basename(dst_path):
                        if not _is_file(connection, dst_path):
                            dst_path = os.path.join(dst_path, os.path.basename(src_path))
                    push_file(src_path, dst_path)
                    cls._logger.debug(
                        f"File: {src_path} successfully uploaded to {connection.host}:{dst_path}"
                    )
            elif os.path.isdir(src_path):
                # Upload files from the local directory recursively
                push_dir(src_path, dst_path)
                cls._logger.debug(f"Directory '{src_path}' uploaded to '{dst_path}'")
            else:
                cls._logger.error("Source path is neither a file nor a directory")
                return DeviceFailedProcess(args=[f"push {src_path}, {dst_path}"])

            # Set permissions to 755 (read/write/execute for owner,
            # read/execute for group and others)
            connection.sendcmd(f"SITE CHMOD 755 {dst_path}")

        except ftplib.all_errors as e:
            cls._logger.error(e)
            return DeviceFailedProcess(
                args=[f"push {src_path}, {dst_path}"], stdout=None, stderr=str(e), orig_error=e
            )

        return DeviceCompletedProcess(args=[f"push {src_path}, {dst_path}"], stdout=None)

    @classmethod
    def pull(
        cls,
        src_path: str | os.PathLike,
        dst_path: str | os.PathLike,
        connection: ftplib.FTP,
        *,
        cwd: Optional[str] = None,
    ) -> DeviceReturn:
        """
        This method downloads files from the specified source path on the FTP server to the
        specified destination path. If the destination path does not exist, it will be created
        automatically.

        Args:
            src_path (str | os.PathLike): The source path on the FTP server to download files from.
            dst_path (str | os.PathLike):: The destination path on the local system to save the
                                           downloaded files.
            connection (str | os.PathLike):: An FTP connection object.
            cwd (str | os.PathLike):: The current working directory on the FTP server.
                                      Defaults to None.

        Returns:
            DeviceReturn: A DeviceCompletedProcess object if successful, otherwise a
                          DeviceFailedProcess object.
        """

        def copy_file(source_file, target_file):
            with open(target_file, "wb") as local_file:
                connection.retrbinary(f"RETR {source_file}", local_file.write)

        def copy_directory(source_dir, target_dir):
            try:
                # Create the local directory (if it doesn't exist)
                os.makedirs(target_dir, exist_ok=True)

                connection.cwd(source_dir)
                items = connection.nlst()

                while items:
                    item = items.pop()
                    if _is_file(connection, f"{source_dir}/{item}"):
                        copy_file(f"{source_dir}/{item}", os.path.join(target_dir, item))
                    else:
                        copy_directory(f"{source_dir}/{item}", os.path.join(target_dir, item))

            except ftplib.all_errors as e:
                if "450 No files found" in str(e):
                    # empty directory, return
                    return
                raise e

        try:
            if cwd:
                connection.cwd(cwd)

            # Check if the destination path exists locally
            if os.path.exists(dst_path):
                cls._logger.warning(f"Destination path {dst_path} already exists.  It may be overwritten")

            if _is_file(connection, src_path):
                copy_file(src_path, dst_path)
            else:
                copy_directory(src_path, dst_path)
            cls._logger.debug(f"Directory '{src_path}' downloaded to '{dst_path}'")

        except ftplib.all_errors as e:
            cls._logger.error(e)
            return DeviceFailedProcess(
                args=[f"pull {src_path}, {dst_path}"], stdout=None, stderr=str(e), orig_error=e
            )

        return DeviceCompletedProcess(args=[f"pull {src_path}, {dst_path}"], stdout=None)

    @classmethod
    def make_directory(cls, dst_path: str, connection: ftplib.FTP) -> Optional[DeviceReturn]:
        """
        This function creates an empty directory on the FTP server at the specified location.
        If the directory already exists, this function will do nothing.

        Args:
            dst_path (str): The path where the new directory should be created on the FTP server.
            connection (ftplib.FTP): An FTP connection object to use
        Returns:
            DeviceReturn: A DeviceCompletedProcess object if successful, a DeviceFailedProcess
                          object if a failure
            occurs or none.
        """
        try:
            connection.mkd(dst_path)
            cls._logger.debug(f"Remote directory '{dst_path}' created on {connection.host}")
        except ftplib.all_errors as e:
            if e.args[0][:3] == "550" and "File exists" in str(e):  # Directory already exists
                cls._logger.debug(
                    f"Remote directory '{dst_path}' already exists on {connection.host}. Skipping creation."
                )
                return None
            cls._logger.error(f"Error creating remote directory: {str(e)}")
            return DeviceFailedProcess(
                args=[f"ftplib.FTP.mkd, {dst_path}"], stdout=None, stderr=str(e), orig_error=e
            )

        return DeviceCompletedProcess(args=[f"ftplib.FTP.mkd {dst_path}"], stdout=None)

    @classmethod
    def get_available_devices(cls, destination: str, *_destinations) -> List[str]:
        """
        This method searches all provided destination strings for available devices using the ping
        command.

        Args:
            destination (str): DNS name or ip address
            _destinations (List(str)): DNS name(s) or ip addresses

        Returns:
           List[str]: The list of available devices, or empty list if none are available.
        """

        available_devices, unavailable_devices = ping.get_available_destinations(destination, *_destinations)

        # log un-available devices
        if unavailable_devices:
            cls._logger.debug(f"The following requested devices were not found: {unavailable_devices}")

        return available_devices

    @classmethod
    def remove(cls, target_path: str, connection: ftplib.FTP) -> DeviceReturn:
        """
        This method removes a remote file or directory from the FTP server.

        Args:
            target_path (str): The path of the file or directory on the FTP server.
            connection (ftplib.FTP): An FTP connection object.

        Returns:
            DeviceReturn: A DeviceCompletedProcess object if successful, a DeviceFailedProcess
                          object
        """

        def remove_dir(dir_name):
            paths_in_dir = []
            for path_in_dir in list(connection.mlsd(dir_name)):
                if path_in_dir[1]["type"] not in ["pdir", "cdir"]:
                    paths_in_dir.append(path_in_dir)

            while paths_in_dir:
                path_to_remove = paths_in_dir.pop()
                if path_to_remove[1]["type"] == "file":
                    connection.delete(dir_name + "/" + path_to_remove[0])
                else:
                    remove_dir(dir_name + "/" + path_to_remove[0])
            connection.rmd(dir_name)

        try:
            # Connect to the FTP server
            # Check if the file or directory exists
            connection.nlst(target_path)
        except ftplib.error_temp as e:
            cls._logger.debug(f"{target_path} does not exist on {connection.host}.")
            return DeviceFailedProcess(
                args=[f"ftplib.FTP.delete {target_path}"], stdout=None, stderr=str(e), orig_error=e
            )

        try:
            # get file facts
            target_path_facts = list(connection.mlsd(target_path))
            if target_path_facts[0][1]["type"] in ("dir", "cdir"):
                remove_dir(target_path_facts[0][0])
            else:
                connection.delete(target_path)
        except ftplib.all_errors as e:
            cls._logger.error(f"Error removing remote directory: {str(e)}")
            return DeviceFailedProcess(
                args=[f"ftplib.FTP.delete {target_path}"], stdout=None, stderr=str(e), orig_error=e
            )

        return DeviceCompletedProcess(args=[f"ftplib.FTP.delete {target_path}"], stdout=None)

    @classmethod
    def close(cls, connection: ftplib.FTP) -> DeviceReturn:
        """
        This method closes an existing FTP connection.

        Args:
            connection (ftplib.FTP): An FTP connection object.

        Returns:
            DeviceReturn: A DeviceCompletedProcess object if successful, a DeviceFailedProcess if
            a failure occurs
        """
        try:
            if connection in cls._CONNECTED_INSTANCE_CACHE.values():
                del cls._CONNECTED_INSTANCE_CACHE[connection.host]
            connection.quit()
        except ftplib.all_errors as e:
            cls._logger.error(f"Error closing connection: {str(e)}")
            return DeviceFailedProcess(args=["ftplib.FTP.quit"], stdout=None, stderr=str(e), orig_error=e)

        return DeviceCompletedProcess(args=["ftplib.FTP.quit"], stdout=None)
