# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
import stat
from pathlib import Path, PosixPath, PurePath, PureWindowsPath
from typing import ClassVar, Dict, List, Optional

import paramiko

from qti.aisw.tools.core.utilities.devices.utils import ping
from qti.aisw.tools.core.utilities.devices.utils.device_code import *
from qti.aisw.tools.core.utilities.devices.utils.device_utils import SingletonABC
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger

from .protocol_helper import ProtocolHelper


def remote_dir_exists(client: paramiko.SFTPClient, remote_path: str):
    try:
        attr = client.lstat(remote_path)
        return stat.S_ISDIR(attr.st_mode) if attr.st_mode else False
    except (FileNotFoundError, IOError):
        return False


def remote_file_exists(client: paramiko.SFTPClient, remote_path: str):
    try:
        if attr := client.lstat(remote_path):
            return not stat.S_ISDIR(attr.st_mode) if attr.st_mode else False
    except (FileNotFoundError, IOError):
        return False


def format_remote_path(remote_path: str):
    # ensure mixed paths have only one type of separator, \\ takes precedence
    if "\\" in remote_path:
        return str(PureWindowsPath(remote_path))
    else:
        return str(PurePath(remote_path))


def get_remote_path_parts(remote_path: str):
    if "\\" in remote_path:
        remote_win_path = PureWindowsPath(remote_path)
        return str(remote_win_path.parent), str(remote_win_path.name)
    else:
        remote_path_var = PurePath(remote_path)
        return str(remote_path_var.parent), str(remote_path_var.name)


class SSHProtocolHelper(ProtocolHelper, metaclass=SingletonABC):
    _logger = QAIRTLogger.get_logger(name=__name__, level="INFO")
    _SSH_DEFAULT_TIMEOUT = 30
    _SSH_DEFAULT_PORT = 22
    _CONNECTED_INSTANCE_CACHE: ClassVar[Dict[str, paramiko.SSHClient]] = {}

    @classmethod
    def setup_ssh_connection(
        cls,
        host: str,
        username: str = "",
        password: str = "",
        *,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Optional[paramiko.SSHClient]:
        """
        This method sets up an SSH connection to the specified host and returns the
        connection object

        Args:
            host: The host or IP address of the target connection.
            username: The username used to authenticate with the target connection.
                      Defaults to empty string.
            password: The password used to authenticate with the target connection.
                      Defaults to empty string.
            port: The port number to connect to. Defaults to the default SSH port (22).
            timeout: The socket timeout value in seconds. Defaults to _SSH_DEFAULT_TIMEOUT

        Returns:
            Optional[paramiko.SSHClient]: An SSH connection object if successful, otherwise None.
        """
        if host in cls._CONNECTED_INSTANCE_CACHE:
            cls._logger.info(f"Connection has already been established for host: {host}")
            return cls._CONNECTED_INSTANCE_CACHE[host]

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        port = port if port else cls._SSH_DEFAULT_PORT
        timeout = timeout if timeout else cls._SSH_DEFAULT_TIMEOUT
        try:
            client.connect(
                host, port=port, username=username, password=password, allow_agent=False, timeout=timeout
            )
            cls._logger.debug(f"SSH connection successful. {host}")
            cls._CONNECTED_INSTANCE_CACHE[host] = client
            return client
        except (paramiko.AuthenticationException, paramiko.SSHException) as e:
            cls._logger.error(f"Authentication failed when connecting to {host} : {e}")
        except Exception as e:
            cls._logger.error(f"Unable to connect to host: {host} due to error: {e}")
        return None

    @classmethod
    def is_connected(cls, client: paramiko.SSHClient) -> bool:
        transport = client.get_transport()
        return transport is not None and transport.is_active()

    @classmethod
    def get_ftp_client(cls, client: paramiko.SSHClient) -> Optional[paramiko.SFTPClient]:
        if cls.is_connected(client):
            try:
                sftp_client = client.open_sftp()
                return sftp_client
            except Exception as e:
                cls._logger.error(f"Could not establish sftp connection: {e}")
        return None

    @classmethod
    def push(cls, src_path: str, dst_path: str, client: paramiko.SFTPClient) -> DeviceReturn:
        """
        This method pushes file(s) from the specified source path to the specified destination path
        on the FTP server.If the destination path does not exist, it will be created automatically.

        Args:
            src_path: The source path on the local system to copy files from. Can be
                       either a single file or directory.
            dst_path: The destination path on the FTP server to copy files to.
                       Should be a directory.
            client (paramiko.SFTPClient): An instance of an SSH sftp session

        Returns:
            DeviceReturn: A DeviceCompletedProcess object if successful,
                          otherwise a DeviceFailedProcess object.
        """

        def push_file(local_file_path, remote_dir_path):
            if os.path.isfile(local_file_path):
                file_path = os.path.join(remote_dir_path, os.path.basename(local_file_path))
                client.put(local_file_path, format_remote_path(file_path))

        def push_dir(local_dir_path, remote_dir_path):
            cls.make_directory(remote_dir_path, client)
            paths = os.listdir(local_dir_path)

            if not paths:
                return

            while paths:
                path = paths.pop()
                full_path = os.path.join(local_dir_path, path)
                if os.path.isfile(full_path):
                    push_file(full_path, remote_dir_path)
                elif os.path.isdir(full_path):
                    push_dir(full_path, os.path.join(remote_dir_path, path))
                else:
                    raise TypeError(f"{full_path} is not a file or directory")

        try:
            # Ensure the source path exists
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Error: Source path '{src_path}' does not exist.")

            # Determine if source is a file or directory
            if os.path.isfile(src_path):
                # upload the file
                push_file(src_path, dst_path)
            elif os.path.isdir(src_path):
                # Upload files from the local directory recursively
                push_dir(src_path, dst_path)
                cls._logger.debug(f"Directory '{src_path}' uploaded to '{dst_path}'")
            else:
                raise IOError(f"{src_path} is neither a file nor a directory")
        except Exception as e:
            cls._logger.error(f"Could not push from path: {src_path}  Received error: {e!s}")
            return DeviceFailedProcess(
                args=[f"sftp.put {src_path} {dst_path}"],
                returncode=DeviceCode.DEVICE_UNKNOWN_ERROR,
                stderr=str(e),
                orig_error=e,
            )
        return DeviceCompletedProcess(
            args=[f"sftp.put {src_path} {dst_path}"], returncode=DeviceCode.DEVICE_SUCCESS
        )

    @classmethod
    def pull(cls, src_path: str, dst_path: str, client: paramiko.SFTPClient) -> DeviceReturn:
        """
        This method pulls file(s) from the specified remote source path to the specified
        destination path on the local host. If the destination path does not exist, it will be
        created automatically.

        Args:
            src_path: The source path on the remote FTP server. Can be
                       either a single file or directory.
            dst_path: The destination path on the local host to copy files to.
                       Should be a directory.
            client (paramiko.SFTPClient): An instance of an SSH sftp session

        Returns:
            DeviceReturn: A DeviceCompletedProcess object if successful,
                          otherwise a DeviceFailedProcess object.
        """

        def pull_file(remote_file_path, local_file_path):
            client.get(remote_file_path, local_file_path)

        def pull_dir(remote_dir_path, local_dir_path):
            paths = client.listdir(remote_dir_path)

            if not paths:
                return

            while paths:
                path = paths.pop()
                full_path = format_remote_path(os.path.join(remote_dir_path, path))
                if remote_file_exists(client, full_path):
                    pull_file(full_path, os.path.join(local_dir_path, path))
                elif remote_dir_exists(client, full_path):
                    pull_dir(full_path, os.path.join(local_dir_path, path))
                else:
                    raise TypeError(f"{full_path} is not a file or directory")

        try:
            # Ensure the source path exists
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            # Determine if source is a file or directory
            if remote_file_exists(client, src_path):
                pull_file(src_path, dst_path)
            elif remote_dir_exists(client, src_path):
                # Upload files from the local directory recursively
                pull_dir(src_path, dst_path)
                cls._logger.debug(f"Directory '{src_path}' pulled to '{dst_path}'")
            else:
                raise IOError(f"{src_path} is neither a file nor a directory")
        except Exception as e:
            cls._logger.error(f"Could not pull from path: {src_path}  Received error: {e!s}")
            return DeviceFailedProcess(
                args=[f"sftp.get {src_path} {dst_path}"],
                returncode=DeviceCode.DEVICE_UNKNOWN_ERROR,
                stderr=str(e),
                orig_error=e,
            )
        return DeviceCompletedProcess(
            args=[f"sftp.get {src_path} {dst_path}"], returncode=DeviceCode.DEVICE_SUCCESS
        )

    @classmethod
    def execute(cls, command: str, client: paramiko.SSHClient) -> DeviceReturn:
        """
        Executes the given command on the remote connection via SSH and returns the result.

        Args:
            command (str): The command to be executed on the remote connection.
            client (paramiko.SSHClient): The SSH connection to use for executing the command.

        Returns:
            DeviceReturn: A DeviceReturn object containing information about the execution status
                          and output.
        """

        try:
            if not cls.is_connected(client):
                raise ValueError("Could not connect to client")

            ssh_stdin, ssh_stdout, ssh_err = client.exec_command(command)

            # Read the output and error
            output = ssh_stdout.read().decode()
            err = ssh_err.read().decode()
            exit_status = 1

            if ssh_out_code := ssh_stdout.channel.recv_exit_status() and not err:
                exit_status = ssh_out_code

            if exit_status:
                cls._logger.debug(f"Executed command : '{command}' on target")
                return DeviceCompletedProcess(
                    args=command, returncode=DeviceCode.DEVICE_SUCCESS, stdout=output, stderr=err
                )
            else:
                raise RuntimeError(f"Execution failed with error: {exit_status}")

        except Exception as e:
            cls._logger.error(f"Could not execute command: {command} on client. Received error: {e!s}")
            return DeviceFailedProcess(
                args=command, returncode=DeviceCode.DEVICE_UNKNOWN_ERROR, stderr=str(e), orig_error=e
            )

    @classmethod
    def make_directory(cls, dst_path: str, client: paramiko.SFTPClient) -> Optional[DeviceReturn]:
        """
        This function creates an empty directory on the SFTP server at the specified location.
        If the directory already exists, this function will do nothing.

        Args:
            dst_path: The path where the new directory should be created on the sFTP server.
            client (paramiko.SFTPClient): An sftp client instance
        Returns:
            DeviceReturn: A DeviceCompletedProcess object if successful, a DeviceFailedProcess
                          object if a failure
            occurs or none.
        """
        if remote_dir_exists(client, dst_path):
            return None
        try:
            base_dir = dst_path
            if not remote_dir_exists(client, base_dir):
                base_dir, _ = get_remote_path_parts(base_dir)
                cls.make_directory(base_dir, client)
            client.mkdir(dst_path)
        except Exception as e:
            return DeviceFailedProcess(
                args=[f"sftp mkdir {dst_path}"], stdout=None, stderr=str(e), orig_error=e
            )

        return DeviceCompletedProcess(
            args=[f"sftp mkdir {dst_path}"], stdout=None, returncode=DeviceCode.DEVICE_SUCCESS
        )

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
    def remove(cls, target_path: str, client: paramiko.SFTPClient) -> Optional[DeviceReturn]:
        """
        This method removes a remote file or directory from the SFTP server.

        Args:
            target_path (str): The path of the file or directory on the SFTP server.
            client (paramiko.SFTPClient): An SFTP connection object.

        Returns:
            DeviceReturn: A DeviceCompletedProcess object if successful, a DeviceFailedProcess
                          object if a failure occurs or none.
        """
        try:
            if remote_dir_exists(client, target_path):
                paths = client.listdir(target_path)
                for path in paths:
                    cls.remove(os.path.join(target_path, path), client)
                # Delete folder
                client.rmdir(target_path)
            elif remote_file_exists(client, target_path):
                # Delete file
                client.remove(target_path)
            else:
                # nothing to do
                return None
        except Exception as e:
            cls._logger.error(f"Error removing remote directory: {str(e)}")
            return DeviceFailedProcess(
                args=[f"sftp.remove {target_path}"], stdout=None, stderr=str(e), orig_error=e
            )

        return DeviceCompletedProcess(
            args=[f"sftp.remove {target_path}"], stdout=None, returncode=DeviceCode.DEVICE_SUCCESS
        )

    @classmethod
    def close(cls, client: paramiko.SSHClient) -> DeviceReturn:
        """
        Closes the given SSH client.

        Args:
            client (SSH): The SSH connection to be closed.
        """
        try:
            if client in cls._CONNECTED_INSTANCE_CACHE.values():
                for host_name in client.get_host_keys():
                    del cls._CONNECTED_INSTANCE_CACHE[host_name]
            client.close()
        except Exception as e:
            cls._logger.error(f"Error closing connection: {str(e)}")
            return DeviceFailedProcess(args=["sshclient.close"], stdout=None, stderr=str(e), orig_error=e)
        cls._logger.debug("SSH client closed successfully")
        return DeviceCompletedProcess(
            args=["sshclient.close"], stdout=None, returncode=DeviceCode.DEVICE_SUCCESS
        )
