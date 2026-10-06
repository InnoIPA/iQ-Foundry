# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import time
from telnetlib import Telnet
from typing import ClassVar, Dict, List, Optional

from qti.aisw.tools.core.utilities.devices.protocol_helpers.protocol_helper import ProtocolHelper
from qti.aisw.tools.core.utilities.devices.utils import ping
from qti.aisw.tools.core.utilities.devices.utils.device_code import (
    DeviceCode,
    DeviceCompletedProcess,
    DeviceFailedProcess,
    DeviceReturn,
)
from qti.aisw.tools.core.utilities.devices.utils.device_utils import SingletonABC
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class TelnetProtocolHelper(ProtocolHelper, metaclass=SingletonABC):
    """
    Telnet protocol helper class that provides methods for connecting to a connection using Telnet.
    """

    _logger = QAIRTLogger.get_logger(name=__name__, level="INFO")
    _TELNET_DEFAULT_TIMEOUT = 10
    _CONNECTED_INSTANCE_CACHE: ClassVar[Dict[str, Telnet]] = {}

    @classmethod
    def setup_telnet_connection(
        cls,
        host: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        *,
        port: int = 23,
        timeout: Optional[int] = None,
        debug_level: int = 0,
    ) -> Optional[Telnet]:
        """
        Connects to the specified host using Telnet and returns a telnet object

        Args:
            host (str): Hostname or IP address of the connection.
            username (str): Username
            password (str):
            port (int): Port number to use for the Telnet connection. Default is 23 (auto-select).
            timeout (Optional[int]): Timeout in seconds before giving up on establishing the Telnet
                                     connection. Default is _TELNET_DEFAULT_TIMEOUT = 10.
            debug_level (int): Debug level for the Telnet connection. Default is 0.

        Raises:
            EOFError: If there's an EOF error during Telnet setup
            ConnectionRefusedError: If the connection attempt was refused

        Returns:
            Optional[Telnet]: A Telnet object representing the established connection, or None if
                              connection failed
        """
        if host in cls._CONNECTED_INSTANCE_CACHE:
            cls._logger.info(f"Connection has already been established for host: {host}")
            return cls._CONNECTED_INSTANCE_CACHE[host]

        timeout = timeout if timeout else cls._TELNET_DEFAULT_TIMEOUT
        try:
            connection = Telnet(host, port, timeout)
            connection.set_debuglevel(debug_level)
            connection.read_until(b"login: ", timeout)

            if username:
                connection.write(username.encode("ascii") + b"\n")

            if password:
                connection.read_until(b"Password:")
                connection.write(password.encode("ascii") + b"\n")

            cls._logger.debug("Telnet connection established successfully")
            cls._CONNECTED_INSTANCE_CACHE[host] = connection
        except (EOFError, ConnectionRefusedError) as e:
            cls._logger.error(f"Unable to connect {host}. Received error code {e!s}")
            return None

        return connection

    @classmethod
    def execute(cls, command: str, connection: Telnet, *, close_connection: bool = False) -> DeviceReturn:
        """
        Executes the given command on the remote connection via Telnet and returns the result.

        Args:
            command (str): The command to be executed on the remote connection.
            connection (Telnet): The Telnet connection to use for executing the command.
            close_connection (bool): Whether to close the Telnet connection after running the
                                     command. Defaults to False.

        Returns:
            DeviceReturn: A DeviceReturn object containing information about the execution status
                          and output.
        """
        try:
            # Append a string to the command to indicate execution is complete
            command += "\n\nRemote execution complete"
            # Execute the command
            connection.write(command.encode("ascii") + b"\n")

            # Read all output from the command
            time.sleep(1)  # Small delay to allow the server to process the command
            # Reading response until the given string is hit
            output = connection.read_until(b"Remote execution complete").decode("ascii")
            if not close_connection:
                cls._logger.warning("Telnet connection must be closed when capture_all_output is set")
                close_connection = True

            cls._logger.debug("Telnet command executed successfully")

            # close connection
            if close_connection:
                cls.close(connection)

        except (EOFError, ConnectionRefusedError) as e:
            cls._logger.error(
                f"Could not execute command: {command} on host {connection.host}. Received error: {e!s}"
            )
            return DeviceFailedProcess(
                args=command, returncode=DeviceCode.DEVICE_UNKNOWN_ERROR, stderr=str(e), orig_error=e
            )

        return DeviceCompletedProcess(args=command, stdout=output)

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

        # log available devices
        cls._logger.info(f"The following devices were found: {available_devices}")

        # log unavailable devices
        cls._logger.warning(f"The following requested devices were not found: {unavailable_devices}")

        return available_devices

    @classmethod
    def close(cls, connection: Telnet):
        """
        Closes the given Telnet connection.

        Args:
            connection (Telnet): The Telnet connection to be closed.
        """
        if connection in cls._CONNECTED_INSTANCE_CACHE.values():
            if connection.host is not None:
                del cls._CONNECTED_INSTANCE_CACHE[connection.host]
        connection.close()
        cls._logger.debug("Telnet connection closed successfully")
