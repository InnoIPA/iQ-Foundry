# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import platform
import shutil
from typing import List, Tuple

from qti.aisw.tools.core.utilities.devices.utils import subprocess_helper
from qti.aisw.tools.core.utilities.devices.utils.device_code import *
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger

logger = QAIRTLogger.get_logger(name=__name__, level="INFO")


def get_available_destinations(destination: str, *_destinations: str) -> Tuple[List[str], List[str]]:
    """
    This method checks that the provided destinations are reachable using ping.

    Args:
        destination (str): DNS name or ip address
        _destinations (str): DNS name(s) or ip addresses

    Returns:
       List[str]: The list of available devices, or empty list if none are available.
    """

    destinations: List[str] = [destination] + list(_destinations)

    device_returns = [get_status(_dest) for _dest in destinations]

    available_devices = [
        _destinations[idx]
        for idx, _device_return in enumerate(device_returns)
        if isinstance(_device_return, DeviceCompletedProcess)
    ]
    unavailable_devices = list(filter(lambda dest: dest not in available_devices, _destinations))

    return available_devices, unavailable_devices


def get_status(destination: str) -> Optional[DeviceReturn]:
    """
    Get the status of a device using ping command.

    Args:
        destination (str): DNS name or ip address

    Returns:
       DeviceReturn:
    """
    ping_param = "-n" if platform.system().lower() == "windows" else ""

    if shutil.which("ping"):
        device_return = subprocess_helper.execute("ping", [ping_param, "-c", "1", destination], logger=logger)
    else:
        return None
    return device_return
