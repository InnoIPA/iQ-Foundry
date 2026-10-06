# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import ABC, abstractmethod
from typing import Any, List


class ProtocolHelper(ABC):
    """
    Abstract class for a protocol that provides methods to interact with a device.
    """

    @classmethod
    @abstractmethod
    def get_available_devices(cls, *args, **kwargs) -> List[Any]:
        """
        Abstract method to get the list of available devices.
        Must be implemented by subclasses.

        Returns:
            List[Any]: The list of available devices, or empty list if none are available.
        """
        raise NotImplementedError("Function has not been implemented")
