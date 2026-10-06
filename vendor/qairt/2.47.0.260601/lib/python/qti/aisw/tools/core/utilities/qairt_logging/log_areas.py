# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import List, Optional

from aenum import Enum, extend_enum

class LogAreas(Enum):
    """
    Log areas enum class
    """

    @classmethod
    def list_log_areas(cls) -> List[str]:
        """
        Returns the list of log areas

        Returns:
            List[str]: The list of log areas
        """
        return [area.value for area in LogAreas]

    @classmethod
    def register_log_area(cls, area: str) -> "LogAreas":
        """
        Registers a new log area

        Args:
            area (str): The name of the log area to be registered

        Returns:
            LogAreas: The newly created log area
        """
        if area not in cls._member_names_:
            extend_enum(cls, area, str(area))

        return cls[area]

    @classmethod
    def get_log_area_by_name(cls, name: str) -> Optional["LogAreas"]:
        """
        Returns the log area enum member given its name or value.

        Args:
            name (str): The name or value of the log area

        Returns:
            Optional[LogAreas]: The corresponding LogAreas member, or None if not found
        """
        # Try to match by member name
        if name in cls._member_names_:
            return cls[name]

        # Try to match by value
        for area in cls:
            if area.value == name:
                return area

        return None
