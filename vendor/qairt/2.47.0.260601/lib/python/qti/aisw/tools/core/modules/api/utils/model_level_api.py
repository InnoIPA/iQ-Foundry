# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from __future__ import annotations

import qti.aisw.core.model_level_api as mlapi
from qti.aisw.tools.core.modules.api.definitions.common import Target
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType


def resolve_hostname(identifier) -> str | None:
    """Returns hostname if available, else ip_addr; None if identifier is None."""
    if not identifier:
        return None
    return identifier.hostname or identifier.ip_addr


def create_mlapi_target(target: Target) -> mlapi.Target:
    """Creates an mlapi target based on the provided target configuration.
    Args:
        target (Target): The target configuration object.
    Returns:
        mlapi.Target: The corresponding mlapi target object.
    Raises:
        ValueError: If an unknown target type is provided.
    """
    # handle hostname/port when support is added to model-level API
    soc_model = target.soc_model if target.soc_model else None
    if target.type == DevicePlatformType.ANDROID:
        device_id = target.identifier.serial_id if target.identifier else None
        hostname = target.identifier.hostname if target.identifier else None
        adb_server_port = target.identifier.port if target.identifier else None
        return mlapi.AndroidTarget(
            device_id=device_id, hostname=hostname, adb_server_port=adb_server_port, soc_model=soc_model
        )
    elif target.type == DevicePlatformType.X86_64_LINUX:
        return mlapi.X86LinuxTarget()
    elif target.type == DevicePlatformType.X86_64_WINDOWS_MSVC:
        return mlapi.X86WindowsTarget()
    elif target.type == DevicePlatformType.WOS:
        return mlapi.WOSTarget()
    elif target.type == DevicePlatformType.QNX:
        hostname = resolve_hostname(target.identifier)
        username = target.credentials.username if target.credentials else None
        password = target.credentials.password if target.credentials else None
        return mlapi.QNXTarget(hostname=hostname, username=username, password=password, soc_model=soc_model)
    elif target.type == DevicePlatformType.LINUX_EMBEDDED:
        hostname = resolve_hostname(target.identifier)
        device_id = target.identifier.serial_id if target.identifier else None
        adb_server_port = target.identifier.port if target.identifier else None
        return mlapi.OELinuxTarget(
            device_id=device_id, hostname=hostname, adb_server_port=adb_server_port, soc_model=soc_model
        )
    else:
        raise ValueError(f"Unknown target type: {target.type}")
