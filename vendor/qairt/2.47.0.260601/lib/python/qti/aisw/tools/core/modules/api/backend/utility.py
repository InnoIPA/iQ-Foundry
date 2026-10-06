# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging
import os
import platform
import shutil
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

from qti.aisw.tools.core.modules.api.definitions.common import BackendType
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType


def get_backend_library(device: DevicePlatformType, backend_type: BackendType) -> str:
    """Returns the name of the backend library based on the host type, architecture, and backend type.

    Args:
        device (DevicePlatformType): Target device type.
        backend_type (str): The type of backend (e.g., 'cpu', 'gpu', 'htp').

    Returns:
        str: The name of the backend library.

    Raises:
        ValueError: If the system is unsupported.
    """
    libraries: Dict[str, Dict] = {
        BackendType.CPU: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: "QnnCpu.dll",
            DevicePlatformType.WOS: "QnnCpu.dll",
            DevicePlatformType.X86_64_LINUX: "libQnnCpu.so",
            DevicePlatformType.ANDROID: "libQnnCpu.so",
            DevicePlatformType.LINUX_EMBEDDED: "libQnnCpu.so",
        },
        BackendType.GPU: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: "QnnGpu.dll",
            DevicePlatformType.WOS: "QnnGpu.dll",
            DevicePlatformType.X86_64_LINUX: "libQnnGpu.so",
            DevicePlatformType.ANDROID: "libQnnGpu.so",
            DevicePlatformType.LINUX_EMBEDDED: "libQnnGpu.so",
        },
        BackendType.HTP: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: "QnnHtp.dll",
            DevicePlatformType.WOS: "QnnHtp.dll",
            DevicePlatformType.X86_64_LINUX: "libQnnHtp.so",
            DevicePlatformType.ANDROID: "libQnnHtp.so",
            DevicePlatformType.QNX: "libQnnHtp.so",
            DevicePlatformType.LINUX_EMBEDDED: "libQnnHtp.so",
        },
        BackendType.HTP_MCP: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: None,
            DevicePlatformType.WOS: None,
            DevicePlatformType.X86_64_LINUX: "libQnnHtpMcp.so",
            DevicePlatformType.QNX: "libQnnHtpMcp.so",
        },
        BackendType.LPAI: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: "QnnLpai.dll",
            DevicePlatformType.WOS: None,
            DevicePlatformType.X86_64_LINUX: "libQnnLpai.so",
        },
        BackendType.AIC: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: None,
            DevicePlatformType.WOS: None,
            DevicePlatformType.X86_64_LINUX: "libQnnAic.so",
        },
    }

    if device in libraries.get(backend_type, {}):
        return libraries[backend_type][device]
    else:
        raise NotImplementedError(f"Unsupported target device: {device} or backend type: {backend_type}")


def get_backend_extension_library(device: DevicePlatformType, backend_type: BackendType) -> str | None:
    """Returns the name of the backend extension library based on the host type and backend type.

    Args:
        device (DevicePlatformType): Target device type.
        backend_type (BackendType): The type of backend.

    Returns:
        str | None: The name of the backend extension library or None if not supported.

    Raises:
        NotImplementedError: If the device or backend type is unsupported.
    """
    extension_libraries: Dict[str, Dict] = {
        BackendType.CPU: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: "QnnCpuNetRunExtensions.dll",
            DevicePlatformType.WOS: "QnnCpuNetRunExtensions.dll",
            DevicePlatformType.X86_64_LINUX: "libQnnCpuNetRunExtensions.so",
            DevicePlatformType.LINUX_EMBEDDED: "libQnnCpuNetRunExtensions.so",
            DevicePlatformType.ANDROID: "libQnnCpuNetRunExtensions.so",
        },
        BackendType.GPU: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: None,
            DevicePlatformType.WOS: "QnnGpuNetRunExtensions.dll",
            DevicePlatformType.X86_64_LINUX: "libQnnGpuNetRunExtensions.so",
            DevicePlatformType.ANDROID: "libQnnGpuNetRunExtensions.so",
            DevicePlatformType.LINUX_EMBEDDED: "libQnnGpuNetRunExtensions.so",
        },
        BackendType.HTP: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: "QnnHtpNetRunExtensions.dll",
            DevicePlatformType.WOS: "QnnHtpNetRunExtensions.dll",
            DevicePlatformType.X86_64_LINUX: "libQnnHtpNetRunExtensions.so",
            DevicePlatformType.ANDROID: "libQnnHtpNetRunExtensions.so",
            DevicePlatformType.LINUX_EMBEDDED: "libQnnHtpNetRunExtensions.so",
        },
        BackendType.HTP_MCP: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: None,
            DevicePlatformType.WOS: None,
            DevicePlatformType.X86_64_LINUX: "libQnnHtpMcpNetRunExtensions.so",
        },
        BackendType.LPAI: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: "QnnLpaiNetRunExtensions.dll",
            DevicePlatformType.WOS: None,
            DevicePlatformType.X86_64_LINUX: "libQnnLpaiNetRunExtensions.so",
        },
        BackendType.AIC: {
            DevicePlatformType.X86_64_WINDOWS_MSVC: None,
            DevicePlatformType.WOS: None,
            DevicePlatformType.X86_64_LINUX: "libQnnAicNetRunExtensions.so",
        },
    }

    if backend_type in extension_libraries and device in extension_libraries[backend_type]:
        return extension_libraries[backend_type][device]
    else:
        raise NotImplementedError(f"Unsupported target device: {device} or backend type: {backend_type}")


class HexagonEnvironment(ABC):
    @classmethod
    @abstractmethod
    def target(cls) -> str:
        """Return the hexagon target version string."""
        raise NotImplementedError("HexagonEnvironment.target() must be implemented by subclass")

    @classmethod
    def _htp_lib_root(cls) -> str:
        return os.path.join(os.environ["QAIRT_SDK_ROOT"], "lib", "arm64x-windows-msvc")

    @classmethod
    def _htp_libs(cls) -> List[str]:
        return ["QnnHtp.dll", f"QnnHtpV{cls.target()}Stub.dll"]

    @classmethod
    def _hexagon_lib_root(cls) -> str:
        return os.path.join(os.environ["QAIRT_SDK_ROOT"], "lib", f"hexagon-v{cls.target()}", "unsigned")

    @classmethod
    def _hexagon_libs(cls) -> List[str]:
        return [f"libQnnHtpV{cls.target()}Skel.so", "libQnnSystem.so"]

    @classmethod
    @abstractmethod
    def working_dir_initialized(cls) -> bool:
        """Return True if the working directory has been created."""
        raise NotImplementedError(
            "HexagonEnvironment.working_dir_initialized() must be implemented by subclass"
        )

    @classmethod
    @abstractmethod
    def working_dir(cls) -> TemporaryDirectory:
        """Return the TemporaryDirectory used for the environment."""
        raise NotImplementedError("HexagonEnvironment.working_dir() must be implemented by subclass")

    @classmethod
    def activate_environment(cls) -> None:
        initialized = cls.working_dir_initialized()
        working_dir = cls.working_dir()
        if not initialized:
            for lib in cls._htp_libs():
                shutil.copy2(os.path.join(cls._htp_lib_root(), lib), os.path.join(working_dir.name, lib))
            for lib in cls._hexagon_libs():
                shutil.copy2(os.path.join(cls._hexagon_lib_root(), lib), os.path.join(working_dir.name, lib))
            for file in os.listdir(cls._hexagon_lib_root()):
                file_path = os.path.join(cls._hexagon_lib_root(), file)
                if os.path.isfile(file_path) and os.path.splitext(file)[1] == ".cat":
                    shutil.copy2(file_path, os.path.join(working_dir.name, file))

        if not os.environ["PATH"].startswith(working_dir.name):
            os.environ["PATH"] = working_dir.name + ";" + os.environ["PATH"]

    @classmethod
    def deactivate_environment(cls) -> bool:
        """
        Will need to use some form of reference counting to only 'deactivate' the environment when all users have
        deactivated the environment. Since we currently only have one possible environment, on WoS, this will do
        nothing and the destructor will clean up the temporary directory on exit.
        """
        return True


class _HexagonV73Environment(HexagonEnvironment):
    _working_dir: Union[TemporaryDirectory, None] = None

    @classmethod
    def target(cls) -> str:
        return "73"

    @classmethod
    def working_dir_initialized(cls) -> bool:
        return _HexagonV73Environment._working_dir is not None

    @classmethod
    def working_dir(cls) -> TemporaryDirectory:
        if not _HexagonV73Environment._working_dir:
            _HexagonV73Environment._working_dir = TemporaryDirectory(ignore_cleanup_errors=True)
            return _HexagonV73Environment._working_dir
        else:
            return _HexagonV73Environment._working_dir


class _HexagonV79Environment(HexagonEnvironment):
    _working_dir: Union[TemporaryDirectory, None] = None

    @classmethod
    def target(cls) -> str:
        return "79"

    @classmethod
    def working_dir_initialized(cls) -> bool:
        return _HexagonV79Environment._working_dir is not None

    @classmethod
    def working_dir(cls) -> TemporaryDirectory:
        if not _HexagonV79Environment._working_dir:
            _HexagonV79Environment._working_dir = TemporaryDirectory(ignore_cleanup_errors=True)
            return _HexagonV79Environment._working_dir
        else:
            return _HexagonV79Environment._working_dir


class _HexagonV81Environment(HexagonEnvironment):
    _working_dir: Union[TemporaryDirectory, None] = None

    @classmethod
    def target(cls) -> str:
        return "81"

    @classmethod
    def working_dir_initialized(cls) -> bool:
        return _HexagonV81Environment._working_dir is not None

    @classmethod
    def working_dir(cls) -> TemporaryDirectory:
        if not _HexagonV81Environment._working_dir:
            _HexagonV81Environment._working_dir = TemporaryDirectory(ignore_cleanup_errors=True)
            return _HexagonV81Environment._working_dir
        else:
            return _HexagonV81Environment._working_dir


class HexagonEnvironmentManager:
    _hexagon_envs: Dict[str, HexagonEnvironment] = {
        # See https://github.com/python/mypy/issues/3115 for information on ignore[abstract] below
        f"v{x.target()}": x()  # type: ignore[abstract]
        for x in [_HexagonV73Environment, _HexagonV79Environment, _HexagonV81Environment]
    }
    _logger = logging.getLogger(name=__name__)

    def __init__(self):
        raise RuntimeError("HexagonEnvironmentManager should not be instantiated")

    @classmethod
    def activate_hexagon_env(cls, target: str) -> bool:
        if not (platform.system() == "Windows" and "ARMv8" in platform.processor()):
            cls._logger.warning(f"Hexagon environments are only needed for WoS platforms")
            return False

        if target.lower() not in cls._hexagon_envs:
            cls._logger.warning(f"Unknown target hexagon environment requested: {target}")
            return False
        else:
            cls._hexagon_envs[target.lower()].activate_environment()

        return True

    @classmethod
    def deactivate_hexagon_env(cls, target: str) -> bool:
        """
        Will eventually need to manage user count and only deactivate if no active users remain
        """
        cls._logger.warning(f"Manually deactivating {target} hexagon environments is currently unsupported.")
        return False
