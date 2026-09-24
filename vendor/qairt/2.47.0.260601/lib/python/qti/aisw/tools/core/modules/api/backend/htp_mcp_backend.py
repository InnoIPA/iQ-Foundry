# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import glob
import json
import os
from pathlib import Path
from shutil import copyfile
from typing import List, Optional

from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.tools.core.modules.api.backend.backend import Backend
from qti.aisw.tools.core.modules.api.backend.utility import get_backend_extension_library, get_backend_library
from qti.aisw.tools.core.modules.api.definitions.common import BackendType, QNNContextConfig
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DevicePlatformType,
)


class HtpMcpBackend(Backend):
    def __init__(
        self,
        target: Optional[Target] = None,
        config_file: Optional[str] = None,
        config_dict: Optional[dict] = None,
        context_config: Optional[QNNContextConfig] = None,
    ):
        """Initializes the class with the given target, configuration file, and configuration dictionary.

        Args:
            target (Optional[Target]): The target for the backend. Defaults to host machine.
            config_file (Optional[str]): The path to the configuration file. Defaults to None.
            config_dict (Optional[dict]): The configuration dictionary. Defaults to None.
            context_config (Optional[QNNContextConfig]): The context configuration. Defaults to None.
        """
        super().__init__(target)
        self._schematic_bins: Optional[List[str]] = None

        if config_file:
            with open(config_file, "r") as f:
                self._config = json.load(f)

        if config_dict:
            self._config.update(config_dict)

        if context_config:
            self._context_config = context_config

    @property
    def backend_library(self) -> str | None:
        """Returns the name of the backend library."""
        return get_backend_library(self.target.target_platform_type, BackendType.HTP_MCP)

    @property
    def backend_extensions_library(self) -> str | None:
        """Returns the name of the backend extensions library."""
        return get_backend_extension_library(self.target.target_platform_type, BackendType.HTP_MCP)

    def get_required_device_artifacts(self, sdk_root: str) -> list:
        """Returns the list of required artifacts in the SDK root.

        Args:
            sdk_root (str): The root directory of the SDK.
        """
        return super().get_required_device_artifacts(sdk_root)

    def get_profiling_artifacts(self) -> list:
        """Returns the list of profiling artifacts."""
        artifacts: List[Path] = list()
        if self._schematic_bins:
            artifacts = [Path(schematic_bin).absolute() for schematic_bin in self._schematic_bins]
        return artifacts

    def clear_profiling_artifacts(self) -> None:
        """Clears the profiling artifacts."""
        self._schematic_bins = None

    def before_generate_hook(self, temp_directory: str, sdk_path: str) -> None:
        """A hook that is called after a context binary is generated.

        Args:
            temp_directory(str): A path to the working directory where the context binary was
                                 generated (not the directory where it was written to)
            sdk_path(str): A path to the root of the SDK
        """
        mcp_elf_path = Path(sdk_path, "lib", "hexagon-v68", "unsigned", "libQnnHtpMcpV68.elf")
        if not mcp_elf_path.exists():
            raise FileNotFoundError(f"Could not find HTP MCP elf file {mcp_elf_path}")

        copyfile(mcp_elf_path, Path(temp_directory, "network.elf"))

    def after_generate_hook(self, temp_directory: str, sdk_path: str) -> None:
        """A hook that is called after a context binary is generated.

        Args:
            temp_directory(str): A path to the working directory where the context binary was
                                 generated (not the directory where it was written to)
            sdk_path(str): A path to the root of the SDK
        """
        if self.target.target_platform_type in [
            DevicePlatformType.X86_64_LINUX,
            DevicePlatformType.X86_64_WINDOWS_MSVC,
        ]:
            self._schematic_bins = glob.glob(f"{temp_directory}{os.sep}*schematic.bin")
