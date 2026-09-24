# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
from pathlib import Path
from typing import Optional

from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.tools.core.modules.api.backend.backend import Backend
from qti.aisw.tools.core.modules.api.backend.utility import get_backend_extension_library, get_backend_library
from qti.aisw.tools.core.modules.api.definitions.common import BackendType, QNNContextConfig
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DevicePlatformType,
)


class LpaiBackend(Backend):
    def __init__(
        self,
        target: Optional[Target] = None,
        config_file: Optional[str] = None,
        config_dict: Optional[dict] = None,
        context_config: Optional[QNNContextConfig] = None,
    ):
        """Initializes the class with the given target, configuration file, and configuration dictionary.

        Args:
            target (Optional[mlapi_Target]): The target for the backend. Defaults to host machine.
            config_file (Optional[str]): The path to the configuration file. Defaults to None.
            config_dict (Optional[dict]): The configuration dictionary. Defaults to None.
            context_config (Optional[QNNContextConfig]): The context configuration. Defaults to None.
        """
        super().__init__(target)
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
        return get_backend_library(self.target.target_platform_type, BackendType.LPAI)

    @property
    def backend_extensions_library(self) -> str | None:
        """Returns the name of the backend extensions library."""
        return get_backend_extension_library(self.target.target_platform_type, BackendType.LPAI)

    def get_required_device_artifacts(self, sdk_root: str) -> list:
        """Returns the list of required artifacts in the SDK root.

        Args:
            sdk_root (str): The root directory of the SDK.
        """
        artifacts = []
        if self.target.target_platform_type == DevicePlatformType.ANDROID:
            lpai_stub = Path(sdk_root, "lib", "aarch64-android", "libQnnLpaiV79Stub.so")
            lpai_skel = Path(sdk_root, "lib", "hexagon-v79", "unsigned", "libQnnLpaiV79Skel_v5.so")
            artifacts = [str(lpai_stub), str(lpai_skel)]

        return artifacts
