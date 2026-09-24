# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
from typing import Optional

from qti.aisw.core.model_level_api.target.android import AndroidTarget
from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.tools.core.modules.api.backend.backend import Backend, BackendConfig
from qti.aisw.tools.core.modules.api.backend.utility import get_backend_extension_library, get_backend_library
from qti.aisw.tools.core.modules.api.definitions.common import BackendType, QNNContextConfig


class GpuBackend(Backend):
    graph_names = BackendConfig()
    precision_mode = BackendConfig()
    disable_memory_optimizations = BackendConfig()
    disable_node_optimizations = BackendConfig()
    kernel_repo_path = BackendConfig()
    invalidate_kernel_repo = BackendConfig()
    disable_queue_recording = BackendConfig()
    perf_hint = BackendConfig()

    def __init__(
        self,
        target: Optional[Target] = None,
        config_file: Optional[str] = None,
        config_dict: Optional[dict] = None,
        context_config: Optional[QNNContextConfig] = None,
        **kwargs,
    ):
        """Initializes the GpuBackend with the given target. If no target is provided,
        defaults to AndroidTarget.

        Args:
            target (Optional[Target]): The target for the backend. Defaults to host machine.
            config_file (Optional[str]): The path to the configuration file. Defaults to None.
            config_dict (Optional[dict]): The configuration dictionary. Defaults to None.
            context_config (Optional[QNNContextConfig]): The context configuration. Defaults to None.
            **kwargs: Additional keyword arguments to set attributes.

        Raises:
            AttributeError: Raised if a keyword argument does not correspond to an existing attribute.
        """

        if target is None:
            target = AndroidTarget()

        super().__init__(target)
        self.logger.debug(f"Target not provided, defaulting to {type(target).__name__}")

        if config_file:
            with open(config_file, "r") as f:
                self._config = json.load(f)

        if config_dict:
            self._config.update(config_dict)

        if context_config:
            self._context_config = context_config

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{type(self).__name__} does not have a config for key: {key}")

        self.logger.debug(f"Config dictionary after processing all provided configs: {self._config}")

    @property
    def backend_library(self) -> str:
        """Returns the name of the backend library."""
        return get_backend_library(self.target.target_platform_type, BackendType.GPU)

    @property
    def backend_extensions_library(self) -> str | None:
        """Returns the name of the backend extensions library."""
        return get_backend_extension_library(self.target.target_platform_type, BackendType.GPU)

    def get_required_device_artifacts(self, sdk_root: str) -> list:
        """Returns the list of required artifacts in the SDK root.

        Args:
            sdk_root (str): The root directory of the SDK.
        """
        return super().get_required_device_artifacts(sdk_root)
