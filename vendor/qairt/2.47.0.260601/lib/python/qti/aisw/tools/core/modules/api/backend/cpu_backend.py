# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Optional

from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.core.model_level_api.workflow.workflow import WorkflowMode
from qti.aisw.tools.core.modules.api.backend.backend import Backend
from qti.aisw.tools.core.modules.api.backend.utility import get_backend_extension_library, get_backend_library
from qti.aisw.tools.core.modules.api.definitions.common import BackendType, QNNContextConfig


class CpuBackend(Backend):
    def __init__(self, target: Optional[Target] = None, context_config: Optional[QNNContextConfig] = None):
        """Initializes the CpuBackend with the given target. If no target is provided,
        defaults to corresponding target based on the host and architecture.

        Args:
            target (Optional[mlapi.Target]): The target for the backend. Defaults to host machine.
            context_config (Optional[QNNContextConfig]): The context configuration. Defaults to None.
        """
        super().__init__(target)

        if context_config:
            self._context_config = context_config

    @property
    def backend_library(self) -> str:
        """Returns the name of the backend library."""
        return get_backend_library(self.target.target_platform_type, BackendType.CPU)

    @property
    def backend_extensions_library(self) -> str | None:
        """Returns the name of the backend extensions library."""
        # no backend extensions support
        return get_backend_extension_library(self.target.target_platform_type, BackendType.CPU)

    def update_config(self, config_dict: Optional[dict], config_file: Optional[str]):
        """Updates the configuration using the provided dictionary and/or configuration file.

        Args:
            config_dict (Optional[dict]): The configuration dictionary.
            config_file (Optional[str]): The path to the configuration file.
        """
        raise NotImplementedError("CPU backend does not support custom configs")

    def get_required_device_artifacts(self, sdk_root: str) -> list:
        """Returns the list of required artifacts in the SDK root.

        Args:
            sdk_root (str): The root directory of the SDK.
        """
        return super().get_required_device_artifacts(sdk_root)

    def _workflow_mode_setter_hook(self, mode: WorkflowMode):
        """
        Sets the target based on the workflow mode.

        Args:
            mode (WorkflowMode): The workflow mode to set the target for.

        Raises:
            ValueError: If an invalid workflow mode is provided.
        """
        if mode == WorkflowMode.CONTEXT_BINARY_GENERATION:
            raise ValueError("CPU backend does not support context binary generation")
