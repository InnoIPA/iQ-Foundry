# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
from typing import Optional

from qti.aisw.core.model_level_api.target.x86_linux import X86LinuxTarget
from qti.aisw.tools.core.modules.api import Target
from qti.aisw.tools.core.modules.api.backend.backend import Backend, BackendConfig
from qti.aisw.tools.core.modules.api.backend.utility import get_backend_extension_library, get_backend_library
from qti.aisw.tools.core.modules.api.definitions.common import BackendType, QNNContextConfig


class AicBackend(Backend):
    graph_names = BackendConfig()
    compiler_compilation_target = BackendConfig()
    compiler_hardware_version = BackendConfig()
    compiler_num_of_cores = BackendConfig()
    compiler_convert_to_FP16 = BackendConfig()
    compiler_do_host_preproc = BackendConfig()
    compiler_stat_level = BackendConfig()
    compiler_printDDRStats = BackendConfig()
    compiler_printPerfMetrics = BackendConfig()
    compiler_perfWarnings = BackendConfig()
    compiler_compilationOutputDir = BackendConfig()
    compiler_enableDebug = BackendConfig()
    compiler_buffer_dealloc_delay = BackendConfig()
    compiler_genCRC = BackendConfig()
    compiler_stats_batch_size = BackendConfig()
    compiler_crc_stride = BackendConfig()
    compiler_enable_depth_first = BackendConfig()
    compiler_overlap_split_factor = BackendConfig()
    compiler_depth_first_mem = BackendConfig()
    compiler_VTCM_working_set_limit_ratio = BackendConfig()
    compiler_size_split_granularity = BackendConfig()
    compiler_compileThreads = BackendConfig()
    compiler_userDMAProducerDMAEnabled = BackendConfig()
    compiler_do_DDR_to_multicast = BackendConfig()
    compiler_combine_inputs = BackendConfig()
    compiler_combine_outputs = BackendConfig()
    compiler_directApi = BackendConfig()
    compiler_force_VTCM_spill = BackendConfig()
    compiler_PMU_recipe_opt = BackendConfig()
    compiler_PMU_events = BackendConfig()
    compiler_cluster_sizes = BackendConfig()
    compiler_max_out_channel_split = BackendConfig()
    runtime_device_id = BackendConfig()
    runtime_num_activations = BackendConfig()
    runtime_submit_timeout = BackendConfig()
    runtime_submit_num_retries = BackendConfig()
    runtime_threads_per_queue = BackendConfig()
    runtime_process_lock = BackendConfig()

    def __init__(
        self,
        target: Optional[Target] = X86LinuxTarget(),
        config_file: Optional[str] = None,
        config_dict: Optional[dict] = None,
        context_config: Optional[QNNContextConfig] = None,
        **kwargs,
    ):
        """Initializes the class with the given target, configuration file, and configuration dictionary.
        Additional keyword arguments can be used to set attributes.

        Args:
            target (Optional[Target]): The target for the backend. Defaults to X86LinuxTarget().
            config_file (Optional[str]): The path to the configuration file. Defaults to None.
            config_dict (Optional[dict]): The configuration dictionary. Defaults to None.
            context_config (Optional[QNNContextConfig]): The context configuration. Defaults to None.
            **kwargs: Additional keyword arguments to set attributes.

        Raises:
            AttributeError: Raised if a keyword argument does not correspond to an existing attribute.
        """
        super().__init__(target)
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
    def backend_library(self) -> str | None:
        """Returns the name of the backend library."""
        return get_backend_library(self.target.target_platform_type, BackendType.AIC)

    @property
    def backend_extensions_library(self) -> str | None:
        """Returns the name of the backend extensions library."""
        return get_backend_extension_library(self.target.target_platform_type, BackendType.AIC)

    def get_required_device_artifacts(self, sdk_root: str) -> list:
        """Returns the list of required artifacts in the SDK root.

        Args:
            sdk_root(str): A path to the root of the SDK
        """
        return []
