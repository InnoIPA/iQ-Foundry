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
from typing import List, Optional

from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.tools.core.modules.api.backend.backend import Backend
from qti.aisw.tools.core.modules.api.backend.utility import (
    get_backend_extension_library,
    get_backend_library,
)
from qti.aisw.tools.core.modules.api.definitions.common import (
    BackendType,
    QNNContextConfig,
)
from qti.aisw.tools.core.utilities.devices.android.android_device import AndroidDevice
from qti.aisw.tools.core.utilities.devices.android.android_device_constants import (
    UNKNOWN_SOC_NAME,
)
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DevicePlatformType,
)
from qti.aisw.tools.core.utilities.devices.api.device_factory import DeviceFactory


class HtpBackend(Backend):
    def __init__(
        self,
        target: Optional[Target] = None,
        config_file: Optional[str] = None,
        config_dict: Optional[dict] = None,
        context_config: Optional[QNNContextConfig] = None,
        **kwargs,
    ):
        """Initializes the HtpBackend with the given target.

        Args:
            target (Optional[Target]): The target for the backend. Defaults to host machine.
            config_file (Optional[str]): The path to the configuration file. Defaults to None.
            config_dict (Optional[dict]): The configuration dictionary. Defaults to None.
            context_config (Optional[QNNContextConfig]): The context configuration. Defaults to None.
            **kwargs: Additional keyword arguments to set attributes.

        Raises:
            AttributeError: Raised if a keyword argument does not correspond to an existing attribute.
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

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{type(self).__name__} does not have a config for key: {key}")

        self.logger.debug(f"Config dictionary after processing all provided configs: {self._config}")

    @property
    def backend_library(self) -> str | None:
        """Returns the name of the backend library."""
        return get_backend_library(self.target.target_platform_type, BackendType.HTP)

    @property
    def backend_extensions_library(self) -> str | None:
        """Returns the name of the backend extensions library."""
        return get_backend_extension_library(self.target.target_platform_type, BackendType.HTP)

    def get_required_device_artifacts(self, sdk_root: str) -> list:
        """Returns the list of required artifacts in the SDK root.

        Args:
            sdk_root (str): The root directory of the SDK.
        """
        artifacts = []
        device_libs = []
        device_info = self.target.device.device_info
        soc_model = (
            device_info.identifier.soc_model
            if device_info and device_info.identifier and device_info.identifier.soc_model
            else None
        )
        if not soc_model and isinstance(self.target.device, AndroidDevice):
            try:
                detected = self.target.device.get_soc_name()
                soc_model = detected if detected != UNKNOWN_SOC_NAME else None
            except Exception as e:
                self.logger.warning(
                    f"Could not auto-detect soc_model from device, all HTP arch libs will be pushed: {e}"
                )

        device_lib_dir = sdk_root + "/lib/" + self.target.target_name + "/"
        if (
            self.target.target_platform_type == DevicePlatformType.ANDROID
            or self.target.target_platform_type == DevicePlatformType.LINUX_EMBEDDED
        ):
            device_libs = [device_lib_dir + "libQnnHtpPrepare.so"]

        hexagon_libs = []
        htp_archs = ["v68", "v69", "v73", "v75", "v79", "v81", "v85"]

        # Fetch required stub/skel combinations
        if soc_model:
            device_info = DeviceFactory.get_device_soc_details("HTP", soc_model)
            if device_info and device_info.dsp_arch:
                soc_dsp_arch = "v" + str(device_info.dsp_arch)
                if soc_dsp_arch in htp_archs:
                    htp_archs = [soc_dsp_arch]

        for arch in htp_archs:
            htp_arch_stub = device_lib_dir + "libQnnHtp" + arch.upper() + "Stub.so"
            if Path(htp_arch_stub).is_file():
                device_libs.append(htp_arch_stub)

            hexagon_arch_lib_dir = sdk_root + "/lib/hexagon-" + arch + "/unsigned/"
            htp_arch_skel = hexagon_arch_lib_dir + "libQnnHtp" + arch.upper() + "Skel.so"
            if Path(htp_arch_skel).is_file():
                hexagon_libs.append(htp_arch_skel)

        artifacts.extend(device_libs)
        artifacts.extend(hexagon_libs)
        return artifacts

    def get_profiling_artifacts(self) -> list:
        """Returns the profiling artifacts."""
        artifacts: List[Path] = []
        if self._schematic_bins:
            artifacts = [Path(schematic_bin).absolute() for schematic_bin in self._schematic_bins]
        return artifacts

    def clear_profiling_artifacts(self) -> None:
        """Clears the profiling artifacts."""
        self._schematic_bins = None

    def after_generate_hook(self, temp_directory: str, sdk_path: str):
        """A hook that is called after a context binary is generated.

        Args:
            temp_directory(str): A path to the working directory where the context binary was
                                 generated (not the directory where it was written to)
            sdk_path(str): A path to the root of the SDK
        """
        if self.target.target_platform_type in {
            DevicePlatformType.X86_64_LINUX,
            DevicePlatformType.X86_64_WINDOWS_MSVC,
            DevicePlatformType.WOS,
            DevicePlatformType.LINUX_EMBEDDED,
        }:
            self._schematic_bins = glob.glob(f"{temp_directory}{os.sep}*schematic.bin")
