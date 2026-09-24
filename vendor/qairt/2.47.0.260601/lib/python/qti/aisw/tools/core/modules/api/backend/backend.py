# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.core.model_level_api.workflow.workflow import WorkflowMode
from qti.aisw.tools.core.modules.api.definitions.common import QNNContextConfig
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class Backend(ABC):
    def __init__(self, target: Optional[Target] = None, **kwargs):
        """Initializes the base class with the given target."""
        self._default_target = target is None
        if target is None:
            target = Target.create_host_target()
        self.target: Target = target
        self._workflow_mode = None
        self._config: Dict = {}
        self._context_config: Optional[QNNContextConfig] = None
        self._op_packages: List = []
        self.log_area = LogAreas.register_log_area(self.__class__.__name__)
        self.logger = QAIRTLogger.register_area_logger(area=self.log_area, level="INFO")

    @property
    def workflow_mode(self):
        """Returns the current workflow mode."""
        return self._workflow_mode

    @workflow_mode.setter
    def workflow_mode(self, mode: WorkflowMode):
        """Sets the workflow mode.

        Args:
            mode (WorkflowMode): Specifies the workflow type
        """
        self._workflow_mode = mode
        self._workflow_mode_setter_hook(mode)

    def _workflow_mode_setter_hook(self, mode: WorkflowMode):
        """Implement this method in a backend subclass if:

        - The default target varies by workflow (e.g., HTP typically generates context
        binaries on x86 but runs inferences on Android).In this scenario, `self.target`
        should be initialized to `None` in the constructor and set within this method if
        `self._default_target` is `True` (i.e., if the class was instantiated without a target).
        - The backend does not support a particular workflow (e.g., CPU does not support context
        binary generation). In this case, raise a `ValueError` if an unsupported workflow is requested.

        Args:
            mode (WorkflowMode): Specifies the workflow type
        """
        pass

    @property
    @abstractmethod
    def backend_library(self) -> str | None:
        """Returns the name of the backend library."""
        pass

    @property
    @abstractmethod
    def backend_extensions_library(self) -> str | None:
        """Returns the name of the backend extensions library."""
        pass

    def get_required_device_artifacts(self, sdk_root: str) -> list:
        """Returns the list of required artifacts in the SDK root.

        Args:
            sdk_root (str): The root directory of the SDK.
        """
        return []

    def get_config_json(self):
        """Returns the configuration as a JSON string."""
        return json.dumps(self._config, indent=2) if self._config else None

    def update_config(self, config_dict: Optional[dict], config_file: Optional[str]):
        """Updates the configuration using the provided dictionary and/or configuration file.

        Args:
            config_dict (Optional[dict]): The configuration dictionary.
            config_file (Optional[str]): The path to the configuration file.
        """
        new_config = {}

        if config_file:
            with open(config_file, "r") as f:
                new_config = json.load(f)

        if config_dict:
            new_config.update(config_dict)

        if new_config:
            self._config = new_config

    def get_context_config(self):
        """Returns the context configuration as a dictionary."""
        return self._context_config

    def register_op_package(
        self,
        path: str,
        interface_provider: str,
        target: Optional[str] = None,
        cpp_stl_path: Optional[str] = None,
    ) -> None:
        """Registers an op package with the given path, interface provider, target, and optional STL path.

        Args:
            path (str): The path to the op package library.
            interface_provider (str): The interface provider for the op package.
            target (Optional[str]): The target platform for the op package. Defaults to None.
            cpp_stl_path (Optional[str]): The optional path to the C++ STL library. Defaults to None.

        Raises:
            FileNotFoundError: If the op package library or the cpp_stl_path cannot be found.
        """
        op_package_path = Path(str(path))
        if not op_package_path.is_file():
            raise FileNotFoundError(f"Could not find op package library: {op_package_path}")
        if cpp_stl_path is not None:
            possible_cpp_stl = Path(str(cpp_stl_path))
            if not possible_cpp_stl.is_file():
                raise FileNotFoundError(f"Could not find cpp_stl_path: {possible_cpp_stl}")
        self._op_packages.append([op_package_path.resolve(), interface_provider, target, cpp_stl_path])

    def get_registered_op_packages(self):
        """Returns the list of registered op packages."""
        return self._op_packages

    def get_profiling_artifacts(self):
        """Returns the profiling artifacts."""
        return None

    def clear_profiling_artifacts(self):
        """Clears the profiling artifacts."""
        pass

    def before_run_hook(self, temp_directory: str, sdk_path: str):
        """A hook that is called before an inference is run. If there are any backend-specific
        workflows that must be done before inferences are run, this function should be implemented
        by the corresponding Backend class.

        Args:
            temp_directory(str): A path to the working directory where the inference will be run.
            sdk_path(str): A path to the root of the SDK
        """
        pass

    def after_run_hook(self, temp_directory: str, sdk_path: str):
        """A hook that is called after an inference is run. If there are any backend-specific
        workflows that must be done after inferences are run, this function should be implemented
        by the corresponding Backend class.

        Args:
            temp_directory(str): A path to the working directory where the inference was  run
            sdk_path(str): A path to the root of the SDK
        """
        pass

    def before_generate_hook(self, temp_directory: str, sdk_path: str):
        """A hook that is called before a context binary is generated. If there are any
        backend-specific workflows that must be done before context binary generation, this function
        should be implemented by the corresponding Backend class.

        Args:
            temp_directory(str): A path to the working directory where the context binary will be
                                 generated (not the directory where it will be written)
            sdk_path(str): A path to the root of the SDK
        """
        pass

    def after_generate_hook(self, temp_directory: str, sdk_path: str):
        """A hook that is called after a context binary is generated. If there are any backend-specific
        workflows that must be done after context binary generation, this function should be
        implemented by the corresponding Backend class.

        Args:
            temp_directory(str): A path to the working directory where the context binary was
                                 generated (not the directory where it was written to)
            sdk_path(str): A path to the root of the SDK
        """
        pass

    def enable_debug(self) -> None:
        """Enable debugging behaviour for the module"""
        level = logging.DEBUG
        QAIRTLogger.set_area_logger_level(self.log_area, level)


# a descriptor to store backend configs in a dictionary that can be directly serialized to json
# while also providing dot notation access to individual configs, allowing them to be set
# dynamically. This strategy can be used for all backends whose configs are solely key-value pairs
class BackendConfig:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return instance._config.get(self._name)

    def __set__(self, instance, value):
        instance._config[self._name] = value
