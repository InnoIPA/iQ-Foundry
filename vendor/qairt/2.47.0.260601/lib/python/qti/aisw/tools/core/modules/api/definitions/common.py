# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator, model_validator

from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DeviceCredentials,
    DevicePlatformType,
    RemoteDeviceIdentifier,
)

"""
This module contains the definition for AISWBaseModel class which is a pydantic class derived
from BaseModel, and AISWVersion which is a pydantic class that stores fields that categorize a
semantic versioning scheme.
"""


class AISWBaseModel(BaseModel):
    """Internal variation of a BaseModel"""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, protected_namespaces=(), populate_by_name=True
    )


class OpPackageIdentifier(AISWBaseModel):
    """Defines the custom op package parameters for the net runner module"""

    package_path: Union[str, PathLike]
    interface_provider: str
    target_name: Optional[str] = None
    cpp_stl_path: Optional[Union[str, PathLike]] = None


class AISWVersion(AISWBaseModel):
    """A dataclass that conveys when modifications are made to a module's interface
    or its properties.
    """

    _MAJOR_VERSION_MAX = 15
    _MINOR_VERSION_MAX = 40
    _PATCH_VERSION_MAX = 15
    _PRE_RELEASE_MAX_LENGTH = 26

    major: int = Field(ge=0, le=_MAJOR_VERSION_MAX)  # Backwards incompatible changes to a module
    minor: int = Field(ge=0, le=_MINOR_VERSION_MAX)  # Backwards compatible changes
    patch: int = Field(ge=0, le=_PATCH_VERSION_MAX)  # Backwards compatible bug fixes
    pre_release: str = Field(default="", max_length=_PRE_RELEASE_MAX_LENGTH)

    @model_validator(mode="after")
    def check_allowed_sem_ver(self):
        """Sanity checks a version to ensure it is not all zeros

        Raises:
            ValueError if no version is set
        """
        if self.major == self.minor == self.patch == 0:
            raise ValueError(f"Version: {self.__repr__()} is not allowed")
        return self

    def __str__(self):
        """Formats the version as a string value: "major.minor.patch"
        or "major.minor.patch" if the release tag is set
        """
        if not self.pre_release:
            return f"{self.major}.{self.minor}.{self.patch}"
        return f"{self.major}.{self.minor}.{self.patch}-{self.pre_release}"


class QNNCommonConfig(AISWBaseModel):
    """Specifies the shared parameters supported by both the Context-bin generator and the Net Runner Module.

    Attributes:
        log_level: Specifies max logging level to be set
        set_output_tensors: Specifies a comma-separated list of intermediate output tensor names, for which the outputs
                                              will be written in addition to final graph output tensors
        profiling_level: Option to Enable Profiling
        profiling_option: Option to Set profiling options
        platform_options: Specifies values to pass as platform options
    """

    log_level: Optional[str] = None
    set_output_tensors: Optional[List[str]] = None
    profiling_level: Optional[str] = None
    profiling_option: Optional[str] = None
    platform_options: Optional[Union[str, Dict[str, str]]] = None


class BackendType(str, Enum):
    """
    Enum representing backend types that are supported by a module.
    """

    CPU = "CPU"
    GPU = "GPU"
    HTP = "HTP"
    HTP_MCP = "HTP_MCP"
    AIC = "AIC"
    LPAI = "LPAI"

    @classmethod
    def offline_preparable_backends(cls):
        return [cls.HTP, cls.AIC, cls.HTP_MCP]

    @classmethod
    def quantizable_backends(cls):
        return [cls.CPU, cls.HTP, cls.AIC, cls.HTP_MCP, cls.LPAI]

    @classmethod
    def from_id(cls, backend_id: int) -> "BackendType":
        _ID_TO_BACKEND = {3: cls.CPU, 4: cls.GPU, 6: cls.HTP, 11: cls.HTP_MCP, 8: cls.AIC, 12: cls.LPAI}
        return _ID_TO_BACKEND[backend_id]

    @staticmethod
    def is_valid_backend(backend_str: str):
        return any(backend_str == backend for backend in BackendType._member_map_)

    @staticmethod
    def backend_to_id(backend_str: str):
        return BackendType(backend_str).id

    @property
    def id(self):
        _BACKEND_TO_ID = {"CPU": 3, "GPU": 4, "HTP": 6, "HTP_MCP": 11, "AIC": 8, "LPAI": 12}

        return _BACKEND_TO_ID[self._value_]


class Target(AISWBaseModel):
    """Defines the type of device to be used by the module, optionally including device identifiers
    and connection parameters for remote devices.

    Attributes:
        type (DevicePlatformType): The type of device platform to be used
        identifier (Optional[RemoteDeviceIdentifier]): The identifier of the device.
                                                        Defaults to None.
        credentials (Optional[DeviceCredentials]): The credentials for the device. Defaults to
        None.
        soc_model (Optional[str]): The soc name of the device ex: SA8295. Defaults to
        None.
    """

    type: DevicePlatformType
    identifier: Optional[RemoteDeviceIdentifier] = None
    credentials: Optional[DeviceCredentials] = None
    soc_model: Optional[str] = None


class ModelType(str, Enum):
    """Enum representing the different types of models that can be used.

    The model type is determined by the file extension of the model file.

    Attributes:
        QNN_MODEL_LIBRARY (str): QNN model library type (file extension: .so)
        QNN_CONTEXT_BINARY (str): QNN context binary type (file extension: .bin)
        DLC (str): DLC type (file extension: .dlc)
    """

    QNN_MODEL_LIBRARY = "QnnModelLibrary"
    QNN_CONTEXT_BINARY = "QnnContextBinary"
    DLC = "DLC"

    @classmethod
    def from_file_extension(cls, file_extension: str) -> "ModelType":
        """Returns the model type corresponding to the given file extension."""
        model_type_mapping = {
            ".so": cls.QNN_MODEL_LIBRARY,
            ".bin": cls.QNN_CONTEXT_BINARY,
            ".dlc": cls.DLC,
        }
        try:
            return model_type_mapping[file_extension.lower()]
        except KeyError:
            raise ValueError(f"Unsupported file extension: {file_extension}")


class ModelConfig(AISWBaseModel):
    """Describes a model in a form that can be consumed by a module.

    Attributes:
        path (FilePath): Path to a model file.
    """

    path: FilePath

    @field_validator("path")
    @classmethod
    def validate_path_extension(cls, path_value: FilePath) -> FilePath:
        """Validates that the file path has a valid extension."""
        file_ext = path_value.suffix.lower()
        if ModelType.from_file_extension(file_ext) is None:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        return path_value

    @property
    def model_type(self) -> ModelType:
        """Returns the type of model represented by this instance.

        Returns:
            ModelType: The type of model (e.g. 'QnnModelLibrary', 'QnnContextBinary', or 'DLC')
        """
        file_ext = self.path.suffix.lower()
        return ModelType.from_file_extension(file_ext)

    @property
    def name(self) -> str:
        """Returns the name of model represented by this instance.

        Returns:
            str: The name of model (e.g. 'model_name.so', 'model_name.bin', or 'model_name.dlc')
        """
        return self.path.name

    def __repr__(self):
        """Returns a string representation of the ModelConfig object."""
        return f"ModelConfig(type='{self.model_type}', name='{self.name}', path='{self.path}')"


class ProfilingLevel(str, Enum):
    """Enum representing profiling levels that are supported by a module."""

    BASIC = "basic"
    DETAILED = "detailed"
    CLIENT = "client"
    BACKEND = "backend"


class ProfilingOption(str, Enum):
    """Enum representing profiling options that are supported by a module."""

    OPTRACE = "optrace"


class ProfilingData(AISWBaseModel):
    """Defines a module's profiling output.

    Attributes:
        profiling_log (Path): A path to the generated profiling log.
        backend_profiling_artifacts: An optional list of paths to any backend-specific profiling
                                     artifacts that were generated.
    """

    profiling_log: PathLike
    backend_profiling_artifacts: Optional[List[Path]] = None


@dataclass
class QNNContextConfig:
    """Configuration to define options related to context priority and graph switching,
    to be given as input to context-bin-generator and net-runner module.

    Attributes:
        context_priority (str): To specify priority of the context
        async_execute_queue_depth (int): Number of executions that can be in the queue at a given time
        enable_graphs (List[str]): List of graphs to enable
        memory_limit_hint (int): Peak memory limit hint of a deserialized context in MBs
        is_persistent_binary (bool): Boolean to indicate availability of context binary pointer
                                     during creation of context from binary
        cache_compatibility_mode (str): Mode to check whether cache record is optimal for the device
    """

    context_priority: Optional[str] = None
    async_execute_queue_depth: Optional[int] = 0
    enable_graphs: Optional[List[str]] = None
    memory_limit_hint: Optional[int] = 0
    is_persistent_binary: Optional[bool] = False
    cache_compatibility_mode: Optional[str] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validates the given context_config."""
        valid_priorities = {"low", "normal", "normal_high", "high"}
        valid_cache_modes = {"permissive", "strict"}

        if self.context_priority and self.context_priority not in valid_priorities:
            raise ValueError(
                f"Invalid context_priority: {self.context_priority}. Expected one of {valid_priorities}."
            )

        if self.cache_compatibility_mode and self.cache_compatibility_mode not in valid_cache_modes:
            raise ValueError(
                f"Invalid cache_compatibility_mode: {self.cache_compatibility_mode}. "
                f"Expected one of {valid_cache_modes}."
            )
