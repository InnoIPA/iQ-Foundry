# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# =============================================================================

from abc import ABC, abstractmethod
from typing import Any, Generic, List, Mapping, Optional, Type, TypeVar, Union

from .schema import ModuleSchema, ModuleSchemaVersion, SchemaVersionError

"""
This module contains a class `Module` that serves as an abstract base class for other modules.
While modules may be freely extended, it is expected that module implementations adhere to the 
interface. Additionally, modules should
follow compliance rules which can be found in directory <compliance>.
"""

ModuleSchemaT = TypeVar("ModuleSchemaT", bound=ModuleSchema)


class Module(ABC, Generic[ModuleSchemaT]):
    """Abstract base class for modules. This class provides a structure that other concrete Modules
    inherit from.
    It manages a schema (ModuleSchema) which describes its properties
    """

    _SCHEMA: Optional[Type[ModuleSchemaT]] = None
    _PREVIOUS_SCHEMAS: List[Type[ModuleSchemaT]] = []
    _LOGGER: Any = None

    def __init__(self, logger=None):
        """Initializes a new instance of the `Module` class.

        Args:
           logger: The logger to be used by the module. If not provided, logging will be determined
            by the module.
        """
        self._logger = logger if logger else self._LOGGER

    @classmethod
    def _all_schemas(cls) -> List[Type[ModuleSchemaT]]:
        """Returns:
        All the schemas associated with this module in a list
        starting from the current schema followed by any previous schemas.
        """
        if not cls._SCHEMA:
            return []
        if cls._PREVIOUS_SCHEMAS:
            return [cls._SCHEMA, *cls._PREVIOUS_SCHEMAS]
        return [cls._SCHEMA]

    @abstractmethod
    def properties(self) -> Mapping[str, Any]:
        """This function should describe some basic properties of this module. Ideally, it should be
        serializable and query-able. Suggested types are JSON and YAML. YAML should be used for
        properties that can be freely edited,
        while JSON should be used for generated properties that should remain fixed.

        Returns:
            A dictionary-like object that describes the properties of this module. The default
            should be pydantic's model_json_schema if no customization is needed.

        """

    @classmethod
    def get_schema(
        cls, version: Optional[Union[str, ModuleSchemaVersion]] = None
    ) -> Optional[Type[ModuleSchemaT]]:
        """This functions returns the module schema that matches the version specified.

        Args:
            version: A string that resolves to a valid ModuleSchemaVersion or a valid
                     ModuleSchemaVersion or None.

        Returns:
            The module schema that matches the specified version, or the latest schema if version
            is None
        Raises:
            May raise an AttributeError if no schema is defined for this module
            May raise a SchemaVersionError if the version is not known
            May raise a TypeError if the version passed is not a str or ModuleSchemaVersion
        """
        if not cls._SCHEMA:
            raise AttributeError("No schema defined for this module")

        if not version:
            return cls._SCHEMA

        if not isinstance(version, (str, ModuleSchemaVersion)):
            raise TypeError(
                f"Unknown type passed for version:{version!r} expected {type(str)} or"
                f" {ModuleSchemaVersion!r} "
            )

        if isinstance(version, str):
            previous_schema_match = list(
                filter(
                    lambda previous_schema: previous_schema.check_version_str(version), cls._PREVIOUS_SCHEMAS
                )
            )
        else:
            previous_schema_match = list(
                filter(
                    lambda previous_schema: version == previous_schema.get_version(), cls._PREVIOUS_SCHEMAS
                )
            )

        if previous_schema_match:
            cls._LOGGER.info(f"Requested version: {version} matches previous schema version")
            return previous_schema_match[0]
        elif isinstance(version, str) and cls._SCHEMA.check_version_str(version):
            return cls._SCHEMA
        elif version == cls._SCHEMA.get_version():
            return cls._SCHEMA
        else:
            raise SchemaVersionError(f"Unknown version provided: {version}")

    @classmethod
    def get_schemas(cls) -> List[Type[ModuleSchemaT]]:
        """This functions returns all the Module Schemas associated with this module.
        Note the return type is the Module Schema class, not an instance of a Module Schema.

        Returns:
            A list of types for all the module's schemas
        """
        return cls._all_schemas()

    @abstractmethod
    def get_logger(self) -> Any:
        """This should return an instance of the logger that is used. The type hint should
        reflect the actual logger type.


        Returns:
            The logger used by the module.
        """

    @abstractmethod
    def enable_debug(self, debug_level: int, **kwargs: Any) -> Optional[bool]:
        """Abstract method that should be implemented by any concrete class that inherits
        from `Module`. This method should enable debugging behavior for the module.

        Args:
            debug_level (int): The level of debugging to be enabled.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            True if debugging is enabled, False otherwise. If no debugging is possible at all,
            then this function may return None.
        """
