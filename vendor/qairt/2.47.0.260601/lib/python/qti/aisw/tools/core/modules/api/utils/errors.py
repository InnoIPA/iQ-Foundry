# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================


class SchemaVersionError(AttributeError):
    """Intended for version related errors"""

    pass


class SchemaFieldTypeError(TypeError):
    """Intended for schema field errors"""

    pass


class SchemaFieldValueError(ValueError):
    """Intended for schema field value errors"""

    pass


class ModuleComplianceError(AttributeError):
    """Intended for module compliance errors"""

    pass
