# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import importlib
import logging
import sys
from typing import Any, Optional

import pkg_resources


class Helper:
    """Utility class contains common utility methods
    To use:
    >>>Helper.safe_import_package(package_name='torch', recommended_package_version='1.13.1')
    >>>Helper.safe_import_package(package_name='numpy')
    """

    mismatched_packages = []

    @classmethod
    def safe_import_package(
        cls, package_name: str, recommended_package_version: Optional[str] = None
    ) -> Any | None:
        """Imports a Python package safely, handling potential import errors.

        Args:
            package_name (str): The name of the package to import.
            recommended_package_version (str): The version of the package that
                                    is recommended for use with this library.

        Returns:
            Any: The imported package or an error message if import fails.
        """
        try:
            # Attempt to import the package using importlib
            package = importlib.import_module(package_name)
        except ImportError:
            # Handle import errors and log a warning message
            logging.error(
                f"Failed to import {package_name}. Kindly refer to SDK documentation"
                f" and install supported version of {package_name}"
            )
            sys.exit(1)
        else:
            if recommended_package_version is not None:
                # Attempt to get the installed package version using pkg_resources
                try:
                    detected_package_version = pkg_resources.get_distribution(package_name).version
                except Exception:
                    # Fallback to getting the version from the package itself from __version__ attribute
                    try:
                        detected_package_version = package.__version__
                    except AttributeError:
                        logging.error(f"Failed to detect installed version of {package_name}")

                if (
                    detected_package_version != recommended_package_version
                    and package_name not in cls.mismatched_packages
                ):
                    logging.warning(
                        f"{package_name} installed version: {detected_package_version}"
                        f", and Recommended version: {recommended_package_version}"
                    )
                    cls.mismatched_packages.append(package_name)
            return package
