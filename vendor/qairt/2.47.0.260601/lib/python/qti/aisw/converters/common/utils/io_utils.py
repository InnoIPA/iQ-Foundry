# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
from qti.aisw.converters.common.utils.converter_utils import *


def check_validity(resource, *, is_path=False, is_directory=False, must_exist=True, create_missing_directory=False,
                   extensions=[]):
    resource_path = os.path.abspath(resource)
    # Check to see if output_path is what needs to be validated
    if create_missing_directory:
        log_debug("Checking if output_path directory exists")
        if is_directory:
            directory = resource_path
        else:
            # Split the path into the directory and filename
            directory, filename = os.path.split(resource_path)

        # Now check if directory exists, if not then create the directory
        if not os.path.exists(directory):
            try:
                log_debug("Creating output_path directory: " + str(directory))
                os.makedirs(directory)
            except OSError as error:
                raise OSError(str(error) + '\n{} is not a valid directory path'.format(str(directory)))

    if is_path and os.path.isdir(resource_path):
        # For the case that resource path can be either dir or file
        is_directory = True
    if must_exist and not os.path.exists(resource_path):
        raise IOError('{} does not exist'.format(str(resource)))
    elif not is_directory:
        if must_exist and os.path.exists(resource_path) and not os.path.isfile(resource_path):
            raise IOError('{} is not a valid {} file'.format(str(resource), str(extensions)))
        if extensions and \
                not any([os.path.splitext(resource_path)[1] == str(extension) for extension in extensions]):
            raise IOError("{} is not a valid file extension: {}".format(resource, str(extensions)))
    else:
        if os.path.exists(resource_path) and not os.path.isdir(resource_path):
            raise IOError('{} is not a valid directory'.format(str(resource)))
        elif extensions:
            raise ValueError("Directories cannot have a file extension".format(str(resource)))


def get_default_output_directory(base_filepath, new_dir_name):
    base_directory = os.path.dirname(os.path.abspath(base_filepath))
    output_dir = os.path.join(base_directory, new_dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir