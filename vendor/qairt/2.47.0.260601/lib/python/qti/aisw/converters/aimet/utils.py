# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import re
import os
import logging
logger = logging.getLogger('AIMET Quantizer')


def replace_invalid_chars_for_variable(name: str) -> str:
    """
    Replace invalid chars such as dot, slash, ...
    :param name: name to replace invalid chars
    :return: cleansed string can be used Python variable name
    """
    if not name.isidentifier():
        found_list = re.findall(r"\w+", name)
        if found_list:
            name = "_".join(found_list)
        if name[0].isdigit():
            name = '_' + name
        if not name.isidentifier():
            error_str = f"Unable to produce a valid model name name from {name}"
            logger.error(error_str)
            raise RuntimeError(error_str)
    return name


def get_input_list_abs(input_list, output_names, datadir):
    """ Get the absolute path for modified input list """
    with open(input_list, "r") as file:
        input_0 = file.readline()

    path_example = input_0.split()[0]
    if ":=" in path_example:
        path_example = path_example.split(":=")[1]

    if os.path.exists(path_example):
        return input_list

    else:
        # Re-write input paths
        new_lines = []
        input_file_dir = os.path.dirname(input_list)
        with open(input_list, "r") as file:
            lines = file.readlines()
            if lines[0][0] == '#' or lines[0][0] == '%':
                output_names = lines[0][1:].split(' ')
                lines = lines[1:]
            for line in lines:
                new_line = []
                for input_path in line.split():
                    if ":=" in input_path:
                        input_name, path = input_path.split(":=")
                        new_path = f"{input_name}:={os.path.join(input_file_dir, path)}"
                    else:
                        new_path = os.path.join(input_file_dir, line)
                    new_line.append(new_path)
                new_line = " ".join(new_line)
                new_lines.append(new_line)

        temp_input_list = os.path.join(datadir, 'temp_input_list.txt')
        with open(temp_input_list, 'w') as f:
            for line in new_lines:
                f.write(line+"\n")

        return temp_input_list
