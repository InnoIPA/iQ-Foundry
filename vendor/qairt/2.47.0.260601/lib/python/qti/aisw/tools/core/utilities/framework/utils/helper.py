# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import os
import re
from os import PathLike
from typing import Dict

import numpy as np
from qti.aisw.tools.core.utilities.framework.utils.constants import (
    FrameworkExecuteReturn,
    MaxLimits,
)
from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger


helper_log_area = LogAreas.register_log_area("Helper")


class Helper:
    """Helper class for common functions.

    Attributes:
        None
    """

    @classmethod
    def transform_node_names(cls, node_name: str) -> str:
        """Transforms the node names to follow converter's node naming conventions.

        All consecutive special characters will be replaced by an underscore '_',
        and node names not beginning with an alphabet will be prepended with an underscore '_'.

        Args:
            node_name (str): The original node name.

        Returns:
            str: The transformed node name.

        Raises:
            None
        """
        if not isinstance(node_name, str):
            node_name = str(node_name)

        transformed_name = re.sub(r"[^a-zA-Z0-9_]+", "_", node_name)

        # Prepend an underscore if the transformed name does not start with an alphabet or _
        if not (transformed_name[0].isalpha() or transformed_name[0] == "_"):
            transformed_name = "_" + transformed_name

        return transformed_name

    @classmethod
    def save_outputs(cls, data: np.ndarray, data_path: str | PathLike, data_type: str) -> None:
        """Saves the output data to a binary file.

        Args:
            data (np.ndarray): The output data to be saved.
            data_path (str | PathLike): The path where the output file should be saved.
            data_type (str): The data type of the output data.

        Returns:
            None

        Raises:
            None
        """
        data.astype(data_type).tofile(data_path)

    @classmethod
    def save_to_json_file(cls, data: str, json_path: str | PathLike) -> None:
        """Saves the given data to a JSON file at the specified path.

        Args:
            data (str): The data to be saved as JSON.
            json_path (str | PathLike): The path where the JSON file should be saved.

        Returns:
            None
        """
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @classmethod
    def save_output_to_file(
        cls, output_info: Dict[str, FrameworkExecuteReturn], output_dir: str | PathLike
    ) -> None:
        """Saves the output information to a file in JSON format.

        Args:
            output_info (Dict[str, FrameworkExecuteReturn]): A dictionary containing the output information.
            output_dir (str | PathLike): The directory path where the output file should be saved.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        data_path = os.path.join(output_dir, "{}{}")
        tensor_info = {}

        for output_tensor, data in output_info.items():
            # Transform output tensor name to avoid dumping outputs in sub-folders.
            output_tensor = Helper.transform_node_names(output_tensor)

            """
            Most of the systems does not allow to create a file name, more than 255 bytes,
            hence skipping to dump those tensor files.
            """
            if len((output_tensor + ".raw").encode("utf-8")) > MaxLimits.max_file_name_size.value:
                continue

            file_path = data_path.format(output_tensor, ".raw")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            Helper.save_outputs(data, file_path, "float32")

            try:
                if type(data) is list:
                    data = np.array(data, dtype=np.float32)
            except Exception as e:
                raise Exception("Encountered Error: {}".format(str(e)))

            if not data.size or data.dtype == bool:
                if data.size == 0:
                    tensor_info[output_tensor] = (
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    )
                else:
                    tensor_info[output_tensor] = (
                        str(data.dtype),
                        data.shape,
                        data.tolist(),
                        data.tolist(),
                        data.tolist(),
                    )
            else:
                tensor_info[output_tensor] = (
                    f"type = {str(data.dtype)}",
                    f"shape = {data.shape}",
                    f"min = {str(round(np.min(data), 3))}",
                    f"max = {str(round(np.max(data), 3))}",
                    f"median = {str(round(np.median(data), 3))}",
                )

        tensor_info_json = data_path.format("profile_info", ".json")
        Helper.save_to_json_file(tensor_info, tensor_info_json)


logger = QAIRTLogger.register_area_logger(
    helper_log_area, level="INFO", formatter_val="extended", handler_list=["dev_console"]
)
