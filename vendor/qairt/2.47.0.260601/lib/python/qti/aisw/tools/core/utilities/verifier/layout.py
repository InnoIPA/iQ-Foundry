# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Any

from pydantic import FilePath
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper


class TensorLayout:
    """This class provides the information of layout for a given dlc file."""

    def get_layout_info_from_dlc(self, dlc: FilePath) -> dict:
        """This method extracts the permute order and dimensions from .dlc file

        Args:
            dlc (FilePath): Input path to dlc file.

        Returns:
            dict : Dictionary containing layout info of intermediate tensors in IRgraph.
        """
        ir_graph = self._get_ir_graph(str(dlc))
        layout_info = dict()

        for tensor_name, tensor_info in ir_graph.get_tensor_map().items():
            transformed_name = Helper.transform_node_names(tensor_name)
            layout_info[transformed_name] = {
                "permute_order_to_src": tensor_info.get_permute_order_to_src(),
                "dims": tensor_info.dims(),
            }

        return layout_info

    @staticmethod
    def _get_ir_graph(dlc: FilePath) -> Any:
        """Get IrGraph object from dlc.

        Args:
            dlc (FilePath): Path to dlc file.

        Raises:
            RuntimeError if unable to read the input dlc file.

        Returns:
            IrGraph instance from dlpacker reader.
        """
        try:
            from qti.aisw.dlc_utils import modeltools

            model_reader = modeltools.IrDlcReader()
            model_reader.open(dlc)
            ir_graph = model_reader.get_ir_graph()
            model_reader.close()
        except Exception as e:
            raise RuntimeError("Unable to load Dlc Model using modeltools: {}".format(str(e)))

        return ir_graph
