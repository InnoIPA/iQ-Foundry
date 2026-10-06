# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Dict, List, Optional, Text, Tuple, Union

import numpy as np
import onnx
from onnx import ModelProto
from qti.aisw.converters.common.model_runtime_interface import (
    IModelRuntime,
)
from qti.aisw.converters.common.utils.framework_utils import TensorInfo
from qti.aisw.converters.onnx.util import (
    get_input_info,
    get_output_info,
    run_ort_inference,
)


class ONNXModelRuntime(IModelRuntime):
    def __init__(
        self,
        model_or_proto: Union[Text, ModelProto],
        input_dims_dict: Dict = None,
        define_symbols: Dict[str, str] = None,
        batch: int = None,
    ) -> None:
        """
        Initializing ONNX runtime object.

        :param Union[Text,ModelProto] model_path: Onnx model path or model proto.
        :param Dict input_dims_dict: Mapping of graph input names with their
            shapes, defaults to None.
        :param Dict[str, str] define_symbols: Mapping of onnx defined symbols
            and their replacement values, defaults to None.
        :param int batch: Value of batch size to be overridden, defaults to None.
        :raises AttributeError: If neither of model path nor onnx model proto
            is provided.
        """
        if isinstance(model_or_proto, str):
            self.model = onnx.load(model_or_proto)
        elif isinstance(model_or_proto, ModelProto):
            self.model = model_or_proto
        else:
            raise AttributeError(
                f"Expected Path or Model Proto but received: {type(model_or_proto)}"
            )
        self.input_dims_dict = input_dims_dict
        self.define_symbols = define_symbols
        if self.define_symbols:
            self.define_symbols = {
                symbol: int(value) for symbol, value in self.define_symbols.items()
            }
        if self.input_dims_dict is not None:
            for name, dims in self.input_dims_dict.items():
                dims = list(map(int, dims.split(",")))
                self.input_dims_dict[name] = dims
        self.batch = batch

    def get_input_info(self) -> Dict[Text, TensorInfo]:
        """
        Gets input tensor information.

        :return Dict[Text, TensorInfo]: Input names to tensor information.
        """
        graph_inputs = {inp.name: inp for inp in self.model.graph.input}
        input_info = get_input_info(self.model, use_onnx_def_symbols=True)
        if self.define_symbols:
            for input_name, input_data in input_info.items():
                shape = input_data.shape
                has_dynamic_shapes = any([isinstance(s, str) for s in shape])
                if not has_dynamic_shapes:
                    continue
                for i, s in enumerate(shape):
                    if isinstance(s, str):
                        if s not in self.define_symbols:
                            raise RuntimeError(
                                f"Input '{input_name}' have "
                                f"dynamic shape '{s}' at index '{i}'. No "
                                "value found in onnx defined symbol for "
                                "this shape."
                            )
                        input_info[input_name].shape[i] = self.define_symbols[s]
        if self.batch:
            for input_name, input_data in input_info.items():
                input_info[input_name].shape[0] = self.batch
                graph_inputs[input_name].type.tensor_type.shape.dim[
                    0
                ].dim_value = self.batch

        if self.input_dims_dict:
            for input_name in self.input_dims_dict:
                if input_name not in graph_inputs:
                    raise RuntimeError(
                        f"Intermediate tensor '{input_name}' "
                        "has been made as graph input. This is not supported "
                        "in validator functionality."
                    )
                new_shape = self.input_dims_dict[input_name]
                input_info[input_name].shape = new_shape
                for i, s in enumerate(new_shape):
                    graph_inputs[input_name].type.tensor_type.shape.dim[i].dim_value = s
        return input_info

    def get_output_info(self) -> Dict[Text, TensorInfo]:
        """
        Gets output tensor information

        :return Dict[Text, TensorInfo]: Output names to tensor information.
        """
        return get_output_info(self.model)

    def execute_inference(
        self, inputs: Dict[Text, np.ndarray], output_names: Optional[List] = []
    ) -> Tuple[bool, Dict[Text, np.ndarray]]:
        """
        Run the inference of given model.

        :param Dict[Text, np.ndarray] input_data: Dict containing input tensor name
            to corresponding tensor data.
        :param Optional[List] output_names:Optional list of output names for
            inference.
        :return Tuple[bool, Dict[Text, np.ndarray]]: Tuple of two values. 1st value
            represents the status of inference and 2nd value represents the Dict
            containing output tensor name as key and its computed numpy array output as value.
        """
        return run_ort_inference(
            self.model, input_data=inputs, output_node_names=output_names
        )
