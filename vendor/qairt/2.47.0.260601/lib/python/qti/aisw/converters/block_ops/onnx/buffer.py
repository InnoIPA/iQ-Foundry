# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""ONNX Buffer Block Operator code"""

import onnx
from onnx import helper
from onnxscript import script
from onnxscript.onnx_types import BOOL
from onnxscript.values import OnnxFunction
from typing import Optional
from qti.aisw.converters.block_ops.onnx.onnx_block_op_base import QnnOnnxBlockOp, _get_onnx_opset_version, qcom_block_op_domain
from qti.aisw.converters.common.converter_ir.op_properties.buffer import (
    BUFFER_MODE_DEFAULT_VAL,
    BUFFER_STRIDE_DEFAULT_VAL,
    BUFFER_PADDING_DEFAULT_VAL
)


class Buffer(QnnOnnxBlockOp):
    """ONNX Buffer Block Operator class

    This class represents a Buffer BlockOperator in ONNX. This operator is
    defined in onnxscript, so both getOnnxScriptFunc and getOnnxFuncProto are
    implemented.

    :ivar min_opset: Initialized to 8.
    :ivar max_opset: Initialized to 21.
    """

    def __init__(self, onnx_opset_version: int, aisw_opset_version: int = 1):
        self.name = "Buffer"
        self.min_opset = 8
        self.max_opset = 21
        super(Buffer, self).__init__(onnx_opset_version, aisw_opset_version)

    def getOnnxScriptFunc(self) -> OnnxFunction:
        opset = self.opset
        qti_aisw = self.aisw_opset

        @script(qti_aisw, default_opset=opset)
        def Buffer(
            input,
            reset: BOOL = opset.Constant(value=False),
            # Attributes
            buffer_size: int = None,
            buffer_dim: int = None,
            buffer_padding: int = BUFFER_PADDING_DEFAULT_VAL,
            stride: int = BUFFER_STRIDE_DEFAULT_VAL,
            mode: int = BUFFER_MODE_DEFAULT_VAL,
        ):
            """Buffer operator in ONNX script.

            This is a dummy implementation for running it on our test framework.
            The output will not match the QNN op def and it should not be used
            as a golden reference implementation.
            """
            return opset.Identity(input)

        return Buffer

    def getOnnxFuncProto(self) -> onnx.FunctionProto:
        return self.getOnnxScriptFunc().to_function_proto()


def addOnnxBufferBlockOp(
    model: onnx.ModelProto,
    input_tensor_name: str,
    buffer_size: int,
    buffer_dim: int,
    reset_tensor: Optional[onnx.TensorProto] = None,
    mode: int = BUFFER_MODE_DEFAULT_VAL,
    stride: int = BUFFER_STRIDE_DEFAULT_VAL,
    padding: int = BUFFER_PADDING_DEFAULT_VAL,
) -> onnx.ModelProto:
    """Insert a Buffer Block Op node onto an existing tensor in the graph.

    Unlike StatefulLstm/StatefulGru, there is no existing ONNX Buffer op to
    replace. This helper inserts a new Buffer node that reads from an existing
    tensor in the graph and produces a new buffered output tensor.

    Tensor names are preserved exactly as given, so numerically-named tensors
    (e.g. ``"19"``) will produce ``"19_reset"`` and ``"19_buffered"`` rather
    than the mangled names that ONNXScript would generate.

    :param model: The model to modify.
    :type model: onnx.ModelProto
    :param input_tensor_name: Name of the existing tensor to buffer.
    :type input_tensor_name: str
    :param buffer_size: Number of frames to buffer.
    :type buffer_size: int
    :param buffer_dim: Dimension along which to buffer.
    :type buffer_dim: int
    :param reset_tensor: The reset tensor. If not provided, a new tensor named
        ``<input_tensor_name>_reset`` is created and added to the graph inputs.
        If provided, the caller is responsible for adding it to the graph inputs.
    :type reset_tensor: Optional[onnx.TensorProto]
    :param mode: Buffer mode attribute (default: BUFFER_MODE_DEFAULT_VAL).
    :type mode: int
    :param stride: Buffer stride attribute (default: BUFFER_STRIDE_DEFAULT_VAL).
    :type stride: int
    :param padding: Buffer padding attribute (default: BUFFER_PADDING_DEFAULT_VAL).
    :type padding: int
    :return: The modified model.
    :rtype: onnx.ModelProto
    :raises ValueError: If an ONNX opset version cannot be found in the model.
    """
    if reset_tensor is None:
        reset_tensor = onnx.helper.make_tensor_value_info(
            f"{input_tensor_name}_reset",
            onnx.TensorProto.BOOL,
            [],  # reset is 0D, so specify shape as empty
        )
        model.graph.input.append(reset_tensor)

    output_tensor_name = f"{input_tensor_name}_buffered"
    buffer_node = onnx.helper.make_node(
        "Buffer",
        inputs=[input_tensor_name, reset_tensor.name],
        outputs=[output_tensor_name],
        domain=qcom_block_op_domain(),
        buffer_size=buffer_size,
        buffer_dim=buffer_dim,
        mode=mode,
        stride=stride,
        buffer_padding=padding,
    )

    first_consumer_idx = None
    for node_idx, node in enumerate(model.graph.node):
        for inp_idx, inp in enumerate(node.input):
            if inp == input_tensor_name:
                node.input[inp_idx] = output_tensor_name
                if first_consumer_idx is None:
                    first_consumer_idx = node_idx

    if first_consumer_idx is not None:
        model.graph.node.insert(first_consumer_idx, buffer_node)
    else:
        model.graph.node.append(buffer_node)

    model_version = _get_onnx_opset_version(model)
    existing_func_names = {f.name for f in model.functions}
    if "Buffer" not in existing_func_names:
        model.functions.extend([Buffer(model_version).getOnnxFuncProto()])
    existing_opset_domains = {o.domain for o in model.opset_import}
    if qcom_block_op_domain() not in existing_opset_domains:
        model.opset_import.extend([helper.make_opsetid(qcom_block_op_domain(), 1)])
    return model
