# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""ONNX StatefulLstm Block Operator code"""

import logging
import onnx
from onnx import helper
from onnxscript import script
from onnxscript.values import OnnxFunction
from onnxscript.onnx_types import BOOL
from typing import Optional
from qti.aisw.converters.block_ops.onnx.onnx_block_op_base import (
    QnnOnnxBlockOp,
    _get_onnx_opset_version,
    qcom_block_op_domain,
)
from qti.aisw.converters.common.converter_ir.op_properties.stateful_lstm import (
    IR_RESET_IDX,
    IR_TO_ONNX_INDICES,
)

_logger = logging.getLogger(__name__)

ONNX_RESET_IDX = IR_TO_ONNX_INDICES[IR_RESET_IDX]


class StatefulLstm(QnnOnnxBlockOp):
    """ONNX StatefulLstm Block Operator class

    This class represents a StatefulLstm BlockOperator in ONNX. This operator is
    defined in onnxscript, so both getOnnxScriptFunc and getOnnxFuncProto are
    implemented.

    :ivar min_opset: Initialized to 7.
    :ivar max_opset: Initialized to 13.
    """

    def __init__(self, onnx_opset_version: int, aisw_opset_version: int = 1):
        self.name = "StatefulLstm"
        self.min_opset = 7
        self.max_opset = 13
        super(StatefulLstm, self).__init__(onnx_opset_version, aisw_opset_version)

    def getOnnxScriptFunc(self) -> OnnxFunction:
        opset = self.opset
        qti_aisw = self.aisw_opset

        @script(qti_aisw, default_opset=opset)
        def StatefulLstm(
            X,
            W,
            R,
            hidden_size: int,
            B=None,
            sequence_lens=None,
            initial_h=None,
            initial_c=None,
            P=None,
            # This parameter is always false in ONNX runtime
            reset: BOOL = opset.Constant(value=False),
            # Attributes
            clip: float = None,
            direction: str = "forward",
            input_forget: int = 0,
        ):
            """See ONNX LSTM documentation for op overview.

            StatefulLstm adds a reset input in addition to the standard ONNX
            LSTM inputs. See QNN MasterOpDef for documentation on the behavior
            of the reset input.

            NOTE: This op currently does not have the reset functionality in
            ONNX runtime.
            """
            return opset.LSTM(
                X,
                W,
                R,
                B=B,
                sequence_lens=sequence_lens,
                initial_h=initial_h,
                initial_c=initial_c,
                P=P,
                clip=clip,
                direction=direction,
                hidden_size=hidden_size,
                input_forget=input_forget,
            )

        return StatefulLstm

    def getOnnxFuncProto(self) -> onnx.FunctionProto:
        return self.getOnnxScriptFunc().to_function_proto()


def replaceOnnxLstmWithBlockOp(node: onnx.NodeProto, reset_tensor: onnx.TensorProto):
    """Given an ONNX node and a reset tensor, convert the node to a StatefulLstm
       Block Op node if it is an ONNX LSTM node and add the reset tensor as an
       input to the node. If the node is not an ONNX LSTM node, it is returned
       unchanged.

    :param node: The node to modify.
    :type node: onnx.NodeProto
    :param reset_tensor: The reset tensor to add to the inputs.
    :type reset_tensor: onnx.TensorProto
    :return: The modified node, or the original node if it is not an LSTM node.
    :rtype: onnx.NodeProto
    :raises ValueError: If an unknown or malformed ONNX LSTM node is found while
        trying to transform.

    """

    if node.op_type != "LSTM":
        return node
    num_empty_args = ONNX_RESET_IDX - len(node.input)
    if num_empty_args < 0:
        raise ValueError(
            "Too many arguments provided to ONNX LSTM "
            f"node {node.name}. Expected at most "
            f"{ONNX_RESET_IDX} arguments, got "
            f"{len(node.input)}."
        )
    node.input.extend([""] * num_empty_args)
    node.input.append(reset_tensor.name)
    node.op_type = "StatefulLstm"
    node.domain = qcom_block_op_domain()
    _logger.debug("Added reset tensor (%s) to %s", reset_tensor.name, node.name)
    return node


def replaceAllOnnxLstmWithBlockOp(model: onnx.ModelProto, reset_tensor: Optional[onnx.TensorProto] = None):
    """Given an onnx ModelProto, all ONNX LSTM nodes are replaced with
       StatefulLstm BlockOps. `reset` inputs are added to each StatefulLstm
       node. If the reset_tensor argument was supplied, all replaced
       StatefulLstm nodes have their reset input set to the provided
       reset_tensor. If reset_tensor is None, a new tensor with the name
       <node_name>_reset is created for each new StatefulLstm. StatefulLstm ONNX
       function proto is added to the model if any nodes were replaced.

    :param model: The model to modify.
    :type model: onnx.ModelProto
    :param reset_tensor: The reset tensor to input to the new stateful
         ops. If it is not provided, a new tensor is created for each
         StatefulLstm node, with the name <node_name>_reset.
    :type reset_tensor: onnx.TensorProto
    :return: The modified model.
    :rtype: onnx.ModelProto
    :raises ValueError: If an unknown or malformed ONNX LSTM node is found while
        trying to transform.
    :raises ValueError: If an ONNX opset version cannot be found in the model.

    """
    replaced_any_lstm_node = False
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "LSTM":
            if reset_tensor is not None:
                node_reset_tensor = reset_tensor
            else:
                node_id = node.name if node.name else f"lstm_{idx}"
                node_reset_tensor = onnx.helper.make_tensor_value_info(f"{node_id}_reset", onnx.TensorProto.BOOL, [])
                model.graph.input.append(node_reset_tensor)
            _ = replaceOnnxLstmWithBlockOp(node, node_reset_tensor)
            replaced_any_lstm_node = True

    if replaced_any_lstm_node:
        model_version = _get_onnx_opset_version(model)
        existing_func_names = {f.name for f in model.functions}
        if "StatefulLstm" not in existing_func_names:
            model.functions.extend([StatefulLstm(model_version).getOnnxFuncProto()])
        existing_opset_domains = {o.domain for o in model.opset_import}
        if qcom_block_op_domain() not in existing_opset_domains:
            model.opset_import.extend([helper.make_opsetid(qcom_block_op_domain(), 1)])
    return model
