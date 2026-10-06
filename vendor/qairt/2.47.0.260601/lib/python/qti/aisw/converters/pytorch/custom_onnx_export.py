# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import functools

import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from qti.aisw.converters.pytorch.torchscript_to_onnx import (
    CONFIG_PATH_TO_OP_DEF_COLLECTION_MAPPING,
    OPSET_VERSION,
)
from torch.onnx import (
    _constants,
    symbolic_helper,
)
from torch.onnx import (
    symbolic_opset9 as opset9,
)
from torch.onnx._internal import jit_utils, registration
from torch.onnx._onnx_supported_ops import all_symbolics_schemas

# ------------------------------------------------------------------------------
#   Define additional op mapping from PyTorch op to ONNX op
#   because this PyTorch op is not present in the list of TorchScript operators
#   supported by ONNX
#
#   - A mapping is registered by PyTorch op type and corresponding ONNX op(s)
# ------------------------------------------------------------------------------

__all__ = [
    "addcdiv",
]


_onnx_symbolic = functools.partial(
    registration.onnx_symbolic, opset=_constants.ONNX_DEFAULT_OPSET
)


@_onnx_symbolic("aten::addcdiv")
@symbolic_helper.parse_args("v", "v", "v", "f")
def addcdiv(g: jit_utils.GraphContext, self, tensor1, tensor2, value=1.0):
    value_tens = g.op("Constant", value_t=torch.tensor([value]))
    return opset9.add(
        g, self, opset9.mul(g, opset9.div(g, tensor1, tensor2), value_tens)
    )


symbolics_schemas = all_symbolics_schemas()


# Define custom symbolic function
def symbolic_fn(g: jit_utils.GraphContext, *inputs, **attrs):
    op_type = g.original_node.kind()
    new_op_type = op_type
    # This op has official support from torch to onnx.
    # Because we want to mark this op as a custom op, we will change the domain of
    # pytorch op so that it is different from official op.
    # E.g., from aten::softmax to custom::softmax
    if op_type in symbolics_schemas:
        new_op_type = "::".join(["custom", op_type.split("::")[-1]])
    return g.op(new_op_type, *inputs, **attrs)


def register_custom_op():
    """
    The purpose of this function is to register the mapping from pytorch custom op to
    onnx custom op.
    - Register custom symbolic function
    """
    for op_def_collection in CONFIG_PATH_TO_OP_DEF_COLLECTION_MAPPING.values():
        for op_def_name in op_def_collection.op_def_dict:
            # Register custom symbolic function
            torch.onnx.register_custom_op_symbolic(
                op_def_name, symbolic_fn, OPSET_VERSION
            )
