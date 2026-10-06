# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import tvm
from tvm import relay
from tvm.topi.nn.qnn import SQNN_CODE_TO_DTYPE

def get_key_from_expr(expr: relay.expr):
    return hash(expr)

def get_prim_type(v):
    if isinstance(v, tvm.tir.expr.IntImm):
        return v.value
    elif isinstance(v, tvm.tir.expr.FloatImm):
        return v.value
    elif isinstance(v, tvm.ir.container.Array):
        return [get_prim_type(i) for i in list(v)]
    elif isinstance(v, tvm.runtime.container.String):
        return str(v)
    else:
        return v

def get_dtype_from_code(code):
    if isinstance(code, tvm.relay.Constant):
        return SQNN_CODE_TO_DTYPE[code.data.asnumpy().item()]
    else:
        return code