# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from enum import Enum
from qti.aisw.converters.common.utils.converter_utils import log_debug1
from qti.aisw.converters.common.converter_ir import op_adapter
import json


# -----------------------------------------------------------------------------------------------------
#   LORA Helpers
# -----------------------------------------------------------------------------------------------------


def make_tensors_updateable(cpp_graph, graph_quantized=False, tensor_names=[]):
    if tensor_names:
        for tensor_name in tensor_names:
            if cpp_graph.has_tensor(tensor_name):
                log_debug1(f"Marking tensor {tensor_name} as updatable in the graph.")
                cpp_tensor = cpp_graph.get_tensor(tensor_name)
                cpp_tensor.set_updateable(True)

    if graph_quantized:
        cpp_graph = propagate_quantized_updateable_status(cpp_graph)

    return cpp_graph

def propagate_quantized_updateable_status(cpp_graph):
    """
    Applies updateable logic for quantized graphs by making tensors updateable
    based on graph properties and propagating updateable status through invariant operations.
    """
    cpp_graph = make_updateable_from_graph_properties(cpp_graph)
    cpp_graph = propagate_updateable_for_invariant_ops(cpp_graph)
    return cpp_graph

def make_updateable_from_graph_properties(cpp_graph):
    if cpp_graph.is_no_quant_updateable() or cpp_graph.is_adapter_only_quant_updateable():
        return cpp_graph
    else:
        cpp_graph = make_activation_and_bias_tensors_updatable(cpp_graph)

    return cpp_graph

def make_activation_and_bias_tensors_updatable(cpp_graph):
    ops_having_weight = [op_adapter.Conv2dOp.TRANSLATION_KEY,
                         op_adapter.Conv3dOp.TRANSLATION_KEY,
                         op_adapter.TransposeConv2dOp.TRANSLATION_KEY,
                         op_adapter.TransposeConv3dOp.TRANSLATION_KEY,
                         op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY,
                         op_adapter.MatMulOp.TRANSLATION_KEY,
                         op_adapter.FullyConnectedOp.TRANSLATION_KEY,
                         op_adapter.BatchnormOp.TRANSLATION_KEY,
                         op_adapter.LayerNormOp.TRANSLATION_KEY,
                         op_adapter.InstanceNormOp.TRANSLATION_KEY,
                         op_adapter.GroupNormOp.TRANSLATION_KEY,
                         op_adapter.RMSNormOp.TRANSLATION_KEY]

    def is_only_consumed_by_gather_ops(tensor):
        gather_op_types = [
            op_adapter.GatherOp.TRANSLATION_KEY,
            op_adapter.GatherElementsOp.TRANSLATION_KEY,
            op_adapter.GatherNDOp.TRANSLATION_KEY
        ]

        consumers = list(tensor.get_consumers())
        for op in consumers:
            if op.type not in gather_op_types:
                return False
        return True

    def is_weight(tensor):
        consumers = list(tensor.get_consumers())
        is_weight = False
        if tensor.is_static():
            for consumer in consumers:
                if consumer.type in ops_having_weight:
                    inputs = consumer.get_input_names
                    if inputs[1] == tensor.name():
                        is_weight = True
        return is_weight

    tensor_map = cpp_graph.get_tensor_map()

    for key, tensor in tensor_map.items():
        if not tensor.is_quantizable():
            # Do not mark non-quantizable tensors as updatable
            pass
        elif not tensor.is_static_tensor():
            # mark all the non-static tensors (quantizable) to updatable
            log_debug1("Marking the activation tensor {} as updatable".format(key))
            tensor.set_updateable(True)
        elif not is_weight(tensor) and not is_only_consumed_by_gather_ops(tensor):
            # mark all non weight static tensors (quantizable) as updatable
            log_debug1("Marking the non-weight static tensor {} as updatable".format(key))
            tensor.set_updateable(True)
        else:
            pass

    return cpp_graph

def propagate_updateable_for_invariant_ops(cpp_graph):
    data_invariant_ops = [op_adapter.TransposeOp.TRANSLATION_KEY,
                          op_adapter.ReduceOp.TRANSLATION_KEY,
                          op_adapter.CropAndResizeOp.TRANSLATION_KEY,
                          op_adapter.GatherOp.TRANSLATION_KEY,
                          op_adapter.GatherElementsOp.TRANSLATION_KEY,
                          op_adapter.GatherNDOp.TRANSLATION_KEY,
                          op_adapter.PadOp.TRANSLATION_KEY,
                          op_adapter.Pool2dOp.TRANSLATION_KEY,
                          op_adapter.Pool3dOp.TRANSLATION_KEY,
                          op_adapter.ReshapeOp.TRANSLATION_KEY,
                          op_adapter.ResizeOp.TRANSLATION_KEY,
                          op_adapter.StridedSliceOp.TRANSLATION_KEY,
                          op_adapter.SpaceToDepthOp.TRANSLATION_KEY,
                          op_adapter.DepthToSpaceOp.TRANSLATION_KEY,
                          op_adapter.ChannelShuffleOp.TRANSLATION_KEY,
                          op_adapter.SplitOp.TRANSLATION_KEY,
                          op_adapter.TopKOp.TRANSLATION_KEY,
                          op_adapter.BatchPermutationOp.TRANSLATION_KEY,
                          op_adapter.ExpandDimsOp.TRANSLATION_KEY,
                          op_adapter.SqueezeOp.TRANSLATION_KEY]

    # TODO Check and add if any other Ops fall under this category (like CastOp) and update logic accordingly
    # These ops are not data invariant but propagatable status can still be transferred under certain conditions
    # Consider a case If Convert Op is added during the end of Quantization/Float Fallback
    # E.g: like ConvertOp from UFXP8 -> UFXP16 for an updateable static tensor
    # In this case, if input UFXP8 tensor is updateable, then so the input UFXP16 also must be made updateable
    # And vice-versa is also possible like case where ConvertOp is added for an updateable activation tensor
    lora_updatable_invariant_ops = [op_adapter.ConvertOp.TRANSLATION_KEY]

    op_list = cpp_graph.get_ops()

    for op in op_list:
        inputs = list(op.inputs())
        outputs = list(op.outputs())

        if op.type in lora_updatable_invariant_ops:
            is_updateable = False
            for tensor in inputs + outputs:
                if tensor.is_updateable():
                    if tensor.is_quantized() or tensor.is_static_tensor():
                        is_updateable = True
                        break

            if is_updateable:
                for tensor in inputs + outputs:
                    if tensor.is_quantized():
                        tensor.set_updateable(True)

        elif op.type in data_invariant_ops:
            is_updateable = inputs[0].is_updateable()
            if is_updateable:
                for tensor in outputs:
                    if tensor.is_quantized():
                        tensor.set_updateable(True)

    return cpp_graph
