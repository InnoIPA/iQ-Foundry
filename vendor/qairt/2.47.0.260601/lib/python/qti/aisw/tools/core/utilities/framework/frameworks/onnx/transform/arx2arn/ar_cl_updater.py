# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import tempfile
from typing import Optional

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger

ar_cl_log_area = LogAreas.register_log_area("ar_cl")

logger = QAIRTLogger.register_area_logger(
    ar_cl_log_area, level="info", formatter_val="extended", handler_list=["dev_console"]
)


def set_context_length(
    model: onnx.ModelProto, new_cl: int, *, in_place: bool = False, is_lvm: bool = False
) -> onnx.ModelProto:
    """
    API for setting context length for an ONNX model.

    Args:
        model (onnx.ModelProto): The ONNX model to update.
        new_cl (int): The new value for context length.
        in_place (bool): whether to modify model in place or return new model - defaults to False
        is_lvm (bool): Whether the model is an LVM or not - defaults to False

    Returns:
        onnx.ModelProto | None: Returns a new ONNX model with updated context length if `in_place` is False.
                                Returns None if `in_place` is True.

    Raises:
        ValueError: If `new_cl` is not a positive integer.
    """
    graph = gs.import_onnx(model)
    original_ar = _get_original_ar_value(graph, is_lvm)
    original_cl = _get_context_length(graph, original_ar)

    if new_cl == original_cl:
        return model

    if new_cl <= 0:
        raise ValueError("Context length must be greater than 0.")

    _update_input_cl(graph, original_cl, new_cl)
    _update_kv_cache_shapes(graph, original_ar, original_cl, new_cl=new_cl)
    _update_reshape_nodes(graph, original_cl, new_cl, is_cl_update=True)
    _update_constant_tensors(graph, original_cl, new_cl)

    updated_model = gs.export_onnx(graph)

    del updated_model.graph.value_info[:]
    updated_model = _infer_shapes_on_disk(updated_model)

    _validate_model(updated_model)

    if in_place:
        model.CopyFrom(updated_model)
        return model
    return updated_model


def set_ar(
    model: onnx.ModelProto, new_ar: int, *, in_place: bool = False, is_lvm: bool = False
) -> onnx.ModelProto:
    """
    API for setting AR for an ONNX model.

    Args:
        model (onnx.ModelProto): The ONNX model to update.
        new_ar (int): The new value for AR.
        in_place (bool): whether to modify model in place or return new model - defaults to False
        is_lvm (bool): Whether the model is an LVM or not - defaults to False

    Returns:
        onnx.ModelProto | None: Returns a new ONNX model with updated AR if `in_place` is False.
                                Returns None if `in_place` is True.

    Raises:
        ValueError: If `new_ar` is invalid or out of bounds for the model's context length.
    """
    graph = gs.import_onnx(model)
    original_ar = _get_original_ar_value(graph, is_lvm)
    original_cl = _get_context_length(graph, original_ar)

    if new_ar == original_ar:
        return model

    if new_ar <= 0 or new_ar > original_cl - 1:
        raise ValueError(
            f"Invalid AR value '{new_ar}' for context length '{original_cl}'. Supported range is 1 <= AR <= {original_cl - 1}."
        )

    _update_input_output_ar(graph, original_ar, new_ar)
    _update_kv_cache_shapes(graph, original_ar, original_cl, new_ar=new_ar)
    _update_reshape_nodes(graph, original_ar, new_ar, is_cl_update=False)
    _update_constant_tensors(graph, original_ar, new_ar)

    updated_model = gs.export_onnx(graph)

    del updated_model.graph.value_info[:]
    updated_model = _infer_shapes_on_disk(updated_model)

    _validate_model(updated_model)
    if in_place:
        model.CopyFrom(updated_model)
        return model
    return updated_model


def _validate_model(model: onnx.ModelProto) -> None:
    """
    Validates the ONNX model using onnx.checker.check_model.

    If the model is larger than 2GB, it is saved using external data and validated via a
    temporary file path. Otherwise, it is validated in-memory.
    """
    model_size = model.ByteSize()
    model_threshold = onnx.checker.MAXIMUM_PROTOBUF

    QAIRT_TMP_DIR = os.getenv("QAIRT_TMP_DIR", tempfile.gettempdir())

    if model_size > model_threshold:
        with tempfile.TemporaryDirectory(dir=QAIRT_TMP_DIR) as tmp_dir:
            model_path = os.path.join(tmp_dir, "temp_model.onnx")
            copied_model = onnx.ModelProto()
            copied_model.CopyFrom(model)
            onnx.save_model(copied_model, model_path, save_as_external_data=True, location="external.data")
            try:
                onnx.checker.check_model(model_path, full_check=True)
                logger.info(f"ONNX Checker - Large model is valid")
            except onnx.checker.ValidationError as e:
                logger.error(f"ONNX Checker - Large model is invalid: {e}")
                raise
    else:
        try:
            onnx.checker.check_model(model, full_check=True)
            logger.info("ONNX Checker - Model is valid")
        except onnx.checker.ValidationError as e:
            logger.error(f"ONNX Checker - Model is invalid: {e}")
            raise


def _infer_shapes_on_disk(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Infers shapes on an ONNX model. Uses on-disk path if model > 2GB.
    """
    model_size = model.ByteSize()
    model_threshold = 2 * (1024**3)

    QAIRT_TMP_DIR = os.getenv("QAIRT_TMP_DIR", tempfile.gettempdir())

    if model_size > model_threshold:
        with tempfile.TemporaryDirectory(dir=QAIRT_TMP_DIR) as tmp_dir:
            in_path = os.path.join(tmp_dir, "input_model.onnx")
            out_path = os.path.join(tmp_dir, "output_model.onnx")
            data_filename = "external.data"

            onnx.save_model(model, in_path, save_as_external_data=True, location=data_filename)

            onnx.shape_inference.infer_shapes_path(in_path, out_path)
            return onnx.load(out_path)
    else:
        return onnx.shape_inference.infer_shapes(model)


def _update_input_output_ar(graph: gs.Graph, original_ar: int, new_ar: int) -> None:
    """
    Update AR-related dimensions in graph inputs and outputs.

    Args:
        graph (gs.Graph): The ONNX graph.
        original_ar (int): Current AR value.
        new_ar (int): New AR value to set.
    """
    for inp in graph.inputs:
        if inp.name in {"input_ids", "position_ids"}:
            if inp.name == "position_ids":
                if inp.shape[1] == _get_context_length(graph, original_ar):
                    inp.shape[0] = new_ar
                else:
                    inp.shape[-1] = new_ar
            if inp.shape[-1] == original_ar:
                inp.shape[-1] = new_ar
        elif inp.name in {"attention_mask", "position_ids_sin", "position_ids_cos"}:
            if inp.shape[-2] == original_ar:
                inp.shape[-2] = new_ar
        elif len(inp.shape) == 3:
            if inp.shape[1] == original_ar:
                inp.shape[1] = new_ar
        elif inp.name == "swa_attention_mask":
            if inp.shape[-2] == original_ar:
                inp.shape[-2] = new_ar
        else:
            if "past" not in inp.name and original_ar in inp.shape:
                try:
                    idx = inp.shape.index(original_ar)
                    logger.debug(
                        f"Fallback: updating shape of input '{inp.name}', replacing dim {original_ar} -> {new_ar} at index {idx}"
                    )
                    inp.shape[idx] = new_ar
                except ValueError:
                    logger.warning(
                        f"Fallback failed: '{inp.name}' shape {inp.shape} does not contain original_ar={original_ar}"
                    )

    # Now modify outputs
    for out_tensor in graph.outputs:
        # Handle logits output for edge case of original AR being in both first two dimensions
        if (
            "past" not in out_tensor.name
            and len(out_tensor.shape) == 3
            and out_tensor.shape[0] == original_ar
            and out_tensor.shape[1] == original_ar
        ):
            out_tensor.shape[1] = new_ar
        elif "past" not in out_tensor.name and original_ar in out_tensor.shape:
            idx = out_tensor.shape.index(original_ar)
            out_tensor.shape[idx] = new_ar


def _update_input_cl(graph: gs.Graph, original_cl: int, new_cl: int) -> None:
    """
    Update context-length-related dimensions in graph inputs.

    Args:
        graph (gs.Graph): The ONNX graph.
        original_cl (int): Current context length.
        new_cl (int): New context length to set.
    """
    for inp in graph.inputs:
        if inp.name == "attention_mask":
            if inp.shape[-1] == original_cl:
                inp.shape[-1] = new_cl
        elif original_cl in inp.shape:
            idx = inp.shape.index(original_cl)
            inp.shape[idx] = new_cl


def _update_kv_cache_shapes(
    graph: gs.Graph,
    original_ar: int,
    original_cl: int,
    new_ar: Optional[int] = None,
    new_cl: Optional[int] = None,
) -> None:
    """
    Update past key/value cache shapes in the ONNX graph.

    Args:
        graph (gs.Graph): The ONNX graph.
        original_ar (int): Original AR value.
        original_cl (int): Original context length.
        new_ar (int, optional): New AR value.
        new_cl (int, optional): New context length.
    """
    input_map = {inp.name: inp for inp in graph.inputs}
    output_map = {out.name: out for out in graph.outputs}

    ar = new_ar if new_ar is not None else original_ar
    cl = new_cl if new_cl is not None else original_cl

    # Modify past key/values inputs and outputs
    indices = set()
    for name in list(input_map.keys()) + list(output_map.keys()):
        if name.startswith("past_key_") and name.endswith("_in"):
            idx = name[len("past_key_"):-3]  
            indices.add(idx)
    sorted_indices = sorted(indices)

    # Modify past key/value inputs and outputs
    for idx in sorted_indices:
        try:
            past_key_in = input_map[f"past_key_{idx}_in"]
            past_value_in = input_map[f"past_value_{idx}_in"]
            past_key_out = output_map[f"past_key_{idx}_out"]
            past_value_out = output_map[f"past_value_{idx}_out"]
        except KeyError:
            break

        is_transposed = past_key_in.shape != past_key_out.shape
        past_key_seq_idx = -1 if is_transposed else -2

        # Look for Concat-based KV caching where concat is used to combine the past sequence with the current AR tokens
        # In this case, the input shape for past_key/past_value should be:
        #    [batch, heads, context_length - AR, head_dim]
        # And the output shape should hold only the new AR tokens:
        #    [batch, heads, AR, head_dim]
        # If Concat not found, then the inputs and outputs keys/values should match the full context
        #    [batch, heads, context_length, head_dim]
        if past_key_in.outputs and past_key_in.outputs[0].op == "Concat":
            new_seq_len = cl - ar
        else:
            new_seq_len = cl

        past_value_in.shape[-2] = new_seq_len
        past_key_in.shape[past_key_seq_idx] = new_seq_len

        output_new_kv_only = past_value_in.shape[-2] > past_value_out.shape[-2]
        if output_new_kv_only:
            past_value_out.shape[-2] = ar
            past_key_out.shape[past_key_seq_idx] = ar
        else:
            past_value_out.shape[-2] = cl
            past_key_out.shape[past_key_seq_idx] = cl


def _update_reshape_nodes(graph: gs.Graph, original_val: int, new_val: int, is_cl_update: bool = False) -> None:
    """
    Update reshape nodes that contain original_val in their shape constant.

    Args:
        graph (gs.Graph): The ONNX graph.
        original_val (int): Original value to replace.
        new_val (int): New value to apply.
        is_cl_update (bool): Whether this is a context length update (True) or AR update (False).
    """
    for node in [n for n in graph.nodes if n.op == "Reshape"]:
        # Skip MLP nodes only when updating context length
        if is_cl_update:
            node_name = getattr(node, 'name', '').lower()
            if "mlp" in node_name:
                continue
            
        # Extract values from the shape tensor
        shape_array = (
            node.i(1).attrs["value"].values
            if isinstance(node.inputs[1], gs.Variable)
            else node.inputs[1].values
        )
        if original_val in shape_array:
            idx = shape_array.tolist().index(original_val)
            # Handle Edge Case when original AR is 1
            if (original_val == 1 and -1 in shape_array) or (
                original_val == 1 and shape_array[0] == 1 and shape_array[1] != 1 and shape_array[2] == 1
            ):
                continue
            # Handle edge case of original AR being in both first two dimensions
            if len(shape_array) == 3 and shape_array[0] == original_val and shape_array[1] == original_val:
                shape_array[1] = new_val
            else:
                if original_val == 1 and idx == 0:
                    # Handle edge case of original AR1 and update non-first dimensions
                    for index in range(1, len(shape_array)):
                        if shape_array[index] == original_val:
                            shape_array[index] = new_val
                            break
                else:
                    shape_array[idx] = new_val


def _update_constant_tensors(graph: gs.Graph, original_val: int, new_val: int) -> None:
    """
    Update constant tensors shaped by AR or context length.

    Args:
        graph (gs.Graph): The ONNX graph.
        original_val (int): Current value in shape.
        new_val (int): New value to apply.
    """
    for tensor in graph.tensors().values():
        if not isinstance(tensor, gs.Constant):
            continue

        shape = list(tensor.shape)
        if len(shape) == 1 and shape[0] == original_val and (tensor.values == np.arange(shape[0])).all():
            tensor.values = np.arange(new_val, dtype=tensor.dtype)
        elif original_val in shape and (tensor.values == 1).all():
            if len(shape) == 3 and shape[0] == original_val and shape[1] == original_val:
                shape[1] = new_val
            else:
                original_val_index = shape.index(original_val)
                if original_val == 1 and original_val_index == 0:
                    # Handle edge case of original AR1 and update non-first dimensions
                    for idx in range(1, len(shape)):
                        if shape[idx] == original_val:
                            shape[idx] = new_val
                            break
                else:
                    shape[original_val_index] = new_val

            tensor.values = np.ones(shape, dtype=tensor.dtype)


def _get_original_ar_value(graph: gs.Graph, is_lvm: bool) -> int:
    """
    Retrieves the ARx value from the ONNX graph.

    Args:
        graph (gs.Graph): The ONNX graph parsed by GraphSurgeon.
        is_lvm (bool): True if the model is an LVM otherwise False.

    Returns:
        int: The ARx value, representing the number of active tokens in the model.

    Notes:
        - For LLM models, the ARX value is taken from the output tensor's second dimension.
        - For LVM models, it is calculated from input dimensions.
    """
    if not is_lvm:
        return graph.outputs[0].shape[1]
    else:
        return graph.inputs[0].shape[-1] * graph.inputs[0].shape[-2]


def _get_context_length_from_attention_mask(graph: gs.Graph) -> int:
    """
    Attempts to infer the model's context length from the attention mask input tensor.

    Args:
        graph (gs.Graph): The ONNX graph parsed by GraphSurgeon.

    Returns:
        int: The context length inferred from the attention mask input.

    Raises:
        ValueError: If no valid attention mask input with a static shape is found.
    """
    for inp in graph.inputs:
        if "attention_mask" in inp.name:
            if isinstance(inp.shape[-1], int):
                return inp.shape[-1]
    raise ValueError("Cannot determine context length from attention mask")


def _get_context_length_from_concat(graph: gs.Graph, ar_value: int) -> int:
    """
    Attempts to infer the model's context length based on a Concat operation involving past values.

    Args:
        graph (gs.Graph): The ONNX graph parsed by GraphSurgeon.
        ar_value (int): The AR value of the graph used to calculate the total context length.

    Returns:
        int: The calculated context length.

    Raises:
        ValueError: If no suitable Concat node involving past values is found.
    """
    for node in graph.nodes:
        if node.op == "Concat":
            for inp in node.inputs:
                if "past_value" in inp.name:
                    if isinstance(inp.shape[-2], int):
                        return ar_value + inp.shape[-2]
    raise ValueError("No Concat node found with past values from KV Cache")


def _get_context_length_from_past_key(graph: gs.Graph) -> int:
    """
    Attempts to infer the context length directly from past key-value cache input tensors.

    Args:
        graph (gs.Graph): The ONNX graph parsed by GraphSurgeon.

    Returns:
        int: The sequence length inferred from the shape of the past_value tensor.

    Raises:
        ValueError: If no past value input tensor with a valid static shape is found.
    """
    input_map = {inp.name: inp for inp in graph.inputs}
    for name, tensor in input_map.items():
        if "past_value" in name:
            if isinstance(tensor.shape[-2], int):
                past_seq_len = tensor.shape[-2]
                context_length = past_seq_len + tensor.shape[-1]
                return context_length
    raise ValueError("Could not determine context length using past values from KV Cache")


def _get_context_length(graph: gs.Graph, ar_value: int) -> int:
    """
    Attempts to determine the model's context length using multiple heuristics.

    First try to get context length:
    - From the attention mask input tensor.
    - From Concat nodes combining past values.
    - From past key-value cache input tensors.

    Args:
        graph (gs.Graph): The ONNX graph parsed by GraphSurgeon.
        ar_value (int): The AR value of the graph, needed if falling back to Concat-based inference.

    Returns:
        int: The inferred context length.

    Raises:
        ValueError: If all methods fail to infer the context length.
    """
    try:
        return _get_context_length_from_attention_mask(graph)
    except ValueError:
        pass
    try:
        return _get_context_length_from_concat(graph, ar_value)
    except ValueError:
        pass
    try:
        return _get_context_length_from_past_key(graph)
    except ValueError:
        pass

    raise ValueError("Unable to determine context length using available heuristics.")
