import warnings
import contextlib
import numpy as np
import torch
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from qti.aisw.emitter.emitter_ops import QuantizedCustomStatefulLSTM, QuantizedCustomStatefulGru, QuantizedNonBlockingBuffer


@contextlib.contextmanager
def _create_int32_bias_quantizers_for_stateful_ops(model, args, kwargs=None):
    """
    Optional helper context method to create int32 bias quantizers for stateful Ops in Quantsim
    """
    if not isinstance(args, (tuple, list)):
        args = (args,)
    kwargs = kwargs or {}
    handles = []
    lstm_modules = []
    orig_bias_qtzrs = {}

    for module_name, module in model.named_modules():
        if isinstance(module, QuantizedCustomStatefulLSTM):
            lstm_modules.append(module)
            orig_bias_qtzrs[module] = {
                f"b_{gate_suffix}": module.param_quantizers[f"b_{gate_suffix}"]
                for gate_suffix in ["i", "o", "f", "c"]
            }

    try:
        for module in lstm_modules:
            if isinstance(module, QuantizedCustomStatefulLSTM):
                # pylint: disable=protected-access
                handle = module.register_forward_hook(
                    module._create_int32_bias_quantizer
                )
                handles.append(handle)
        try:
            model(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        yield
    finally:
        for module, bias_qtzrs in orig_bias_qtzrs.items():
            for bias_name, bias_qtzr in bias_qtzrs.items():
                module.param_quantizers[bias_name] = bias_qtzr


def _get_closest_producer_wrapper(sim, op):
    """
    Find the closest producer QcQuantizeWrapper and return it

    :param op: Target operation
    :return: QcQuantizerWrapper if exists else None
    """
    wrapper = sim._get_qmodule(op)
    if wrapper:
        if (
                (wrapper.output_quantizers[0] is not None)
                or (wrapper.input_quantizers[0] is not None)
        ):
            return wrapper

        if len(op.input_ops) == 1:
            return _get_closest_producer_wrapper(sim, op.input_ops[0])

        warnings.warn(
            f"A wrapper of %s with output quantization disabled has no input or more than one input "
            f"exists. It's ambiguous to find the nearest producer in this case {str(op.get_module())}",
        )
        return None

    if not op.input_ops:
        warnings.warn(
            "No input exists for navigation for traversal, it's not possible to find the closest producer"
        )
        return None

    if len(op.input_ops) > 1:
        warnings.warn(
            "Multiple input ops exist, traversal to find closest producer is performed based on the "
            "first input"
        )

    return _get_closest_producer_wrapper(sim, op.input_ops[0])


def _get_effective_quantizer(sim, input_quantizer, input_op):
    """
    :param sim: Quantsim instance
    :param input_quantizer: Input quantizer
    :param input_op: Producer op in the connected graph for the input quantizer
    :return: returns the effective quantizer for the input corresponding to the input quantizer
    """
    target_quantizer = None
    if input_quantizer is not None:
        target_quantizer = input_quantizer
    elif input_op:
        closest_producer_wrapper = _get_closest_producer_wrapper(sim, input_op)
        if closest_producer_wrapper:
            target_quantizer = (
                closest_producer_wrapper.output_quantizers[0]
                if closest_producer_wrapper.output_quantizers[0] is not None
                else closest_producer_wrapper.input_quantizers[0]
            )
        else:
            warnings.warn(
                "The closest wrapper could not be found. MatMul exception rule does not apply. "
                "If you haven't used model preparer, consider using it."
            )
    return target_quantizer


def _get_effective_quantizer_for_module_input(sim, module, input_idx):
    """
    Get the effective quantizer for the given model input traversing through the connected graph
    """
    op = sim.connected_graph._module_to_op_dict[module]
    inp_quantizer = module.input_quantizers[input_idx]
    inp_producer = (
        None
        if inp_quantizer is not None
        else op.inputs[input_idx].producer
    )
    return _get_effective_quantizer(sim, inp_quantizer, inp_producer)


def _set_quantizer_to_16bit_symmetric(qtzr):
    """
    :param qtzr: quantizer which should be made 16 bit symmetric
    """
    qtzr.symmetric = True
    if qtzr.bitwidth != 16:
        qtzr.bitwidth = 16
        qtzr.qmin = 0
        qtzr.qmax = 2**16-1


def _apply_stateful_op_qsim_exception_rules(sim: QuantizationSimModel):
    """
    Apply Quantsim exception rules for Stateful Ops
    StatefulLstm:
    1) Tie hidden state and cell state quantizers in input and output.
    2) If hidden state and cell state share same effective quantizer,
       create an input quantizer for the cell state as it should always be sfxp16.
    3) cell state and intermediate gate outputs will always be sfxp16
    Stateful Gru:
    1) Tie input and output hidden state quantizers
    Buffer:
    1) If Buffer Op has output quantizer, make sure to tie the input and output quantizers
    """
    for module_name, module in sim.model.named_modules():
        if isinstance(module, QuantizedCustomStatefulLSTM):
            # Find target quantizer h_0 and tie the target quantizer with output[0] and output[2]
            h_target_quantizer = _get_effective_quantizer_for_module_input(sim, module, 1)
            module.output_quantizers[0] = h_target_quantizer
            module.output_quantizers[2] = h_target_quantizer

            # Find target quantizer c_0 and tie the target quantizer with output[1]
            c_target_quantizer = _get_effective_quantizer_for_module_input(sim, module, 2)

            # If h_0 and c_0 share same target quantizer, create an input quantizer for c_0 and tie it with c_t
            if c_target_quantizer != h_target_quantizer:
                module.output_quantizers[1] = c_target_quantizer
            else:
                qtzr = QuantizeDequantize(shape=(), qmin=-2**15, qmax=2**15-1, symmetric=True)
                module.input_quantizers[2] = qtzr
                module.output_quantizers[1] = qtzr

            # Set (c_0 & C_t) - already tied, matmul_i, matmul_f, matmul_c, matmul_o to 16-bit symmetric always
            _set_quantizer_to_16bit_symmetric(module.output_quantizers[1])
            _set_quantizer_to_16bit_symmetric(module.output_quantizers[3])
            _set_quantizer_to_16bit_symmetric(module.output_quantizers[4])
            _set_quantizer_to_16bit_symmetric(module.output_quantizers[5])
            _set_quantizer_to_16bit_symmetric(module.output_quantizers[6])

        if isinstance(module, QuantizedCustomStatefulGru):
            # Find target quantizer h_0 and tie the target quantizer with output[0]
            h_target_quantizer = _get_effective_quantizer_for_module_input(sim, module, 1)
            module.output_quantizers[0] = h_target_quantizer
            module.output_quantizers[1] = h_target_quantizer

        if isinstance(module, QuantizedNonBlockingBuffer):
            # Find target quantizer for input and tie with the output quantizer
            if module.output_quantizers[0] is not None:
                module.output_quantizers[0] = _get_effective_quantizer_for_module_input(sim, module, 0)


def _reset_qsim_stateful_op_buffers(sim):
    """
    Reset buffers for Stateful Ops to avoid using incorrect buffers before calibration
    """
    for module_name, module in sim.model.named_modules():
        if isinstance(module, (QuantizedCustomStatefulLSTM, QuantizedCustomStatefulGru, QuantizedNonBlockingBuffer)):
            module._reset_buffers_to_zero()
