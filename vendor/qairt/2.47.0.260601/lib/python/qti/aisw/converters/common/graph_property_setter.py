# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.op_graph import QuantUpdatableMode
from qti.aisw.converters.common.utils.converter_utils import log_debug


class GraphPropertySetter(object):
    """
    Stateless utility class for setting graph properties.

    Methods:
        set_graph_properties(self, graph, quant_updatable_mode):
            Infers and sets graph properties based on quant updatable mode and
            list of lora tensor names.
        validate_graph_properties(self, graph, quant_updatable_mode, lora_tensor_names):
            Validates graph properties and tensor configurations.
        copy_ir_graph_properties(self, py_graph, cpp_graph):
            Copy graph properties from an IrGraph instance to IrOpGraph instance.
    """
    def set_graph_properties(self, graph, quant_updatable_mode):
        """
        Given mode and tensor names, sets graph properties related to which
        tensor's values and/or quant encodings are updatable. Currently used for
        lora models.

        Note: This method only sets properties. Call validate_graph_properties()
        separately to perform validation.
        """
        lora_tensor_names = graph.get_all_updatable_tensors()
        self._set_updatable(graph, lora_tensor_names)
        self._set_quant_updatable_flags(graph, quant_updatable_mode, lora_tensor_names)

    def copy_ir_graph_properties(self, py_graph, cpp_graph):
        """
        Copy graph properties from an IrGraph instance to IrOpGraph instance.
        """
        py_graph.update_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_NO_QUANT_UPDATABLE, cpp_graph.is_no_quant_updateable())
        py_graph.update_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_UPDATABLE, cpp_graph.is_updateable())
        py_graph.update_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_ADAPTER_ONLY_QUANT_UPDATABLE, cpp_graph.is_adapter_only_quant_updateable())
        py_graph.update_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_FLOAT_ONLY_QUANT_UPDATABLE, cpp_graph.is_float_only_quant_updateable())
        self._validate_graph_property_consistency(py_graph)

    def _set_updatable(self, graph, lora_tensor_names):
        """Set UPDATABLE flag based on presence of lora tensor names."""
        is_updatable = len(lora_tensor_names) > 0
        graph.update_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_UPDATABLE, is_updatable)

    def _set_quant_updatable_flags(self, graph, quant_updatable_mode, lora_tensor_names):
        """Set quantization updatable flags based on mode without validation."""
        def set_no_quant_updatable(graph):
            graph.update_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_NO_QUANT_UPDATABLE, True)

        def set_float_only_quant_updatable(graph):
            graph.update_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_FLOAT_ONLY_QUANT_UPDATABLE, True)
            log_debug("LoRA graph has float only quant updatable mode")

        def set_adapter_only_quant_updatable(graph, is_adapter_only):
            graph.update_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_ADAPTER_ONLY_QUANT_UPDATABLE, is_adapter_only)

            if is_adapter_only:
                log_debug("LoRA graph has adapter only updatable quantization encodings.")
            else:
                log_debug("LoRA graph does not have adapter only updatable quantization encodings.")

        # Set flags based on mode without validation
        if quant_updatable_mode is None:
            # Backward compatible: assume adapter_only if lora_tensor_names is not empty
            # Actual validation will happen in validate_graph_properties()
            is_adapter_only = self.has_native_and_static_tensors(graph, lora_tensor_names)
            set_adapter_only_quant_updatable(graph, is_adapter_only)
        elif quant_updatable_mode == QuantUpdatableMode.NONE:
            set_no_quant_updatable(graph)
        elif quant_updatable_mode == QuantUpdatableMode.ADAPTER_ONLY:
            set_adapter_only_quant_updatable(graph, True)
        elif quant_updatable_mode == QuantUpdatableMode.FLOAT_ONLY:
            set_float_only_quant_updatable(graph)
        elif quant_updatable_mode == QuantUpdatableMode.ALL:
            # ALL mode doesn't set any special flags, just UPDATABLE
            pass
        else:
            raise ValueError(f"Quant updatable mode, {quant_updatable_mode}, "
                              "is not supported.")

    def has_only_static_tensors(self, graph, lora_tensor_names):
        """Check if all lora tensors are static (weights/biases)."""
        for tensor_name in lora_tensor_names:
            if graph.has_tensor(tensor_name):
                tensor = graph.get_tensor(tensor_name)
                if not tensor.is_static_tensor():
                    return False
        return True

    def has_native_and_static_tensors(self, graph, lora_tensor_names):
        """
        Check if lora tensors contain both static tensors and native (activation) tensors.
        """
        has_updateable_static_tensor = False
        has_updateable_native_tensor = False

        for tensor_name in lora_tensor_names:
            if graph.has_tensor(tensor_name):
                if graph.get_tensor(tensor_name).is_static_tensor():
                    has_updateable_static_tensor = True
                else:
                    has_updateable_native_tensor = True

        return has_updateable_static_tensor and has_updateable_native_tensor

    def _validate_graph_property_consistency(self, graph):
        """Validate that graph property flags are set consistently."""
        is_updatable = graph.get_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_UPDATABLE)
        is_no_quant_updatable = graph.get_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_NO_QUANT_UPDATABLE)
        is_adapter_only_quant_updatable = graph.get_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_ADAPTER_ONLY_QUANT_UPDATABLE)
        is_float_only_quant_updatable = graph.get_graph_property_mask(ir_graph.IR_GRAPH_PROPERTY_MASK_FLOAT_ONLY_QUANT_UPDATABLE)

        if not is_updatable:
            if is_adapter_only_quant_updatable:
                raise RuntimeError(f"Invalid graph property combination: ADAPTER_ONLY_QUANT_UPDATABLE is set but UPDATABLE is not.")
            elif is_no_quant_updatable:
                raise RuntimeError(f"Invalid graph property combination: NO_QUANT_UPDATABLE is set but UPDATABLE is not.")
            elif is_float_only_quant_updatable:
                raise RuntimeError(f"Invalid graph property combination: FLOAT_ONLY_QUANT_UPDATABLE is set but UPDATABLE is not.")

        if is_no_quant_updatable and is_adapter_only_quant_updatable:
            raise RuntimeError(f"Invalid graph property combination: NO_QUANT_UPDATABLE and ADAPTER_ONLY_QUANT_UPDATABLE are both set.")
        elif is_float_only_quant_updatable and is_adapter_only_quant_updatable:
            raise RuntimeError(f"Invalid graph property combination: FLOAT_ONLY_QUANT_UPDATABLE and ADAPTER_ONLY_QUANT_UPDATABLE are both set.")
        elif is_float_only_quant_updatable and is_no_quant_updatable:
            raise RuntimeError(f"Invalid graph property combination: FLOAT_ONLY_QUANT_UPDATABLE and NO_QUANT_UPDATABLE are both set.")

    def validate_post_quantize_graph(self, cpp_graph, quant_updatable_mode):
        """
        Validate post-quantization graph to ensure updatable properties are correctly set.
        Should be called after quantization is complete.

        Validates:
        - NONE mode: Only adapter weight values are updatable, no quant encodings are updatable
        - ALL mode: All adapter weight values are updatable, all quant encodings are updatable
        - ADAPTER_ONLY mode: Adapter weight values are updatable, only LoRA branch quant encodings are updatable

        Note: For ADAPTER_ONLY mode, lora_tensor_names contains both adapter weights and native tensors.

        :param cpp_graph: The quantized C++ graph to validate
        :param quant_updatable_mode: The quantization updatable mode
        :raises RuntimeError: If validation fails
        """
        if quant_updatable_mode is None:
            # Skip validation if mode is not specified
            return

        if (quant_updatable_mode == QuantUpdatableMode.NONE or quant_updatable_mode == QuantUpdatableMode.FLOAT_ONLY):
            self._validate_none_float_only_mode_post_quantize_cpp(cpp_graph, quant_updatable_mode)
        elif quant_updatable_mode == QuantUpdatableMode.ALL:
            self._validate_all_mode_post_quantize_cpp(cpp_graph)
        elif quant_updatable_mode == QuantUpdatableMode.ADAPTER_ONLY:
            self._validate_adapter_only_mode_post_quantize_cpp(cpp_graph)

        log_debug(f"Quant updatable mode - {quant_updatable_mode.value}, post-quantization validation passed")

    def _validate_none_float_only_mode_post_quantize_cpp(self, cpp_graph, quant_updatable_mode):
        """
        Validate NONE/FLOAT_ONLY mode: Only adapter weight values are updatable, no quant encodings are updatable.

        :param cpp_graph: C++ graph (Post-Quantizer)
        :raises RuntimeError: If validation fails
        """
        # Collect problematic tensors by error type
        lora_static_tensors_missing_updatable = []
        lora_non_static_tensors_incorrectly_updatable = []
        non_lora_tensors_incorrectly_updatable = []

        tensor_map = cpp_graph.get_tensor_map()
        lora_tensor_names = cpp_graph.get_all_updatable_tensors()

        for tensor_name, tensor in tensor_map.items():
            if tensor_name in lora_tensor_names:
                # LoRA branch tensors
                if tensor.is_static_tensor():
                    # Rule 1: LoRA static tensors should be updatable
                    if not tensor.is_updateable():
                        lora_static_tensors_missing_updatable.append(tensor_name)
                else:
                    # Rule 2: LoRA non-static tensors should NOT be updatable
                    if tensor.is_updateable():
                        lora_non_static_tensors_incorrectly_updatable.append(tensor_name)
            else:
                # Rule 3: Non-LoRA tensors should NOT be updatable
                if tensor.is_updateable():
                    non_lora_tensors_incorrectly_updatable.append(tensor_name)

        # Build error message with summary
        errors = []
        if lora_static_tensors_missing_updatable:
            errors.append(
                f"{quant_updatable_mode.value} mode: {len(lora_static_tensors_missing_updatable)} LoRA static tensor(s) not marked as updatable. "
                f"Examples: {', '.join(lora_static_tensors_missing_updatable[:5])}"
                + (" ..." if len(lora_static_tensors_missing_updatable) > 5 else "")
            )

        if lora_non_static_tensors_incorrectly_updatable:
            errors.append(
                f"{quant_updatable_mode.value} mode: {len(lora_non_static_tensors_incorrectly_updatable)} LoRA non-static tensor(s) incorrectly marked as updatable. "
                f"Examples: {', '.join(lora_non_static_tensors_incorrectly_updatable[:5])}"
                + (" ..." if len(lora_non_static_tensors_incorrectly_updatable) > 5 else "")
            )

        if non_lora_tensors_incorrectly_updatable:
            errors.append(
                f"{quant_updatable_mode.value} mode: {len(non_lora_tensors_incorrectly_updatable)} non-LoRA tensor(s) incorrectly marked as updatable. "
                f"Examples: {', '.join(non_lora_tensors_incorrectly_updatable[:5])}"
                + (" ..." if len(non_lora_tensors_incorrectly_updatable) > 5 else "")
            )

        if errors:
            raise RuntimeError(
                f"Post-quantization validation failed for {quant_updatable_mode.value} mode:\n" +
                "\n".join(errors)
            )

    def _validate_all_mode_post_quantize_cpp(self, cpp_graph):
        """
        Validate ALL mode: All adapter weight values are updatable, all quant encodings are updatable.

        Note: Non-LoRA static tensors (base model weights) are not validated.

        :param cpp_graph: C++ graph (Post-Quantizer)
        :raises RuntimeError: If validation fails
        """
        # Collect problematic tensors by error type
        lora_static_tensors_missing_updatable = []
        non_quantized_non_static_tensors_incorrectly_updatable = []
        quantized_non_static_tensors_missing_updatable = []

        tensor_map = cpp_graph.get_tensor_map()
        lora_tensor_names = cpp_graph.get_all_updatable_tensors()

        for tensor_name, tensor in tensor_map.items():
            # Rule 1: All LoRA static tensors should have updatable values
            if tensor_name in lora_tensor_names and tensor.is_static_tensor():
                if not tensor.is_updateable():
                    lora_static_tensors_missing_updatable.append(tensor_name)

            # Rules for non-static tensors (both LoRA and non-LoRA)
            if not tensor.is_static_tensor():
                # Rule 2: Non-static tensors without quant encodings should NOT be updatable
                if not (tensor.is_quantized() or tensor.is_quantizable()):
                    if tensor.is_updateable():
                        non_quantized_non_static_tensors_incorrectly_updatable.append(tensor_name)
                else:
                    # Rule 3: All quantized non-static tensors should have updatable quant encodings
                    if not tensor.is_updateable():
                        quantized_non_static_tensors_missing_updatable.append(tensor_name)

        # Build error message with summary
        errors = []
        if lora_static_tensors_missing_updatable:
            errors.append(
                f"ALL mode: {len(lora_static_tensors_missing_updatable)} LoRA static tensor(s) not marked as updatable. "
                f"Examples: {', '.join(lora_static_tensors_missing_updatable[:5])}"
                + (" ..." if len(lora_static_tensors_missing_updatable) > 5 else "")
            )

        if non_quantized_non_static_tensors_incorrectly_updatable:
            errors.append(
                f"ALL mode: {len(non_quantized_non_static_tensors_incorrectly_updatable)} non-quantized non-static tensor(s) incorrectly marked as updatable. "
                f"Examples: {', '.join(non_quantized_non_static_tensors_incorrectly_updatable[:5])}"
                + (" ..." if len(non_quantized_non_static_tensors_incorrectly_updatable) > 5 else "")
            )

        if quantized_non_static_tensors_missing_updatable:
            errors.append(
                f"ALL mode: {len(quantized_non_static_tensors_missing_updatable)} quantized non-static tensor(s) not marked as updatable. "
                f"Examples: {', '.join(quantized_non_static_tensors_missing_updatable[:5])}"
                + (" ..." if len(quantized_non_static_tensors_missing_updatable) > 5 else "")
            )

        if errors:
            raise RuntimeError(
                "Post-quantization validation failed for ALL mode:\n" +
                "\n".join(errors)
            )

    def _is_valid_lora_producer(self, cpp_graph, tensor_name):
        """Check if producer is Conv2D, Matmul, Fc, Convert, or ElementwiseBinary(MULTIPLY)"""
        tensor = cpp_graph.get_tensor(tensor_name)
        if not tensor:
            return False

        producer = tensor.get_producer()
        if not producer:
            return False

        # Check if producer is Conv2D, Matmul, Fc or Convert
        if producer.type in [ir_graph.QNN_OP_CONV_2D, ir_graph.QNN_OP_FULLY_CONNECTED,
                             ir_graph.QNN_OP_MAT_MUL, ir_graph.QNN_OP_CONVERT]:
            return True

        # Check ElementwiseBinary with MULTIPLY
        if producer.type == ir_graph.IR_OP_ELTWISE_BINARY:
            if producer.attrs.has(ir_graph.QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION):
                operation = producer.attrs.get_uint32(ir_graph.QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION)
                return operation == ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY

        return False

    def _validate_adapter_only_mode_post_quantize_cpp(self, cpp_graph):
        """
        Validate ADAPTER_ONLY mode: Adapter weight values are updatable, only LoRA branch quant encodings are updatable.

        In ADAPTER_ONLY mode:
        - lora_tensor_names contains both static tensors (LoRA weights) and non-static tensors (LoRA activations)
        - LoRA static tensors: values should be updatable
        - LoRA non-static tensors: must be outputs of conv2d, matmul, fc, convert or mul ops
        - LoRA quantized non-static tensors: quant encodings should be updatable (marked as updatable)
        - Non-LoRA tensors: nothing should be updatable

        Note: For non-static tensors, "updatable" means their quant encodings are updatable, not tensor values.

        :param cpp_graph: C++ graph (Post-Quantizer)
        :raises RuntimeError: If validation fails
        """
        # Collect problematic tensors by error type
        lora_static_tensors_missing_updatable = []
        lora_non_static_tensors_invalid_producer = []
        lora_non_quantized_non_static_tensors_incorrectly_updatable = []
        lora_quantized_non_static_tensors_missing_updatable = []
        non_lora_tensors_incorrectly_updatable = []

        tensor_map = cpp_graph.get_tensor_map()
        lora_tensor_names = cpp_graph.get_all_updatable_tensors()

        for tensor_name, tensor in tensor_map.items():
            if tensor_name in lora_tensor_names:
                # LoRA tensors
                if tensor.is_static_tensor():
                    # Rule 1: All LoRA static tensors should have updatable values
                    if not tensor.is_updateable():
                        lora_static_tensors_missing_updatable.append(tensor_name)
                else:
                    # LoRA non-static tensors
                    # Rule 2: LoRA non-static tensors must be outputs of conv2d or mul or convert
                    if not self._is_valid_lora_producer(cpp_graph, tensor_name):
                        lora_non_static_tensors_invalid_producer.append(tensor_name)

                    if not tensor.is_quantized():
                        # Rule 3: LoRA non-quantized non-static tensors should NOT be updatable
                        if tensor.is_updateable():
                            lora_non_quantized_non_static_tensors_incorrectly_updatable.append(tensor_name)
                    else:
                        # Rule 4: LoRA quantized non-static tensors should be updatable
                        if not tensor.is_updateable():
                            lora_quantized_non_static_tensors_missing_updatable.append(tensor_name)
            else:
                # Rule 5: Non-LoRA tensors (both static and non-static) should NOT be updatable
                if tensor.is_updateable():
                    non_lora_tensors_incorrectly_updatable.append(tensor_name)

        # Build error message with summary
        errors = []
        if lora_static_tensors_missing_updatable:
            errors.append(
                f"ADAPTER_ONLY mode: {len(lora_static_tensors_missing_updatable)} LoRA static tensor(s) not marked as updatable (values should be updatable). "
                f"Examples: {', '.join(lora_static_tensors_missing_updatable[:5])}"
                + (" ..." if len(lora_static_tensors_missing_updatable) > 5 else "")
            )

        if lora_non_static_tensors_invalid_producer:
            errors.append(
                f"ADAPTER_ONLY mode: {len(lora_non_static_tensors_invalid_producer)} LoRA non-static tensor(s) have invalid producers (must be conv2d, convert, or mul). "
                f"Examples: {', '.join(lora_non_static_tensors_invalid_producer[:5])}"
                + (" ..." if len(lora_non_static_tensors_invalid_producer) > 5 else "")
            )

        if lora_non_quantized_non_static_tensors_incorrectly_updatable:
            errors.append(
                f"ADAPTER_ONLY mode: {len(lora_non_quantized_non_static_tensors_incorrectly_updatable)} LoRA non-quantized non-static tensor(s) incorrectly marked as updatable. "
                f"Examples: {', '.join(lora_non_quantized_non_static_tensors_incorrectly_updatable[:5])}"
                + (" ..." if len(lora_non_quantized_non_static_tensors_incorrectly_updatable) > 5 else "")
            )

        if lora_quantized_non_static_tensors_missing_updatable:
            errors.append(
                f"ADAPTER_ONLY mode: {len(lora_quantized_non_static_tensors_missing_updatable)} LoRA quantized non-static tensor(s) not marked as updatable (quant encodings should be updatable). "
                f"Examples: {', '.join(lora_quantized_non_static_tensors_missing_updatable[:5])}"
                + (" ..." if len(lora_quantized_non_static_tensors_missing_updatable) > 5 else "")
            )

        if non_lora_tensors_incorrectly_updatable:
            errors.append(
                f"ADAPTER_ONLY mode: {len(non_lora_tensors_incorrectly_updatable)} non-LoRA tensor(s) incorrectly marked as updatable. "
                f"Examples: {', '.join(non_lora_tensors_incorrectly_updatable[:5])}"
                + (" ..." if len(non_lora_tensors_incorrectly_updatable) > 5 else "")
            )

        if errors:
            raise RuntimeError(
                "Post-quantization validation failed for ADAPTER_ONLY mode:\n" +
                "\n".join(errors)
            )