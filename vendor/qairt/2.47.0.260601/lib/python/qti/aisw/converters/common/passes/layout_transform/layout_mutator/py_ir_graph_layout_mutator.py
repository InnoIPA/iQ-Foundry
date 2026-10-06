# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define layout mutator for PyIrGraph."""

import numpy as np

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir import op_adapter, op_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker, AxisOrder
from qti.aisw.converters.common.passes.layout_transform import util
from qti.aisw.converters.common.passes.layout_transform.layout_defs import (
    DEFAULT_INPUT_LAYOUT_TABLE,
)
from qti.aisw.converters.common.passes.layout_transform.layout_mutator import layout_mutator_base


class PyIrGraphLayoutMutator(layout_mutator_base.LayoutMutatorBase):
    """Layout mutator for Python IR Graph.

    Attributes:
        new_graph: An instance of PyIrGraph after layout transform.
        src_graph: An instance of PyIrGraph before layout transform.

    Methods:
        add_op_to_new_graph: Add op to new graph.
        get_src_buffer_shapes: Get shapes for given source buffers.
        get_src_input_buffer_names: Get input buffer names for source node.
        get_src_output_buffer_names: Get output buffer names for source node.
        preserve_graph_output_layout: Preserve graph output layout if necessary.
    """

    def __init__(self, src_graph):
        """Initialize mutator with source graph.

        Args:
            src_graph: Refer to Attributes.
        """
        super().__init__()
        self.src_graph = src_graph
        self.new_graph = self._create_empty_new_graph(src_graph)

    def _add_reshape_if_not_exist(
        self,
        reshape_name,
        input_buffer_name,
        output_buffer_name,
        shape,
        src_info=None
    ):
        """Add Reshape into new graph if not exist."""
        if self.new_graph.has_buffer(output_buffer_name):
            return False

        reshape_op = op_adapter.ReshapeOp(reshape_name, shape=shape)
        node = self.new_graph.add(
            reshape_op,
            [input_buffer_name],
            [output_buffer_name],
            [AxisTracker.AxisFormat.NONTRIVIAL],
            perms_to_src=[list(util.get_src_perm_seq(len(shape)))]
        )

        # Set op trace info.
        self.new_graph.set_trace_info(node, src_info)
        self.new_graph.set_trace_info(self.new_graph.get_buffer(output_buffer_name), src_info)
        return True

    def _add_transpose(
        self,
        transpose_op_name,
        input_buffer_name,
        output_buffer_name,
        current_perm_seq,
        target_perm_seq,
        src_info=None,
        is_forward_tracing=False
    ):
        """Add Transpose op into new graph."""
        # Calculate perm seq of transpose to target one.
        lt_perm_seq = util.calculate_perm_seq(current_perm_seq, target_perm_seq)
        lt_op = op_adapter.TransposeOp(transpose_op_name, lt_perm_seq)
        node = self.new_graph.add(
            lt_op,
            [input_buffer_name],
            [output_buffer_name],
            [AxisTracker.AxisFormat.NONTRIVIAL],
            perms_to_src=[util.get_perm_to_src(target_perm_seq)]
        )

        # Set op trace info.
        self.new_graph.set_trace_info(node, src_info)
        src_buffer_name = input_buffer_name if is_forward_tracing else output_buffer_name
        self.new_graph.set_trace_info(self.new_graph.get_buffer(src_buffer_name), src_info)

    def _create_empty_new_graph(self, src_graph):
        """Create new graph from source graph for layout transform."""
        new_graph = op_graph.IROpGraph(
            naming_policy=src_graph.naming_policy,
            shape_inference_policy=src_graph.shape_inference_policy,
            input_types=[],
            input_dtypes=[],
            input_encodings=[],
            src_axis_order=AxisOrder(),
            input_layouts=[],
        )
        src_vars = vars(src_graph)
        # TODO: Revisit the feature like "custom_IO".
        # Check how to support this feature with new layout transform algorithm.
        # ignore "int64_input_cast_map" because this datatype check has already done in src graph
        ignored_list = [
            "naming_policy",
            "shape_inference_policy",
            "src_axis_order",
            "input_axis_formats",
            "preserve_layout_tensors",
            "preserve_io",
            "preserve_io_layout_passed",
            "nodes_by_name",
            "nodes_in_order",
            "buffers",
            "total_macs",
            "total_params_count",
            "int64_input_cast_map",
        ]
        # Copy required variables to new graph.
        for var_key in src_vars:
            if var_key not in ignored_list:
                setattr(new_graph, var_key, src_vars[var_key])

        return new_graph

    def _get_shape_by_name(self, graph, name):
        """Get shape from graph by given name."""
        # In Lstm/Gru cases, some inputs are optional, so its name may be empty string
        # For this case, return empty list
        return graph.buffers[name].shape.dims if name else []

    def _update_op_attribute(self, op, new_attrs):
        """"Update op attribute."""
        if not new_attrs:
            return

        if isinstance(op, op_adapter.ReshapeOp):
            # TODO: Fix data_type of shape attribute to list[int] in Reshape op
            # Transform shape params from list[int] to IrStaticTensor
            new_shape = new_attrs["shape"]
            new_shape_tensor = ir_graph.IrStaticTensor(
                ir_graph.IR_OP_RESHAPE_PARAM_SHAPE,
                [len(new_shape)],
                np.array(new_shape, dtype=np.int32),
                ir_graph.QNN_DATATYPE_INT_32,
            )
            new_attrs["shape"] = new_shape_tensor

        if isinstance(op, op_adapter.StridedSliceOp) and 'ranges' in new_attrs:
            # TODO: Revise pybind to accept numpy array instead of IrStaticTensor.
            ranges = np.array(new_attrs['ranges'], dtype=np.int32)
            new_attrs['ranges'] = ir_graph.IrStaticTensor(
                ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                list(ranges.shape),
                ranges,
                ir_graph.QNN_DATATYPE_INT_32
            )

        for attr_key, attr_val in new_attrs.items():
            setattr(op, attr_key, attr_val)

    def _update_custom_op_output_shape(self, node, output_perm_seqs):
        """Update output buffer shapes for CustomOp.

        The output buffer shapes of CustomOp is explicitly permuted here as a workaround for its
        shape inference function not able to be invoked twice.
        """
        output_buffers = self.new_graph.get_output_buffers(node)
        for buffer, perm_seq in zip(output_buffers, output_perm_seqs):
            buffer.shape = [buffer.shape[axis] for axis in perm_seq]

    def add_op_to_new_graph(
        self,
        src_node,
        new_attrs,
        new_input_buffer_names,
        output_perm_seqs,
        layout_recorder,
    ):
        """Add op to new graph.

        The overall process of adding op to new graph is divided into three steps:
            1. Update op attributes.
            2. Add op to new graph.
            3. Update layout memo.

        Args:
            src_node: An instance of OpNode on source graph.
            new_attrs: A dict containing attributes to be updated.
            new_input_buffer_names: A list of strs specifying the input names for new node.
            output_perm_seqs: A list of output permute sequences for new node.
            layout_recorder: An instance of LayoutRecorder.

        Returns:
            new_node: An instance of OpNode added on new graph.
        """
        op = src_node.op

        # Step 1.
        self._update_op_attribute(op, new_attrs)

        src_output_buffers = self.src_graph.get_output_buffers(src_node)
        output_buffer_names = self.get_src_output_buffer_names(src_node)

        # Step 2.
        new_node = self.new_graph.add(
            op,
            new_input_buffer_names,
            output_buffer_names,
            [AxisTracker.AxisFormat.NONTRIVIAL] * len(output_buffer_names),
            sparse_params=src_output_buffers[0].sparse_params,
            perms_to_src=util.get_perms_to_src(output_perm_seqs)
        )

        if isinstance(op, op_adapter.CustomOp):
            self._update_custom_op_output_shape(new_node, output_perm_seqs)

        # Step 3.
        layout_recorder.update_perm_seqs(
            output_buffer_names, output_buffer_names, output_perm_seqs
        )

        return new_node

    def get_src_buffer_shapes(self, buffer_names):
        """Get shapes for given source buffers.

        Args:
            buffer_names: A list of strs specifying buffers on source graph.

        Returns:
            A list of list of ints specifying the shapes for target buffers.
        """
        return [
            self._get_shape_by_name(self.src_graph, buffer_name) for buffer_name in buffer_names
        ]

    def get_src_input_buffer_names(self, src_node):
        """Get input buffer names for source node.

        Args:
            src_node: An instance of OpNode on source graph.

        Returns:
            A list of strs specifying the input buffer names of given node.
        """
        if isinstance(src_node.op, op_adapter.InputOp):
            # input op doesn't have input buffer,
            # but we need it's output buffer name to get its custom_layout.
            # Thus, we return output buffer name for input op case.
            return self.get_src_output_buffer_names(src_node)
        input_bufs = self.src_graph.get_input_buffers(src_node)
        return [input_buf.name for input_buf in input_bufs]

    def get_src_output_buffer_names(self, src_node):
        """Get output buffer names for source node.

        Args:
            src_node: An instance of OpNode on source graph.

        Returns:
            A list of strs specifying the output buffer names of given node.
        """
        output_bufs = self.src_graph.get_output_buffers(src_node)
        return [output_buf.name for output_buf in output_bufs]

    def transform_output_buffers(
        self, output_buffer_names, output_perm_seqs, new_node, layout_recorder
    ):
        """Transform output buffers.

        Case 1. Buffer is graph input, and source_model_input_layout is specified.
                Layout-Transform would implicitly insert transpose behind to propagate channel-last
                permute sequence to the following nodes.
                For example, source_model_input_layout is "NCHW". Transpose with perm (0,2,3,1) will
                be inserted, which can populate (0,2,3,1) from graph input.
        Case 2. Buffer is graph output, and custom layout isn't specified in custom_layout_table.
                For example, custom_layout_table is {}.
                4D buffer called "input" would be transformed back to src-format,
                and the target output permute sequence would be (0,1,2,3).
        Case 3. Buffer is graph output, and custom layout is specified in custom_layout_table.
                For example, custom_layout_table is {"input": {"Source":"NCHW", "Desired":"NHWC"}}.
                4D buffer called "input" will be transformed into NHWC,
                and the target output permute sequence would be (0,2,3,1).

        Args:
            output_buffer_names: A list of strs specifying the output buffer names for new node.
            output_perm_seqs: A list of output permute sequences for new node.
            new_node: An instance of OpNode newly added on new graph.
            layout_recorder: An instance of LayoutRecorder.
        """
        for idx, output_buffer_name in enumerate(output_buffer_names):
            output_perm_seq = output_perm_seqs[idx]
            rank = len(output_perm_seq)

            if isinstance(new_node.op, op_adapter.InputOp) and 3 <= rank <= 5:
                target_output_perm_seq = output_perm_seq
                if output_buffer_name in self.src_graph.input_axis_formats:
                    # Case 1.
                    # Get user-specified input layout and calculate desired output perm_seq.
                    src_layout = self.src_graph.input_axis_formats[output_buffer_name]

                    if src_layout == "NONTRIVIAL":
                        # When NONTRIVIAL is specified for this input buffer, no transpose would be
                        # inserted.
                        continue
                    elif src_layout in DEFAULT_INPUT_LAYOUT_TABLE:
                        desired_layout = DEFAULT_INPUT_LAYOUT_TABLE[src_layout]
                        target_output_perm_seq = util.calculate_perm_seq(src_layout, desired_layout)
                    else:
                        raise ValueError(
                            f"Invalid source_model_input_layout {src_layout} is specified."
                        )

                if target_output_perm_seq == output_perm_seq:
                    continue

                new_out_buf_name = util.generate_new_buffer_name(
                    output_buffer_name, target_output_perm_seq
                )

                # Add transpose to transform output layout.
                self._add_transpose(
                    new_out_buf_name,
                    output_buffer_name,
                    new_out_buf_name,
                    output_perm_seq,
                    target_output_perm_seq,
                    [(output_buffer_name, op_graph.TraceType.TENSOR)]
                )

                # Update transposed buffer into memo.
                layout_recorder.update_perm_seq(
                    output_buffer_name, new_out_buf_name, target_output_perm_seq
                )

            elif output_buffer_name in self.new_graph.output_names:
                target_output_perm_seq  = layout_recorder.get_custom_perm_seq(output_buffer_name)
                if not target_output_perm_seq:
                    # Case 2.
                    # Custom layout is not specified, so we use src permute sequence as output
                    # permute sequence
                    target_output_perm_seq = util.get_src_perm_seq(rank)
                if target_output_perm_seq == output_perm_seq:
                    continue
                # Rename current output buffer to make sure that new graph's output name unchanged,
                # where current output buffer name will be inherited by the Transpose op inserted
                # afterwards.
                new_out_buf_name = util.generate_new_buffer_name(
                    output_buffer_name, output_perm_seq
                )

                # Update naming in node.
                new_node.output_names[idx] = new_out_buf_name
                # Update naming in buffer.
                curr_output_buf = self.new_graph.buffers.pop(output_buffer_name)
                curr_output_buf.name = new_out_buf_name
                # Update naming in graph.
                self.new_graph.buffers[new_out_buf_name] = curr_output_buf
                # Update naming in layout recorder.
                layout_recorder.update_perm_seq(
                    output_buffer_name, new_out_buf_name, output_perm_seq
                )

                # Add transpose behind to transform output layout.
                self._add_transpose(
                    new_out_buf_name,
                    new_out_buf_name,
                    output_buffer_name,
                    output_perm_seq,
                    target_output_perm_seq,
                    [(output_buffer_name, op_graph.TraceType.TENSOR)],
                    is_forward_tracing=True
                )

                # Update quantization_params if output buffer name is changed.
                quant_params = self.new_graph.quantization_params.get(new_node.op.name)
                if quant_params:
                    for quant_param in quant_params['output_encodings']:
                        if quant_param['name'] == output_buffer_name:
                            quant_param['name'] = new_out_buf_name

                # Update transposed buffer into memo.
                layout_recorder.update_perm_seq(
                    output_buffer_name, output_buffer_name, target_output_perm_seq
                )
