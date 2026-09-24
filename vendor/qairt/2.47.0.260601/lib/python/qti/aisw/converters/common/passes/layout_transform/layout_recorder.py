# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define layout recorder for tracing layouts during layout transform."""

from qti.aisw.converters.common.utils.converter_utils import log_assert
from qti.aisw.converters.common.passes.layout_transform import util


class LayoutRecorder(object):
    """A container for tracing layouts of graph buffers.

    This recorder aims to keep track of all buffers' layouts during layout transform regardless any
    specific frontend. It provides several methods to access layouts in list or dict formats
    according to the expected usage. Users must as well correctly update the memo through provided
    methods after transforming layouts for any buffer.

    The tracking of buffers' layouts is maintained through a mapping of buffer names on source
    graph to a dict containing pairs of permute sequences and buffer names on new graph. The dict
    represents the buffer on new graph transposed from the corresponding one on source graph by
    the permute sequence as key. Below provides an example of layout memo structure:

        -----------------------------------------------------------------------
        | buffer name on src graph | perm_seq -> buffer_names on new graph    |
        |--------------------------|------------------------------------------|
        | 'b1'                     | {(0,1,2,3): 'b1', (0,2,3,1): 'b1_0231'}  |
        |--------------------------|------------------------------------------|
        | 'b2'                     | {(0,2,3,1): 'b2'}                        |
        -----------------------------------------------------------------------

    Attributes:
        custom_layout_table: A dict recording custom layouts for specified buffer names.
        desired_layout_table: A dict recording desired layouts for heavily layout-sensitive ops.
            Refer to `layout_defs.py` for details.
        layout_record: A dict mapping source graph buffers to new graph ones by permute sequences.
        preferred_layout_table: A dict recording preferred permute sequences in order for
            layout-agnostic ops.
        src_layout_table: A dict recording source layouts for heavily layout-sensitive ops. Refer to
            `layout_defs.py` for details.

    Methods:
        get_buffer_name_on_new_graph: Get buffer name on new graph by given source buffer name and
            permute sequence
        get_custom_perm_seq: Get custom permute sequence for target buffer name.
        get_desired_layouts: Get desired layouts for target op type and rank.
        get_layout_map: Get layout map for target buffer.
        get_perm_seqs: Get permute sequences for target buffer.
        get_perm_seqs_table: Get permute sequences table for target buffers.
        get_preferred_perm_seqs: Get preferred permute sequences by rank.
        get_src_layouts: Get source layouts for target op type and rank.
        update_perm_seq: Update permute sequence for given buffer.
        update_perm_seqs: Update permute sequences for given buffers.
    """

    def __init__(
        self,
        src_layout_table,
        desired_layout_table,
        preferred_perm_seq_table,
        custom_layout_table
    ):
        """Initialize layout recorder.

        Args:
            src_layout_table: Refer to Attributes.
            desired_layout_table: Refer to Attributes.
            preferred_perm_seq_table: Refer to Attributes.
            custom_layout_table: Refer to Attributes.
        """
        self.src_layout_table = src_layout_table
        self.desired_layout_table = desired_layout_table
        self.preferred_perm_seq_table = preferred_perm_seq_table
        self._custom_layout_table = custom_layout_table

        self.layout_record = dict()

    def update_perm_seq(self, src_buffer_name, new_buffer_name, new_perm_seq):
        """Update permute sequence for given buffer.

        Args:
            src_buffer_name: A str specifying the key in layout memo.
            new_buffer_name: A str specifying the value to be updated in layout memo.
            new_perm_seq: A tuple of ints specifying the second-level key in layout memo.
        """
        if src_buffer_name not in self.layout_record:
            self.layout_record[src_buffer_name] = {new_perm_seq: new_buffer_name}
        else:
            # TODO: Check perm seq exists or not.
            layout_map = self.layout_record[src_buffer_name]
            layout_map[new_perm_seq] = new_buffer_name

    def update_perm_seqs(self, src_buffer_names, new_buffer_names, new_perm_seqs):
        """Update permute sequences for given buffers.

        Args:
            src_buffer_names: A list of strs specifying the keys in layout memo.
            new_buffer_names: A list strs specifying the values to be updated in layout memo.
            new_perm_seqs: A list of tuple of ints specifying the second-level keys in layout memo.
        """
        for src_buffer_name, new_buffer_name, new_perm_seq in zip(
            src_buffer_names, new_buffer_names, new_perm_seqs
        ):
            self.update_perm_seq(src_buffer_name, new_buffer_name, new_perm_seq)

    def get_perm_seqs(self, buffer_name, rank=None):
        """Get permute sequences for target buffer.

        Args:
            buffer_name: A str specifying the target buffer.
            rank: An int specifying the target rank for permute sequences to be acquired. Defaults
                to None where no filtering will be performed.

        Returns:
            A list of tuple of ints containing the permute sequences for target buffer.
        """
        perm_seqs = list(self.layout_record[buffer_name].keys())
        if rank is None:
            return perm_seqs

        # If rank is provided, only those perm seqs in specified rank will be returned.
        return util.get_perm_seqs_by_rank(perm_seqs, rank)

    def get_perm_seqs_table(self, buffer_names, ranks=None):
        """Get permute sequences table for target buffers.

        For example of buffer 'b1' and 'b2', the table may be like:

            -----------------------------------------------------------
            | buffer name on src graph | perm seqs on new graph       |
            |--------------------------|------------------------------|
            | 'b1'                     | [(0,1,2,3), (0,2,3,1)]       |
            |--------------------------|------------------------------|
            | 'b2'                     | [(0,2,3,1)]                  |
            -----------------------------------------------------------

        Args:
            buffer_names: A list of strs specifying the target buffers.
            ranks: A list of ints specifying the target ranks for permute sequences to be acquired.
                Defaults to None where no filtering will be performed.

        Returns:
            perm_seqs_table: A dict mapping buffer names to list of tuple of ints containing the
                permute sequences for corresponding target buffers.
        """
        if ranks:
            log_assert(
                len(buffer_names) == len(ranks),
                "Number of buffer_names and ranks must be same, but got {} buffer_names and {} ranks.",
                len(buffer_names),
                len(ranks),
            )
        else:
            ranks = [None] * len(buffer_names)

        perm_seqs_table = dict()
        for buffer_name, rank in zip(buffer_names, ranks):
            perm_seqs = list(self.layout_record[buffer_name].keys())
            if rank is None:
                perm_seqs_table[buffer_name] = perm_seqs
            else:
                # If ranks are provided, only those perm seqs in specified ranks will be returned.
                perm_seqs_table[buffer_name] = util.get_perm_seqs_by_rank(
                    perm_seqs, rank
                )
        return perm_seqs_table

    def get_layout_map(self, buffer_name):
        """Get layout map for target buffer.

        For example of below layout memo and target buffer 'b1':

            -----------------------------------------------------------------------
            | buffer name on src graph | perm_seq -> buffer_names on new graph    |
            |--------------------------|------------------------------------------|
            | 'b1'                     | {(0,1,2,3): 'b1', (0,2,3,1): 'b1_0231'}  |
            |--------------------------|------------------------------------------|
            | 'b2'                     | {(0,2,3,1): 'b2'}                        |
            -----------------------------------------------------------------------

        Then the dict {(0,1,2,3): 'b1', (0,2,31): 'b1_02341'} will be returned.

        Args:
            buffer_name: A str specifying the target buffer for acquiring layout.

        Returns:
            A dict mapping permute sequences to buffer names on new graph.
        """
        return self.layout_record[buffer_name]

    def get_buffer_name_on_new_graph(self, src_input_buffer_name, perm_seq):
        """Get buffer name on new graph by given source buffer name and permute sequence.

        Args:
            src_input_buffer_name: A str specifying the target buffer name on source graph.
            perm_seq: A tuple of ints specifying target permute sequence.

        Returns:
            A str specifying the target buffer name on new graph.
        """
        return self.layout_record[src_input_buffer_name][perm_seq]

    def get_preferred_perm_seqs(self, rank):
        """Get preferred permute sequences by rank.

        Defaults to source permute sequence if target rank is not specified in the table.

        Args:
            rank: An int specifying the target rank.

        Returns:
            A list of preferred permute sequences.
        """
        if rank not in self.preferred_perm_seq_table:
            return [tuple(range(rank))]
        return self.preferred_perm_seq_table[rank]

    def get_src_layouts(self, op_type, rank):
        """Get source layouts for target op type and rank.

        Args:
            op_type: A str specifying the target op type.
            rank: An int specifying the target rank.

        Returns:
            A list of source layouts.
        """
        return self.src_layout_table[op_type][rank]

    def get_desired_layouts(self, op_type, rank):
        """Get desired layouts for target op type and rank.

        Args:
            op_type: A str specifying the target op type.
            rank: An int specifying the target rank.

        Returns:
            A list of desired layouts.
        """
        return self.desired_layout_table[op_type][rank]

    def get_custom_perm_seq(self, buffer_name):
        """Get custom permute sequence for target buffer name.

        Args:
            buffer_name: A str specifying target buffer name.

        Returns:
            A tuple of ints specifying custom permute sequence or None.
        """
        custom_layout  = self._custom_layout_table.get(buffer_name)
        if custom_layout:
            # buffer name is specified in custom_layout_table
            # accepted layouts: NCDHW, NDHWC, NCHW, NHWC, NFC, NCF, NTF, TNF, NF, NC, F, NONTRIVIAL
            # If NONTRIVIAL is specified, which means that buffer should remain in src-format
            if custom_layout['Source'] == 'NONTRIVIAL' and custom_layout['Desired'] == 'NONTRIVIAL':
                return None

            custom_perm_seq = util.calculate_perm_seq(
                custom_layout['Source'], custom_layout['Desired']
            )

            # check custom_perm_seq is valid permute sequence to propagate
            if (
                len(custom_perm_seq) <= 5
                and custom_perm_seq in self.preferred_perm_seq_table[len(custom_perm_seq)]
            ):
                return custom_perm_seq
            else:
                raise ValueError(f"Custom_IO layout parameters for {buffer_name} aren't valid.\
                    Src Layout is {custom_layout['Source']}.\
                    Desired Layout is {custom_layout['Desired']}."
                )

        # buffer name is not specified in custom_layout_table
        return None
