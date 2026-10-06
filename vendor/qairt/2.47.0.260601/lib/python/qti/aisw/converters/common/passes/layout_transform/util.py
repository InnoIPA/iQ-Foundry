# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Define utility functions for layout transform.

Functions:
    align_rank: Unsqueeze the shape into given rank.
    calculate_perm_seq: Calculate permute sequence.
    find_majority_perm_seqs: Find majority permute sequences.
    generate_new_buffer_name: Generate new buffer name.
    get_cf_perm_seq: Get permute sequence in channel-first (CF) format.
    get_cl_perm_seq: Get permute sequence in channel-last (CL) format.
    get_custom_op_layouts: Get source and desired layouts for CustomOp.
    get_perm_seqs_by_rank: Get permute sequences by rank.
    get_perm_to_src: Get permute sequence to transform buffer back to source format.
    get_perms_to_src: Get permute sequences to transform buffers back to source formats.
    get_ranks: Get ranks for given shapes.
    get_src_perm_seq: Get permute sequence in source format.
    get_target_perm_seq_by_vote: Get target permute sequence by vote.
    search_preferred_perm_seqs_in_order: Search preferred permute sequence in order.
"""

from collections import Counter
import itertools

from qti.aisw.converters.common.utils import converter_utils


def get_src_perm_seq(rank):
    """Get permute sequence in source format.

    For example of rank=4, permute sequence in source format will be (0,1,2,3).

    Args:
        rank: An int specifying the target rank.

    Returns:
        A tuple of ints specifying the permute sequence in source format.
    """
    return tuple(range(rank))


def get_cf_perm_seq(rank):
    """Get permute sequence in channel-first (CF) format.

    For example of rank=4, permute sequence in CF format will be (0,3,1,2).

    Args:
        rank: An int specifying the target rank.

    Returns:
        A tuple of ints specifying the permute sequence in CF format.
    """
    converter_utils.log_assert(
        3 <= rank <= 5, f"Only rank 3 to 5 are valid for channel-first layout but got rank {rank}."
    )
    src_perm_seq = get_src_perm_seq(rank)
    return (src_perm_seq[0], src_perm_seq[-1], *src_perm_seq[1:-1])


def get_cl_perm_seq(rank):
    """Get permute sequence in channel-last (CL) format.

    For example of rank=4, permute sequence in CL format will be (0,2,3,1).

    Args:
        rank: An int specifying the target rank.

    Returns:
        A tuple of ints specifying the permute sequence in CL format.
    """
    converter_utils.log_assert(
        3 <= rank <= 5, f"Only rank 3 to 5 are valid for channel-last layout but got rank {rank}."
    )
    src_perm_seq = get_src_perm_seq(rank)
    return (src_perm_seq[0], *src_perm_seq[2:], src_perm_seq[1])


def calculate_perm_seq(input_perm_seq, target_perm_seq):
    """Calculate permute sequence.

    Calculate the permute order to transpose input permute sequence to target one. For example:

        case 1: input_perm_seq (0,1,2,3), target_perm_seq (0,2,3,1) --> (0,2,3,1)
        case 2: input_perm_seq (0,2,3,1), target_perm_seq (0,1,2,3) --> (0,3,1,2)

    Args:
        input_perm_seq:
        target_perm_seq:

    Return:
        A tuple of ints specifying the permute sequence for transpose.
    """
    return tuple(input_perm_seq.index(axis) for axis in target_perm_seq)


def generate_new_buffer_name(buf_name, perm_seq):
    """Generate new buffer name.

    For example of buf_name='b1' and perm_seq=(0,2,3,1), new buffer name will be 'b1_0231'.

    Args:
        buf_name: A str specifying base name.
        perm_seq: A tuple of ints specifying the postfix for buffer name.

    Returns:
        A str in {buf_name}_{perm_seq} format.
    """
    return buf_name + "_" + "".join(map(str, perm_seq))


def find_majority_perm_seqs(perm_seqs_table):
    """Find majority permute sequences.

    For example:

        -----------------------------------------------------------
        | buffer name on src_graph | perm_seqs exist on new_graph |
        |--------------------------|------------------------------|
        | 'b1'                     | [(0,1,2,3), (0,2,3,1)]       |
        |--------------------------|------------------------------|
        | 'b2'                     | [(0,2,3,1)]                  |
        -----------------------------------------------------------

        where the counting results will be {(0,2,3,1): 2, (0,1,2,3): 1}, and therefore the permute
        sequence with maximum occurrence is (0,2,3,1).

    Notes
        1. This function expects all permute sequences in the table having identical ranks.
        2. If multiple permute sequences have the same occurence, they will all be returned.

    Args:
        perm_seqs_table: A dict containing buffer names and corresponding existing permute
            sequences.

    Returns:
        A list of tuple of ints containing permute sequences with maximum occurence.
    """
    perm_seq_statistical_table = Counter(
        itertools.chain.from_iterable(perm_seqs_table.values())
    )
    max_value = max(perm_seq_statistical_table.values())
    majority_perm_seqs = [
        k for k, v in perm_seq_statistical_table.items() if v == max_value
    ]
    return majority_perm_seqs


def search_preferred_perm_seqs_in_order(preferred_perm_seqs, perm_seqs):
    """Search preferred permute sequence in order.

    Each of the preferred permute sequence is orderly checked whether presenting in the given
    permute sequences. Therefore, the order of preferred sequences is significant, where the first
    matched one is returned. Note that if none of the preferred ones are matched, the first one of
    the given permute sequences is returned.

    Args:
        preferred_perm_seqs: A list of tuple of ints specifying the preferred permute sequences.
        perm_seqs: A list of tuple of ints specifying the permute sequences to be searched over.

    Returns:
        A tuple of ints specifying the searched permute sequence.
    """
    for preferred_perm_seq in preferred_perm_seqs:
        if preferred_perm_seq in perm_seqs:
            return preferred_perm_seq

    # No match, select the first one.
    return perm_seqs[0]


def align_rank(shape, rank):
    """Unsqueeze the shape into given rank.

    This function aims to unsqueeze the shape for broadcastability by prepending 1 in the given
    shape until matching the target rank. For example of shape=[1,16,48] and rank=4, unsqueezed
    shape will be [1,1,16,48].

    Args:
        shape: A list of ints specifying the shape to be unsqueezed.
        rank: An int specifying the target rank after unsqueeze.

    Returns:
        A list of ints specifying the unsqueezed shapes.
    """
    broadcast_prefix = [1] * (rank - len(shape))
    return broadcast_prefix + shape


def get_perm_seqs_by_rank(perm_seqs, rank):
    """Get permute sequences by rank.

    This function aims to filter permute sequences with undesired ranks.

    Args:
        perm_seqs: A list of tuple of ints containing permute sequences.
        rank: A int specifying the target rank.

    Returns:
        A list of tuple of ints containing permute sequences with target rank.
    """
    return [perm_seq for perm_seq in perm_seqs if len(perm_seq) == rank]


def get_ranks(shapes):
    """Get ranks for given shapes.

    Args:
        shapes: A list of list of ints containing shapes.

    Returns:
        A list of ints specifying corresponding ranks for given shapes.
    """
    return [len(shape) for shape in shapes]


def get_target_perm_seq_by_vote(perm_seqs_table, preferred_perm_seqs):
    """Get target permute sequence by vote.

    This function first attempts to acquire target permute sequence through majority voting and
    then handle tied cases by determining from given preferred ones. Note that all permute
    sequences in the given table are expected to have identical ranks.

    Args:
        perm_seqs_table: A dict mapping buffer names to existing permute sequences.
        preferred_perm_seqs: A list of preferred permute sequences in order.

    Returns:
        A tuple of ints as target permute sequence.
    """
    # Find majority permute sequences.
    majority_perm_seqs = find_majority_perm_seqs(perm_seqs_table)

    if len(majority_perm_seqs) > 1:
        # Handle tied vote case by searching preferred perm seqs.
        target_perm_seq = search_preferred_perm_seqs_in_order(
            preferred_perm_seqs, majority_perm_seqs
        )

        # Select the first one if no preferred is matched.
        if not target_perm_seq:
            target_perm_seq = majority_perm_seqs[0]
    else:
        target_perm_seq = majority_perm_seqs[0]

    return target_perm_seq


def get_perm_to_src(perm_seq):
    """Get permute sequence to transform buffer back to source format.

    Args:
        perm_seq: A list of ints specifying buffer's current permute sequence.

    Returns:
        A list of ints specifying permute sequence back to source format.
    """
    return [perm_seq.index(axis) for axis in get_src_perm_seq(len(perm_seq))]


def get_perms_to_src(perm_seqs):
    """Get permute sequences to transform buffers back to source formats.

    Args:
        perm_seqs: A list of a list of ints specifying buffers' current permute sequences.

    Returns:
        A list of list of ints specifying permute sequences back to source formats.
    """
    return list(map(get_perm_to_src, perm_seqs))


def get_custom_op_layouts(xml_layouts):
    """Get source and desired layouts for CustomOp.

    The source and desired layouts for CustomOp are parsed from the given dict specified from the
    XML file. Those unspecified input/output names and those non-NSC buffers will not be presented
    in the returned layout dict, and the caller must handle them.

    Note that only 4D layout can be specified for CustomOp currently, and therefore NCHW and NHWC
    are adopted as source and desired layouts, respectively.

    Args:
        xml_layouts: A dict specifying layouts for CustomOp input/output buffers.

    Returns:
        A dict mapping I/O names to source and desired layouts.
    """
    return {
        name: {'Source': 'NCHW', 'Desired': 'NHWC'}
        for name, layout in xml_layouts.items()
        if layout == 'NSC'
    }
