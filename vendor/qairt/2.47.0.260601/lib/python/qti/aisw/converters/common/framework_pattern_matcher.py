# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Callable, List, Tuple

from qti.aisw.converters.common.converter_ir.op_graph import BufferCriteria
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils.converter_utils import (
    log_assert,
    log_debug1,
    log_error,
)


class FrameworkPatternMatcher:
    @staticmethod
    def verify_op_types_exist(op_list: List[str], supported_op_list: List[str]) -> bool:
        """
        Verify whether the given op type list contains supported ops or not.

        :param List[str] op_list: List of op types to be checked.
        :param List[str] supported_op_list: List of supported op types.
        :return bool: Boolean status indicating whether the ops are supported or not.
        """
        if type(op_list) is not list:
            op_list = [op_list]

        for op in op_list:
            if op not in supported_op_list:
                log_error(f"Unknown operator with op type '{op}' found.")
                return False
        return True

    @staticmethod
    def validate_buffers(
        expected_buffer_info: Tuple[str, List],
        actual_buffers: List[str],
        supported_op_list: List[str],
    ) -> bool:
        """
        Validates the actual buffers(inputs or outputs of nodes) against the
        criteria set in the expected buffers.

        :param Tuple[str, List] expected_buffers: A tuple with BufferCriteria
            for matching the list of buffers, list of tuple pairs with each
            tuple containing the type of op and a buffer criteria.
            e.g. (BufferCriteria.<criteria>, [(op_type, BufferCriteria.<criteria>)
                                              (op_type, BufferCriteria.<criteria>)
                                              ...])
        :param List[str] actual_buffers: List of actual buffer types for the
            current node being evaluated
        :param List[str] supported_op_list: List of supported op types.

        :raises ValueError: If unknown buffer criteria is provided.
        :raises ValueError: If ALL criteria given and there exists more expected
            inputs
        :raises AssertionError: If unknown matching criteria is provided.
        :raises AssertionError: If unsupported op_type is provided.
        :raises AssertionError: If buffer index is more than length of actual
            buffers for MATCH_NUM_BUFS criteria.

        :return bool: True if actual buffers pass criteria set in the expected
            buffers, False otherwise
        """
        # remove matching criteria from expected buffers and validate
        matching_criteria, expected_buffers = expected_buffer_info
        matching_criteria = matching_criteria.upper()
        acceptable_matching_criteria = [
            BufferCriteria.MATCH_NUM_BUFS,
            BufferCriteria.FLEXIBLE_NUM_BUFS,
            BufferCriteria.MATCH_BUFS_AT_INDEX,
        ]
        log_assert(
            matching_criteria in acceptable_matching_criteria,
            f"Framework Pattern matching: Expected {acceptable_matching_criteria} for matching criteria of buffers, Got {matching_criteria}",
        )

        if matching_criteria == BufferCriteria.MATCH_NUM_BUFS and len(
            expected_buffers
        ) != len(actual_buffers):
            return False

        for op_type, buf_criteria in expected_buffers:
            log_assert(
                FrameworkPatternMatcher.verify_op_types_exist(
                    op_type, supported_op_list
                )
                is True,
                f"Not all requested op_type(s) '{op_type}' for matching sequence supported. Please verify that each op_type supported by model's framework.",
            )

            if type(buf_criteria) == int:
                if matching_criteria == BufferCriteria.MATCH_NUM_BUFS:
                    # User knows the number of input/output buffers to expect, hence it is an error to request
                    # an out-of-range index
                    log_assert(
                        buf_criteria < len(actual_buffers),
                        f"Framework Pattern matching: Index {buf_criteria} for buffer list length {len(actual_buffers)} not valid. Please adjust optimization criteria",
                    )

                    if actual_buffers[buf_criteria] != op_type:
                        return False

                elif matching_criteria == BufferCriteria.MATCH_BUFS_AT_INDEX:
                    # In this case, user doesnt know/care for the number of input/output buffers of a node but want to
                    # match ops that fit a certain criteria e.g. when the 2nd input is a particular op type;
                    # in this instance an out-of-range index is not an error.

                    if (
                        buf_criteria >= len(actual_buffers)
                        or actual_buffers[buf_criteria] != op_type
                    ):
                        return False
                elif matching_criteria == BufferCriteria.FLEXIBLE_NUM_BUFS:
                    # In this case, user knows exactly how many of this type to expect but does not care
                    # about the position in the inputs
                    op_type_count = len(
                        [
                            actual_op_type
                            for actual_op_type in actual_buffers
                            if actual_op_type == op_type
                        ]
                    )
                    if op_type_count != buf_criteria:
                        return False
            elif buf_criteria.upper() == BufferCriteria.ALL:
                if len(expected_buffers) != 1:
                    raise ValueError(
                        f"Framework Pattern matching: There should only be one expected buffer provided when Buffer criteria is 'ALL'. Got {len(expected_buffers)}"
                    )
                if not all(buf == op_type for buf in actual_buffers):
                    return False

            elif buf_criteria.upper() == BufferCriteria.ANY:
                if not any(buf == op_type for buf in actual_buffers):
                    return False

            elif buf_criteria.upper() == BufferCriteria.NONE:
                if any(buf == op_type for buf in actual_buffers):
                    return False

            # Unknown buffer criteria, so raise error
            else:
                raise ValueError(
                    f"Framework Pattern matching: Expected {['ALL', 'ANY', 'NONE']} or int(Index) for buffer criteria, Got {buf_criteria}"
                )

        return True

    @staticmethod
    def match(
        sequence: List[Tuple],
        loader: FrameworkModelLoader,
        validator: Callable = None,
        ignore_constants: bool = False,
    ) -> List:
        """
        Traverses graph to find the requested pattern using DFS.

        :param List[Tuple] sequence: A list of tuple of node op types with their inputs and outputs.
            i.e. Each tuple contains (op_type, ([inputs]), ([outputs]))
            The tuple for inputs/outputs should state BufferCriteria to verify list length;
            Additionally, each input/output should state specific BufferCriteria
            to determine how many(if any) if the buffer should be in the matched
            sequence.
            E.g. for format:
            sequence = [
                # node type A
                (op_type,
                    # inputs
                    (BufferCriteria.<criteria>, [(op_type, BufferCriteria.<criteria>),
                                                 (op_type, BufferCriteria.<criteria>),
                                                 ...]),
                    # outputs
                    (BufferCriteria.<criteria>, [(op_type, BufferCriteria.<criteria>),
                                                 (op_type, BufferCriteria.<criteria>),
                                                 ...])
                )
                # node type B
                (op_type,
                    # inputs
                    (),
                    # outputs
                    ()
                )
                ...
            ]
            E.g (Channel Shuffle).
                Note: we can pass strings instead of class.xxx for convenience,
                      this function handles both.
             sequence = [
                        ("Reshape",
                            (),
                            ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
                        ),
                        ("Transpose",
                            (),
                            ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
                        ),
                        ("Reshape",
                            (),
                            ()
                        )
                       ]
            Note 1: Both inputs and outputs should also be part of sequence node.
                So that the graph dependency can be identified.
            Note 2: BufferCriteria can either be one of the BufferCriteria Enums
                or an INT to match a specific index
            Note 3: It is not required to have inputs or outputs, they can be
                left empty.
        :param FrameworkModelLoader loader: Loader object required to parse and
            traverse model graph during pattern matching.
        :param Callable validator: A function to run if a match is found based
            on sequence. The matched sequence will be passed as
            {"node_tuples": (nodes_matched)} If not provided, function will
            return based on only matching the sequence as criteria.,
            defaults to None
        :param bool ignore_constants: If constant nodes need to be filtered
            during matching, this flag will be set to True., defaults to False
        :return List: List of node tuples that match the sequence provided,
            where each tuple contains the corresponding nodes for each <op_type>
            in the sequence.
        """
        requested_types_seq = [entry[0] for entry in sequence]
        nodes_list = loader.get_nodes()

        if ignore_constants:
            nodes_list = [
                node
                for node in nodes_list
                if loader.get_op_type(node) not in loader.const_op_type
            ]

        log_debug1(f"Evaluating to match Sequence {requested_types_seq}...")

        # Adding 'Input' and 'Output' as additional op so that we can define
        # sequence in pattern matching to identify model's inputs and outputs.
        io_op_list = ["Input", "Output"]
        supported_op_list = loader.get_supported_operators() + io_op_list
        # we want to allow use of strings for op translation_keys(i.e op_types) to make sequence length minimal
        # so validate user has asked to match op_types that are supported in op_adapter
        log_assert(
            FrameworkPatternMatcher.verify_op_types_exist(
                requested_types_seq, supported_op_list
            )
            is True,
            f"Not all requested op_type(s) '{requested_types_seq}' for "
            "matching sequence supported. Please verify that each op_type "
            "supported by model's framework.",
        )

        matched_nodes = []
        first_sequence_node_op_type = requested_types_seq[0]
        for model_node in nodes_list:
            if loader.get_op_type(model_node) == first_sequence_node_op_type:
                start_node = model_node

                start_node_depth = 0
                stack = [[start_node, start_node_depth, [start_node]]]

                matched_node_tuples = []
                while len(stack) != 0:
                    _node, _node_level, _path = stack.pop()
                    if _node_level >= len(requested_types_seq):
                        # This means we have checked all the elements in the
                        # pattern. Now we should not traverse further into
                        # children nodes.
                        continue
                    if loader.get_op_type(_node) == requested_types_seq[_node_level]:
                        inputs_actual = loader.get_input_op_types(_node)
                        outputs_actual = loader.get_output_op_types(_node)
                        inputs_expected, outputs_expected = sequence[_node_level][1:]

                        # providing inputs_expected and outputs_expected is not required from user
                        # since user might just care to match a sequence of node types for any given inputs/outputs
                        if (
                            len(inputs_expected)
                            and not FrameworkPatternMatcher.validate_buffers(
                                inputs_expected, inputs_actual, supported_op_list
                            )
                        ) or (
                            len(outputs_expected)
                            and not FrameworkPatternMatcher.validate_buffers(
                                outputs_expected, outputs_actual, supported_op_list
                            )
                        ):
                            continue

                        if _node_level + 1 == len(sequence):
                            # This means we reached at the end of the pattern
                            # Pattern fully matched.
                            # _path will represent all the matched nodes.
                            matched_node_tuples.append(_path)

                        _children_nodes = loader.get_children_nodes(_node)
                        stack.extend(
                            [
                                [_c_node, _node_level + 1, [*_path, _c_node]]
                                for _c_node in _children_nodes
                            ]
                        )

                for matched_node_tuple in matched_node_tuples:
                    if validator is None or validator(matched_node_tuple):
                        if len(matched_node_tuple) != len(sequence):
                            log_debug1(
                                "Matched node list length must be same as "
                                f"requested sequence. Expected {len(sequence)}, "
                                f"Got {len(matched_node_tuple)}"
                            )
                            continue
                        matched_nodes.append(matched_node_tuple)
                        matched_node_names = [n.name for n in matched_node_tuple]
                        log_debug1(f"Found match: {matched_node_names}")

        log_debug1(f"Found {len(matched_nodes)} match(es)")
        return matched_nodes
