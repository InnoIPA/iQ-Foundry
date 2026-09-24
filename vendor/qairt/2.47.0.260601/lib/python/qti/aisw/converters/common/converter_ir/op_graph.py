# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import inspect
import json
import yaml
import numpy as np
from enum import IntEnum
import copy
import random
from collections.abc import MutableSet, Sequence
from collections import OrderedDict

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders, AxisTracker
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import code_to_message, translation_utils
from qti.aisw.converters.common.converter_ir.op_adapter import type_to_code


class OpNode(object):
    def __init__(self, op, input_names, output_names, axis_order=None):
        self.op = op
        self.input_names = input_names
        self.output_names = output_names
        self.axis_order = axis_order

    def __repr__(self):
        return str(self.op)

    def is_equal(self, other_node):
        if not isinstance(other_node, self.__class__):
            return False
        node_vars = dict(self.__dict__)
        other_node_vars = dict(other_node.__dict__)
        for var in list(node_vars.keys()):
            if node_vars[var] != other_node_vars[var]:
                return False, "Node for {} does not match current node for attr {}. \nExpected node: {}. \nGot {}". \
                    format(other_node.op.name, var, node_vars[var], other_node_vars[var])
        return True, "Node for {} matches current Node instance.".format(other_node.op.name)

    def encode(self, graph):
        data = {"name": self.op.name, "type": self.op.type}
        input_dict = []
        for name in self.input_names:
            input_dict.append({name:  graph.get_buffer(name).encode(graph)})
        data["inputs"] = input_dict

        output_dict = {}
        for name in self.output_names:
            output_dict[name] = graph.get_buffer(name).encode(graph)
        data["outputs"] = output_dict

        attrs_dict = self.op.encode()
        data["attrs"] = attrs_dict

        return data


class OrderedSet(MutableSet, Sequence):
    def __init__(self, iterable=None):
        self._data = {}
        if iterable is not None:
            for item in iterable:
                self.add(item)

    def add(self, item):
        """Add an element to the end of the OrderedSet if it doesn't exist."""
        self._data[item] = None

    def discard(self, item):
        """Remove an element from the OrderedSet if it exists."""
        self._data.pop(item, None)

    def remove(self, item):
        """Remove an element from the OrderedSet. Raises KeyError if element is not contained in the OrderedSet."""
        self._data.pop(item)

    def clear(self):
        """Remove all elements from the OrderedSet."""
        self._data.clear()

    def update(self, iterable):
        """Update the OrderedSet, adding elements from all others."""
        for item in iterable:
            self.add(item)

    def copy(self):
        """Return a shallow copy of the OrderedSet."""
        return OrderedSet(self._data)

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        if not self._data:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}({list(self._data)})"

    def __eq__(self, other):
        if isinstance(other, (set, frozenset)):
            return set(self._data) == other
        if isinstance(other, OrderedSet):
            return list(self._data) == list(other._data)
        return NotImplemented

    def __getitem__(self, index):
        if isinstance(index, slice):
            return OrderedSet(list(self._data.keys())[index])
        return list(self._data.keys())[index]

    def __reversed__(self):
        return reversed(list(self._data.keys()))

    def union(self, other):
        result = OrderedSet(self)
        result.update(other)
        return result

    def intersection(self, other):
        other_set = set(other)
        return OrderedSet(item for item in self if item in other_set)

    def difference(self, other):
        other_set = set(other)
        return OrderedSet(item for item in self if item not in other_set)

    def symmetric_difference(self, other):
        other_set = set(other)
        result = OrderedSet(item for item in self if item not in other_set)
        result.update(item for item in other if item not in self._data)
        return result

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)


class Buffer(object):
    def __init__(
            self,
            name,
            shape,
            dtype,
            producer,
            axis_format=None,
            type=None,
            sparse_params=None,
            perm_to_src=None,
    ):
        self.name = name
        self.producer = producer
        self.consumers = OrderedSet()
        self.shape = shape
        self.dtype = dtype
        self.axis_format = AxisTracker.AxisFormat.NOT_YET_DEFINED if axis_format is None else axis_format
        self.type = BufferType.REGULAR if type is None else type
        self.src_axis_format = AxisTracker.AxisFormat.NOT_YET_DEFINED
        self.sparse_params = ir_graph.IrSparseParamsInfo() if sparse_params is None else sparse_params

        # save permute order from layout-transform
        self._perm_to_src = None
        self.perm_to_src = perm_to_src

    def __repr__(self):
        return self.name

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, vals):
        if isinstance(vals, op_adapter.BufferShape):
            self._shape = vals
        else:
            self._shape = op_adapter.BufferShape(list(vals))

    def rank(self):
        return len(self.shape)

    def get_buf_dims(self):
        return self.shape

    def set_buf_dims(self, shape):
        self.shape = shape

    def get_axis_format(self):
        return self.axis_format

    def set_axis_format(self, axis_format):
        self.axis_format = axis_format

    def get_src_axis_format(self):
        return self.src_axis_format

    def set_src_axis_format(self, src_axis_format):
        self.src_axis_format = src_axis_format

    def get_sparse_params(self):
        return self.sparse_params

    def set_sparse_params(self, sparse_params):
        self.sparse_params = sparse_params

    @property
    def perm_to_src(self):
        """return permute sequence to source format."""
        return self._perm_to_src

    @perm_to_src.setter
    def perm_to_src(self, perm_seq):
        """set permute sequence to source format."""
        log_assert(
            isinstance(perm_seq, (type(None), list)),
            "Expecting None or a list for perm seq."
        )
        if isinstance(perm_seq, list):
            log_assert(len(perm_seq) == self.rank(),"Perm_seq and shape have different rank.")

        self._perm_to_src = perm_seq

    def get_consumers(self):
        return self.consumers

    def populate_axis_format(self, axis_order, encodings, time_series_format=False):
        # get_axis_format returns format with implicit assumption for image input when buf is 4D
        # and format for time-series style network when buf is 3D. Depending on InputEncoding provided by user
        # we need to account for non-trivial case
        input_in_encodings = [input_encoding[0] for input_encoding in encodings]
        if InputEncodings.OTHER in input_in_encodings and len(self.get_buf_dims()) in [3, 4]:
            self.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            # Override time_series_format based on encoding
            if InputEncodings.TIME_SERIES in input_in_encodings and len(self.get_buf_dims()) == 3:
                time_series_format = True
            self.axis_format = axis_order.get_axis_format(self.rank(),
                                                          time_series_format=time_series_format)

    def get_axis_annotations(self):
        """Translate AxisFormat enum to modeltools axis order list"""
        if self.axis_format == 'ANY':
            return [AxisTracker.AxisAnnotations.ANY for _ in range(len(self.shape))]
        else:
            return AxisTracker.get_axis_annotation_from_format(self.axis_format)

    def is_null(self):
        return self.type == BufferType.NULL

    def is_equal(self, other_buffer):
        """
        Compares another buf instance to current one based attribute matching
        :param other_buffer: a Buffer object
        :return: bool, msg. True if type and attr/params match, False otherwise. Plus message detailing what was
                            different
        """
        # attr/param list equality check
        other_buf_params = dict(other_buffer.__dict__)
        current_buf_params = dict(self.__dict__)
        for attr_ in list(current_buf_params.keys()):
            if attr_ == "producer":  # node comparison
                is_attr_eq = current_buf_params[attr_].is_equal(other_buf_params[attr_])
            elif attr_ == "consumers":
                cur_consumers = current_buf_params[attr_]
                other_consumers = other_buf_params[attr_]
                if len(cur_consumers) != len(other_consumers):
                    return False, "Attribute match error for Buffer:{}. Consumers length mismatch. Expected {} Got {}." \
                        .format(str(other_buffer.name), len(cur_consumers), len(other_consumers))
                is_attr_eq = True
                for cur_cons, other_cons in zip(cur_consumers, other_consumers):
                    is_attr_eq = cur_cons.is_equal(other_cons)
                    if not is_attr_eq:
                        break
            elif attr_ == "sparse_params":
                cur_params = current_buf_params[attr_]
                other_params = other_buf_params[attr_]
                if cur_params.layout == other_params.layout:
                    if cur_params.cooInfo.numSpecifiedElements == other_params.cooInfo.numSpecifiedElements:
                        if cur_params.cooInfo.numSparseDimensions == other_params.cooInfo.numSparseDimensions:
                            is_attr_eq = True
                        else:
                            return False, ("Attribute match error for Buffer:{} Num Sparse Dimensions mismatch. "
                                           "Expected {}, Got {} ").format(
                                str(other_buffer.name), str(other_params.cooInfo.numSparseDimensions),
                                str(cur_params.cooInfo.numSparseDimensions))
                    else:
                        return False, ("Attribute match error for Buffer:{} Num Specified Elements mismatch. "
                                       "Expected {}, Got {} ").format(
                            str(other_buffer.name), str(other_params.cooInfo.numSpecifiedElements),
                            str(cur_params.cooInfo.numSpecifiedElements))
                else:
                    return False, ("Attribute match error for Buffer:{} Sparse layout mismatch. "
                                   "Expected {}, Got {} ").format(
                        str(other_buffer.name), str(other_params.layout),
                        str(cur_params.layout))

            elif attr_ in ["shape", "dtype"]:
                is_attr_eq = (other_buf_params[attr_] == current_buf_params[attr_])
            else:
                is_attr_eq = translation_utils.compare_values(other_buf_params[attr_], current_buf_params[attr_])
            if not is_attr_eq:
                return False, "Attribute match error for Buffer:{} Attribute: {}. Expected {}, Got {} ".format(
                    str(other_buffer.name), attr_, str(current_buf_params[attr_]), str(other_buf_params[attr_]))

        return True, "Buffer {} is equal to current Buffer instance".format(other_buffer)

    def encode(self, graph):
        buf_dict = {}
        buf_dict["layout"] = self.axis_format
        buf_dict["shape"] = self.shape.to_dict()
        buf_dict["dtype"] = to_str(self.dtype)
        return buf_dict

class BufferCriteria(object):
    """
    Class(enum) to use for setting buffer criteria on inputs/outputs for validating matched node sequences
    """
    # to be used for individual buffers
    ALL = "ALL"  # all the buffer(s) must be this same expected op_type
    ANY = "ANY"  # There can be one or more of this op_type as buffer(s)
    NONE = "NONE"  # None of the buffer(s) should be of this type

    # to be used for set of buffers
    MATCH_NUM_BUFS = "MATCH_NUM_BUFS"  # the expected number of buffers must be same length as matched buffers
    FLEXIBLE_NUM_BUFS = "FLEXIBLE_NUM_BUFS"  # the expected number of buffers doesnt need to be equal to matched buffers
    MATCH_BUFS_AT_INDEX = "MATCH_BUFS_AT_INDEX" # matches the buffers at the specified index


class BufferType(object):
    """
    Class(enum) to use for setting buffer type and corresponding to the Qnn_TensorType_t in IrTensor for the C++ alignment work
    """
    REGULAR = "REGULAR"  # stands for all the IrTensor TensorType other than QNN_TENSOR_TYPE_NULL
    NULL = "NULL"  # stands for the QNN_TENSOR_TYPE_NULL in IrTensor TensorType


class InputType(object):
    """
    Contains supported input types. This will be used by DSP to determine quantization
    """
    IMAGE = "image"  # input is float between 0-255 and the input's mean is 0.0f and the input's max is 255.0f
    DEFAULT = "default"  # pass the input as floats to the dsp directly and the DSP will quantize it
    OPAQUE = "opaque"  # assumes input is float because the consumer layer(i.e next layer) requires it as float,
    # therefore it won't be quantized by DSP

    @classmethod
    def get_supported_types(cls):
        return [cls.IMAGE, cls.DEFAULT, cls.OPAQUE]

    @classmethod
    def is_valid_type(cls, input_type):
        return input_type in cls.get_supported_types()


class TraceType(IntEnum):
    """
    Enumeration class used for setting framework trace type. This will be used to trace op/tensor type and pass to backend.
    """
    TENSOR = 0  # means this is a tensor type
    OP = 1      # means this is a node(op) type


class QuantUpdatableMode(Enum):
    """
    Enumeration class for quant updatable modes.
    """
    NONE = "none"
    ADAPTER_ONLY = "adapter_only"
    ALL = "all"
    FLOAT_ONLY = "float_only"


class InputLayout(object):
    """
    Contains supported input layouts. This will be used to validate the input argument
    """
    NCDHW = "NCDHW" # NCDHW
    NDHWC = "NDHWC" # NDHWC
    NCHW = "NCHW"   # NCHW
    NHWC = "NHWC"   # NHWC
    HWIO = "HWIO"   # HWIO
    OIHW = "OIHW"   # OIHW
    NFC = "NFC"     # NFC
    NCF = "NCF"     # NCF
    NTF = "NTF"     # NTF
    TNF = "TNF"     # TNF
    NF = "NF"       # NF
    NC = "NC"       # NC
    FEATURE = "F"   # Feature
    NONTRIVIAL = 'NONTRIVIAL'

    @classmethod
    def get_supported_types(cls):
        return [cls.NCDHW, cls.NDHWC, cls.NCHW, cls.NHWC, cls.HWIO, cls.OIHW, cls.NFC, cls.NCF, cls.NTF, cls.TNF, cls.NF, cls.NC, cls.FEATURE, cls.NONTRIVIAL]

    @classmethod
    def get_supported_types_v2(cls):
        return [cls.NCDHW, cls.NDHWC, cls.NCHW, cls.NHWC, cls.HWIO, cls.OIHW, cls.NFC, cls.NCF, cls.NTF, cls.TNF, cls.NF, cls.NC, cls.FEATURE]

    @classmethod
    def get_axis_format(cls, layout):
        layout_to_axis_format_dict = {cls.NDHWC: AxisTracker.AxisFormat.NDHWC,
                                      cls.NCDHW: AxisTracker.AxisFormat.NCDHW,
                                      cls.NHWC: AxisTracker.AxisFormat.NSC,
                                      cls.NCHW: AxisTracker.AxisFormat.NCS,
                                      cls.HWIO: AxisTracker.AxisFormat.HWIO,
                                      cls.OIHW: AxisTracker.AxisFormat.OIHW,
                                      cls.NFC: AxisTracker.AxisFormat.NFC,
                                      cls.NCF: AxisTracker.AxisFormat.NCF,
                                      cls.NTF: AxisTracker.AxisFormat.NTF,
                                      cls.TNF: AxisTracker.AxisFormat.TNF,
                                      cls.NF: AxisTracker.AxisFormat.NF,
                                      cls.NC: AxisTracker.AxisFormat.NC,
                                      cls.FEATURE: AxisTracker.AxisFormat.ANY,
                                      cls.NONTRIVIAL: AxisTracker.AxisFormat.NONTRIVIAL}
        return layout_to_axis_format_dict[layout]

    @classmethod
    def is_valid_type(cls, input_layout):
        return input_layout in cls.get_supported_types()

    # Currently v2 validation is not being used, for backward compatibility with 'NONTRIVIAL' layout
    # TODO: Use this validation once preserve IO related issues are resolved for layout sensitive Ops and backends
    @classmethod
    def is_valid_type_v2(cls, input_layout):
        return input_layout in cls.get_supported_types_v2()


class InputEncodings(object):
    """
    Contains supported input encodings
    """
    BGR = "bgr"
    RGB = "rgb"
    RGBA = "rgba"
    ARGB32 = "argb32"
    NV21 = "nv21"
    NV12 = "nv12"
    TIME_SERIES = "time_series"
    OTHER = "other"

    valid_transformations = [
        (ARGB32, RGB),
        (ARGB32, BGR),
        (RGBA, RGB),
        (RGBA, BGR),
        (NV21, RGB),
        (NV21, BGR),
        (NV12, RGB),
        (NV12, BGR),
    ]

    @classmethod
    def get_supported_encodings(cls):
        return [cls.BGR, cls.RGB, cls.RGBA, cls.ARGB32, cls.NV21, cls.NV12, cls.TIME_SERIES, cls.OTHER]

    @classmethod
    def get_supported_encodings_v2(cls):
        return [cls.BGR, cls.RGB, cls.RGBA, cls.ARGB32, cls.NV21, cls.NV12]

    @classmethod
    def is_valid_encoding(cls, input_encoding):
        return input_encoding in cls.get_supported_encodings()

    @classmethod
    def is_valid_transformation(cls, input_encoding, out_encoding):
        return (input_encoding, out_encoding) in cls.valid_transformations


class QuantParams(object):
    """
    Contains supported quantization params
    """
    BN_PARAMS = "bn_params"
    OUTPUT_ENCODINGS = "output_encodings"
    PARAM_ENCODINGS = "param_encodings"

    @classmethod
    def get_supported_quant_params(cls):
        return [cls.BN_PARAMS, cls.OUTPUT_ENCODINGS, cls.PARAM_ENCODINGS]

    @classmethod
    def is_valid_quant_param(cls, param):
        return param in cls.get_supported_quant_params()


class QuantUpdatableType(IntEnum):
    """
    Enumeration class for specifying the updatable status of tensors.
    Constants:
    - NOT_SET: The updatable status has not been set. This is used primarily for validation purposes.
    - UNDETERMINED: The updatable status is not certain. This could be the case for tensors that
      are not explicitly marked as updatable but might need to be updated for specific reasons.
    - UPDATABLE: The tensor is confirmed to be updatable
    - NON_UPDATABLE: The tensor is confirmed to be non-updatable
    """
    NOT_SET = 0
    UNDETERMINED = 1
    UPDATABLE = 2
    NON_UPDATABLE = 3

class IROpGraph(object):
    def __init__(self,
                 naming_policy,
                 shape_inference_policy,
                 input_types,
                 input_dtypes,
                 input_encodings,
                 src_axis_order,
                 input_layouts=[],
                 source_output_layouts=[],
                 desired_io_layouts=[],
                 quantization_overrides=None,
                 user_custom_io={},
                 preserve_io=[],
                 keep_quant_nodes=False,
                 output_nodes=[],
                 keep_int64_inputs=False,
                 enable_framework_trace=False,
                 qairt_converter=False,
                 remove_unused_inputs=False,
                 use_quantize_v2=False,
                 lora_tensor_names=[]):
        # Policies
        self.naming_policy = naming_policy
        self.shape_inference_policy = shape_inference_policy

        # custom_io_layouts, similar to the input_layouts list, contains the buffer_name and corresponding layout provide by the user
        # in the YAML file. (similar to the --input_layout option)
        self.user_custom_io = user_custom_io
        # Initialize custom_io_layouts with source_output_layouts
        # Custom IO Input layouts will be populated later during Parsing of Custom IO
        self.custom_io_layouts = source_output_layouts
        self.custom_datatype_tensors = {}
        self.dump_qairt_io_config_yaml = ""
        self.dump_yaml_file_data = ""
        self.dump_yaml_file_name = ""
        self.qairt_converter = qairt_converter
        self.remove_unused_inputs = remove_unused_inputs
        self.use_quantize_v2 = use_quantize_v2
        if user_custom_io:
            self.parse_custom_io_config(user_custom_io, input_layouts + source_output_layouts, desired_io_layouts)

        # Input and SRC graph details
        self.src_axis_order = src_axis_order
        self.inputs_type_dict = self._create_input_types_dict(input_types)
        self.input_dtypes_dict = self._create_input_dtypes_dict(input_dtypes)
        self.inputs_encoding_dict = self._create_input_encodings_dict(input_encodings)
        self.input_axis_formats = self._create_input_layouts_dict(input_layouts, qairt_converter)
        self.custom_io_axis_formats = self._create_input_layouts_dict(self.custom_io_layouts, qairt_converter)
        # desired_io_layouts is already being handled in user_custom_io for unification of IO Config & Commandline behavior
        # This call here is just for validating layouts provided in desired_io_layouts
        _ = self._create_input_layouts_dict(desired_io_layouts, qairt_converter)

        # Flag to enable framework trace functionality
        self.enable_trace = True if enable_framework_trace else False
        # dict to store op tracing information from current node(op)/buffer(tensor) to its source op/tensor
        # It should be {TraceType.OP:     {'op_name':     [(source_name, source_type),
        #                                                  ...],
        #                                  ...},
        #               TraceType.TENSOR: {'tensor_name': [(source_name, source_type),
        #                                                  ...],
        #                                  ...}
        self.trace_dict = {TraceType.OP: dict(),
                           TraceType.TENSOR: dict()}
        # Used to store source topology when enable_trace is True
        # This is meant to save the topology of the source framework after graph transformations are done (if any)
        self.source_topology = None
        # source framework graph op/tensor names for framework tracing validation
        self.fw_op_names = []
        self.fw_tensor_names = []

        # A set containining all the IO tensors for which the layout has to be preserved
        self.preserve_layout_tensors = set()
        self.preserve_io = preserve_io

        # preserve_io_layout_passed is used to track the various usages of the preserve_io option
        # preserve_io_layout_passed = 0 indicates that the preserve_io option is not passed
        # preserve_io_layout_passed = 1 indicates that the user has explicitly mentioned the tensors to preserve layout for
        #                        using the 'layout' field. Usage example: "--preserve_io layout input1"
        # preserve_io_layout_passed = 2 indicates that the user wants to preserve the layout for all the inputs and outputs.
        #                        Usage example: "--preserve_io layout" or "--preserve_io"
        self.preserve_io_layout_passed = 0

        # A dict containining all the IO tensors for which the datatype has to be preserved
        self.preserve_datatype_tensors = {}

        # preserve_io_datatype_passed is used to track the various usages of the preserve_io option
        # preserve_io_datatype_passed = 0 indicates that the preserve_io option is not passed
        # preserve_io_datatype_passed = 1 indicates that the user has explicitly mentioned the tensors to preserve datatype for
        #                        using the 'datatype' field. Usage example: "--preserve_io datatype input1"
        # preserve_io_datatype_passed = 2 indicates that the user wants to preserve the datatype for all the inputs and outputs.
        #                        Usage example: "--preserve_io datatype" or "--preserve_io"
        self.preserve_io_datatype_passed = 0

        if len(preserve_io) > 0: # condition to check if preserve_io option is used or not
            self.preserve_io_datatype_passed, self.preserve_io_layout_passed = self._validate_preserve_io(preserve_io)

        # Internal Data Structures
        self.nodes_by_name = {}
        self.nodes_in_order = []
        self.buffers = {}
        # One null buffer with empty name linked to all the null inputs
        self.null_buffer = Buffer(name='', shape=[], dtype=ir_graph.QNN_DATATYPE_FLOAT_32,
                                  producer=None, axis_format=AxisTracker.AxisFormat.NULL,
                                  type=BufferType.NULL)
        self.output_names = output_nodes
        self.quantization_params = {}
        self.src_graph_op_info = {}

        # Performance tracking
        self.total_macs = 0
        self.total_params_count = 0

        # Quantization variables
        self.keep_quant_nodes = keep_quant_nodes
        self.user_quantization_overrides = {
            'activation_encodings': {},
            'param_encodings': {}
        }
        if quantization_overrides:
            log_info('Processing user provided quantization encodings: ', quantization_overrides)
            f = open(quantization_overrides)
            self.user_quantization_overrides = json.load(f)

        # Prevent user from invoking quantizer v2 (use_quantize_v2) with older (v0.6.0 and v1.0.0) encodings.
        if quantization_overrides and use_quantize_v2:
            if ((not ('version' in self.user_quantization_overrides)) or ('version' in self.user_quantization_overrides and self.user_quantization_overrides['version'] != '2.0.0')):
                raise ValueError("--use_quantize_v2 must be used with quantization overrides json schema v2.0.0.")

        # For the new version of encoding file (version 1.0.0), we pre-process the encoding file and convert it to
        # old version (0.0.6) encoding format
        if 'version' in self.user_quantization_overrides and self.user_quantization_overrides['version'] == '1.0.0':
            self.user_quantization_overrides['activation_encodings'] = IROpGraph.process_encoding_dict(self.user_quantization_overrides['activation_encodings'])
            self.user_quantization_overrides['param_encodings'] = IROpGraph.process_encoding_dict(self.user_quantization_overrides['param_encodings'])
        elif 'version' in self.user_quantization_overrides and self.user_quantization_overrides['version'] == '2.0.0':
            self.user_quantization_overrides['encodings'] = IROpGraph.process_encoding_dict(self.user_quantization_overrides['encodings'])

        # Add updatable attribute to encodings
        # updatable_without_enc_names is a list of tensor names whose corresponding tensors do not have encodings but are still updatable,
        # assuming these tensor names are not changed during op optimization, this assumption will be validated in get_all_updatable_tensors()
        self.updatable_without_enc_names = IROpGraph.embed_updatable_attributes(self.user_quantization_overrides, lora_tensor_names)

        # Bitmask for boolean properties to be transferred to IrGraph and serialized
        self.graph_property_bitmask = 0

        # Misc flags
        self.keep_int64_inputs = keep_int64_inputs
        self.int64_input_cast_map = {}

        self.input_dynamic_axes = {}

        # Track unique op types inserted in the graph
        self.op_types = set()

        # Track external data path which is dumped to tempdir to be deleted after serialization
        self.external_data_path = None

        # Lora tensors
        self.lora_tensor_names = lora_tensor_names

    @staticmethod
    def process_encoding_dict(encodings_list):
        encodings_dict = dict()
        for encoding in encodings_list:
            if 'name' not in encoding:
                raise AttributeError('Name must be present in every encoding')
            tensor_name = encoding['name']
            encoding.pop('name')
            encodings_dict[tensor_name] = [encoding]
        return encodings_dict

    @staticmethod
    def embed_updatable_attributes(user_quantization_overrides,
                                   updatable_tensors):
        if updatable_tensors is None:
            updatable_tensors_set = set()
        else:
            updatable_tensors_set = set(updatable_tensors)

        processed_tensors = set()
        for enc_field in ["activation_encodings", "param_encodings", "encodings"]:
            if enc_field not in user_quantization_overrides:
                continue
            for tensor_name, enc in user_quantization_overrides[enc_field].items():
                if isinstance(enc, list):
                    for enc_element in enc: # for onnx/tf model, enc is always a list, regardless of the encoding type
                        if tensor_name in updatable_tensors_set:
                            enc_element["updatable"] = QuantUpdatableType.UPDATABLE
                            processed_tensors.add(tensor_name)
                        else:
                            # May be updatable or non-updatable afeter optimization
                            enc_element["updatable"] = QuantUpdatableType.UNDETERMINED
                else:
                    # for other situation, (like torch model, enc is a dict)
                    # we don't embed the updatable attributes into IROpGraph,
                    # so that the updatable attributes will not be tracked in the op optimizations
                    # Note: This combination (torch model with an updatable list) is not yet systematically supported.
                    pass

        # some tensors has not encodings, but also updatable
        # for example indices for lorav3 (int), in this case we store them in the graph
        updatable_without_enc_names = [x for x in updatable_tensors if x not in processed_tensors]

        return updatable_without_enc_names

    def get_all_updatable_tensors(self, report_error=False):
        """
        Retrieve all updatable tensors in the graph and perform validation.

        This method iterates through the quantization parameters to identify tensors that are marked as updatable.
        It also checks for tensors that are specified as updatable but do not have associated encodings.

        Note: Tracking of updatable tensors is currently not supported for Q-Dq models, as their encodings are added
        during the translations of QuantizeLinear and DequantizeLinear op. Additional handling, such as removing
        the tensor name from updatable_without_enc_names, should be required in those translations steps.

        :param report_error: If True, log an error message for tensors that are updatable but lack encodings.
        :return: A list of names of tensors that are updatable.
        """
        updatable_tensor_list = []
        log_func = log_error if report_error else log_warning
        has_error = False

        # if tensor has encodings, then it should have updatable attribute
        for node in self.list_nodes():
            if node.op.name not in self.quantization_params:
                continue
            for i, enc in enumerate(self.quantization_params[node.op.name][QuantParams.OUTPUT_ENCODINGS]):
                if "updatable" not in enc or enc["updatable"] == QuantUpdatableType.NOT_SET:
                    log_func(f"updatable attribute of Tensor {enc['name']} is not tracked")
                    has_error = True
                    continue
                if enc["updatable"] == QuantUpdatableType.UPDATABLE:
                    updatable_tensor_list.append(enc['name'])

        # for the updatable tensors that without encodings, they should be still in the graph
        tensors_in_graph = set(x.name for x in self.list_buffers())
        for tensor_name in self.updatable_without_enc_names:
            if tensor_name not in tensors_in_graph:
                log_func(f"tensor {tensor_name} are set to updatable but missing in the graph:\n")
                has_error = True
                continue
            updatable_tensor_list.append(tensor_name)

        if has_error and report_error:
            raise Exception("updatable attribute of some tensors are not correctly tracked, see the log above")
        return updatable_tensor_list

    def __iter__(self):
        return iter(self.nodes_in_order)

    def dump_json(self, filename):
        graph = self

        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, OpNode):
                    return obj.encode(graph)
                elif isinstance(obj, np.int32):
                    return int(obj)
                elif isinstance(obj, np.int64):
                    return int(obj)
                else:
                    try:
                        import tvm
                        if isinstance(obj, tvm.tir.expr.IntImm):
                            return int(obj)
                    except ImportError:
                        pass
                    # Let the base class default method raise the TypeError
                    return json.JSONEncoder.default(self, obj)

        if filename[-5:] != ".json":
            filename = filename + ".json"
        with open(filename, "w") as f:
            # Create a dict order by nodes_in_order instead of node_by_name to debug insert order issues
            dict_in_order = {node.op.name: node for node in self.nodes_in_order}
            json.dump(dict_in_order, f, cls=Encoder,
                      indent=4, separators=(',', ': '))

    def dump_qairt_converter_io_config_yaml(self, args):
        """
        Constructs QAIRT equivalent Commandline Arguments and IO Config File (which can be provided to
        the QAIRT Converter) based on the currently provided Commandline Arguments.
        :param args: Argparse Namespace object
        """
        yaml_data = []
        qairt_quantizer_command_list = ["input_dlc", "output_dlc", "input_list", "float_fallback", "param_quantizer",
                                        "act_quantizer", "algorithms", "bias_bitwidth", "bias_bw", "act_bitwidth", "act_bw",
                                        "weights_bitwidth", "weight_bw", "ignore_encodings", "use_per_channel_quantization",
                                        "use_per_row_quantization", "use_native_input_files", "use_native_output_files",
                                        "restrict_quantization_steps", "use_dynamic_16_bit_weights", "pack_4_bit_weights",
                                        "act_quantizer_calibration", "param_quantizer_calibration", "act_quantizer_schema",
                                        "param_quantizer_schema", "percentile_calibration_value", "use_aimet_quantizer", "op_package_lib"]

        cmdline_not_required_args = ["input_dtype", "input_layout", "input_dim", "input_encoding", "input_type",
                                     "dump_qairt_io_config_yaml", "input_network", "custom_io", "preserve_io",
                                     "dump_qairt_quantizer_command", "out_names", "disable_relu_squashing"]

        modified_args_dict = {"no_simplification": "onnx_no_simplification",
                              "batch": "onnx_batch",
                              "define_symbol": "onnx_define_symbol",
                              "no_optimization": "tf_no_optimization",
                              "show_unconsumed_nodes": "tf_show_unconsumed_nodes",
                              "saved_model_tag": "tf_saved_model_tag",
                              "saved_model_signature_key": "tf_saved_model_signature_key",
                              "validate_models": "tf_validate_models",
                              "signature_name": "tflite_signature_name"}

        input_args = sys.argv

        file_name = ""
        if hasattr(args, 'input_network'):
            file_name = args.input_network
        yaml_file_name = args.dump_qairt_io_config_yaml
        self.dump_qairt_io_config_yaml = args.dump_qairt_io_config_yaml
        input_dtype = []
        if hasattr(args, 'input_dtype'):
            input_dtype = args.input_dtype
        input_layout = []
        if hasattr(args, 'input_layout'):
            input_layout = args.input_layout
        input_encoding = []
        if hasattr(args, 'input_encoding'):
            input_encoding = args.input_encoding
        input_dim = []
        if hasattr(args, 'input_dim'):
            input_dim = args.input_dim
        qairt_converter_command = f"qairt-converter --input_network {file_name}"

        for argument,value in vars(args).items():
            if (str(argument) in qairt_quantizer_command_list) or (str(argument) in cmdline_not_required_args):
                continue
            elif str(argument) == "output_path" and value is not None:
                qairt_converter_command += f" -o {value}"
            elif str(argument) in modified_args_dict:
                if f"--{argument}" in input_args:
                    if str(argument) == "define_symbol":
                        for arg in value:
                            qairt_converter_command += f" --onnx_define_symbol {arg[0]} {arg[1]}"
                    else:
                        qairt_converter_command += f" --{modified_args_dict[str(argument)]} {value}"
            elif type(value) == bool and value:
                if f"--{argument}" in input_args:
                    qairt_converter_command += f" --{argument}"
            elif type(value) == int and value > 0:
                if f"--{argument}" in input_args:
                    qairt_converter_command += f" --{argument} {value}"
            elif type(value) == list and len(value) > 0:
                for arg in value:
                    if len(arg) == 1:
                        qairt_converter_command += f" --{argument} {arg}"
                    else:
                        qairt_converter_command += f" --{argument}"
                        for multi_arg in arg:
                            qairt_converter_command += f' "{multi_arg}"'
            elif type(value) == str and len(value) > 0:
                if f"--{argument}" in input_args:
                    qairt_converter_command += f" --{argument} {value}"

        qairt_converter_command += f" --io_config {yaml_file_name}"
        self.qairt_converter_command = qairt_converter_command
        qairt_converter_command = f"# QAIRT Converter Command Line:\n# " + qairt_converter_command
        yaml_data.append(qairt_converter_command)

        input_dtype_dict = dict(input_dtype)
        input_layout_dict = dict(input_layout)
        input_encoding_dict = {}
        for input_enc in input_encoding:
            input_encoding_dict[input_enc[0]] = input_enc[1]
        input_dim_dict = {}
        if input_dim is not None:
            input_dim_dict = dict(input_dim)

        custom_io_dtype_dict = {}
        custom_io_qparams = {}
        if self.user_custom_io:
            for entry in self.user_custom_io:
                buffer_name = str(entry['IOName'])
                if 'Datatype' in entry:
                    custom_io_dtype_dict[buffer_name] = entry['Datatype']
                if 'QuantParam' in entry:
                    custom_io_qparams[buffer_name] = [entry['QuantParam']['Scale'],entry['QuantParam']['Offset']]

        supported_input_layouts = ['NCDHW', 'NDHWC', 'NCHW', 'NHWC', 'OIHW', 'HWIO', 'NFC', 'NCF', 'NTF', 'TNF', 'NF', 'NC', 'F']
        supported_dtype = ['float32', 'float16', 'int64', 'int32', 'int8', 'uint32', 'uint8', 'bool']

        comments = "\n# I/O configuration template for QAIRT model.\n" \
                   "# Each Input/Output field has two sub fields: Src Model Parameters and Desired Model Parameters. \n" \
                   "# Src Model Parameters: Specify the fields of the buffer in the original model. Default value is obtained from the model. \n" \
                   "# Desired Model Parameters: Specify the desired fields of the buffer. Needs to be filled by the user. \n" \
                   "# Src Model Parameters field has two sub fields: DataType and Layout. \n" \
                   "# Desired Model Parameters field has total of five sub fields: DataType, Layout and QuantParams (For both Inputs and Outputs). \n" \
                   "#                                                              Shape and Color Conversion (Only for Inputs) \n\n" \
                   "# DataType: Default values for input buffer are obtained from the model.\n" \
                   "#           Accepted values are: float32, float16, int64, int32, int8, uint32, uint8 and bool.\n\n" \
                   "# Layout: Layout value for input/output buffer. Supports valid QNN Layout. \n" \
                   "#         Accepted values are: NCDHW, NDHWC, NCHW, NHWC, OIHW, HWIO, NFC, NCF, NTF, TNF, NF, NC, F\n" \
                   "#         where, N = Batch, C = Channels, D = Depth, H = Height, W = Width, F = Feature, T = Time\n" \
                   "# Note: Cannot provide only either Source/Desired Layout. Must provide both incase atleast one of them is provided.\n\n" \
                   "# QuantParam field has two sub fields: Scale and Offset \n" \
                   "# Scale and Offset fields will be ignored for an I/O if the DataType field of I/O is not set to uint8 \n\n" \
                   "# Shape: Shape of the input buffers to the network specified in comma-seperated dimension format. \n" \
                   "#        For example: 1,224,224,3 \n\n" \
                   "# Color Conversion: Input encoding of the network inputs. Default is bgr.\n" \
                   "#                   Accepted values are: bgr, rgb, rgba, argb32, nv21, nv12.\n"

        yaml_data.append(comments)

        visited_buffers = set()
        out_tensor_names = []
        for node in self.get_output_nodes_of_graph():
            for buffer_name in node.output_names:
                # Don't dump duplicate output buffers
                if buffer_name in visited_buffers:
                    continue
                visited_buffers.add(buffer_name)
                out_tensor_names.append(buffer_name)

        out_tensor_data = f"Converted Graph:\n  - Output Tensors: {out_tensor_names}"
        yaml_data.append(out_tensor_data)

        i_str = "\nInput Tensor Configuration:\n"
        input_num = 0
        visited_buffers.clear()

        for node in self.get_input_nodes_to_graph():
            for buffer_name in node.output_names:
                # Don't dump duplicate input buffers
                if buffer_name in visited_buffers:
                    continue
                visited_buffers.add(buffer_name)
                input_num += 1
                i_str += "  # Input " + str(input_num) + "\n"
                i_str += f"  - Name: '{buffer_name}'\n"
                i_str += "    Src Model Parameters:\n"
                i_str += "        DataType: " + str(input_dtype_dict.get(buffer_name,"")) + "\n"
                src_layout = self.buffers[buffer_name].src_axis_format
                if src_layout not in supported_input_layouts:
                    # Don't dump Source Layout incase of Unsupported Layout
                    src_layout = ""
                i_str += "        Layout: " + str(src_layout) + "\n"
                i_str += "        Shape: " + input_dim_dict.get(buffer_name,"") + "\n"
                i_str += "    Desired Model Parameters:\n"
                desired_dtype = custom_io_dtype_dict.get(buffer_name,"")
                # Don't dump Desired Datatype incase datatype is not supported
                if desired_dtype not in supported_dtype:
                    desired_dtype = ""
                i_str += "        DataType: " + str(desired_dtype) + "\n"
                desired_layout = self.buffers[buffer_name].axis_format
                # Don't dump Desired Layout also, incase Source Layout is not supported
                if desired_layout not in supported_input_layouts or src_layout == "":
                    desired_layout = ""
                i_str += "        Layout: " + str(desired_layout) + "\n"
                i_str += "        Color Conversion: " + input_encoding_dict.get(buffer_name,"") + "\n"
                i_str += "        QuantParams:\n"
                scale = ""
                offset = ""
                if buffer_name in custom_io_qparams:
                    scale = str(custom_io_qparams[buffer_name][0])
                    offset = str(custom_io_qparams[buffer_name][1])
                i_str += "          Scale: " + scale + "\n"
                i_str += "          Offset: " + offset + "\n\n"
        yaml_data.append(i_str)

        o_str = "Output Tensor Configuration:\n"
        output_num = 0
        visited_buffers.clear()

        for node in self.get_output_nodes_of_graph():
            for buffer_name in node.output_names:
                # Don't dump duplicate output buffers
                if buffer_name in visited_buffers:
                    continue
                visited_buffers.add(buffer_name)
                output_num += 1
                o_str += "  # Output " + str(output_num) + "\n"
                o_str += f"  - Name: '{buffer_name}'\n"
                o_str += "    Src Model Parameters:\n"
                o_str += "        DataType: " + str(input_dtype_dict.get(buffer_name,"")) + "\n"
                src_layout = self.buffers[buffer_name].src_axis_format
                if src_layout not in supported_input_layouts:
                    # Keep both Source and Desired Layouts same incase of 'NOT_YET_DEFINED' layout
                    # Else don't dump Source Layout incase if Source/Desired Layout is Unsupported
                    if src_layout == 'NOT_YET_DEFINED':
                        src_layout = self.buffers[buffer_name].axis_format
                    if self.buffers[buffer_name].axis_format not in supported_input_layouts:
                        src_layout = ""
                o_str += "        Layout: " + str(src_layout) + "\n"
                o_str += "    Desired Model Parameters:\n"
                desired_dtype = custom_io_dtype_dict.get(buffer_name,"")
                # Don't dump Desired Datatype incase datatype is not supported
                if desired_dtype not in supported_dtype:
                    desired_dtype = ""
                o_str += "        DataType: " + str(desired_dtype) + "\n"
                desired_layout = self.buffers[buffer_name].axis_format
                # Don't dump Desired Layout also, incase Source/Desired Layout is not supported
                if desired_layout not in supported_input_layouts or src_layout == "":
                    desired_layout = ""
                o_str += "        Layout: " + str(desired_layout) + "\n"
                o_str += "        QuantParams:\n"
                scale = ""
                offset = ""
                if buffer_name in custom_io_qparams:
                    scale = str(custom_io_qparams[buffer_name][0])
                    offset = str(custom_io_qparams[buffer_name][1])
                o_str += "          Scale: " + scale + "\n"
                o_str += "          Offset: " + offset + "\n\n"

        # Don't dump Output Tensor Configuration if Output Buffers are empty
        if o_str != 'Output Tensor Configuration:\n':
            yaml_data.append(o_str)

        self.dump_yaml_file_data = yaml_data
        self.dump_yaml_file_name = yaml_file_name

    @staticmethod
    def _create_input_types_dict(input_types):
        log_assert(all(InputType.is_valid_type(type_) for _, type_ in input_types),
                   code_to_message.get_error_message("ERROR_UNSUPPORTED_INPUT_TYPE")(InputType.get_supported_types()))
        return {input_name: input_type for input_name, input_type in input_types}

    @staticmethod
    def _create_input_dtypes_dict(input_dtypes):
        return {input_name: np.dtype(input_dtype) for input_name, input_dtype in input_dtypes}

    @staticmethod
    def _create_input_encodings_dict(input_encodings):
        # Populate input encodings out with default encoding of BGR
        for i, input_encoding in enumerate(input_encodings):
            if len(input_encoding) == 2:
                input_encodings[i].append(InputEncodings.BGR)

        log_assert(all(InputEncodings.is_valid_encoding(encoding_in) and InputEncodings.is_valid_encoding(encoding_out)
                       for _, encoding_in, encoding_out in input_encodings),
                   code_to_message.get_error_message("ERROR_UNSUPPORTED_INPUT_ENCODING")
                   (input_encodings, InputEncodings.get_supported_encodings()))
        # Return dictionary of the form - input_name: (input_encoding_in, input_encoding_out)
        return {input_encoding[0]: tuple(input_encoding[1:]) for input_encoding in input_encodings}

    @staticmethod
    def _create_input_layouts_dict(input_layouts, qairt_converter):
        if not qairt_converter:
            log_assert(all(InputLayout.is_valid_type(layout_) for _, layout_ in input_layouts),
                       code_to_message.get_error_message("ERROR_UNSUPPORTED_INPUT_LAYOUT")
                       (input_layouts, InputLayout.get_supported_types()))
        else:
            log_assert(all(InputLayout.is_valid_type(layout_) for _, layout_ in input_layouts),
                       code_to_message.get_error_message("ERROR_UNSUPPORTED_INPUT_LAYOUT")
                       (input_layouts, InputLayout.get_supported_types_v2()))
            non_trivial_layout_buffer_names = []
            for name, layout in input_layouts:
                if layout == 'NONTRIVIAL':
                    non_trivial_layout_buffer_names.append(name)
            if non_trivial_layout_buffer_names:
                log_warning(f"NONTRIVIAL input layout is deprecated, provided for buffers: {non_trivial_layout_buffer_names}")
        return {input_name: InputLayout.get_axis_format(input_layout) for input_name, input_layout in input_layouts}

    def parse_custom_io_config(self, user_custom_io, source_io_layouts, desired_io_layouts):
        log_info('Validating user provided custom IO')
        buffers = set()
        source_io_layout_dict = dict(source_io_layouts)
        desired_io_layout_dict = dict(desired_io_layouts)
        for entry in user_custom_io:
            for field in entry.keys():
                if field not in ['IOName', 'Layout', 'Datatype', 'QuantParam', 'Optional']:
                    log_assert("Incorrect field %s provided in the custom IO YAML file. Valid fields are: 'IOName', 'Layout', 'Datatype', 'QuantParam', 'Optional'",
                               field)
            log_assert('IOName' in entry,"No IOName provided in custom IO YAML file. IOName is a mandatory field")
            buffer_name = entry['IOName']
            log_assert(buffer_name not in buffers,"Multiple entries provided for buffer {} in custom IO YAML file".format(buffer_name))
            if 'Layout' in entry:
                if desired_io_layout_dict and buffer_name in desired_io_layout_dict:
                    log_assert((buffer_name in source_io_layout_dict),
                               f"Providing Desired layout requires both Desired and Source Model Layout to be specified, "
                               f"provided for the buffer '{buffer_name}'")
                log_assert(('Custom' in entry['Layout'] and 'Model' in entry['Layout']),
                           "Both Custom layout and Model layout should be provided in the custom IO YAML file.")
                # The Model layout in the Custom IO YAML file is equivalent to the --input_layout command line option.
                self.custom_io_layouts.append([entry['IOName'], entry['Layout']['Model']])
            custom_datatype = None
            if 'Datatype' in entry:
                custom_datatype = entry['Datatype']
                log_assert((custom_datatype in ['float32','float16','uint8','uint16','uint32',
                                                'bool','int8','int16','int32','int64']),
                           "Custom datatype {} of buffer {} is not supported.".format(entry['Datatype'], buffer_name))
                if 'QuantParam' not in entry:
                    self.custom_datatype_tensors[str(entry['IOName'])] = str(entry['Datatype'])
            if 'QuantParam' in entry:
                quant_param = entry['QuantParam']
                for sub_field in quant_param.keys():
                    if sub_field not in ['Type', 'Scale', 'Offset']:
                        log_assert("Incorrect field %s provided in QuantParam of the custom IO YAML file for buffer %s. Valid fields are: 'Type', 'Scale', 'Offset'",
                                   field, buffer_name)
                log_assert('Type' in quant_param,"No Type provided for QuantParam in custom IO YAML file for buffer {}."
                           .format(buffer_name))
                log_assert(quant_param['Type'] == 'QNN_DEFINITION_DEFINED',"Type must be set to 'QNN_DEFINITION_DEFINED' if user wants to provide scale and offset.\
                    Invalid Type provided for buffer {}".format(buffer_name))
                log_assert('Scale' in quant_param and 'Offset' in quant_param,"Scale and/or Offset not provided for buffer {} in the custom IO YAML file."
                           .format(buffer_name))
                log_assert(isinstance(quant_param['Scale'], float) or isinstance(quant_param['Scale'], int),"Scale provided for buffer {} in the custom IO YAML file\
                    is of incorrect datatype. It should be a number.".format(buffer_name))
                log_assert(isinstance(quant_param['Offset'], int), "Offset provided for buffer {} in the custom IO YAML file\
                    is of incorrect datatype. It should be an integer.".format(buffer_name))
                if custom_datatype is not None:
                    log_assert(custom_datatype in ['uint8', 'int8','uint16','int16'], "Valid datatypes for quantized input/output are int8, uint8, int16 and uint16\
                                for custom IO. Invalid datatype {} provided for buffer {}.".format(custom_datatype, buffer_name))
            buffers.add(buffer_name)

    @staticmethod
    def _validate_preserve_io(preserve_io):
        preserve_io_datatype_passed, preserve_io_layout_passed = 0, 0
        for arg in preserve_io:
            if len(arg) == 0:
                # preserve both layout and datatypes for all IO tensors by just passing the option as follows:
                # --preserve_io
                preserve_io_datatype_passed = 2
                preserve_io_layout_passed = 2
            else:
                log_assert((arg[0] == 'layout' or arg[0] == 'datatype'),"Incorrect usage of the --preserve_io option.")
                if arg[0] == 'layout':
                    # preserve the layout for all/some the inputs and outputs of the graph by passing the option as follows:
                    # --preserve_io layout
                    # --preserve_io layout input1 input2 output1
                    if len(arg) > 1:
                        preserve_io_layout_passed = 1
                    else:
                        preserve_io_layout_passed = 2
                elif arg[0] == 'datatype':
                    # preserve the datatype for all/some the inputs and outputs of the graph by passing the option as follows:
                    # --preserve_io datatype
                    # --preserve_io datatype input1 input2 output1
                    if len(arg) > 1:
                        preserve_io_datatype_passed = 1
                    else:
                        preserve_io_datatype_passed = 2
        return preserve_io_datatype_passed, preserve_io_layout_passed

    def get_input_type(self, input_name):
        # use input_type: default as the default for all inputs
        return self.inputs_type_dict.get(input_name, InputType.DEFAULT)

    def get_input_dtype(self, input_name):
        dtype = self.input_dtypes_dict.get(input_name, np.dtype("float32"))

        # TODO: Remove this once we support int64 dtype
        if dtype == np.dtype('int64') and not self.is_int64_input_preserved(input_name):
            return np.dtype('int32')
        else:
            return dtype

    def has_input_dtype(self, input_name):
        return input_name in self.input_dtypes_dict

    def is_int64_input_preserved(self, input_name):
        if input_name in self.preserve_datatype_tensors and self.preserve_datatype_tensors[input_name] == 'int64':
            return True
        return self.keep_int64_inputs

    def get_input_types(self):
        return self.inputs_type_dict.values()

    def get_input_encoding(self, input_name):
        # use input_encoding: bgr as the default for all inputs
        # tensor name can follow (colon):num which is stripped to check for command-line name
        input_name_ = input_name
        if ":" in input_name and input_name not in self.inputs_type_dict:
            input_name_ = input_name[0: input_name.index(':')]
        return self.inputs_encoding_dict.get(input_name_, (InputEncodings.BGR, InputEncodings.BGR))

    def get_input_encodings(self):
        return self.inputs_encoding_dict.values()

    def is_time_series_input(self):
        return InputEncodings.TIME_SERIES in self.get_input_encodings()

    def add_src_op_info(self, op_name, inputs: list = None, outputs: list = None):
        """
        Adds a mapping of the original op's inputs and outputs to the original op name

        :param op_name: The name of the op whose inputs and outputs were added
        :param inputs: The inputs in the original source framework for the given op (params)
        :param outputs: The outputs in the original source framework for the given op
        """
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        self.src_graph_op_info[op_name] = {'inputs': inputs,
                                           'outputs': outputs}

    def add_quantization_params(self, op_name, **kwargs):
        """
        Adds quantization params to an IR graph object for a given op_name. The dictionary provided
        is expected to contain one/all of output_encodings, param_encodings or bn_params as a key(s).

        :param op_name: The name of the op whose quantization params will be added.
        :param kwargs: The dictionary containing the output encodings, param encodings and bn_params for that op
        :raises: An assertion error if the quantization params are not valid.
        """
        log_assert(all(QuantParams.is_valid_quant_param(param) for param, _ in kwargs.items()),
                   code_to_message.get_error_message("ERROR_UNSUPPORTED_QUANT_PARAM")
                   (QuantParams.get_supported_quant_params()))

        new_output_encs = translation_utils.to_list(kwargs.get(QuantParams.OUTPUT_ENCODINGS, []))
        new_param_encs = translation_utils.to_list(kwargs.get(QuantParams.PARAM_ENCODINGS, []))

        bn_params = {}
        output_encodings = []
        param_encodings = []
        if op_name in self.quantization_params:
            # update defaults to existing params if we are updating a layer
            bn_params = self.quantization_params[op_name][QuantParams.BN_PARAMS]
            output_encodings = self.quantization_params[op_name][QuantParams.OUTPUT_ENCODINGS]
            param_encodings = self.quantization_params[op_name][QuantParams.PARAM_ENCODINGS]


        # Replace exist encodings if found, or Extend existing lists(if exist) for encodings
        for o in new_output_encs:
            found_encoding = False
            for i,current_enc in enumerate(output_encodings):
                if o['name'] == current_enc['name']:
                    output_encodings[i] = o
                    found_encoding = True
            if not found_encoding:
                output_encodings.extend(translation_utils.to_list(o))

        for p in new_param_encs:
            found_encoding = False
            for i,current_enc in enumerate(param_encodings):
                if p['name'] == current_enc['name']:
                    param_encodings[i] = p
                    found_encoding = True
            if not found_encoding:
                param_encodings.extend(translation_utils.to_list(p))

        self.quantization_params.update({
            op_name: {
                QuantParams.BN_PARAMS: kwargs.get(QuantParams.BN_PARAMS, bn_params),
                QuantParams.OUTPUT_ENCODINGS: output_encodings,
                QuantParams.PARAM_ENCODINGS: param_encodings
            }
        })

    def has_quantization_param(self, layer_name):
        return layer_name in self.quantization_params

    def remove_quantization_params(self, layer_name):
        if self.has_quantization_param(layer_name):
            del self.quantization_params[layer_name]

    def get_layer_quantization_param(self, layer_name):
        log_assert(layer_name in self.quantization_params,
                   code_to_message.get_error_message("ERROR_LAYER_NOT_FOUND_IN_QUANT_PARAM")(layer_name))
        return self.quantization_params[layer_name]

    def replace_quantization_param(self, old_layer_name, new_layer_name):
        if self.has_quantization_param(old_layer_name) and old_layer_name != new_layer_name:
            quant_params = self.get_layer_quantization_param(old_layer_name)
            self.add_quantization_params(new_layer_name, **quant_params)
            self.remove_quantization_params(old_layer_name)

    def copy_quantization_param(self, src_op_name, dest_op_name, src_buffer_name, dest_buffer_name, encoding_type=QuantParams.OUTPUT_ENCODINGS):
        """
        Copy the output encodings from source op to destination op
        :param source_op_name: the layer/op name for the source Op
        :param destination_op_name: the layer/op name for the destination Op
        :param src_buffer_name: the output tensor name for the source op
        :param dest_buffer_name: the output tensor name for the destination op
        :param encoding_type: The type of encoding to update
        source op and destination op can keep different buffer names and just copy the encodings
        """
        if self.has_quantization_param(src_op_name):
            src_encodings = copy.deepcopy(self.quantization_params[src_op_name][encoding_type])
            if not isinstance(src_encodings, list):
                src_encodings = [src_encodings]
            override_encodings = []
            for i in range(len(src_encodings)):
                if (src_encodings[i]["name"] == src_buffer_name):
                    src_encodings[i]["name"] = dest_buffer_name
                    override_encodings.append(src_encodings[i])
                    break

            if len(override_encodings) > 0:
                dest_old_encodings = {}
                if self.has_quantization_param(dest_op_name):
                    dest_old_encodings = self.quantization_params[dest_op_name]
                else:
                    dest_old_encodings = {QuantParams.BN_PARAMS: {}, QuantParams.PARAM_ENCODINGS: [], QuantParams.OUTPUT_ENCODINGS: []}
                    self.quantization_params[dest_op_name] = dest_old_encodings

                target_encodings = dest_old_encodings[encoding_type]
                if isinstance(target_encodings, list):
                    for i, encodings in enumerate(target_encodings):
                        if dest_buffer_name == encodings["name"]:
                            del target_encodings[i]
                            break
                    target_encodings.extend(override_encodings)
                else:
                    target_encodings = override_encodings[0] if len(override_encodings) > 0 else {}

    def merge_quantization_params(self, source_op_name, destination_op_name, pre_merge_dest_tensor_name,
                                  post_merge_dest_tensor_name, encoding_type=QuantParams.OUTPUT_ENCODINGS,
                                  use_dest_encoding=False):
        """
        Merges the output encodings for source op to destination
        :param source_op_name: the layer/op name for the source Op
        :param destination_op_name: the layer/op name for the destination Op
        :param pre_merge_dest_tensor_name: the output tensor name for the destination op before merging source op
        :param post_merge_dest_tensor_name: the output tensor name for the destination op after merging source op
        :param encoding_type: The type of encoding to update
        :param use_dest_encoding: If set and source name is not in quant params, then use the dest encoding
        Change the output tensor name after mergeing
        """

        if source_op_name in self.quantization_params:
            # verify this is proper merging. i.e. source op with more than one output/param is not allowed
            if len(self.quantization_params[source_op_name][encoding_type]) != 1:
                log_warning('Can only merge 1 encoding for src op: {} w/tensor: {}, but found {}'.format(
                    source_op_name, pre_merge_dest_tensor_name, len(self.quantization_params[source_op_name][encoding_type])))
                return

            source_encodings = self.quantization_params[source_op_name][encoding_type]

            overridden_encodings = []
            dest_encodings = { QuantParams.BN_PARAMS: {}, QuantParams.PARAM_ENCODINGS: [], QuantParams.OUTPUT_ENCODINGS: [] }
            if destination_op_name in self.quantization_params:
                overridden_encodings = self.quantization_params[destination_op_name][encoding_type]
                dest_encodings = {
                    QuantParams.BN_PARAMS : self.quantization_params[destination_op_name][QuantParams.BN_PARAMS],
                    QuantParams.PARAM_ENCODINGS : self.quantization_params[destination_op_name][QuantParams.PARAM_ENCODINGS],
                    QuantParams.OUTPUT_ENCODINGS : self.quantization_params[destination_op_name][QuantParams.OUTPUT_ENCODINGS]
                }
                # remove the entry for the destination encoding as that will be replaced with the source op's encoding
                for i, encodings in enumerate(overridden_encodings):
                    if pre_merge_dest_tensor_name == encodings["name"]:
                        del self.quantization_params[destination_op_name][encoding_type][i]

            # Note: only need output encoding from source since the weights/bias will be merged into destination op
            source_encodings[0]["name"] = post_merge_dest_tensor_name  # replace output tensor name
            overridden_encodings.extend(source_encodings)
            dest_encodings[encoding_type] = overridden_encodings
            self.quantization_params.update({ destination_op_name: dest_encodings })

            # remove quantization entry for source op as the op will be merged
            del self.quantization_params[source_op_name]
        elif destination_op_name in self.quantization_params and use_dest_encoding:
            # Fall back to destination op name, if source op name is not present and
            # the op can use destination encoding i.e. data movement op
            # change the output name for destination encoding to the post merge name
            dest_encodings = self.quantization_params[destination_op_name][encoding_type]
            # In case of nodes with more than one outputs, when encodings are passed against
            # multiple output tensors, it is important to check whether the output encoding
            # present is for the pre_merge_dest_tensor_name tensor and merge only with that.
            for i in range(len(dest_encodings)):
                if (dest_encodings[i]["name"] == pre_merge_dest_tensor_name):
                    dest_encodings[i]["name"] = post_merge_dest_tensor_name
                    self.quantization_params[destination_op_name][encoding_type] = dest_encodings
                    break

    def eval_macs_params(self):
        """ Evaluates macs and params count for each Op in graph and adds to total"""
        self.reset_macs_params()
        for node in self.list_nodes():
            input_shapes = self.get_input_shapes(node)
            output_shapes = self.get_output_shapes(node)
            node.op.set_macs_params(input_shapes, output_shapes, self.src_axis_order)
            self.total_macs += node.op.macs
            self.total_params_count += node.op.params_count

    def reeval_macs_params(self):
        """ Re Calculates total macs and params count for graph"""
        self.reset_macs_params()
        for node in self.list_nodes():
            self.total_macs += node.op.macs
            self.total_params_count += node.op.params_count

    def reset_macs_params(self):
        """ Resets total macs and params count for graph"""
        self.total_macs = 0
        self.total_params_count = 0

    def update_constant_node_tensor(self, node, required_shape):
        """ Update the shape of Constant node tensor """
        # Check if the node is a constant node
        if node.op.type != op_adapter.ConstantOp.TRANSLATION_KEY:
            raise ValueError(f"Node {node.op.name} is not a Constant node. Got type: {node.op.type}")

        node.op.tensor = np.reshape(node.op.tensor, required_shape)
        # Updating the output buffer shape
        buffer_name = node.output_names[0]
        buffer = self.get_buffer(buffer_name)
        buffer.shape = required_shape

    # A method to change the buffer name
    def change_buffer_name(self, old_buffer_name, new_buffer_name):
        # Update the name of the buffer
        node_output_buffer = self.get_buffer(old_buffer_name)
        trace_info = self.get_trace_info(node_output_buffer)
        self.remove_trace_info(node_output_buffer)
        prev_node = node_output_buffer.producer
        node_output_buffer.name = new_buffer_name
        self.set_trace_info(node_output_buffer, trace_info)
        self.buffers[new_buffer_name] = node_output_buffer
        # Update the consumer nodes.
        for consumer in node_output_buffer.consumers:
            in_idx = consumer.input_names.index(old_buffer_name)
            consumer.input_names[in_idx] = new_buffer_name
        # Update the output names.
        out_idx = prev_node.output_names.index(old_buffer_name)
        prev_node.output_names[out_idx] = new_buffer_name
        del self.buffers[old_buffer_name]

    def __insert_node(
            self,
            node,
            output_shapes,
            output_dtypes,
            axis_formats=None,
            idx=-1,
            sparse_params=None,
            perms_to_src=None
    ):
        """Insert a node into the graph's internal data structures.

        node: Node to be inserted
        output_shapes: shapes of the node's output buffers, which must be created.
        axis_formats: List of axis_format to override for each output Buffer
        idx: index in nodes_in_order at which to insert. By default, appends to
             the list.
        sparse_params: sparse params for the output buffers of the node
        perms_to_src: a list of permute sequences, which can transform tensors to src-format.
        """
        # Step 1: Connect node as consumer to input buffers
        for name in node.input_names:
            # Add consumer if the input is not null
            if name:
                self.buffers[name].consumers.add(node)

        # Step 2: Insert Node in Graph
        if node in self.nodes_in_order:
            raise IndexError("Node by name {} already exists at index {}".format(
                node.op.name, self.nodes_in_order.index(node)))
        if node.op.name in self.nodes_by_name:
            # Check if the existing node and the node to be inserted are actually same
            node_equal, _ = node.op.is_equal(self.nodes_by_name[node.op.name].op)
            if node_equal:
                raise ValueError("Duplicate node name, {} already exists".format(node.op.name))
            else:
                while node.op.name in self.nodes_by_name:
                    node.op.name = node.op.name + "__" + str(random.randint(100000, 999999))

        self.nodes_by_name[node.op.name] = node

        # add the current op type in the op_types list
        self.op_types.add(node.op.type)
        if idx == -1:
            self.nodes_in_order.append(node)
        else:
            self.nodes_in_order.insert(idx, node)

        # Step 3: Populate Data Axis Format. Do this after connection between existing buffers
        # and nodes is established in the graph, since in some cases this function needs to
        # change the graph and update the connections. e.g. EltwiseBinary Op
        node.op.populate_data_axis_formats(self, self.get_input_buffers(node))

        # Step 4: Add Output Buffers
        for i, (name, shape, dtype) in enumerate(zip(node.output_names, output_shapes, output_dtypes)):
            if self.has_buffer(name):
                raise ValueError("Duplicate buffer name, {} already exists".format(name))

            if perms_to_src:
                buf = Buffer(
                    name,
                    shape,
                    dtype,
                    node,
                    axis_formats[i],
                    sparse_params=sparse_params,
                    perm_to_src=perms_to_src[i]
                )
            elif axis_formats:
                buf = Buffer(name, shape, dtype, node, axis_formats[i], sparse_params=sparse_params)
            else:
                buf = Buffer(name, shape, dtype, node, sparse_params=sparse_params)
                node.op.populate_axis_format(
                    self,
                    buf,
                    self.src_axis_order,
                    self.get_input_encodings(),
                    self.get_input_buffers(node)
                )

            self.buffers[name] = buf
            log_debug1("Added buffer named {0} of shape {1}", name, shape)


    def add(
            self,
            op,
            input_names,
            output_names,
            axis_formats=None,
            idx=-1,
            sparse_params=None,
            perms_to_src=None
    ):
        """
        Adds op to graph by creating a node and corresponding buffer, as well as update
        input and output buffers for node.
        :param op: an operation from op_adapter class
        :param input_names: inputs to node. (This will be the buffer input names)
        :param output_names: output buffer names of node
        :param axis_formats: axis format of output buffers
        :param idx: index in nodes_in_order at which to insert. By default, appends to the list.
        :param sparse_params: sparse params for the output buffer.
        :param perms_to_src: a list of permute sequences, which can transform tensors to src-format.
        :return: The created node for op.
        """
        op.name = self.naming_policy.get_op_name(op)

        if not isinstance(input_names, list):
            input_names = [input_names]
        for input_idx, in_name in enumerate(input_names):
            if in_name in self.int64_input_cast_map:
                input_names[input_idx] = self.int64_input_cast_map[in_name]
        input_names = self.naming_policy.get_input_names(op, input_names)

        input_shapes = []
        input_dtypes = []
        for name in input_names:
            if name and name not in self.buffers:
                raise KeyError("Graph has no buffer %s, referred to as input for %s" % (name, op.name))
            # To maintain consistency in the length of input_shapes, append empty shape if the input is null
            input_shapes.append(self.buffers[name].shape if name else self.null_buffer.shape)
            input_dtypes.append(self.buffers[name].dtype if name else self.null_buffer.dtype)

        if not isinstance(output_names, list):
            output_names = [output_names]
        output_names = self.naming_policy.get_output_names(op, output_names)

        # TODO: Move output_shapes, input_shapes inside OpNode.
        node = OpNode(op, input_names, output_names, self.src_axis_order)
        log_debug1("\n")
        log_debug1("Added OpNode with name {0}, in_names {1}, out_names {2}".format(op.name, input_names, output_names))
        try:
            input_axis_formats = []
            for name in input_names:
                # Append null axis format if the input is null
                input_axis_formats.append(self.get_buffer(name).axis_format if name else self.null_buffer.axis_format)
            output_shapes = op.infer_shape(input_shapes, input_axis_formats, len(output_names), self.src_axis_order)
            output_dtypes = op.infer_data_type(input_names, input_shapes, input_dtypes,
                                               input_axis_formats, len(output_shapes))
        except NotImplementedError as e:
            if self.shape_inference_policy:
                try:
                    output_shapes = self.shape_inference_policy.infer_shape(op, input_shapes)
                except KeyError as e:
                    log_error("Node %s: %s", op.name, e)
                    raise e
            else:
                log_error("Node %s: %s", op.name, e)
                raise e

        if len(output_shapes) != len(output_names):
            raise ValueError("Op %s: produced %d output shapes, but have %d outputs" % (op.name, len(output_shapes),
                                                                                        len(output_names)))
        if len(output_dtypes) != len(output_shapes):
            raise ValueError("Op %s: produced %d output shapes, but produced %d datatypes" % (op.name, len(output_shapes),
                                                                                              len(output_dtypes)))
            # at this point everything should be error free, so it's fine to actually
        # touch the data structures
        self.__insert_node(
            node,
            output_shapes,
            output_dtypes,
            axis_formats,
            idx=idx,
            sparse_params=sparse_params,
            perms_to_src=perms_to_src
        )

        op.update_param_quant_overrides(self, node)

        # return the added node
        return node

    def add_chained_eltwise_ops(self, operation, src_op_name, src_input_names, src_output_name):
        """
        Adds source elementwise op nodes that have > 2 inputs as a chain of binary elementwise nodes
        :param operation: The param value for the elementwise operation we're adding
        :param src_op_name: The name of the original node in the source framework
        :param src_input_names: The input names of the original node in the source framework
        :param src_output_name: The output name of the original node in the source framework
        """

        # Base case where the number of inputs is equal to two
        if len(src_input_names) == 2:
            new_ir_op = op_adapter.ElementwiseBinaryOp(src_op_name, operation=operation)
            node = self.add(new_ir_op, src_input_names, src_output_name)
            self.set_trace_info(node, [(src_op_name, TraceType.OP)])
            for output_name in node.output_names:
                self.set_trace_info(self.get_buffer(output_name), [(output_name, TraceType.TENSOR)])
            return node

        log_debug(code_to_message.get_debugging_message("DEBUG_ELEMENTWISEBINARY_CHAIN")
                  (src_op_name, op_adapter.ElementwiseBinaryOp.operation_to_legacy[operation], src_input_names))

        # For an elementwise node with n inputs, n-1 nodes must be added to the graph. The first node in the chain takes
        # the first two inputs to the original node. Subsequent nodes take their first input from the previous node in the
        # chain and take their second input from the next input of the original node. Inserted nodes are named according
        # to their index in the chain. The last node maintains the original node's name and output name.
        new_nodes_names = []
        for i in range(len(src_input_names) - 1):
            if i == 0:
                new_node_inputs = [src_input_names[i], src_input_names[i+1]]
            else:
                new_node_inputs = [new_nodes_names[-1], src_input_names[i+1]]

            if i != len(src_input_names) - 2:
                new_nodes_names.append("%s_chain_%d" % (src_op_name, i))
                new_node = self.add(op_adapter.ElementwiseBinaryOp(new_nodes_names[-1], operation=operation), new_node_inputs, new_nodes_names[-1])
                self.set_trace_info([new_node, self.get_buffer(new_nodes_names[-1])], [(src_op_name, TraceType.OP)])
            else:
                new_nodes_names.append(src_op_name)
                new_node = self.add(op_adapter.ElementwiseBinaryOp(new_nodes_names[-1], operation=operation), new_node_inputs, src_output_name)
                self.set_trace_info(new_node, [(src_op_name, TraceType.OP)])
                for output_name in new_node.output_names:
                    self.set_trace_info(self.get_buffer(output_name), [(output_name, TraceType.TENSOR)])

        return new_node

    def add_constant_op(self, op_name, output_names, constant_tensor, axis_formats=None, idx=-1, sparse_params=None,
                        transform_manager=None):
        if isinstance(constant_tensor, np.ndarray):
            constant_tensor = ir_graph.PyIrConstantTensor(constant_tensor)
            constant_tensor.set_name(op_name)
            if transform_manager:
                constant_tensor.transform_manager.copy_from(transform_manager)
        constant_op = op_adapter.ConstantOp(op_name, tensor=constant_tensor)
        constant_node = self.add(constant_op, [], output_names, axis_formats, idx, sparse_params)
        self.add_src_op_info(op_name, None, constant_node.output_names)
        return constant_node

    def update_node(self, node_name, new_node):
        if node_name in self.nodes_by_name:
            # Update the node in nodes_by_name
            self.nodes_by_name[node_name] = new_node
        idx = self.nodes_in_order.index(new_node)
        # Update the node in nodes_in_order
        self.nodes_in_order[idx] = new_node

    def remove_node_as_consumer(self, node, buffer_name):
        """
        Removes node as consumer from buffer with name buffer_name. Removes all instances of buffer_name from node's
        input_name list
        :param node: the node that should be removed as a consumer of the buffer specified by buffer_name
        :param buffer_name: the name of the buffer that this node is no longer a consumer of
        """
        if not isinstance(node, OpNode):
            raise TypeError("Passed node is not an instance of class OpNode. Passed type: {}".format(type(node)))
        buffer = self.get_buffer(buffer_name)
        if node in buffer.consumers:
            buffer.consumers.remove(node)
        # Note: Removes all instances of buffer_name from node's input_names
        while buffer_name in node.input_names:
            node.input_names.remove(buffer_name)
        # Prune the producing node if node was the only consumer of buffer buffer_name
        if not len(buffer.consumers):
            self.prune(buffer.producer)

    def update_consumer_data_axis_formats(self, op_node):
        for consumer in self.get_op_output_nodes(op_node):
            consumer_input_buffers = self.get_input_buffers(consumer)
            consumer.op.populate_data_axis_formats(self, consumer_input_buffers)

    def replace(self, old_op, new_op):
        old_node = self.nodes_by_name[old_op.name]
        input_buffers = self.get_input_buffers(old_node)
        output_buffers = self.get_output_buffers(old_node)
        input_names = [buf.name for buf in input_buffers]
        output_names = [buf.name for buf in output_buffers]

        # Create OpNode for the new op
        new_op.name = self.naming_policy.get_op_name(new_op)
        new_node = OpNode(new_op, input_names, output_names, self.src_axis_order)

        # update the new_node op_trace_data by the old_node for it is a direct replace
        self.update_trace_info(new_node, old_node, remove_source=True)

        # Replace the op in buffers
        # loop through as set to support scenarios where a node is listed as input more than once
        input_shapes = []
        input_dtypes = []
        for buf in uniques(input_buffers):
            buf.consumers.remove(old_node)
            buf.consumers.add(new_node)
            input_shapes.append(buf.shape)
            input_dtypes.append(buf.dtype)

        try:
            input_axis_formats = [buf.axis_format for buf in input_buffers]
            output_shapes = new_op.infer_shape(input_shapes, input_axis_formats, len(output_names), self.src_axis_order)
            output_dtypes = new_op.infer_data_type(input_names, input_shapes, input_dtypes,
                                                   input_axis_formats, len(output_shapes))

        except NotImplementedError as e:
            if self.shape_inference_policy:
                try:
                    output_shapes = self.shape_inference_policy.infer_shape(new_op, input_shapes)
                except KeyError as e:
                    log_error("Node %s: %s", new_op.name, e)
                    raise e
            else:
                log_error("Node %s: %s", new_op.name, e)
                raise e

        new_op.populate_data_axis_formats(self, input_buffers)
        for i, buf in enumerate(output_buffers):
            buf.producer = new_node
            buf.shape = output_shapes[i]
            buf.dtype = output_dtypes[i]
            new_op.populate_axis_format(self,
                                        buf,
                                        self.src_axis_order,
                                        self.get_input_encodings(),
                                        input_buffers)

        # Replace the op in op-lists
        idx = self.nodes_in_order.index(old_node)
        self.nodes_by_name[new_op.name] = new_node
        if idx == -1:
            self.nodes_in_order.append(new_node)
        else:
            self.nodes_in_order.insert(idx, new_node)

        # If the old and new Op names are the same, no need to delete the entry.
        # The value should have been updated with the new node.
        if new_op.name != old_op.name:
            del self.nodes_by_name[old_node.op.name]
        self.nodes_in_order.remove(old_node)
        self.replace_quantization_param(old_op.name, new_op.name)

        #updating op_types list with new_op type
        self.op_types.add(new_op.type)

    def add_input(self, name, shape, axis_format=None, input_type=None, input_dtype=None):
        if not input_type:
            input_type = self.get_input_type(name)

        # command-line takes precedence
        if not input_dtype or self.has_input_dtype(name):
            input_dtype = self.get_input_dtype(name)

        # cast float16 datatype to float32. The IR constructed for input should be fp32.
        if input_dtype == np.dtype('float16'):
            input_dtype = np.dtype('float32')

        if input_dtype == np.dtype('int64') and not self.is_int64_input_preserved(name):
            input_dtype = np.dtype('int32')

        input_encoding_in, input_encoding_out = self.get_input_encoding(name)

        if input_encoding_in == InputEncodings.TIME_SERIES:
            log_assert(len(shape) == 3,
                       code_to_message.get_error_message("ERROR_TIMESERIES_UNEXPECTED_RANK")
                       (name, len(shape)))
        if input_encoding_in == InputEncodings.OTHER:
            axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        # validate color transformation for image based encodings
        if input_encoding_in != input_encoding_out and \
                input_encoding_in not in [InputEncodings.TIME_SERIES, InputEncodings.OTHER]:

            log_assert(len(shape) == 4 and
                       self.src_axis_order.extract_2d_spatial_dims(shape)[-1] == 3,
                       code_to_message.get_error_message("ERROR_INVALID_COLOR_TRANSFORM_INPUT_SHAPE")
                       (shape, name, "[b,h,w,3]"))

            log_assert(InputEncodings.is_valid_transformation(input_encoding_in, input_encoding_out),
                       code_to_message.get_error_message("ERROR_UNSUPPORTED_COLOR_TRANSFORMATION")
                       (input_encoding_in, input_encoding_out))

        dynamic_axes = set()
        if name in self.input_dynamic_axes:
            dynamic_axes = self.input_dynamic_axes[name]
        op = op_adapter.InputOp(name, shape, dynamic_axes,
                                input_encoding_in=input_encoding_in,
                                input_encoding_out=input_encoding_out,
                                input_type=input_type,
                                input_dtype=input_dtype)
        output_names = self.naming_policy.get_output_names(op, [name])

        node = OpNode(op, [], output_names, self.src_axis_order)
        axis_formats = [axis_format] if axis_format is not None else (
            [self.input_axis_formats[name]] if name in self.input_axis_formats else (
                None if name not in self.custom_io_axis_formats else [self.custom_io_axis_formats[name]]
            ))
        input_op_output_dtype = op.numpy_to_qnn_datatype(op.input_dtype)
        self.__insert_node(node, [op.shape], [input_op_output_dtype], axis_formats)

        # TODO: Remove this once we support int64 dtype
        if input_dtype == np.dtype('int64'):
            cast_node = self.add(op_adapter.CastOp(name + "_cast_int32", from_type="int64", to_type="int32"),
                                 [name],
                                 [name + "_cast_int32"])
            self.set_trace_info([cast_node, self.get_buffer(cast_node.output_names[0])],
                                (name, TraceType.TENSOR))
            self.int64_input_cast_map[name] = name + "_cast_int32"

        # return the added input node
        return node

    def inject(self, op, input_name, output_name, consumer_names=None, axis_format=None, set_consumers_trace_info=True):
        """
        Inject node into graph.
        If the argument 'set_consumers_trace_info' is set to True, the injected node's trace information is set by merging
        all the previous output consumers. If it is set to False, the injected node's trace information is set by
        merging all the previous input producers.
        """
        if not isinstance(input_name, str):
            raise TypeError("Input name {} needs to be String, received {}".format(input_name, type(input_name)))
        if not isinstance(output_name, str):
            raise TypeError("Output name {} needs to be String, received {}".format(output_name, type(output_name)))

        op.name = self.naming_policy.get_op_name(op)
        if input_name not in self.buffers:
            raise KeyError("Cannot inject op %s onto nonexistent buffer %s" % (op.name, input_name))

        input_buffer = self.buffers[input_name]
        if consumer_names is None:
            old_consumers = list(input_buffer.consumers)
            input_buffer.consumers.clear()
        else:
            old_consumers = []
            for name in consumer_names:
                if name not in self.nodes_by_name:
                    raise KeyError("Cannot inject op %s with nonexistent consumer %s" % (op.name, name))
                consumer = self.nodes_by_name[name]
                if consumer not in input_buffer.consumers:
                    raise KeyError("Cannot inject op %s, specified consumer %s does not actually consume input"
                                   " buffer %s" % (op.name, name, input_name))

                old_consumers.append(consumer)
                input_buffer.consumers.remove(consumer)

        output_name = self.naming_policy.get_output_names(op, [output_name])[0]
        producer_idx = self.nodes_in_order.index(input_buffer.producer)

        try:
            output_shapes = op.infer_shape([input_buffer.shape], [input_buffer.axis_format], 1, self.src_axis_order)
            output_dtypes = op.infer_data_type([input_buffer.name], [input_buffer.shape], [input_buffer.dtype],
                                               [input_buffer.axis_format], 1)
        except NotImplementedError as e:
            if self.shape_inference_policy:
                try:
                    output_shapes = self.shape_inference_policy.infer_shape(op, [input_buffer.shape])
                except KeyError as e:
                    log_error("Node %s: %s", op.name, e)
                    raise e
            else:
                log_error("Node %s: %s", op.name, e)
                raise e
        node = OpNode(op, [input_name], [output_name], self.src_axis_order)
        axis_formats = None if axis_format is None else [axis_format]
        self.__insert_node(node, output_shapes, output_dtypes, axis_formats=axis_formats, idx=producer_idx+1)
        output_buffer = self.buffers[output_name]
        for consumer in old_consumers:
            output_buffer.consumers.add(consumer)
            for i, name in enumerate(consumer.input_names):
                if name == input_name:
                    consumer.input_names[i] = output_name
        # update the node and its output op_trace_data by merging all the previous output consumers'
        # trace information if the flag set_consumers_trace_info set to True
        if set_consumers_trace_info:
            self.update_trace_info([node, output_buffer], old_consumers)
        # update the node and its output op_trace_data by merging all the previous input producers'
        # trace information if the flag set_consumers_trace_info set to True
        else:
            self.update_trace_info([node, output_buffer], input_buffer.producer)
        return node

    @staticmethod
    def get_implicit_permute_node_name(input_name, target_format):
        return str(input_name + '.' + target_format.lower())

    def inject_implicit_permute(self, input_name, target_format, permute_order, consumers:[str] = None):
        permute_name = self.get_implicit_permute_node_name(input_name, target_format)

        input_buf = self.get_buffer(input_name)
        if consumers:
            for consumer_name in consumers:
                if self.nodes_by_name[consumer_name] not in input_buf.consumers:
                    log_warning("Attempting to remove non-existent consumer {} from buffer {}"
                                .format(consumer_name, input_buf.name))
        log_assert(input_buf.rank() == len(permute_order),
                   "Error: length of buf to permute({}) does not match length of permute order({})"
                   " for input name: {}",
                   input_buf.rank(), len(permute_order), input_name)

        implicit_permute = op_adapter.TransposeOp(permute_name, permute_order)

        if self.has_buffer(permute_name) and self.get_buffer(permute_name).axis_format == target_format:
            permuted_buf = self.get_buffer(permute_name)
            producer_op = self.get_buffer(permute_name).producer.op
            if isinstance(producer_op, op_adapter.TransposeOp) and producer_op.perm == permute_order:
                # Permute already exists
                if consumers:
                    for consumer_name in consumers:
                        consumer = self.get_node_by_name(consumer_name)
                        in_ids = [i for i in range(len(consumer.input_names)) if consumer.input_names[i] == input_name]
                        for in_idx in in_ids:
                            consumer.input_names[in_idx] = permute_name
                        permuted_buf.consumers.add(consumer)
                        if consumer in input_buf.consumers:
                            # Check if the consumer has already been removed
                            # (might come if input_buf is a shared buffer)
                            input_buf.consumers.remove(consumer)
                return permuted_buf.producer

        # since the implicit permute won't be visited in this pass, go
        # ahead and set the correct order for its buffer here.
        return self.inject(implicit_permute, input_name, permute_name, consumers, axis_format=target_format)

    def prune(self, node, force_remove=False, merge_trace_info_to_previous=False, merge_trace_info_to_next=False):
        """Remove a node and its output buffers from the graph completely.
        :param force_remove: if set to False and the node has any successors, an exception will be raised
        :param merge_trace_info_to_previous: Boolean indicate to merge op trace information into node input's producer op
        :param merge_trace_info_to_next: Boolean indicate to merge op trace information into node output's consumer op
        """

        # Disconnect output nodes
        input_buffers = self.get_input_buffers(node)
        output_buffers = self.get_output_buffers(node)
        consumers = []
        for buf in output_buffers:
            consumers.extend(buf.consumers)

        if len(consumers) > 0:
            if force_remove:
                for buf in output_buffers:
                    for c in buf.consumers:
                        c.input_names = [input_name for input_name in c.input_names if input_name != buf.name]
            else:
                consumer_name_list = [c.op.name for c in consumers]
                raise RuntimeError("Cannot prune node %s, which has the following successors: %s"
                                   % (node.op.name, consumer_name_list))

        # merge the pruned node and its outputs framework trace info into its input's producer node
        if merge_trace_info_to_previous:
            for buf in input_buffers:
                merge_src = [node, buf.producer]
                merge_src.extend(output_buffers)
                for in_buf in self.get_input_buffers(node):
                    # If there is constant input, it is needed that merged target node should
                    # inherit this static input's framework tracing source information
                    if in_buf.producer.op.type == "constant":
                        merge_src.append([in_buf, in_buf.producer])
                self.update_trace_info(buf.producer, merge_src)

        # merge the pruned node and its output framework trace info into its output's consumer node
        if merge_trace_info_to_next:
            for buf in output_buffers:
                for c in buf.consumers:
                    merge_src = [node, self.get_buffer(buf.name), c]
                    print(node)
                    for in_buf in self.get_input_buffers(node):
                        print(in_buf)
                        # If there is constant input, it is needed that merged target node should
                        # inherit this static input's framework tracing source information
                        if in_buf.producer.op.type == "constant":
                            merge_src.extend([in_buf, in_buf.producer])
                    self.update_trace_info(c, merge_src)

        # remove op trace info of the prune nodes and output buffers
        for buf in output_buffers:
            self.remove_trace_info(buf)
        self.remove_trace_info(node)

        for buf in output_buffers:
            del self.buffers[buf.name]
            self.naming_policy.remove_output_name(buf.name)

        # Disconnect input nodes
        # loop through as set to support scenarios where a node is listed as input more than once
        for buf in set(self.get_input_buffers(node)):
            # This can create dangling buffers and subgraphs, but we let remove_disconnected_nodes to handle it
            if buf.consumers:
                buf.consumers.remove(node)
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)

    def squash_identity(self, node: OpNode, is_data_movement_node=False):
        """
        Squashes IdentityOp node into the input buffer. Basically change the output name of the previous op to be
        the output of current opnode. Since we are only changing output name, there is no need to check for number
        of consumers or other restrictions
        :param node: OpNode
        :param is_data_movement_node: Boolean indicating if the node is a data movement node
        :return: throws Error if node is not of type Identity, otherwise return nothing
        """

        if len(node.input_names) != 1:
            raise ValueError("Op {} expected to have only 1 input, has {}".format(node.op.type, len(node.input_names)))
        if len(node.output_names) != 1:
            raise ValueError("Op {} expected to have only 1 output, has {}".format(node.op.type, len(node.output_names)))
        input_name = node.input_names[0]
        output_name = node.output_names[0]

        input_buffer = self.get_buffer(input_name)
        output_buffer = self.get_buffer(output_name)
        use_dest_encoding = False

        # Squash into previous is only possible if the producing op is not an input
        # and if input is not one of the graph's output
        # also don't squash if it's a constant -> identity reshape -> graph output, identity reshape is for static graph output
        if input_buffer.producer.op.type != op_adapter.InputOp.TRANSLATION_KEY and \
                input_name not in self.output_names:

            if input_buffer.shape == output_buffer.shape and \
                node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY and \
                input_buffer.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                output_name in self.output_names:

                raise ValueError(f"Unable to squash node: \'{node}\' into previous node: \'{input_buffer.producer}\' as node produces a static output")

            # Change the input names of input buffer's consumers to consume output_name instead.
            # (ideally consumers other than current node,
            # but its ok to change here because the output name was cached above)
            #
            # No need to change the input_buffer's consumers set since the buffer will be deleted anyway
            for consumer in input_buffer.consumers:
                while input_name in consumer.input_names:
                    input_idx = consumer.input_names.index(input_name)
                    consumer.input_names[input_idx] = output_name
                    if consumer.op.name != node.op.name:
                        output_buffer.consumers.add(consumer)

            # Change the output name of prev_node
            prev_node = input_buffer.producer
            output_idx = prev_node.output_names.index(input_name)
            prev_node.output_names[output_idx] = output_name

            # Change the producer of output buffer to prev_node
            output_buffer.producer = prev_node

            # A data movement node is a node whose op only shapes, reshapes or selects a portion of its input
            # It does not change the encodings of its input, so after squashing the destination encoding should
            # be unchanged except for the new output tensor name
            if is_data_movement_node:
                use_dest_encoding = True

            # update the quantization_params when prev_node output name is change
            self.merge_quantization_params(node.op.name, prev_node.op.name, input_name, output_name,
                                           use_dest_encoding=use_dest_encoding)

            # update op_trace_data
            self.update_trace_info(prev_node, [prev_node, node, input_buffer])

            # Delete dangling buffer and nodes
            del self.buffers[input_name]
            del self.nodes_by_name[node.op.name]
            self.nodes_in_order.remove(node)
            self.remove_trace_info([node, input_buffer])

            # Make sure that the correct prev_node's output name is reflected in the output_encodings
            if prev_node.op.name in self.quantization_params.keys():
                for encoding_dict in self.quantization_params[prev_node.op.name]['output_encodings']:
                    if encoding_dict['name'] == input_name:
                        encoding_dict['name'] = output_name
        else:
            self.squash(node, input_name, squash_into_next=True, is_data_movement_node=is_data_movement_node)

    def squash(self, node: OpNode, input_name:str, squash_into_next=False, is_data_movement_node=False) -> bool:
        """
        Squashes OpNode after which the OpNode will no longer exist.
        Attempt to squash into previous node first,
        and if that's not possible, then squash into next node.
        :param node: the node to be squashed
        :param input_name: the name of the input buffer coming from the previous node
        :param squash_into_next: Force squash into next. e.g. Used when lowering, squash into previous is not
                                 possible since the previous node is already lowered. Hence force squash into next
        :param is_data_movement_node: Boolean indicating if the node is a data movement node

        :return returns False on failure, else True
        """

        if len(node.output_names) != 1:
            raise ValueError("Node {} must have exactly one output to be squashed. Got {}".format(
                node.op.name, len(node.output_names)))

        if not isinstance(input_name, str):
            raise TypeError("Input name {} needs to be String, received {}".format(input_name, type(input_name)))

        if input_name not in node.input_names:
            raise ValueError("Input name {} must be an input to Node {}. Node's inputs: {}".format(
                input_name, node.op.name, node.input_names))

        output_name = node.output_names[0]
        input_buffer = self.get_buffer(input_name)
        output_buffer = self.get_buffer(output_name)

        previous_node = input_buffer.producer
        next_op_nodes = self.get_op_output_nodes(node)

        # Do not squash if it's a constant -> identity reshape -> graph output, as reshape is added for static graph output
        if previous_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
            node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY and \
            (output_name in self.output_names or \
            output_name.split(":")[0] in self.output_names) and \
            input_buffer.shape == output_buffer.shape:
            raise ValueError(f"Unable to squash node: \'{node}\' into previous node: \'{input_buffer.producer}\' as node produces a static output")

        # don't squash if the buffer which will be replaced is one of the graph's output.
        # 1. input buffer if squashing into previous node (squash_into_next == False)
        # 2. output buffer if squashing into next node (squash_into_next == True)
        # Cannot squash into previous if the previous node's op is InputOp or
        # if the input_buffer has more than 1 consumer
        if previous_node.op.type != op_adapter.InputOp.TRANSLATION_KEY and \
                len(input_buffer.consumers) == 1 and \
                not squash_into_next and \
                input_name not in self.output_names and \
                input_name.split(":")[0] not in self.output_names:
            # Change Previous Node output as current Output Buffer
            output_idx = previous_node.output_names.index(input_name)
            legacy_output_buffer = self.get_buffer(previous_node.output_names[output_idx])
            previous_node.output_names[output_idx] = output_name
            output_buffer.producer = previous_node
            # update op_trace_data to previous since it is squash to previous
            self.update_trace_info(previous_node, [previous_node, node, input_buffer])
            self.remove_trace_info([legacy_output_buffer, node])

            # A data movement node is a node whose op only shapes, reshapes or selects a portion of its input
            # It does not change the encodings of its input, so after squashing the destination encoding should
            # be unchanged except for the new output tensor name
            use_dest_encoding = False
            if is_data_movement_node:
                use_dest_encoding = True

            # update the quantization_params when prev_node output name is change
            self.merge_quantization_params(node.op.name, previous_node.op.name, input_name, output_name,
                                           use_dest_encoding=use_dest_encoding)


            # Generate other_input_names as set in case input names are listed twice
            other_input_names = set([in_name for in_name in node.input_names if in_name != input_name])
            for in_buf_name in other_input_names:
                in_buf = self.get_buffer(in_buf_name)
                in_producer = self.get_producer_node(in_buf_name)
                # Simply prune if producing node has exactly one consumer, the node we are squashing
                if len(in_buf.consumers) == 1 and len(in_producer.output_names) == 1:
                    self.prune(in_producer, True)
                else:
                    # Remove node to be squashed from input buffer consumers
                    in_buf.consumers.remove(node)
                    # If this input buffer has no more consumers after removing producing node, then remove it
                    if not len(in_buf.consumers):
                        # merge in_buf op_trace_data to its producer since it has no consumer and will be deleted
                        self.update_trace_info(in_producer, [in_producer, in_buf])
                        in_producer.output_names.remove(in_buf_name)
                        del self.buffers[in_buf_name]
                        self.naming_policy.remove_output_name(in_buf_name)

            # Remove the input buffer from the graph
            del self.buffers[input_name]
            self.naming_policy.remove_output_name(input_name)
        elif len(next_op_nodes) and \
                output_name not in self.output_names and \
                output_name.split(":")[0] not in self.output_names:
            # Change Next Node input to be current Input Buffer
            for next_op_node in next_op_nodes:
                # There can be multiple same inputs to the op
                while output_name in next_op_node.input_names:
                    input_idx = next_op_node.input_names.index(output_name)
                    legacy_input_buffer = self.get_buffer(next_op_node.input_names[input_idx])
                    next_op_node.input_names[input_idx] = input_name
                    input_buffer.consumers.add(next_op_node)
                    # update op_trace_data to next since it is squash to next
                    merge_src = [node, output_buffer, next_op_node]
                    for in_buf in self.get_input_buffers(node):
                        # If there is constant input, it is needed that squashed node should
                        # inherit this static input's framework source information
                        if in_buf.producer.op.type == "constant":
                            merge_src.append(in_buf)
                    self.update_trace_info(next_op_node, merge_src)
                    self.remove_trace_info([legacy_input_buffer, node])
            # Copy current_node's output encoding to previous_node if is_data_movement_node
            if is_data_movement_node:
                self.merge_quantization_params(node.op.name, previous_node.op.name, input_name, input_name)
            del self.buffers[output_name]
            self.naming_policy.remove_output_name(output_name)
            self.remove_quantization_params(node.op.name)
            input_buffer.consumers.remove(node)
        else:
            # Squashing not possible. Raise error to catch the failures. The caller can use try-except to failsafe
            raise RuntimeError("Squash not possible for op {}".format(node))

        # Remove the squashed node from the graph
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)
        return True

    def update_preserve_datatype_tensors(self):
        '''
        Iterates over the preserve_datatype_tensors (tensors whose dtypes are preserved by passing
        --preserve_io datatype argument). Since source frameworks like ONNX can't distinguish between
        Fixed point types and integer types, this function helps distinguishing them.
        If an output tensor's producer is a Quantize op, its output dtype needs to be changed to
        FXP with the same bitwidth. Similarly, if an input tensor's consumers are Dequant ops, then
        the input dtype needs to be changed to FXP with the same bitwidth.

        Return: True if any of the dtypes have been modified, else false.
        '''
        graph_input_nodes = self.get_input_nodes_to_graph()
        graph_output_nodes = self.get_output_nodes_of_graph()
        graph_input_names = [node.output_names[0] for node in graph_input_nodes]
        graph_output_names = [node.output_names[0] for node in graph_output_nodes]
        modified = False
        for tensor in self.preserve_datatype_tensors:
            # If a preserve_datatype_tensor is an input to the graph, and Dequantize op is its consumer,
            # then the dtype of the input will be in FXP and not INT. Therefore, update the datatype
            # to FXP.
            if tensor in graph_input_names:
                orig_dtype = self.preserve_datatype_tensors[tensor]
                new_dtype = None
                consumers = self.get_buffer(tensor).consumers
                for consumer in consumers:
                    if isinstance(consumer.op, op_adapter.DequantizeOp):
                        new_dtype = int_to_fxp(orig_dtype)
                if new_dtype is not None:
                    self.preserve_datatype_tensors[tensor] = new_dtype
                    modified = True
            # Similarly, if a tensor is an output to the graph and Quantize op is its producer,
            # then the dtype of the output will be in FXP and not INT. Therefore, update the datatype
            # to FXP.
            elif tensor in graph_output_names:
                orig_dtype = self.preserve_datatype_tensors[tensor]
                new_dtype = None
                producer = self.get_buffer(tensor).producer
                if isinstance(producer.op, op_adapter.QuantizeOp):
                    new_dtype = int_to_fxp(orig_dtype)
                if new_dtype is not None:
                    self.preserve_datatype_tensors[tensor] = new_dtype
                    modified = True
            else:
                continue
        return modified

    def get_matched_nodes_v2(self, sequence, validator=None, ignore_constants=False, unmatched_nodes_idx=None):
        """
        Traverses graph to find the requested pattern using DFS. This function is
        superior to get_matched_nodes as this will work in case of single
        start node and multiple end nodes of the given pattern.
        :param sequence: list[tuples] a list of node translation keys with their inputs and outputs. i.e:
                         each tuple contains ("opdapter.<op_name>.TRANSLATION_KEY", ([inputs]), ([outputs]))
                         The tuple for inputs/outputs should state BufferCriteria to verify list length; additionally,
                         each input/output should state specific BufferCriteria to determine how many(if any) of the
                         buffer should be in the matched sequence.
             E.g for format:
             sequence = [
                   # node type A
                   (op_adapter.<op_name>.TRANSLATION_KEY,
                       # inputs
                       (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...]),
                       # outputs
                       (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...])
                   ),
                   # node type B
                   (op_adapter.<op_name>.TRANSLATION_KEY,
                       # inputs
                       (),
                       # outputs
                       ()
                   ),
                   ...
             ]
             E.g (Channel Shuffle). Note: we can pass strings instead of class.xxx for convenience,
                                          this function handles both.
             sequence = [
                        ("reshape",
                            (),
                            ("MATCH_NUM_BUFS", [("permute", "ALL")])
                        ),
                        ("permute",
                            (),
                            ("MATCH_NUM_BUFS", [("reshape", "ALL")])
                        ),
                        ("reshape",
                            (),
                            ()
                        )
                       ]
             Note 1: both inputs and outputs should also be translation keys
             Note 2: BufferCriteria can either be one of the BufferCriteria Enums or an INT to match a specific index
             Note 3: it is not required to have inputs or outputs, they can be left empty.
        :param validator: function to run if a match is found based on sequence. The matched sequence will be passed as
                          {"node_tuples": (nodes_matched)}
                          If not provided, function will return based on only matching the sequence as criteria.
        :param ignore_constants: if constant nodes need to be filtered during matching, this flag will be set to True.
        :return: list of node tuples that match the sequence provided, where each tuple contains the corresponding nodes
                 for each TRANSLATION_KEY in the sequence.
        """
        requested_types_seq = [entry[0] for entry in sequence]

        # Skip the pattern matching if any op in the sequence is not present in the graph
        for op_type in requested_types_seq:
            if op_type not in self.op_types:
                return []

        nodes_list = self.list_nodes()

        if ignore_constants:
            nodes_list = [node for node in nodes_list if node.op.type != op_adapter.ConstantOp.TRANSLATION_KEY]

        if unmatched_nodes_idx is not None:
            nodes_list = [nodes_list[idx] for idx in unmatched_nodes_idx]

        log_debug2("Evaluating to match Sequence {}...", requested_types_seq)

        # we want to allow use of strings for op translation_keys(i.e op_types) to make sequence length minimal
        # so validate user has asked to match op_types that are supported in op_adapter
        log_assert(self.verify_op_types_exist(requested_types_seq) is True,
                   code_to_message.get_error_message("ERROR_UNKNOWN_OP_TYPE(S)_FOUND")(requested_types_seq))

        matched_nodes = []
        first_sequence_node_op_type = requested_types_seq[0]
        for model_node in nodes_list:
            if model_node.op.type != first_sequence_node_op_type:
                continue
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
                elif self.is_output_node(_node):
                    # This means we reached end of graph but the pattern to
                    # be matched is still has some nodes.
                    continue
                if _node.op.type == requested_types_seq[_node_level]:
                    inputs_actual = self.get_input_op_types(_node)
                    outputs_actual = self.get_output_op_types(_node)
                    inputs_expected, outputs_expected = sequence[_node_level][1:]

                    # providing inputs_expected and outputs_expected is not required from user
                    # since user might just care to match a sequence of node types for any given inputs/outputs
                    if (len(inputs_expected) and not self._validate_buffers(inputs_expected, inputs_actual)) or \
                            (len(outputs_expected) and not self._validate_buffers(outputs_expected, outputs_actual)):
                        continue

                    if _node_level + 1 == len(sequence):
                        # This means we reached at the end of the pattern
                        # Pattern fully matched.
                        # _path will represent all the matched nodes.
                        matched_node_tuples.append(_path)

                    _children_nodes = self.get_op_output_nodes(_node)
                    stack.extend([[_c_node, _node_level + 1, [*_path, _c_node]] for _c_node in _children_nodes])

            for matched_node_tuple in matched_node_tuples:
                if validator is None or validator(matched_node_tuple):
                    if len(matched_node_tuple) != len(sequence):
                        log_debug("Matched node list length must be same as "
                                  "requested sequence. Expected {}, Got {}", \
                                  len(sequence), len(matched_node_tuple))
                        continue
                    matched_nodes.append(tuple(matched_node_tuple))
                    log_debug2(f"Found match: {matched_node_tuple}")

        log_debug2("Found {} match(es)", len(matched_nodes))

        return matched_nodes

    def get_matched_nodes(self, sequence, validator=None, ignore_constants=False, use_dfs=False):
        """
        Traverses each node in graph to find the requested pattern
        :param sequence: list[tuples] a list of node translation keys with their inputs and outputs. i.e:
                         each tuple contains ("opdapter.<op_name>.TRANSLATION_KEY", ([inputs]), ([outputs]))
                         The tuple for inputs/outputs should state BufferCriteria to verify list length; additionally,
                         each input/output should state specific BufferCriteria to determine how many(if any) of the
                         buffer should be in the matched sequence.
             E.g for format:
             sequence = [
                   # node type A
                   (op_adapter.<op_name>.TRANSLATION_KEY,
                       # inputs
                       (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...]),
                       # outputs
                       (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...])
                   ),
                   # node type B
                   (op_adapter.<op_name>.TRANSLATION_KEY,
                       # inputs
                       (),
                       # outputs
                       ()
                   ),
                   ...
             ]
             E.g (Channel Shuffle). Note: we can pass strings instead of class.xxx for convenience,
                                          this function handles both.
             sequence = [
                        ("reshape",
                            (),
                            ("MATCH_NUM_BUFS", [("permute", "ALL")])
                        ),
                        ("permute",
                            (),
                            ("MATCH_NUM_BUFS", [("reshape", "ALL")])
                        ),
                        ("reshape",
                            (),
                            ()
                        )
                       ]
             Note 1: both inputs and outputs should also be translation keys
             Note 2: BufferCriteria can either be one of the BufferCriteria Enums or an INT to match a specific index
             Note 3: it is not required to have inputs or outputs, they can be left empty.
        :param validator: function to run if a match is found based on sequence. The matched sequence will be passed as
                          {"node_tuples": (nodes_matched)}
                          If not provided, function will return based on only matching the sequence as criteria.
        :param ignore_constants: if constant nodes need to be filtered during matching, this flag will be set to True.
        :return: list of node tuples that match the sequence provided, where each tuple contains the corresponding nodes
                 for each TRANSLATION_KEY in the sequence.
        """

        matched_nodes = []
        requested_types_seq = [entry[0] for entry in sequence]
        # Skip the pattern matching if any op in the sequence is not present in the graph
        for op_type in requested_types_seq:
            if op_type not in self.op_types:
                return []

        # create a list of op code from the op_types
        requested_code_seq = []
        for type in requested_types_seq:
            requested_code_seq.append(type_to_code[type])

        start = 0
        end = len(sequence)
        nodes_list = self.list_nodes()

        if ignore_constants:
            constant_op_code = type_to_code[op_adapter.ConstantOp.TRANSLATION_KEY]
            nodes_list = [node for node in nodes_list if node.op.op_code != constant_op_code]

        log_debug2("Evaluating to match Sequence {}...", requested_types_seq)

        # we want to allow use of strings for op translation_keys(i.e op_types) to make sequence length minimal
        # so validate user has asked to match op_types that are supported in op_adapter
        log_assert(self.verify_op_types_exist(requested_types_seq) is True,
                   code_to_message.get_error_message("ERROR_UNKNOWN_OP_TYPE(S)_FOUND")(requested_types_seq))

        first_sequence_node_op_type = requested_code_seq[0]
        unmatched_nodes_idx = []
        while end <= len(nodes_list):
            # Check if first op is matching and do early exit
            # if first_sequence_node_op_type != nodes_list[start].op.type:
            if first_sequence_node_op_type != nodes_list[start].op.op_code:
                start += 1
                end = start + len(sequence)
                continue
            nodes_tuple = tuple(nodes_list[start:end])  # get number of nodes based on length of sequence
            # current_types_seq = [node.op.type for node in nodes_tuple]
            current_types_seq = [node.op.op_code for node in nodes_tuple]
            output_name = nodes_list[start].output_names[0]
            # The following IF condtiion handles the case where the length of the sequence to be matched is 2.
            # We make sure that the first node in the requested sequence is not a model output.
            if len(requested_code_seq) == 2 and len(nodes_list[start].output_names) == 1 and \
                    self.buffers[output_name].consumers:
                node = list(self.buffers[output_name].consumers)[0]
                candidates_nodes_tuples = tuple([nodes_list[start], node])
                if (node.op.op_code == requested_code_seq[1] and self._validate_nodes_topology(candidates_nodes_tuples, sequence)) and \
                        (validator is None or validator(candidates_nodes_tuples)):
                    matched_nodes.append(candidates_nodes_tuples)
                # nodes_list can sometimes have nodes which are parallel as neighbours. So, do not skip over the length
                # of the sequence matched, rather we just increment the start pointer by 1.
                start += 1
                end = start + len(sequence)
                continue
            # Validate the node topology and add it to the list of the matched nodes if it satisfies it.
            elif (current_types_seq == requested_code_seq and self._validate_nodes_topology(nodes_tuple, sequence)) and \
                    (validator is None or validator(nodes_tuple)):
                start = end  # start next node by skipping over the length of the sequence matched
                end += len(sequence)
                matched_nodes.append(nodes_tuple)
                continue
            elif use_dfs:
                # Add the index of the node if it does not get matched and also use_dfs option is enabled for it.
                unmatched_nodes_idx.append(start)
            start += 1
            end = start + len(sequence)

        # Call get_matched_nodes_v2 function only if there are some unmatched sequence and also use_dfs option is enabled
        # for them.
        if use_dfs and unmatched_nodes_idx:
            log_debug2("DFS based pattern matching is enabled to match Sequence {}...", requested_types_seq)
            matched_nodes_v2 = self.get_matched_nodes_v2(sequence, validator=validator, ignore_constants=True, unmatched_nodes_idx=unmatched_nodes_idx)
            for node in matched_nodes_v2:
                if node not in matched_nodes:
                    matched_nodes.append(node)

        log_debug2("Found {} match(es)", len(matched_nodes))

        return matched_nodes

    def _validate_nodes_topology(self, nodes_tuple, sequence):
        """
        validates the input and output buffers for each matched node sequence in graph

        :param nodes_tuple: a tuple of matched nodes based on pattern
        :param sequence: the original list of sequences provided by user
        :return: True if each node's input and output buffer match the expected ones in sequence, False otherwise
        :raises: AssertionError if length and node types of node_list and sequence do not match
        """

        log_assert(len(nodes_tuple) == len(sequence), "Matched node list length must be same as requested sequence. "
                                                      "Expected {}, Got {}", len(nodes_tuple), len(sequence))

        for i in range(0, len(nodes_tuple)):
            node_type_actual = nodes_tuple[i].op.type
            node_type_expected = sequence[i][0]
            log_assert(node_type_actual == node_type_expected,
                       "Cannot validate topology for nodes of different types. Expected {}, Got{}",
                       node_type_expected, node_type_actual)

            inputs_actual = self.get_input_op_types(nodes_tuple[i])
            outputs_actual = self.get_output_op_types(nodes_tuple[i])
            inputs_expected, outputs_expected = sequence[i][1:]

            # providing inputs_expected and outputs_expected is not required from user
            # since user might just care to match a sequence of node types for any given inputs/outputs
            if (len(inputs_expected) and not self._validate_buffers(inputs_expected, inputs_actual)) or \
                    (len(outputs_expected) and not self._validate_buffers(outputs_expected, outputs_actual)):
                log_debug2("Sequence pattern {} matched, but not input/output buffers for node {} of type {} in "
                           "sequence.", [entry[0] for entry in sequence], nodes_tuple[i].op.name,
                           nodes_tuple[i].op.type)
                return False

        return True

    def _validate_buffers(self, expected_buffers, actual_buffers):
        """
        validates the actual buffers(inputs or outputs of nodes) against the criteria set in the expected buffers
        :param expected_buffers: a tuple with BufferCriteria for matching the list of buffers, list of tuple pairs
                                 with each tuple containing the type of op and a buffer criteria
                        (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...])
        :param actual_buffers: list of actual buffer types for the current node being evaluated
        :return: true if actual buffers pass criteria set in the expected buffers, False otherwise

        raises Assertion error: if unknown buffer criteria,
               Value error: if ALL criteria given and there exists more expected inputs
        """

        # remove matching criteria from expected buffers and validate
        matching_criteria, expected_buffers = expected_buffers
        matching_criteria = matching_criteria.upper()
        log_assert(matching_criteria in [BufferCriteria.MATCH_NUM_BUFS, BufferCriteria.FLEXIBLE_NUM_BUFS, BufferCriteria.MATCH_BUFS_AT_INDEX],
                   code_to_message.get_error_message("ERROR_UNKNOWN_MATCHING_CRITERIA")
                   ([BufferCriteria.MATCH_NUM_BUFS, BufferCriteria.FLEXIBLE_NUM_BUFS, BufferCriteria.MATCH_BUFS_AT_INDEX], matching_criteria))

        if matching_criteria == BufferCriteria.MATCH_NUM_BUFS and len(expected_buffers) != len(actual_buffers):
            return False

        for op_type, buf_criteria in expected_buffers:
            log_assert(self.verify_op_types_exist(op_type) is True,
                       code_to_message.get_error_message("ERROR_UNKNOWN_OP_TYPE(S)_FOUND")(op_type))

            if type(buf_criteria) == int:
                if matching_criteria == BufferCriteria.MATCH_NUM_BUFS:
                    # User knows the number of input/output buffers to expect, hence it is an error to request
                    # an out-of-range index
                    log_assert(buf_criteria < len(actual_buffers),
                               code_to_message.get_error_message("ERROR_BUFFER_CRITERIA_INDEX")
                               (op_type, buf_criteria, len(actual_buffers)))

                    if actual_buffers[buf_criteria] != op_type:
                        return False

                elif matching_criteria == BufferCriteria.MATCH_BUFS_AT_INDEX:
                    # In this case, user doesnt know/care for the number of input/output buffers of a node but want to
                    # match ops that fit a certain criteria e.g. when the 2nd input is a particular op type;
                    # in this instance an out-of-range index is not an error.

                    if buf_criteria >= len(actual_buffers) or actual_buffers[buf_criteria] != op_type:
                        return False
                elif matching_criteria == BufferCriteria.FLEXIBLE_NUM_BUFS:
                    # In this case, user knows exactly how many of this type to expect but does not care
                    # about the position in the inputs
                    op_type_count = len([actual_op_type for actual_op_type in actual_buffers
                                         if actual_op_type == op_type])
                    if op_type_count != buf_criteria:
                        return False
            elif buf_criteria.upper() == BufferCriteria.ALL:
                if len(expected_buffers) != 1:
                    raise ValueError(code_to_message.get_error_message("ERROR_BUFFER_CRITERIA_ALL")
                                     (op_type, len(expected_buffers)))
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
                raise ValueError(code_to_message.get_error_message("ERROR_UNKNOWN_BUFFER_CRITERIA")
                                 (op_type, ["ALL", "ANY", "NONE"], buf_criteria))

        return True

    @staticmethod
    def verify_op_types_exist(op_list):
        if type(op_list) is not list:
            op_list = [op_list]
        # get all supported op_types in op_adapter module
        supported_op_list = [class_[1].TRANSLATION_KEY if hasattr(class_[1], 'TRANSLATION_KEY') else ''
                             for class_ in inspect.getmembers(op_adapter, inspect.isclass)]
        supported_op_list.extend(op_adapter.ElementwiseTernaryOp.ir_to_legacy_type.values())
        supported_op_list.extend(op_adapter.ElementwiseUnaryOp.operation_to_legacy.values())
        supported_op_list.extend(op_adapter.ElementwiseBinaryOp.operation_to_legacy.values())
        supported_op_list.extend(op_adapter.ReduceOp.ir_to_legacy_type.values())
        return all(op in supported_op_list for op in op_list)

    def get_input_buffers(self, node):
        node.op.c_op.inputs()
        input_buffers = []
        for name in node.input_names:
            input_buffers.append(self.buffers[name] if name else self.null_buffer)
        return input_buffers

    def get_output_buffers(self, node):
        node.op.c_op.outputs()
        return [self.buffers[name] for name in node.output_names]

    def get_input_shapes(self, node):
        node.op.c_op.get_input_shapes()
        input_shapes = []
        for name in node.input_names:
            input_shapes.append(self.buffers[name].shape if name else self.null_buffer.shape)
        return input_shapes

    def get_output_shapes(self, node):
        node.op.c_op.get_output_shapes()
        return [self.buffers[name].shape for name in node.output_names]

    def get_input_axis_formats(self, node):
        input_axis_formats = []
        for name in node.input_names:
            input_axis_formats.append(self.buffers[name].axis_format if name else self.null_buffer.axis_format)
        return input_axis_formats

    def get_output_axis_formats(self, node):
        return [self.buffers[name].axis_format for name in node.output_names]

    def save_src_axis_formats(self):
        for buffer in self.buffers.values():
            src_axis_format = buffer.get_axis_format()
            buffer.set_src_axis_format(src_axis_format)

    @staticmethod
    def get_input_buffer_idx(node, buf_name):
        buf_idx = -1
        for i, input_name in enumerate(node.input_names):
            if buf_name == input_name:
                buf_idx = i
        log_assert(buf_idx != -1, "Unable to find input buffer {} in Node {} input buffer list [{}]"
                   .format(buf_name, node.op.name, node.input_names))
        return buf_idx

    @staticmethod
    def get_output_buffer_idx(node, buf_name):
        buf_idx = -1
        for i, output_name in enumerate(node.output_names):
            if buf_name == output_name:
                buf_idx = i
        log_assert(buf_idx != -1, "Unable to find output buffer {} in Node {} output buffer list [{}]"
                   .format(buf_name, node.op.name, node.output_names))
        return buf_idx

    def add_output_buffer(self, node, name: str, shape: list, axis_format=None):
        buf = Buffer(name, shape, node, axis_format=axis_format)
        self.buffers[name] = buf
        node.output_names.append(name)

    def get_op_output_nodes(self, node):
        output_nodes = set()
        for buf in self.get_output_buffers(node):
            output_nodes.update(buf.consumers)
        return list(output_nodes)

    def get_op_input_nodes(self, node):
        op_input_nodes = []
        for buf in self.get_input_buffers(node):
            op_input_nodes.append(buf.producer if not buf.is_null() else self.null_buffer.producer)
        return op_input_nodes

    def get_input_op_types(self, node):
        input_op_types = []
        for name in node.input_names:
            input_op_types.append(self.buffers[name].producer.op.type if name else self.null_buffer.producer)
        return input_op_types

    def get_output_op_types(self, node):
        consumer_nodes = []
        consumer_nodes_types = []
        for name in node.output_names:
            for consumer in self.buffers[name].consumers:
                # consumer already existing in our list can happen if one consumer takes 2 or more outputs of a node.
                # e.g: if node_a has buf_1, buf_2 as outputs and next layer(node_b) has both of these buffers as input,
                # both buf_1 and buf_2 will list node_b as consumers so we don't want to have [node_b, node_b]
                # for outputs
                if consumer not in consumer_nodes:
                    consumer_nodes.append(consumer)
                    consumer_nodes_types.append(consumer.op.type)
        return consumer_nodes_types

    def get_buffer(self, buffer_name):
        return self.buffers[buffer_name] if buffer_name else self.null_buffer

    def has_buffer(self, buffer_name):
        return buffer_name in self.buffers

    def add_buffer(self, buf):
        if self.has_buffer(buf.name):
            raise ValueError("Duplicate buffer name, {} already exists".format(buf.name))
        self.buffers[buf.name] = buf

    def delete_buffer(self, buffer_name):
        if not self.has_buffer(buffer_name):
            raise ValueError("Buffer ({}) requested for deletion is not found.".format(buffer_name))
        del self.buffers[buffer_name]

    def get_producer_node(self, buffer_name):
        return self.buffers[buffer_name].producer

    def get_producer_op(self, buffer_name):
        return self.buffers[buffer_name].producer.op

    def get_consumer_nodes(self, buffer_name):
        return list(self.buffers[buffer_name].consumers)

    def get_consumer_ops(self, buffer_name):
        return [node.op for node in self.buffers[buffer_name].consumers]

    def get_parent_nodes(self, node):
        par_nodes = []
        for input_buf in self.get_input_buffers(node):
            par_nodes.append(self.get_producer_node(input_buf.name))
        return par_nodes

    def get_children_nodes(self, node):
        children_nodes = []
        for output_buf in self.get_output_buffers(node):
            children_nodes.extend(self.get_consumer_nodes(output_buf.name))
        return children_nodes

    def get_input_nodes_to_graph(self):
        input_nodes = []
        for node in self.list_nodes():
            if node.op.TRANSLATION_KEY == op_adapter.InputOp.TRANSLATION_KEY:
                input_nodes.append(node)
        return input_nodes

    def get_output_nodes_of_graph(self):
        output_nodes = []
        passed_output_buf_names = self.output_names[:]
        # matches list of given output buf names to graph buf names
        for node in self.list_nodes():
            for output_buf_name in self.output_names:
                # Find name prefix before colon(:), similar to check in IrOpGraph to IrGraph translation
                node_output_name_list = [output_name.split(":")[0] for output_name in node.output_names]
                if (output_buf_name in node_output_name_list) or (output_buf_name in node.output_names):
                    output_nodes.append(node)
                    if output_buf_name in passed_output_buf_names:
                        passed_output_buf_names.remove(output_buf_name)

        for name in passed_output_buf_names:
            log_warning("Output Buffer {}, specified via command line, does not exist in graph.".format(name))

        return output_nodes

    def is_output_node(self, node):
        node_output_names1 = [buffer.name for buffer in self.get_output_buffers(node)]
        # TF models may insert ":0" in output names
        node_output_names2 = [buffer.name.split(":")[0] for buffer in self.get_output_buffers(node)]
        for graph_output_name in self.output_names:
            if (graph_output_name in node_output_names1) or (graph_output_name in node_output_names2):
                return True
        return False

    def list_nodes(self):
        return self.nodes_in_order[:]

    def list_buffers(self):
        return list(self.buffers.values())

    def get_graph_output_buffer_names(self):
        return self.output_names[:]

    def has_op(self, op):
        nodes = self.list_nodes()
        for node in nodes:
            if node.op == op:
                return True
        return False

    def has_node(self, node_name):
        return node_name in self.nodes_by_name

    def get_node_by_name(self, node_name):
        return self.nodes_by_name.get(node_name)

    def is_node_present(self, node):
        return node is self.get_node_by_name(node.op.name)

    def get_node_idx_in_order(self, node):
        return self.nodes_in_order.index(node)

    def get_quantizable_tensors(self):
        params = {}
        for node in self.list_nodes():
            if node in self.list_nodes():
                tmp = {'type' : node.op.type}
                for kv in node.attrs.items():
                    if isinstance(kv[1], np.ndarray) and kv[1].dtype == np.float32 and \
                            node.op.hasattr('quantizable') and node.op['quantizable']:
                        tmp[kv[0]] = kv[1]
                if len(tmp) > 1:
                    params[node.name] = tmp
        return params

    def get_overridden_encoding(self, name, is_param=True):
        """
        Returns quantization encoding if provided externally for a given buffer name
        :param name: name of the buffer whose quantization encoding is to be retrieved
        :param is_param: True if given buffer is a parameter such as weights and bias, False for activation buffer
        :return: list of encoding dictionary for the given buffer, if any, otherwise return None
        """
        if self.has_user_quantization_overrides():
            if 'param_encodings' in self.user_quantization_overrides:
                if is_param:
                    param_encodings = self.user_quantization_overrides['param_encodings']
                    if name in param_encodings.keys():
                        return param_encodings[name]
                else:
                    act_encodings = self.user_quantization_overrides['activation_encodings']
                    if name in act_encodings.keys():
                        return act_encodings[name]
            else:
                encodings = self.user_quantization_overrides['encodings']
                if name in encodings.keys():
                    return encodings[name]
        return None
    def set_overridden_encoding(self, name, encoding, is_param=True):
        """
        Set quantization encoding for a given buffer name
        :param name: name of the buffer whose quantization encoding is to be retrieved
        :param is_param: True if given buffer is a parameter such as weights and bias, False for activation buffer
        """
        if 'param_encodings' in self.user_quantization_overrides:
            if is_param:
                param_encodings = self.user_quantization_overrides['param_encodings']
                if name not in param_encodings.keys():
                    param_encodings[name] = encoding
            else:
                act_encodings = self.user_quantization_overrides['activation_encodings']
                if name not in act_encodings.keys():
                    act_encodings[name] = encoding
        else:
            encodings = self.user_quantization_overrides['encodings']
            if name not in encodings.keys():
                encodings[name] = encoding

    def remove_overridden_encoding(self, name, is_param=True):
        """
        Removes user provided quantization encoding for a given buffer name
        :param name: name of the buffer whose quantization encoding is to be deleted
        :param is_param: True if given buffer is a parameter such as weights and bias, False for activation buffer
        """
        if 'param_encodings' in self.user_quantization_overrides:
            if is_param:
                del self.user_quantization_overrides['param_encodings'][name]
            else:
                del self.user_quantization_overrides['activation_encodings'][name]
        else:
            del self.user_quantization_overrides['encodings'][name]

    def has_user_quantization_overrides(self):
        for val in self.user_quantization_overrides.values():
            if val != {}:
                return True
        return False

    def get_trace_target_name(self, target):
        # The argument target could be a OpNode or a Buffer
        if self.enable_trace:
            if isinstance(target, Buffer):
                return target.name
            elif isinstance(target, OpNode):
                return target.op.name
            else:
                log_debug("The passed in target type is '{}', which is not acceptable for framework tracing.".format(type(target)))
                return None

    def get_trace_target_type(self, target):
        # The argument target could be a OpNode or a Buffer
        if self.enable_trace:
            if isinstance(target, Buffer):
                return TraceType.TENSOR
            elif isinstance(target, OpNode):
                return TraceType.OP
            else:
                log_debug("The passed in target type is '{}', which is not acceptable for framework tracing.".format(type(target)))
                return None
        return None

    def get_trace_info(self, targets):
        """
        Function to get the trace information from target(s). Target(s) can be one OpNode or one Buffer or a list of them.
        :param targets: a OpNode or a Buffer of a list of OpNode/Buffer
        :return: a list of source_name and source_type tuples like [(source_name, source_type), ...]
        """
        if not self.enable_trace:
            return None
        if not isinstance(targets, list) and not isinstance(targets, tuple):
            targets = [targets]
        trace_info = list()
        for target in targets:
            target_name = self.get_trace_target_name(target)
            target_type = self.get_trace_target_type(target)
            if target_type is not None:
                if target_name in self.trace_dict[target_type]:
                    trace_info.extend(self.trace_dict[target_type][target_name]) \
                        if self.trace_dict[target_type][target_name] else trace_info.extend(list())
            else:
                log_debug("The passed in target name '{}' with type '{}' is not traced by framework tracing dict."
                          .format(target_name, target_type))
        trace_info = list(OrderedDict.fromkeys(trace_info))
        return trace_info

    def set_trace_info(self, targets, trace_pairs):
        """
        Function to set the trace information to target(s). Target(s) can be one OpNode or one Buffer or a list of them.
        :param targets: a OpNode or a Buffer of a list of OpNode/Buffer
        :param trace_pairs:a single tuple like (source_name, source_type) or
                           a list of tuple like [(source_name, source_type), ...]
        """
        if not self.enable_trace:
            return
        if not isinstance(targets, list) and not isinstance(targets, tuple):
            targets = [targets]
        for target in targets:
            target_name = self.get_trace_target_name(target)
            target_type = self.get_trace_target_type(target)
            if target_name is None or target_type is None:
                continue
            if not isinstance(trace_pairs, list):
                trace_pairs = [trace_pairs]
            self.trace_dict[target_type][target_name] = trace_pairs

    def update_trace_info(self, targets, source_list, remove_source=False):
        """
        Function to update the target(s) trace information by merge all the source_list unit trace information.
        Target(s) can be one OpNode or one Buffer or a list of them.
        :param targets: a OpNode or a Buffer of a list of OpNode/Buffer
        :param source_list: single or a list of source OpNode/Buffer instances
        :param remove_source: remove sources' trace_info after their trace_info are collected.
        """
        if self.enable_trace:
            new_trace_info = self.get_trace_info(source_list)
            if remove_source:
                self.remove_trace_info(source_list)
            self.set_trace_info(targets, new_trace_info)


    def remove_trace_info(self, targets):
        """
        Function to remove the trace information of target(s). Target(s) can be one OpNode or one Buffer or a list of them.
        :param targets: a OpNode or a Buffer of a list of OpNode/Buffer
        """
        if not self.enable_trace:
            return
        if not isinstance(targets, list) and not isinstance(targets, tuple):
            targets = [targets]
        for target in targets:
            target_name = self.get_trace_target_name(target)
            target_type = self.get_trace_target_type(target)
            if target_name is None or target_type is None:
                continue
            if target_name in self.trace_dict[target_type]:
                del(self.trace_dict[target_type][target_name])

    def validate_trace_info_completeness(self, source_op_names=[], source_tensor_names=[], stage="NONE"):
        """
        To validate the trace dict is complete for all source graph ops/tensors and all target graph (current graph) ops/tensors
            - From keys view (target graph) of op trace record:
                - All the names in "keys list" in op trace info MUST be in the target graph.
                It should be a name of a op or a tensor of target graph, if not it MUST give out warnings.
                - All the op and tensor names in the target graph MUST be in the "keys list" of op trace info, if not it MUST give out warnings.
            - From values view (source graph) of op trace record:
                - All the name in the "values list" of each op or tensor in the op trace info MUST be in the Source Model.
                It should be a name of op or a tensor of source graph, if not it MUST give out warnings.
                - All the ops and tensors names in the source graph MUST be in the "value list" of op trace info, if not it MUST give out warnings.
        :param source_op_names: list of source graph op names
        :param source_tensor_names: list of source graph tensor names
        :param stage: the string to describe the translation stage
        """
        if self.enable_trace:
            # prepare target graph op/tensor names
            target_op_names = [node.op.name for node in self.list_nodes()]
            target_tensor_names = [buffer.name for buffer in self.list_buffers()]
            # prepare source info from trace record
            framework_trace_values = []
            for v in self.trace_dict[TraceType.OP].values():
                if isinstance(v, tuple):
                    framework_trace_values.append(v)
                else:
                    framework_trace_values.extend(v)
            for v in self.trace_dict[TraceType.TENSOR].values():
                if isinstance(v, tuple):
                    framework_trace_values.append(v)
                else:
                    framework_trace_values.extend(v)
            op_trace_info_value_names = [pair[0] for pair in framework_trace_values if pair[1]==TraceType.OP]
            tensor_trace_info_value_names = [pair[0] for pair in framework_trace_values if pair[1]==TraceType.TENSOR]
            # prepare target info from trace record
            op_trace_info_key_names = self.trace_dict[TraceType.OP].keys()
            tensor_trace_info_key_names = self.trace_dict[TraceType.TENSOR].keys()
            # start completeness validation
            is_completeness = True
            delta_op_names = set(op_trace_info_key_names) - set(target_op_names)
            if delta_op_names != set():
                is_completeness = False
                log_warning("Op names in framework tracing record are not all contained by target graph "
                            "in stage {}, please check below ops: {}".format(
                    str(stage), list(delta_op_names)
                ))
            delta_op_names = set(target_op_names) - set(op_trace_info_key_names)
            if delta_op_names != set():
                is_completeness = False
                log_warning("Op names in target graph are not all contained by framework tracing record "
                            "in stage {}, please check below ops: {}".format(
                    str(stage), list(delta_op_names)
                ))
            delta_op_names = set(op_trace_info_value_names) - set(source_op_names)
            if delta_op_names != set():
                is_completeness = False
                log_warning("Op names in framework tracing record are not all contained by source graph "
                            "in stage {}, please check below ops: {}".format(
                    str(stage), list(delta_op_names)
                ))
            delta_op_names = set(source_op_names) - set(op_trace_info_value_names)
            if delta_op_names != set():
                is_completeness = False
                log_warning("Op names in source graph are not all contained by framework tracing record "
                            "in stage {}, please check below ops: {}".format(
                    str(stage), list(delta_op_names)
                ))
            delta_tensor_names = set(tensor_trace_info_key_names) - set(target_tensor_names)
            if delta_tensor_names != set():
                is_completeness = False
                log_warning("tensor names in framework tracing record are not all contained by target graph "
                            "in stage {}, please check below tensors: {}".format(
                    str(stage), list(delta_tensor_names)
                ))
            delta_tensor_names = set(target_tensor_names) - set(tensor_trace_info_key_names)
            if delta_tensor_names != set():
                is_completeness = False
                log_warning("tensor names in target graph are not all contained by framework tracing record "
                            "in stage {}, please check below tensors: {}".format(
                    str(stage), list(delta_tensor_names)
                ))
            delta_tensor_names = set(tensor_trace_info_value_names) - set(source_tensor_names)
            if delta_tensor_names != set():
                is_completeness = False
                log_warning("tensor names in framework tracing record are not all contained by source graph "
                            "in stage {}, please check below tensors: {}".format(
                    str(stage), list(delta_tensor_names)
                ))
            delta_tensor_names = set(source_tensor_names) - set(tensor_trace_info_value_names)
            if delta_tensor_names != set():
                is_completeness = False
                log_warning("tensor names in source graph are not all contained by framework tracing record "
                            "in stage {}, please check below tensors: {}".format(
                    str(stage), list(delta_tensor_names)
                ))
            if is_completeness:
                log_info("Framework tracing record is complete in stage {}.".format(str(stage)))

    def get_trace_info_sub_graph(self, node_tuple, output_buffers_exclude=[]):
        """
        Function to fetch all trace_info for nodes in node_tuple and all constant inputsa and
        outputs of these nodes. Use the argument output_buffers_exclude to exclude the specific
        buffers for example the last node output buffer.
        :param node_tuple: list of OpNode
        :param output_buffers_exclude: the buffer should be excluded from node_tuple subgraph,
                                       usually the last one node output also the subgraph output.
                                       Default is empty.
        """
        if self.enable_trace:
            all_nodes_outputs = []
            for node in node_tuple:
                if node not in all_nodes_outputs:
                    all_nodes_outputs.append(node)
                    for input_name in node.input_names:
                        if self.get_buffer(input_name).producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                            all_nodes_outputs.append(self.get_buffer(input_name).producer)
                            all_nodes_outputs.append(self.get_buffer(input_name))
                    for output_name in node.output_names:
                        outbut_buf = self.get_buffer(output_name)
                        if outbut_buf not in all_nodes_outputs and outbut_buf not in output_buffers_exclude:
                            all_nodes_outputs.append(outbut_buf)
            all_trace_info = self.get_trace_info(all_nodes_outputs)
            if all_trace_info is not None:
                all_trace_info = list(OrderedDict.fromkeys(all_trace_info))
            return all_trace_info

    def update_graph_property_mask(self, bool_property_enum, bool_value):
        if bool_value:
            self.graph_property_bitmask |= (bool_property_enum.value)
        else:
            self.graph_property_bitmask &= ~(bool_property_enum.value)

    def get_graph_property_mask(self, bool_property_enum):
        return (self.graph_property_bitmask & bool_property_enum.value) != 0
