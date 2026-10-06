# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2018-2020, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import sys
import argparse
import numpy
import json
from collections import OrderedDict
from enum import Enum
from qti.aisw.converters.common import ir_graph


# -----------------------------------------------------------------------------------------------------
#   Common Functions
# -----------------------------------------------------------------------------------------------------


def sanitize_args(args, args_to_ignore=[]):
    sanitized_args = []
    if isinstance(args, argparse.Namespace):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        raise TypeError("args needs to be of type argparse.Namespace or Dict")

    for k, v in list(sorted(args_dict.items())):
        if k in args_to_ignore:
            continue
        sanitized_args.append('{}={}'.format(k, v))
    return "{}; {}".format(sys.argv[0].split('/')[-1], '; '.join(sanitized_args))


def get_string_from_txtfile(filename):
    if not filename:
        return ""
    if filename.endswith('.txt'):
        try:
            with open(filename, 'r') as myfile:
                file_data = myfile.read()
            return file_data
        except Exception as e:
            logger.error("Unable to open file %s: %s" % (filename, e))
            sys.exit(-1)
    else:
        logger.error("File %s: must be a text file." % filename)
        sys.exit(-1)


# Returns the i-th bit of val
def get_bit(val, i):
    return val & (1 << i)


# Returns the indices of all the set bits in val
def get_bits(val):
    count = 0
    bits = []
    while (val):
        if val & 1:
            bits.append(count)
        count += 1
        val >>= 1
    return bits


def uniques(values):
    """
    :type values: list
    :rtype: list
    """
    dictionary = OrderedDict()
    for v in values:
        if v not in dictionary:
            dictionary[v] = v
    return list(dictionary.keys())


def is_valid_buffer_name(name):
    """ a function to check if a name would cause error"""
    # the file_name for buffer is "<buffer_name>.raw" in bin file,
    # if buffer name length larger than 256, tar will cause "File name too long" error
    # since the file_name could contain path, here we use 200 as a threshold to validate buffer name
    if len(name) >= 200:
        return False
    return True


def rename_user_quantization_overrides(graph, src_name, dst_name):
    # if encoding is provided, rename it to new param name
    if 'param_encodings' in graph.user_quantization_overrides:
        if src_name in graph.user_quantization_overrides['param_encodings']:
            graph.user_quantization_overrides['param_encodings'][dst_name] = graph.user_quantization_overrides['param_encodings'][src_name].copy()
            del graph.user_quantization_overrides['param_encodings'][src_name]
        elif src_name in graph.user_quantization_overrides['activation_encodings']:
            graph.user_quantization_overrides['activation_encodings'][dst_name] = graph.user_quantization_overrides['activation_encodings'][src_name].copy()
            del graph.user_quantization_overrides['activation_encodings'][src_name]
    elif 'encodings' in graph.user_quantization_overrides:
        if src_name in graph.user_quantization_overrides['encodings']:
            graph.user_quantization_overrides['encodings'][dst_name] = graph.user_quantization_overrides['encodings'][src_name].copy()
            del graph.user_quantization_overrides['encodings'][src_name]


def add_new_user_quantization_overrides(graph, tensor_name, quant_param, enc):
    # Add the given encoding to graph as user overridden encodings
    supported_quant_params = ["param_encodings", "activation_encodings", "encodings"]
    if quant_param not in supported_quant_params:
        raise ValueError("Invalid quant param {} passed. Supported values are {}"
                        .format(quant_param, supported_quant_params))
    graph.user_quantization_overrides[quant_param][tensor_name] = enc


def convert_args_dict_to_namespace(args: dict, converter_frontend_argparser: argparse.ArgumentParser) -> argparse.Namespace:
    """Converts args dictionary to the specified converter frontend namespace instance"""

    required_args = []
    required_arg_names = [action.option_strings for action in
                          converter_frontend_argparser.argument_groups['required arguments']._group_actions]
    for arg_name_options in required_arg_names:
        for arg_name in arg_name_options:
            if arg_name[2:] in args.keys():
                arg = args.get(arg_name[2:])
                if isinstance(arg, list):
                    for arg_i in arg:
                        required_args.append(arg_name)
                        if isinstance(arg_i, list):
                            required_args.extend(arg_i)
                        else:
                            required_args.append(arg_i)
                else:
                    required_args.append(arg_name)
                    required_args.append(arg)

    namespace_instance = converter_frontend_argparser.parse_args(required_args)
    namespace_dict = vars(namespace_instance)
    namespace_dict.update(args)
    return argparse.Namespace(**namespace_dict)


# -----------------------------------------------------------------------------------------------------
#   Caffee Common Functions
# -----------------------------------------------------------------------------------------------------

class NpUtils(object):
    def blob2arr(self, blob):
        if hasattr(blob, "shape"):
            return numpy.ndarray(buffer=blob.data, shape=blob.shape, dtype=numpy.float32)
        else:
            # Caffe-Segnet fork doesn't have shape field exposed on blob.
            return numpy.ndarray(buffer=blob.data, shape=blob.data.shape, dtype=numpy.float32)


# -----------------------------------------------------------------------------------------------------
#   Logging
# -----------------------------------------------------------------------------------------------------
# @deprecated
# TODO: remove once cleanup of converters is done to use method below instead
logger = logging.getLogger(__name__)


def setUpLogger(verbose):
    formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
    lvl = logging.INFO
    if verbose:
        lvl = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(lvl)
    formatter = logging.Formatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# --- end of deprecated ---

LOGGER = None
HANDLER = None
LOG_LEVEL = logging.INFO

# Custom Logging
logging.VERBOSE = LOG_LEVEL_VERBOSE = 5
logging.DEBUG_3 = DEBUG_LEVEL_IR_TO_BACKEND = 11
logging.DEBUG_2 = DEBUG_LEVEL_IR_OPTIMIZATION = 12
logging.DEBUG_1 = DEBUG_LEVEL_CONVERTER_TO_IR = 13

# add the custom log-levels
logging.addLevelName(DEBUG_LEVEL_IR_TO_BACKEND, "DEBUG_3")
logging.addLevelName(DEBUG_LEVEL_IR_OPTIMIZATION, "DEBUG_2")
logging.addLevelName(DEBUG_LEVEL_CONVERTER_TO_IR, "DEBUG_1")
logging.addLevelName(LOG_LEVEL_VERBOSE, "VERBOSE")


def setup_logging(debug_lvl, name=None):
    global LOGGER
    global HANDLER
    global LOG_LEVEL

    if debug_lvl == -1:  # --debug is not set
        LOG_LEVEL = logging.INFO
    elif debug_lvl == 0:  # --debug is set with no specific level. i.e: print every debug message.
        LOG_LEVEL = logging.DEBUG
    elif debug_lvl == 1:
        LOG_LEVEL = logging.DEBUG_1
    elif debug_lvl == 2:
        LOG_LEVEL = logging.DEBUG_2
    elif debug_lvl == 3:
        LOG_LEVEL = logging.DEBUG_3
    elif debug_lvl == 4:
        LOG_LEVEL = logging.VERBOSE
    else:
        log_assert("Unknown debug level provided. Got {}", debug_lvl)

    if LOGGER is None:
        LOGGER = logging.getLogger(name)
    LOGGER.setLevel(LOG_LEVEL)

    if HANDLER is None:
        formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
        HANDLER = handler
    HANDLER.setLevel(LOG_LEVEL)


def log_assert(cond, msg, *args):
    assert cond, msg.format(*args)


def log_debug(msg, *args):
    if LOGGER:
        LOGGER.debug(msg.format(*args))


def log_debug1(msg, *args):
    def debug1(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.DEBUG_1):
            LOGGER._log(logging.DEBUG_1, msg, args, kwargs)
    debug1(msg.format(*args))


def log_debug2(msg, *args):
    def debug2(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.DEBUG_2):
            LOGGER._log(logging.DEBUG_2, msg, args, kwargs)
    debug2(msg.format(*args))


def log_debug3(msg, *args):
    def debug3(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.DEBUG_3):
            LOGGER._log(logging.DEBUG_3, msg, args, kwargs)
    debug3(msg.format(*args))


def log_verbose(msg, *args):
    def verbose(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.VERBOSE):
            LOGGER._log(logging.VERBOSE, msg, args, kwargs)
    verbose(msg.format(*args))


def log_error(msg, *args):
    if LOGGER:
        LOGGER.error(msg.format(*args))


def log_info(msg, *args):
    if LOGGER:
        LOGGER.info(msg.format(*args))


def log_warning(msg, *args):
    if LOGGER:
        LOGGER.warning(msg.format(*args))


def get_log_level():
    global LOG_LEVEL
    return LOG_LEVEL


def is_log_level_debug():
    global LOG_LEVEL
    return LOG_LEVEL == logging.DEBUG


def log_debug_msg_as_status(msg, *args):
    log_debug(msg + "."*50, *args)


# -----------------------------------------------------------------------------------------------------
#   Translation Helpers
# -----------------------------------------------------------------------------------------------------
"""
Following functions sanitize op/layer type names and attach converter type for registering translations
"""


def get_op_info(type_name):
    """Return the op name and version, if specified"""
    op_data = str(type_name).split('-')
    if len(op_data) > 1:
        return [op_data[0], int(op_data[1])]
    op_data.append(0)
    return op_data


def op_type(type_name):
    """Return the actual onnx op name"""
    data = get_op_info(type_name)
    return data[0]


def converter_type(type_name, src_converter):
    """Convert an src converter type name string to a namespaced format"""
    return src_converter + '_' + (op_type(type_name)).lower()


# -----------------------------------------------------------------------------------------------------
#   LoRA Helpers
# -----------------------------------------------------------------------------------------------------

def get_lora_tensor_names_from_file(filepath):
    if filepath:
        with open(filepath, 'r') as file:
            tensor_names = file.read()
            tensor_names = tensor_names.splitlines()
            return tensor_names
    else:
        # --lora_weight_list can be passed without any input file,
        # when input Model does not have LoRA branches but activation
        # encodings can change based on selected LoRA adapter during run time
        return []


def track_transform(constant_tensor, transform_type, transform_attributes):
    if constant_tensor.transform_manager:
        if not constant_tensor.transform_manager.track_transform(transform_type, transform_attributes):
            logger.error("Tracking transform type {} failed for tensor {}.".format(
                            ir_graph.convertTransformTypeToString(transform_type),
                            constant_tensor.name))
            sys.exit(-1)

def get_updateable_static_tensors_in_graph(cpp_graph, tensor_names):
    tensor_map = cpp_graph.get_tensor_map()

    return [
        tensor_map[name]
        for name in tensor_names
        if name in tensor_map and
        tensor_map[name].is_updateable() and
        tensor_map[name].is_static_tensor()
    ]

def populate_lora_metadata_json_schema(cpp_graph, lora_tensor_names):
    lora_static_tensors = get_updateable_static_tensors_in_graph(cpp_graph, lora_tensor_names)

    lora_metadata_dict = {
        "version": "1.0.0",
        cpp_graph.name: {
            "lora_tensors": {}
        }
    }

    for tensor in lora_static_tensors:
        if not tensor.transform_manager:
            continue
        serialized_transforms_string = tensor.transform_manager.serialize_transforms()
        lora_metadata_transforms = "{" + serialized_transforms_string + "}"
        transform_list_json = json.loads(lora_metadata_transforms)
        lora_metadata_dict[cpp_graph.name]["lora_tensors"][tensor.name()] = transform_list_json

    return lora_metadata_dict

# Convert QNN Data type to string
def to_str(qnn_data_type):
    dtype_map = {
        ir_graph.QNN_DATATYPE_FLOAT_32: "float32",
        ir_graph.QNN_DATATYPE_FLOAT_16: "float16",
        ir_graph.QNN_DATATYPE_UINT_64:  "uint64",
        ir_graph.QNN_DATATYPE_INT_64:   "int64",
        ir_graph.QNN_DATATYPE_UINT_32:  "uint32",
        ir_graph.QNN_DATATYPE_INT_32:   "int32",
        ir_graph.QNN_DATATYPE_UINT_16:  "uint16",
        ir_graph.QNN_DATATYPE_INT_16:   "int16",
        ir_graph.QNN_DATATYPE_UINT_8:   "uint8",
        ir_graph.QNN_DATATYPE_INT_8:    "int8",
        ir_graph.QNN_DATATYPE_BOOL_8:   "bool",
        ir_graph.QNN_DATATYPE_STRING:   "str",
        ir_graph.QNN_DATATYPE_UFIXED_POINT_4: "quint4",
        ir_graph.QNN_DATATYPE_UFIXED_POINT_8: "quint8",
        ir_graph.QNN_DATATYPE_UFIXED_POINT_16: "quint16",
        ir_graph.QNN_DATATYPE_UFIXED_POINT_32: "quint32",
        ir_graph.QNN_DATATYPE_SFIXED_POINT_4: "qint4",
        ir_graph.QNN_DATATYPE_SFIXED_POINT_8: "qint8",
        ir_graph.QNN_DATATYPE_SFIXED_POINT_16: "qint16",
        ir_graph.QNN_DATATYPE_SFIXED_POINT_32: "qint32",
    }
    return dtype_map.get(qnn_data_type)

# Convert numpy data type to QNN
def numpy_to_qnn_datatype(dtype):
        _numpy_to_qnn_datatype = {
            numpy.dtype('int8'): ir_graph.QNN_DATATYPE_INT_8,
            numpy.dtype('int16'): ir_graph.QNN_DATATYPE_INT_16,
            numpy.dtype('int32'): ir_graph.QNN_DATATYPE_INT_32,
            numpy.dtype('int64'): ir_graph.QNN_DATATYPE_INT_64,
            numpy.dtype('uint8'): ir_graph.QNN_DATATYPE_UINT_8,
            numpy.dtype('uint16'): ir_graph.QNN_DATATYPE_UINT_16,
            numpy.dtype('uint32'): ir_graph.QNN_DATATYPE_UINT_32,
            numpy.dtype('uint64'): ir_graph.QNN_DATATYPE_UINT_64,
            numpy.dtype('float16'): ir_graph.QNN_DATATYPE_FLOAT_16,
            numpy.dtype('float32'): ir_graph.QNN_DATATYPE_FLOAT_32,
            numpy.dtype('bool'): ir_graph.QNN_DATATYPE_BOOL_8,
            numpy.dtype('str'): ir_graph.QNN_DATATYPE_STRING,
        }
        return _numpy_to_qnn_datatype[dtype]

# Convert QNN data type to numpy
def qnn_to_numpy_datatype(dtype):
        qnn_to_numpy_datatype = {
            ir_graph.QNN_DATATYPE_INT_8: numpy.dtype('int8'),
            ir_graph.QNN_DATATYPE_INT_16: numpy.dtype('int16'),
            ir_graph.QNN_DATATYPE_INT_32: numpy.dtype('int32'),
            ir_graph.QNN_DATATYPE_INT_64: numpy.dtype('int64'),
            ir_graph.QNN_DATATYPE_UINT_8: numpy.dtype('uint8'),
            ir_graph.QNN_DATATYPE_UINT_16: numpy.dtype('uint16'),
            ir_graph.QNN_DATATYPE_UINT_32: numpy.dtype('uint32'),
            ir_graph.QNN_DATATYPE_UINT_64: numpy.dtype('uint64'),
            ir_graph.QNN_DATATYPE_FLOAT_16: numpy.dtype('float16'),
            ir_graph.QNN_DATATYPE_FLOAT_32: numpy.dtype('float32'),
            ir_graph.QNN_DATATYPE_BOOL_8: numpy.dtype('bool'),
            ir_graph.QNN_DATATYPE_STRING: numpy.dtype('str'),
        }
        return qnn_to_numpy_datatype[dtype]

# Get Quantized datatype
def get_datatype_from_qinfo(bw, is_sym):
    dtype_map = {
        (4, True): ir_graph.QNN_DATATYPE_SFIXED_POINT_4,
        (4, False): ir_graph.QNN_DATATYPE_UFIXED_POINT_4,
        (8, True): ir_graph.QNN_DATATYPE_SFIXED_POINT_8,
        (8, False): ir_graph.QNN_DATATYPE_UFIXED_POINT_8,
        (16, True): ir_graph.QNN_DATATYPE_SFIXED_POINT_16,
        (16, False): ir_graph.QNN_DATATYPE_UFIXED_POINT_16,
        (32, True): ir_graph.QNN_DATATYPE_SFIXED_POINT_32,
        (32, False): ir_graph.QNN_DATATYPE_UFIXED_POINT_32,
    }
    dtype = dtype_map.get((bw, is_sym))
    if not dtype:
        raise KeyError(f"Unsupported bitwidth {bw}")
    return dtype

# Convert INT to FXP, since framework doesn't distinguish between
# int and fxp types
def int_to_fxp(int_dtype):
    dtype_map = {
        "int4"  :   "qint4",
        "int8"  :   "qint8",
        "int16" :   "qint16",
        "int32" :   "qint32",
        "uint4" :   "quint4",
        "uint8" :   "quint8",
        "uint16":   "quint16",
        "uint32":   "quint32",
    }
    fxp_dtype = dtype_map.get(int_dtype)
    if not fxp_dtype:
        raise KeyError(f"Unrecognized quant datatype {int_dtype}")
    return fxp_dtype

