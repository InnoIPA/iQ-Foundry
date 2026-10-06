# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import ast
import os
from argparse import Namespace
from typing import List, Optional, Tuple, Union

import yaml
from qti.aisw.converters.common.converter_base import ConverterFrontend
from qti.aisw.converters.common.model_validator import Validator
from qti.aisw.converters.common.utils.converter_utils import log_error, log_warning
from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory
from qti.aisw.tools.core.modules.converter.constants import *


def get_framework_extension(framework: str) -> str:
    if framework == OnnxFrameworkInfo.name:
        extension = OnnxFrameworkInfo.extensions[0]
    elif framework == PytorchFrameworkInfo.name:
        extension = PytorchFrameworkInfo.extensions[0]
    elif framework == TFLiteFrameworkInfo.name:
        extension = TFLiteFrameworkInfo.extensions[0]
    elif framework == TensorflowFrameworkInfo.name:
        extension = TensorflowFrameworkInfo.extensions[0]
    else:
        extension = ""
    return extension


def get_framework(extension: str) -> str:
    if OnnxFrameworkInfo.check_framework(extension):
        return OnnxFrameworkInfo.name
    elif PytorchFrameworkInfo.check_framework(extension):
        return PytorchFrameworkInfo.name
    elif TensorflowFrameworkInfo.check_framework(extension):
        return TensorflowFrameworkInfo.name
    elif TFLiteFrameworkInfo.check_framework(extension):
        return TFLiteFrameworkInfo.name
    else:
        raise Exception("Invalid model format specified. Supported types are {}".format(SUPPORTED_EXTENSIONS))


def get_frontend_converter(framework: str, args: dict,
                           validator: Optional[Validator] = None) -> ConverterFrontend:
    if not validator:
        validator = get_validator(framework, args)
    if framework == OnnxFrameworkInfo.name:
        if not args["use_onnx_relay"]:
            from qti.aisw.converters.onnx.onnx_to_ir import OnnxConverterFrontend
            return OnnxConverterFrontend(args, custom_op_factory=QnnCustomOpFactory(), validator=validator)
        else:
            try:
                # use onnx-relay-converter flow
                from qti.aisw.converters.onnx.onnx_to_ir_relay import OnnxRelayConverterFrontend
                return OnnxRelayConverterFrontend(args, custom_op_factory=QnnCustomOpFactory())
            except Exception as e:
                raise Exception(
                    "'use_onnx_relay' is not available. Please remove use_onnx_relay in converter input args.")
    elif framework == TensorflowFrameworkInfo.name:
        from qti.aisw.converters.tensorflow.tf_to_ir import TFConverterFrontend
        if not args["input_dim"] or not args["out_names"]:
            raise Exception("'source_model_input_shape' and 'out_tensor_node' are required for TensorFlow conversion")
        return TFConverterFrontend(args, custom_op_factory=QnnCustomOpFactory(), validator=validator)
    elif framework == TFLiteFrameworkInfo.name:
        from qti.aisw.converters.tflite.tflite_to_ir import TFLiteConverterFrontend
        return TFLiteConverterFrontend(args, custom_op_factory=QnnCustomOpFactory())
    else:
        from qti.aisw.converters.pytorch.pytorch_to_ir import PyTorchConverterFrontend
        from qti.aisw.converters.relay.custom_ops.utils.pytorch_helpers import PytorchCustomOpFactory
        if not args["input_dim"]:
            raise Exception("'source_model_input_shape' is required for PyTorch conversion")
        return PyTorchConverterFrontend(args, custom_op_factory=PytorchCustomOpFactory())


def infer_framework(input_network: str) -> str:

    model_path, model_ext = os.path.splitext(input_network)

    # tensorflow2 takes as input a folder which would have the ".pb" file

    if model_ext not in SUPPORTED_EXTENSIONS:
        model_files = os.listdir(model_path)
        for file in model_files:
            file_ext = os.path.splitext(file)[1]
            if file_ext == '.pb':
                model_ext = '.pb'

    if model_ext not in SUPPORTED_EXTENSIONS:
        raise Exception("Invalid model format specified. Supported types are {}".format(SUPPORTED_EXTENSIONS))
    framework = get_framework(model_ext)
    return framework


def get_num_tensor_configs(tensor_configs: Union[int, str]) -> int:
    # for when there is only one tensor config e.g. tensor_configs == (1,2,3)
    if isinstance(tensor_configs[0], int):
        return 1

    # for when input_dims is passed individually via CLI
    elif isinstance(tensor_configs, str):
        return 1

    # for when there is multiple tensor configs e.g. tensor_configs == ((1,2,3), (4,5,6))
    else:
        return len(tensor_configs)


def get_graph_configs(args: dict) -> list:
    def convert_dimensions_to_string(dims):
        return ",".join([str(dim) for dim in dims])

    num_configurations = get_num_graph_configs(args)
    configurations = []

    for i in range(num_configurations):
        configuration = []
        for tensor_name, tensor_configs in args["input_dim"]:
            if get_num_tensor_configs(tensor_configs) > 1:
                tensor_dims = convert_dimensions_to_string(tensor_configs[i])
            else:
                tensor_dims = convert_dimensions_to_string(tensor_configs)
            configuration.append([tensor_name, tensor_dims])
        configurations.append(configuration)
    return configurations


def get_num_graph_configs(args: dict) -> int:
    def validate_num_configs_is_1_or_n(num_tensor_configs_seen):
        error_message = "Error: Number of tensor configurations can either be 1 or N. \
                       You specified the following number of tensor configurations: {}" \
            .format(num_tensor_configs_seen)
        if len(num_tensor_configs_seen) > 2:
            log_error(error_message)
        elif len(num_tensor_configs_seen) == 2:
            if 1 not in num_tensor_configs_seen:
                log_error(error_message)

    if args["input_dim"] is None or args["input_dim"] == []:
        return 1

    num_tensor_configs_seen = set()
    for tensor_name, tensor_configs in args["input_dim"]:
        num_tensor_configs = get_num_tensor_configs(tensor_configs)
        num_tensor_configs_seen.add(num_tensor_configs)
    validate_num_configs_is_1_or_n(num_tensor_configs_seen)
    return max(num_tensor_configs_seen)


def get_validator(framework: str, args: dict) -> Validator:
    validator = None
    if ((framework == OnnxFrameworkInfo.name or framework == TensorflowFrameworkInfo.name)
            and args["validate_models"]):
        if args.converter_op_package_lib:
            log_warning("Model is having custom ops skipping validation.")
            args["validate_models"] = False
        else:
            validator = Validator()
    return validator


def set_graph_configs(args, config):
    args.input_dim = config


def generate_custom_io_config(tensor: dict) -> dict:
    """This method creates a custom io config from an InputTensor or OutputTensor dict

    Args:
        tensor: InputTensor or OutputTensor dict

    Returns:
        A dictionary containing the custom I/O config for the given tensor.
    """
    custom_io_config = dict()
    custom_io_config["IOName"] = tensor["name"]

    # Handle datatype for both input and output tensors
    datatype = tensor.get("desired_model_input_datatype") or tensor.get("desired_model_output_datatype")

    if datatype:
        custom_io_config["Datatype"] = datatype

    # Handle layout for both input and output tensors
    layout = tensor.get("desired_model_input_layout") or tensor.get("desired_model_output_layout")

    if layout:
        custom_io_config["Layout"] = {"Custom": layout}
        model_layout = tensor.get("source_model_input_layout") or tensor.get("source_model_output_layout")
        if model_layout:
            custom_io_config["Layout"]["Model"] = model_layout

    # Handle quant_param
    if tensor["quant_param"]:
        custom_io_config["QuantParam"] = {}
        custom_io_config["QuantParam"]["Scale"] = tensor["quant_param"]["scale"]
        custom_io_config["QuantParam"]["Offset"] = tensor["quant_param"]["offset"]
        custom_io_config["QuantParam"]["Type"] = "QNN_DEFINITION_DEFINED"

    # Handle optional
    if tensor["optional"]:
        custom_io_config["Optional"] = tensor["optional"]

    return custom_io_config


def convert_args_v2_to_v1(args: dict) -> dict:
    # input_dims is parsed as [['ip1', 'a,b,c,d'], ['ip1', 'd,e,f,g']]
    input_dims = None
    input_encoding = []
    input_layout = []
    input_dtype = []
    output_names = []
    user_custom_io = []
    # in case user provides multiple dimensions for an input, network specialization will be enabled (supported only
    # in onnx) and input_dims will be populated as [['ip1', ((a,b,c), (d,e,f))], ['ip2', ((a',b',c'), (d',e',f'))]]
    network_specialization = False

    if args["io_config"]:
        if isinstance(args["io_config"], list):
            io_tensor_configs = args["io_config"]
            for tensor in io_tensor_configs:
                custom_io_config = generate_custom_io_config(tensor)
                user_custom_io.append(custom_io_config)
        else:
            with open(os.fspath(args["io_config"])) as f:
                io_config_dict = yaml.safe_load(f)
            input_layout_dict = {}
            output_layout_dict = {}

            if "Input Tensor Configuration" in io_config_dict:
                for config in io_config_dict["Input Tensor Configuration"]:
                    name = config.get("Name")
                    if not name:
                        continue

                    src_params = config.get("Src Model Parameters", {})
                    desired_params = config.get("Desired Model Parameters", {})

                    if "DataType" in src_params and src_params["DataType"]:
                        input_dtype.append([name, src_params["DataType"]])
                    if "Layout" in src_params and src_params["Layout"]:
                        input_layout.append([name, src_params["Layout"]])
                        input_layout_dict[name] = src_params["Layout"]

                    if "Shape" in desired_params and desired_params["Shape"]:
                        if input_dims is None:
                            input_dims = []

                        shape = desired_params["Shape"]
                        # for cases when user passes a shape with one dimension
                        # e.g. Shape: 1
                        if isinstance(shape, int):
                            shape = f"({shape},)"
                        dim = ast.literal_eval(shape)
                        # for cases when user passes a shape with one dimension
                        # e.g. Shape: (1)
                        if isinstance(dim, int):
                            dim = (dim,)
                        if type(dim[0]) is tuple:
                            network_specialization = True
                        input_dims.append([name, dim])

                    custom_io_options = {"IOName": name}
                    if "DataType" in desired_params and desired_params["DataType"]:
                        custom_io_options["Datatype"] = desired_params["DataType"]
                    if "Layout" in desired_params and desired_params["Layout"]:
                        custom_io_options["Layout"] = {"Custom": desired_params["Layout"]}
                        # Get the model layout corresponding to the custom layout for current input
                        if name in input_layout_dict:
                            custom_io_options["Layout"]["Model"] = input_layout_dict[name]
                    if "QuantParams" in desired_params and (
                        desired_params["QuantParams"]["Scale"] or desired_params["QuantParams"]["Offset"]
                    ):
                        custom_io_options["QuantParam"] = desired_params["QuantParams"]
                        custom_io_options["QuantParam"]["Type"] = "QNN_DEFINITION_DEFINED"
                    if "Optional" in desired_params:
                        custom_io_options["Optional"] = desired_params["Optional"]
                    if len(custom_io_options) > 1:
                        user_custom_io.append(custom_io_options)
                    if "Color Conversion" in desired_params and desired_params["Color Conversion"]:
                        input_encoding.append([name, desired_params["Color Conversion"]])

            if "Output Tensor Configuration" in io_config_dict:
                for config in io_config_dict["Output Tensor Configuration"]:
                    name = config.get("Name")
                    if not name:
                        continue
                    output_names.append(name)

                    src_params = config.get("Src Model Parameters", {})
                    desired_params = config.get("Desired Model Parameters", {})

                    if "Layout" in src_params and src_params["Layout"]:
                        output_layout_dict[name] = src_params["Layout"]

                    custom_io_options = {"IOName": name}
                    if "Layout" in desired_params and desired_params["Layout"]:
                        custom_io_options["Layout"] = {"Custom": desired_params["Layout"]}
                        if name in output_layout_dict:
                            custom_io_options["Layout"]["Model"] = output_layout_dict[name]
                    if "DataType" in desired_params and desired_params["DataType"]:
                        custom_io_options["Datatype"] = desired_params["DataType"]
                    if "QuantParams" in desired_params and (
                        desired_params["QuantParams"]["Scale"] or desired_params["QuantParams"]["Offset"]
                    ):
                        custom_io_options["QuantParam"] = desired_params["QuantParams"]
                        custom_io_options["QuantParam"]["Type"] = "QNN_DEFINITION_DEFINED"
                    if "Optional" in desired_params:
                        custom_io_options["Optional"] = desired_params["Optional"]
                    if len(custom_io_options) > 1:
                        user_custom_io.append(custom_io_options)

    # update following args only if they were not provided on the commandline
    if not args['input_dim']:
        # convert name:str, dim:tuple to name:str, dim:str if network specialization is disabled
        if input_dims and not network_specialization:
            for i in range(len(input_dims)):
                # convert tuple of dimension to comma separated string
                if type(input_dims[i][1]) is tuple:
                    input_dims[i][1] = ','.join(map(str, input_dims[i][1]))
                # remove whitespaces if any from string of dimension
                elif isinstance(input_dims[i][1], str):
                    input_dims[i][1] = input_dims[i][1].replace(" ", "")

        args["input_dim"] = input_dims

    if not args['input_layout']:
        args['input_layout'] = input_layout
    if not args['input_dtype']:
        args['input_dtype'] = input_dtype
    if not args['input_encoding']:
        args['input_encoding'] = input_encoding

    # following arguments will be unused
    args['input_type'] = []
    args['dump_custom_io_config_template'] = ""

    if not args['out_names']:
        args['out_names'] = output_names

    args['user_custom_io'] = user_custom_io

    # populate preserve_io_arg with [['layout']] to apply it to all inputs/outputs
    args['preserve_io'] = [['layout']]
    if args["disable_preserve_io"]:
        args['preserve_io'] = []
    if args["preserve_io_datatype"]:
        args['preserve_io'].append(['datatype'] + args["preserve_io_datatype"][0])

    return args

def ts_to_onnx(args_dict: dict) -> None:
    from qti.aisw.converters.pytorch.torchscript_to_onnx import to_onnx

    args = Namespace(**args_dict)

    try:
        onnx_model_name = to_onnx(args)
    except Exception as e:
        raise RuntimeError(
            "Converter would convert Torchscript into Onnx, but it failed to convert to onnx!"
        ) from e

    if args.output_path is None:
        args.output_path = args.input_network.rpartition(".")[0] + ".dlc"

    args.input_network = onnx_model_name
    args.perform_sequence_construct_optimizer = True
    args_dict.update(vars(args))


def parse_shape(shape: Union[str, int, list, tuple]) -> Tuple[int, ...]:
    """Parses a shape definition into a tuple of integers.

    Args:
        shape (Union[str, int, list, tuple]): The shape to parse.

    Returns:
        Tuple[int, ...]: The parsed shape as a tuple of integers.

    Raises:
        ValueError: If the shape string is malformed or contains non-integer values.
        TypeError: If the shape type is unsupported.
    """
    if isinstance(shape, str):
        try:
            parsed = tuple(int(s.strip()) for s in shape.strip("()").split(","))
        except ValueError:
            raise ValueError(f"Invalid shape format: {shape}")
    elif isinstance(shape, int):
        parsed = (shape,)
    elif isinstance(shape, (list, tuple)):
        parsed = tuple(shape)
    else:
        raise TypeError(f"Unsupported shape type: {type(shape)}")

    if not all(isinstance(dim, int) for dim in parsed):
        raise ValueError(f"Shape list/tuple must contain only integers: {parsed}")

    return parsed


def get_input_dims_from_yaml(
    io_config: Union[str, os.PathLike],
) -> List[Tuple[str, Union[Tuple[int, ...], List[Tuple[int, ...]]]]]:
    """Extracts input tensor dimensions from a YAML I/O configuration file.

    Args:
        io_config (Union[str, os.PathLike]): Path to the YAML file containing input tensor configuration.

    Returns:
        List[Tuple[str, Union[Tuple[int, ...], List[Tuple[int, ...]]]]]: A list of tuples, each containing
        the tensor name and its shape as a tuple or list of tuples.
    """
    input_dims = []

    with open(os.fspath(io_config)) as f:
        io_config_dict = yaml.safe_load(f)

    if "Input Tensor Configuration" in io_config_dict:
        for config in io_config_dict["Input Tensor Configuration"]:
            name = config.get("Name")
            if not name:
                continue

            desired_params = config.get("Desired Model Parameters", {})
            shape = desired_params.get("Shape")

            if shape is not None:
                if isinstance(shape, str):
                    # Handle multiple shapes in a single string
                    shape_parts = [s.strip() for s in shape.split("),")]
                    parsed_shapes = []
                    for part in shape_parts:
                        if not part.endswith(")"):
                            part += ")"
                        parsed_shapes.append(parse_shape(part))
                    shape = parsed_shapes if len(parsed_shapes) > 1 else parsed_shapes[0]
                elif isinstance(shape, int):
                    shape = parse_shape(shape)
                elif isinstance(shape, (list, tuple)):
                    shape = (
                        [parse_shape(s) for s in shape]
                        if isinstance(shape[0], (list, tuple))
                        else parse_shape(shape)
                    )

                input_dims.append((name, shape))

    return input_dims
