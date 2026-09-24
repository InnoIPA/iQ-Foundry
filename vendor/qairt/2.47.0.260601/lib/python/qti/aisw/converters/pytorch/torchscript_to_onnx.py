# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import copy
import numpy as np
import os
import onnx
import pathlib
import tempfile
import torch
from onnx.numpy_helper import to_array
from packaging.version import parse

torch_ver = torch.__version__
if parse(torch_ver) < parse("2.2"):
    raise ValueError(
        "To support Torchscript Converter, the PT version needs >= 2.2, but got {}".format(
            torch_ver
        )
    )
import torchvision  # noqa


# RAND_FLOAT_SCALE is for making random sample floating input value's range in [0, RAND_FLOAT_SCALE) to avoid the range too narrow.
RAND_FLOAT_SCALE = 10
# OPSET_VERSION is for the onnx opset used in torch.onnx.export
OPSET_VERSION = 17
# Create the mapping from config path to op def collection
CONFIG_PATH_TO_OP_DEF_COLLECTION_MAPPING = {}


def _get_torch_dtype(dtype: str):
    """
    Get torch dtype by string.

    Args:
        dtype: data type in string format.

    Returns:
        torch dtype: data type in torch.
    """
    torch_dtype = {
        "int32": torch.int32,
        "int64": torch.int64,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype in torch_dtype:
        return torch_dtype[dtype]
    else:
        raise ValueError(f"{dtype} is not included in torch_dtype")


def modify_onnx_attribute_type(attr):
    """
    Modify the data type of onnx attribute from tensor type to int/float/string type.

    Args:
        attr: Onnx AttributeProto.

    Returns:
        attr: Onnx AttributeProto.
    """
    value = to_array(attr.t)

    if parse(onnx.version.version) >= parse("1.19.0"):
        tensor_dtype_str = onnx.helper.tensor_dtype_to_string(attr.t.data_type)
    else:
        tensor_dtype_str = onnx.mapping.TENSOR_TYPE_MAP[attr.t.data_type].name

    # float
    if tensor_dtype_str in [
        "TensorProto.FLOAT",
        "TensorProto.FLOAT16",
        "TensorProto.DOUBLE",
        "TensorProto.BFLOAT16",
    ]:
        attr.type = onnx.AttributeProto.FLOAT
        attr.f = float(value)
    # string
    elif tensor_dtype_str == "TensorProto.STRING":
        attr.type = onnx.AttributeProto.STRING
        attr.s = str(value).encode("utf-8")
    # int / boolean
    else:
        attr.type = onnx.AttributeProto.INT
        attr.i = int(value)
    return attr


def update_op_def_collection():
    """
    Update the op def collection stored in CONFIG_PATH_TO_OP_DEF_COLLECTION_MAPPING.
    """
    for op_def_collection in CONFIG_PATH_TO_OP_DEF_COLLECTION_MAPPING.values():
        for op_def_name in op_def_collection.op_def_dict:
            op_def = op_def_collection.op_def_dict[op_def_name]
            # Change the op type field to match the onnx op type
            new_op_def = copy.deepcopy(op_def)
            # There is no "<domain>::" in onnx op type, so we need to remove it in new op def
            new_op_def_name = op_def_name.split("::")[-1]
            new_op_def.name = new_op_def_name
            del op_def_collection.op_def_dict[op_def_name]
            op_def_collection.op_def_dict[new_op_def_name] = new_op_def

            # Change the op type in supported and supplemental dict
            for backend in op_def_collection.get_supported_backends():
                if op_def_collection.is_supported(op_def_name, backend):
                    for runtime in op_def_collection.get_supported_runtimes(backend) + [
                        op_def_collection.ALL
                    ]:
                        op_def_collection.supported_ops[backend][runtime].remove(
                            op_def_name
                        )
                        op_def_collection.supported_ops[backend][runtime].append(
                            new_op_def_name
                        )
                        other_runtime = (
                            runtime if runtime != op_def_collection.ALL else None
                        )
                        if op_def_collection.is_supplemented(
                            op_def_name, backend, other_runtime
                        ):
                            new_supp_op_def_list = []
                            for supp_op_def in op_def_collection.supplemental_op_def[
                                backend
                            ][runtime][op_def_name]:
                                new_supp_op_def = copy.deepcopy(supp_op_def)
                                new_supp_op_def.name = new_op_def_name
                                new_supp_op_def_list.append(new_supp_op_def)
                            del op_def_collection.supplemental_op_def[backend][runtime][
                                op_def_name
                            ]
                            op_def_collection.supplemental_op_def[backend][runtime][
                                new_op_def_name
                            ] = new_supp_op_def_list


def write_op_def_collection(args, custom_op_config_paths):
    """
    Write new op def to XML files.

    Args:
        args: ArgParser.
        custom_op_config_paths: path to original config files (JSON/XML).
    """
    import qti.aisw.op_package_generator.translator.op_def_translator as xml_package_translator

    SCHEMA_PATH = os.path.abspath(os.path.dirname(xml_package_translator.__file__))
    SCHEMA = os.path.join(
        SCHEMA_PATH, xml_package_translator.OpDefTranslator.DEFAULT_SCHEMA
    )
    translator_instance = xml_package_translator.OpDefTranslator(xml_schema=SCHEMA)
    for (
        config_path,
        op_def_collection,
    ) in CONFIG_PATH_TO_OP_DEF_COLLECTION_MAPPING.items():
        idx = custom_op_config_paths.index(config_path)
        new_config_path = config_path.replace(".xml", "_onnx.xml")
        # Also update the path in args
        args.custom_op_config_paths[idx] = new_config_path
        translator_instance.write_op_defs(op_def_collection, new_config_path)


def preprocess_custom_op(args):
    """
    Preprocess Custom Op by parsing config file to op def collection,
    recording the op def collection, importing the custom onnx export extension,
    updating op type in the op def collection, and writing the new op config.

    Args:
        args: argparse.Namespace.

    Returns:
        custom_opsets: dict[str, int].
    """
    # According to pytorch source,
    # A dict with schema:
    #   KEY (str): opset domain name
    #   VALUE (int): opset version
    custom_opsets = {}
    if (
        hasattr(args, "custom_op_config_paths")
        and args.custom_op_config_paths is not None
    ):
        custom_op_config_paths = args.custom_op_config_paths
        try:
            from qti.aisw.converters.snpe_backend.custom_ops.snpe_udo_config import (
                UdoGenerator,
            )

            package_generator = UdoGenerator()
        except:
            from qti.aisw.op_package_generator.generator import QnnOpPackageGenerator

            package_generator = QnnOpPackageGenerator()

        for config_path in custom_op_config_paths:
            # Parse config file to op def collection
            config_path, op_def_collection = (
                package_generator.parse_config_to_op_def_collection(config_path)
            )
            # Record the op def collection
            CONFIG_PATH_TO_OP_DEF_COLLECTION_MAPPING[config_path] = op_def_collection
            # Set custom opset
            for op_def_name in op_def_collection.op_def_dict.keys():
                domain = op_def_name.split("::")[0]
                if (
                    domain not in ["aten", "prim", "onnx"]
                    and domain not in custom_opsets
                ):
                    custom_opsets[domain] = OPSET_VERSION

    # Import custom_onnx_export file after parsing custom op config,
    # so that the pytorch custom op type can be obtained in custom_onnx_export
    import qti.aisw.converters.pytorch.custom_onnx_export  # noqa
    from qti.aisw.converters.pytorch.custom_onnx_export import register_custom_op

    register_custom_op()

    if CONFIG_PATH_TO_OP_DEF_COLLECTION_MAPPING:
        # Since the op type of onnx custom op may be different from pytorch custom op,
        # the op def collection needs to be updated to conform to the new op type.
        update_op_def_collection()

        # Write the new op config whose op type matches onnx format
        write_op_def_collection(args, custom_op_config_paths)

    return custom_opsets


def postprocess_custom_op(onnx_model_path):
    """
    Postprocess Custom Op by separating parameters from the inputs with the help of
    custom config file.

    Args:
        onnx_model_path: The path where the exported onnx model is located.
    """
    if CONFIG_PATH_TO_OP_DEF_COLLECTION_MAPPING:
        from qti.aisw.converters.onnx.util import (
            get_graphs,
            get_nodes,
            save_model,
        )

        # Load exported model
        model = onnx.load(onnx_model_path)
        # Record op def
        op_def_dict = {}
        for op_def_collection in CONFIG_PATH_TO_OP_DEF_COLLECTION_MAPPING.values():
            op_def_dict.update(op_def_collection.op_def_dict)

        # Iterate over each node in the model and
        # separate parameters from the inputs with the help of custom config file.
        const_node_dict = {}
        const_node_to_be_removed_list = []
        for node in get_nodes(model):
            onnx_op_type = node.op_type
            # Store Constant op for further use
            if node.op_type == "Constant":
                const_node_dict[node.output[0]] = node
            # When this op is a custom op,
            # map input that is supposed to be attribute to parameter
            # with name specified in config.
            if onnx_op_type in op_def_dict:
                op_def = op_def_dict[onnx_op_type]
                node_inputs = list(copy.deepcopy(node.input))
                param_count = 0
                for inp in node_inputs:
                    if not inp:
                        node.input.remove(inp)
                    elif param_count != len(op_def.parameters):
                        if inp in const_node_dict:
                            const_node = const_node_dict[inp]
                            # Extract the value from constant node
                            attr = list(const_node.attribute)[0]
                            # When the tensor has no dimensions, it means that
                            # the tensor is of type boolean or int or float or string.
                            if attr.type == attr.TENSOR and not attr.t.dims:
                                # Change attr to correct type
                                attr = modify_onnx_attribute_type(attr)
                            # Reset attr name from op def
                            attr.name = op_def.parameters[param_count].name
                            # Add new attribute to the current custom node
                            node.attribute.append(attr)
                            param_count += 1
                            # Remove the attribute from inputs
                            node.input.remove(inp)
                            if const_node not in const_node_to_be_removed_list:
                                const_node_to_be_removed_list.append(const_node)
        # Remove the constant node that is now an attribute from the graph
        for graph in get_graphs(model):
            tmp_const_node_to_be_removed_list = copy.deepcopy(
                const_node_to_be_removed_list
            )
            for const_node in tmp_const_node_to_be_removed_list:
                if const_node in graph.node:
                    graph.node.remove(const_node)
                    const_node_to_be_removed_list.remove(const_node)
            if not const_node_to_be_removed_list:
                break
        # Save new onnx model
        save_model(model, onnx_model_path)


def to_onnx(args):
    """
    Convert Torchscript to Onnx.

    Args:
        args: argument given from users.

    Returns:
        onnx_model_path: saved onnx model's path.
    """
    # Handle customOp
    if args.pytorch_custom_op_lib:
        for lib in args.pytorch_custom_op_lib.split(","):
            torch.ops.load_library(lib)
    custom_opsets = preprocess_custom_op(args)

    # Load Torchscript Model
    torch_model = torch.jit.load(args.input_network, map_location="cpu").eval()

    # Prepare Onnx Model's name
    model_name = pathlib.Path(args.input_network).stem
    tmp_file = tempfile.NamedTemporaryFile(
        prefix=f"{model_name}_",
        suffix=f"_exportedOnnx_.onnx",
        delete=False
    )
    onnx_model_path = tmp_file.name

    # Fetch user's given input dims and dtype
    dtype_dict = {}
    if args.input_dtype:
        for in_name, in_dtype in args.input_dtype:
            dtype_dict[in_name] = _get_torch_dtype(in_dtype)

    input_info = {}
    input_names = []
    for in_name, in_dims in args.input_dim:
        input_names.append(in_name)
        in_dims = [int(i) for i in in_dims.split(",")]
        if in_name in dtype_dict:
            input_info[in_name] = [in_dims, dtype_dict[in_name]]
        else:
            input_info[in_name] = [in_dims, torch.float32]

    # To run torch.onnx.export, we will prepare the dummy input data.
    sample_inputs = []
    for name in input_names:
        if dtype_dict:
            if "int" in str(input_info[name][1]):
                torch_tensor = torch.randint(
                    high=1, size=input_info[name][0], dtype=input_info[name][1]
                )
            else:
                torch_tensor = (
                    torch.rand(*input_info[name][0], dtype=input_info[name][1])
                    * RAND_FLOAT_SCALE
                )
        else:
            torch_tensor = torch.rand(*input_info[name][0]) * RAND_FLOAT_SCALE
        sample_inputs.append(torch_tensor)

    torch.onnx.export(
        torch_model,
        sample_inputs,
        onnx_model_path,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=input_names,
        custom_opsets=custom_opsets,
    )

    # Remove redundant input in arguments after exporting the onnx model.
    model = onnx.load(onnx_model_path)
    onnx_input = [node.name for node in model.graph.input]

    args.input_dtype = [
        in_dtype for in_dtype in args.input_dtype if in_dtype[0] in onnx_input
    ]
    args.input_dim = [in_dim for in_dim in args.input_dim if in_dim[0] in onnx_input]

    postprocess_custom_op(onnx_model_path)

    return onnx_model_path
