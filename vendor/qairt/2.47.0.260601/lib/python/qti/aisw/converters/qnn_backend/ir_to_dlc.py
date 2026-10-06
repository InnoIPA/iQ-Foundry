# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
import os
import sys
import argparse

# modeltools is present in common as well as dlc_utils
# try importing from common first (currently used by QNN) and if not found import from dlc_utils (used by SNPE)
# TODO: remove modeltools from dlc_utils and update all SNPE tools to use modeltools from common
try:
    from qti.aisw.converters.common import modeltools
except ImportError as ie1:
    from qti.aisw.dlc_utils import modeltools

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common import ir_quantizer
from qti.aisw.converters.common.backend_base import BackendTranslationBase
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import lora_utils, validation_utils
from qti.aisw.converters.common.utils.validation_utils import ExportFormatType
from qti.aisw.converters.common.utils.translation_utils import get_si_notation
from qti.aisw.converters.qnn_backend.qnn_translations import QnnTranslations
from qti.aisw.converters.qnn_backend.qnn_backend_base import QnnConverterBackendBase
from qti.aisw.converters.qnn_backend.qnn_mappings import *
from qti.aisw.converters.common.graph_optimizer import GraphOptimizer, OptimizationStage, OptimizationPassModeManager


from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory as CustomFactory



# TODO: updated inheritance to ConverterBackend once alignment of Ops are complete
class DLCBackend(QnnConverterBackendBase):
    class ArgParser(QnnConverterBackendBase.ArgParser):
        def __init__(self, **kwargs):
            super(DLCBackend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--model_version', type=str, default=None,
                                       help='User-defined ASCII string to identify the model, only first '
                                            '64 bytes will be stored')
            self.add_optional_argument('--validation_target', nargs=2,
                                       action=validation_utils.ValidateTargetArgs,
                                       help="Note: This option is deprecated. \n"
                                            "A combination of processor and runtime target against which model "
                                            "will be validated. \n"
                                            "Choices for RUNTIME_TARGET: \n   {cpu, gpu, dsp}. \n"
                                            "Choices for PROCESSOR_TARGET: \n"
                                            "   {snapdragon_801, snapdragon_820, snapdragon_835}.\n"
                                            "If not specified, will validate model against "
                                            "{snapdragon_820, snapdragon_835} across all runtime targets.",
                                       metavar=('RUNTIME_TARGET', 'PROCESSOR_TARGET'),
                                       default=[], )
            self.add_optional_argument('--strict', dest="enable_strict_validation",
                                       action="store_true",
                                       default=False,
                                       help="Note: This option is deprecated. \n"
                                            "If specified, will validate in strict mode whereby model will not "
                                            "be produced if it violates constraints of the specified validation "
                                            "target. If not specified, will validate model in permissive mode "
                                            "against the specified validation target.")
            self.add_optional_argument("--udo_config_paths", "-udo", nargs='+',
                                       dest="custom_op_config_paths",
                                       action=validation_utils.check_json(),
                                       help="Path to the UDO configs (space separated, if multiple)")

    class ArgParserv2(QnnConverterBackendBase.ArgParserv2):
        def __init__(self, **kwargs):
            super(DLCBackend.ArgParserv2, self).__init__(**kwargs)
            self.add_optional_argument('--model_version', type=str, default=None,
                                       help=argparse.SUPPRESS)
            self.add_optional_argument('--set_model_version', dest='model_version', type=str, default=None,
                                       help='User-defined ASCII string to identify the model, only first '
                                            '64 bytes will be stored')
            self.add_optional_argument('--disable_qnn_op_config_validation', action='store_true',
                                       help=argparse.SUPPRESS, default=False)
            self.add_optional_argument("--export_format", default="DLC_DEFAULT",
                                       action=validation_utils.validate_export_format_option(),
                                       help='DLC_DEFAULT (default)\n'
                                            '- Produce a Float graph given a Float Source graph\n'
                                            '- Produce a Quant graph given a Source graph with provided Encodings\n'
                                            'DLC_STRIP_QUANT\n'
                                            '- Produce a Float Graph with discarding Quant data\n')

    def __init__(self, args):
        super(DLCBackend, self).__init__(args)
        # get converter args for saving dlc
        if self.output_model_path is None:
            filename, _ = os.path.splitext(os.path.realpath(self.input_model_path))
            self.output_path = filename + ".dlc"
        else:
            self.output_path = self.output_model_path

        if '--model_version' in sys.argv:
            log_warning("--model_version option is deprecated, use --set_model_version.")

        self.args = args
        self.model_version = args.model_version
        self.serialize_with_suppl_attr = True
        self.tensor_hashes = {}
        self.context_static_tensors_name_to_id_map = {}

        if hasattr(args, 'validation_target'):
            self.validation_target = args.validation_target
            if args.validation_target:
                log_warning("--validation_target is deprecated.")
        if hasattr(args, 'strict'):
            self.enable_strict_validation = args.enable_strict_validation
            if args.enable_strict_validation:
                log_warning("--strict is deprecated.")

        self.do_qnn_op_config_validation = True
        if hasattr(args, 'disable_qnn_op_config_validation'):
            self.do_qnn_op_config_validation = not args.disable_qnn_op_config_validation

        # Ensure model version fits in 64 bytes to match dlcv3
        model_version = self.model_version
        if model_version:
            model_version = model_version[:64]
        else:
            model_version = ''

        self.optimizationPassManager = OptimizationPassModeManager.get_instance()

        self.applied_float_fallback = False

        self.dump_qairt_quantizer_command = ""
        if '--dump_qairt_quantizer_command' in sys.argv:
            self.dump_qairt_quantizer_command = args.dump_qairt_quantizer_command

        is_strip_ir_data = False
        if self.qairt_converter and self.args.export_format == 'DLC_DEFAULT':
            is_strip_ir_data = True

        self.dlc_serializer = modeltools.IrDlcSerializer(self.output_path,
                                                         self.copyright_str,
                                                         model_version,
                                                         self.converter_command,
                                                         "",
                                                         is_strip_ir_data)

    # TODO: Cleanup when all ops are aligned to QNN
    """ Start of clean up """
    def add_tensor(self, node_name, tensor_name, tensor_type, tensor: np.ndarray,
                   check_encodings=True, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32,
                   src_axis_format=None, tensor_axis_format=None,
                   permute_order_to_src=None, orig_tensor_name=None, is_bias=False,
                   transform_manager=None):

        is_quantizable = True
        if tensor_data_type != ir_graph.QNN_DATATYPE_FLOAT_32 or not check_encodings:
            is_quantizable = False

        data = None
        if tensor_type == qnn_definitions.QNN_TENSOR_TYPE_STATIC:
            data = tensor
        tensor_info = self.create_tensor_info(
            tensor_name,
            tensor_type,
            tensor.shape,
            tensor_data_type,
            src_axis_format,
            tensor_axis_format,
            permute_order_to_src=permute_order_to_src,
            data=data,
            encoding=None,
            is_bias=is_bias,
            quantizable=is_quantizable
        )

        if transform_manager:
            tensor_info["transform_manager"] = transform_manager

        if not self.model.add_tensor(node_name, tensor_info, is_quantizable=is_quantizable):
            raise RuntimeError("Adding Tensor {} for Node {} failed.".format(node_name, tensor_name))

    def add_lazy_tensor(
        self,
        node_name,
        tensor_name,
        tensor_type,
        tensor_shape,
        ext_data_info,
        check_encodings=True,
        tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32,
        src_axis_format=None,
        tensor_axis_format=None,
        permute_order_to_src=None,
        orig_tensor_name=None,
        is_bias=False,
        transform_manager=None
    ):
        is_quantizable = True
        if tensor_data_type != ir_graph.QNN_DATATYPE_FLOAT_32 or not check_encodings:
            is_quantizable = False

        tensor_info = self.create_tensor_info(
            tensor_name,
            tensor_type,
            tensor_shape,
            tensor_data_type,
            src_axis_format,
            tensor_axis_format,
            permute_order_to_src=permute_order_to_src,
            data=None,
            encoding=None,
            is_bias=is_bias,
            quantizable=is_quantizable
        )

        if transform_manager:
            tensor_info["transform_manager"] = transform_manager

        if not self.model.add_lazy_tensor(node_name, tensor_info, ext_data_info, is_quantizable=is_quantizable):
            raise RuntimeError("Adding Lazy Tensor {} for Node {} failed.".format(node_name, tensor_name))

    def add_custom_input_tensor(
        self,
        node_name,
        tensor_name,
        tensor_type,
        tensor: np.ndarray,
        tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32,
        tensor_axis_format=None,
        permute_order_to_src=None,
        quant_params=None,
        params_count=0
    ):
        """
        Function to add a tensor with the quant_params obtained from Custom IO YAML file.
        :param node_name: the IRGraph name for node.
        :param tensor_name: name to use for the tensor
        :param tensor_type: the QNN tensor type. (i.e: NATIVE, APP_WRITE,...)
        :param tensor: np.ndarray object
        :param tensor_data_type: the data type to use for the tensor
        :param tensor_axis_format: the axis format of the QNN tensor
        :param permute_order_to_src: the permute order to transform QNN tensor back to src format
        :param quant_params: Dictionary containing information regarding the scale and offset
                            of custom input tensor.
        :param params_count: the size of weights for the operation, if applicable
        """

        # TODO: Directly accept FXP8 from the config file rather than combination
        #       of INT8 and QuantParams to infer FXP

        if quant_params:
            tensor_data_type = custom_dtype_to_quant_dtype_map.get(tensor_data_type, tensor_data_type)

        tensor_info = self.create_tensor_info(
            tensor_name,
            tensor_type,
            tensor.shape,
            tensor_data_type,
            tensor_axis_format,
            permute_order_to_src=permute_order_to_src,
            data=None,
            encoding=None
        )
        tensor_info['quant_params'] = quant_params
        is_quantizable = False
        if quant_params:
            is_quantizable = True
        if not self.model.add_tensor(node_name, tensor_info, is_quantizable=is_quantizable):
            raise RuntimeError("Adding Tensor {} for Node {} failed.".format(node_name, tensor_name))

    def add_node(self, node_name, node_type, input_names, outputs_info, tensor_params={}, scalar_params={},
                 macs=0, backend_name=None):
        # resolve package names for each node name
        node_package_name = self.resolve_package_names(node_type)

        do_qnn_op_config_validation = self._get_qnn_op_config_validation(node_type)

        if not self.model.add_node(node_name, node_type, node_package_name, tensor_params, scalar_params,
                                   input_names, outputs_info, do_qnn_op_config_validation, backend_name):
            raise RuntimeError("Adding Node {} failed.".format(node_name))

    # Ops for which we need to skip QnnOpConfigValidation.
    # These ops don't have a QNN OpDef yet and are instead handeled by
    # lowering to other ops by an IrOptimizer pass. So while converting it
    # to IrOp (add_op()) we need to skip QnnOpConfig validation.
    # Format: {op_type : 'PASS_NAME'}
    # TODO: Remove entries once the corresponding QNN OpDefs are available.
    _OPS_WITHOUT_QNN_OPDEF_TO_PASS_MAP = {
        'Inverse' : 'EXPAND_INVERSE',
        'Mean'    : 'expand_elementwise_mean'
    }

    def _get_qnn_op_config_validation(self, node_type):
        """
        Decide whether QNN op config validation should be performed for the given node_type.

        Validation is skipped if:
        - The op is listed in _OPS_WITHOUT_QNN_OPDEF_TO_PASS_MAP, and
        - Its corresponding IrOptimizer PASS is enabled.

        :param node_type: str – the QNN translation key of the op.

        :return: bool – True if validation should run, False if it should be skipped.
        """
        if node_type in self._OPS_WITHOUT_QNN_OPDEF_TO_PASS_MAP and \
            self._OPS_WITHOUT_QNN_OPDEF_TO_PASS_MAP[node_type] in self.optimizationPassManager.get_cpp_ir_pass_list():
            log_debug("Setting QnnOpConfigValidation = False for {} op !!".format(node_type))
            return False

        return self.do_qnn_op_config_validation

    @staticmethod
    def sanitize_name(name):
        return name

    @staticmethod
    def _sanitize_tensor_name(tensor_name):
        return tensor_name

    """ End of clean up """

    # overrides the set_package_dict method in qnn_backend_base
    # to correctly set the package dict info for snpe 2.0 udo
    def set_package_dict(self, graph):
        if self.package_name:
            package_name_dict = {self.package_name: [node.op.type for node in graph.list_nodes()[1:]]}
        elif CustomFactory.package_resolver:
            package_name_dict = CustomFactory.package_resolver
        else:
            package_name_dict = dict()

        # if there is no package lib provided, then it is assumed that the default qti package will be
        # will used to quantize any custom ops.
        if self.op_package_lib:
            self.quantize_with_default_package = False

        self.package_name_to_qnn_op_types = package_name_dict

    # overrides the resolve_package_names method in qnn_backend_base
    # to correctly resolve the package names for snpe 2.0 udo
    def resolve_package_names(self, node_type):
        default_package_name = qnn_definitions.QNN_OP_PACKAGE_NAME_QTI_AISW
        package_names = [default_package_name]
        for package_name, node_types in self.package_name_to_qnn_op_types.items():
            if node_type.lower() in node_types:
                package_names.append(package_name)
        return package_names[-1]

    def apply_custom_io_dequant(self, graph):
        for entry in graph.user_custom_io:
            buffer_name = str(entry['IOName'])
            log_assert(buffer_name in graph.buffers,"Incorrect IOName provided in custom IO YAML file. Buffer {} not found in graph"
                       .format(buffer_name))
            if 'Datatype' in entry:
                if entry['Datatype'] not in ['int8', 'uint8']:
                    log_assert(self.c_ir_graph is None,"To pass non-quantized inputs/output to quantized model, use the --input_data_type/--output_data_type\
                        option of qnn-net-run. {} datatype provided for Buffer {}".format(entry['Datatype'], buffer_name))
            if "QuantParam" in entry:
                # Default datatype for quantized model is uint8 in case of custom IO.
                custom_datatype = 'uint8'
                if 'Datatype' in entry:
                    custom_datatype = entry['Datatype']
                if custom_datatype == 'int8':
                    log_assert(self.c_ir_graph is None,"Custom IO does not support int8 inputs to quantized model. int8 datatype provided for Buffer {}"
                               .format(buffer_name))
                isInput = False
                # Check if the buffer name provided is input buffer
                for node in graph.get_input_nodes_to_graph():
                    if buffer_name in node.output_names:
                        isInput = True
                #To handle the case when quantized custom inputs are to be provided to a non-quantized model
                if isInput and entry['QuantParam']['Type'] == 'QNN_DEFINITION_DEFINED':
                    consumers = [str(name) for name in graph.buffers[buffer_name].consumers]

                    # Insert a dequant op after the input node. The params for the dequant op are obtained from graph.quantization_params which
                    # is in-turn filled with the information obtained from the custom IO YAML file.
                    node = graph.buffers[buffer_name].producer
                    node.op.input_dtype = custom_datatype
                    dequant_op = op_adapter.DequantizeOp(buffer_name+"_dequant", bw=graph.quantization_params[buffer_name]['output_encodings'][0]['bw'],
                                                         scale=graph.quantization_params[buffer_name]['output_encodings'][0]['scale'][0],
                                                         offset=graph.quantization_params[buffer_name]['output_encodings'][0]['offset'][0],
                                                         is_symmetric=graph.quantization_params[buffer_name]['output_encodings'][0]['is_symmetric'])
                    graph.inject(dequant_op, buffer_name, buffer_name+"_custom_dequant", consumer_names=consumers)

                # Check if the buffer name provided is output buffer
                isOutput = False
                for node in graph.get_output_nodes_of_graph():
                    if buffer_name in node.output_names:
                        isOutput = True
                        break
                #To handle the case when quantized custom outputs are to be provided to a non-quantized model
                if isOutput and entry['QuantParam']['Type'] == 'QNN_DEFINITION_DEFINED':
                    # Insert a Convert op after the output node. The params for the Convert op are obtained from graph.quantization_params which
                    # is in-turn filled with the information obtained from the custom IO YAML file.
                    node = graph.buffers[buffer_name].producer
                    convert_buffer_name = buffer_name + "_custom_convert"
                    convert_name = buffer_name + "_custom_convert"
                    graph.change_buffer_name(buffer_name, convert_buffer_name)
                    qnn_dtype = numpy_dtype_to_qnn.get(np.dtype(custom_datatype))
                    convert_dtype = custom_dtype_to_quant_dtype_map.get(qnn_dtype, ir_graph.QNN_DATATYPE_FLOAT_32)
                    convert_op = op_adapter.ConvertOp(convert_name, to_type=convert_dtype)
                    graph.inject(convert_op, input_name=convert_buffer_name, output_name=buffer_name, consumer_names=[])
                    graph.add_quantization_params(convert_name,  output_encodings=graph.quantization_params[node.op.name]['output_encodings'][0])
                    graph.remove_quantization_params(node.op.name)

    def cleanup_custom_io_qparams(self, graph):
        for entry in graph.user_custom_io:
            buffer_name = str(entry['IOName'])
            if "QuantParam" not in entry:
                continue

            isInput = False
            # Check if the buffer name provided is input buffer
            for node in graph.get_input_nodes_to_graph():
                if buffer_name in node.output_names:
                    isInput = True

            if (isInput and entry['QuantParam']['Type'] == 'QNN_DEFINITION_DEFINED' and
                    buffer_name in graph.quantization_params):
                graph.remove_quantization_params(buffer_name)

            isOutput = False
            for node in graph.get_output_nodes_of_graph():
                if buffer_name in node.output_names:
                    isOutput = True
                    break

            custom_convert_buffer_name = buffer_name + "_custom_convert"
            if (isOutput and entry['QuantParam']['Type'] == 'QNN_DEFINITION_DEFINED' and
                    custom_convert_buffer_name in graph.quantization_params):
                graph.remove_quantization_params(custom_convert_buffer_name)

    def dump_qairt_cmdline_io_config(self, graph):
        """
        Dumps QAIRT equivalent Commandline Arguments and IO Config File (which can be provided to the QAIRT Converter) based
        on the currently provided Commandline Arguments.
        :param graph: IROpGraph object
        """
        if graph.dump_qairt_io_config_yaml:
            yaml_dump_dir = os.path.dirname(os.path.abspath(self.output_path))
            yaml_file_name = yaml_dump_dir + "/" + graph.dump_yaml_file_name
            f = open(yaml_file_name, 'w')
            f.write('\n'.join(graph.dump_yaml_file_data))
            log_info("Dumped IO config at: %s " % yaml_file_name)
            f.close()
            print("\n------------------------------QAIRT Converter Commandline------------------------------------------------------------------------------------------")
            print(graph.qairt_converter_command)
            print("\nNote: IO Config file is generated at:", yaml_file_name)
            print("---------------------------------------------------------------------------------------------------------------------------------------------------")

        if self.dump_qairt_quantizer_command:
            if os.path.exists(self.dump_qairt_quantizer_command):
                f = open(self.dump_qairt_quantizer_command, "r")
                print("\n------------------------------QAIRT Quantizer Commandline------------------------------------------------------------------------------------------")
                print(f.read())
                print("---------------------------------------------------------------------------------------------------------------------------------------------------\n")
                f.close()

    def initialize(self):
        self.dlc_serializer.initialize()
        log_info(code_to_message.get_progress_message("INFO_INITIALIZATION_SUCCESS"))
        self.num_graphs_in_dlc = 0

    def prepare_py_graph(self, graph):
        # set up the package information for each op type in the graph
        self.set_package_dict(graph)

        # To handle the case when quantized custom inputs are to be provided to the model
        if graph.user_custom_io:
            # Incase if G2G optimization is being triggered, then defer this call here
            if 'PROCESS_CUSTOM_IO_QUANT_INFO' not in self.optimizationPassManager.get_cpp_ir_pass_list():
                self.apply_custom_io_dequant(graph)

        return graph

    def prepare_cpp_graph(self, graph, ir_graph, network_specialization: bool = False):
        if network_specialization:
            # This is Network Specialization case.
            # Update graph name to identify the index of the input shape config used
            ir_graph_tensor_map = ir_graph.get_tensor_map()
            for tensor in ir_graph_tensor_map.values():
                if tensor.is_static():
                    # Set all static tensors of 1st graph as context static
                    if self.num_graphs_in_dlc == 1:
                        tensor.set_tensor_type_as_context_static()

                    if tensor.name() not in self.context_static_tensors_name_to_id_map:
                        self.create_hash_tensor_data(tensor)
                        self.context_static_tensors_name_to_id_map[tensor.name()] = tensor.id()
                    else:
                        status = self.validate_hash_tensor_data(tensor)
                        # If status is True, static tensor can be converted to context static
                        if status:
                            # Set tensor ID same for context static tensors with same name
                            # QnnGraphComposer matches (tensor id, tensor name) to set context static
                            tensor.set_id(self.context_static_tensors_name_to_id_map[tensor.name()])

                            # Set tensor type as context static
                            tensor.set_tensor_type_as_context_static()

            ir_graph.set_name(ir_graph.name + "_configuration_" + str(self.num_graphs_in_dlc))

        if graph.enable_trace:
            self.model.validation_framework_tracing_completeness("prepare_cpp_graph")

        if graph.user_custom_io:
            self.cleanup_custom_io_qparams(graph)
            # Prepare Custom IO entries for cpp graph, which will be used while applying G2G optimizations
            if ('PROCESS_CUSTOM_IO_QUANT_INFO' in self.optimizationPassManager.get_cpp_ir_pass_list() and
               (hasattr(self.args, "io_config") and self.args.io_config)):
                ir_graph.set_user_custom_io(graph.user_custom_io)

        return ir_graph

    def create_hash_tensor_data(self, tensor: ir_graph.IrStaticTensor):
        # hash_data computes hash on the data(if loaded)
        # In case of defer loading, it hashes the ExternalDataInfo metadata
        tensor_hash = tensor.hash_data()
        self.tensor_hashes[tensor.name()] = tensor_hash

    def validate_hash_tensor_data(self, tensor: ir_graph.IrStaticTensor):
        # hash_data computes hash on the data(if loaded)
        # returns True, if static tensor can be converted to context static
        # In case of defer loading, it hashes the ExternalDataInfo metadata
        tensor_hash = tensor.hash_data()
        status = False
        if tensor.name() in self.tensor_hashes:
            if self.tensor_hashes[tensor.name()] != tensor_hash:
                log_debug3(f"Context Static tensor data validation failed for {tensor.name()}: Tensor data mismatch")
            else:
                status = True
        return status

    def set_qairt_converter_float_fallback_default_args(self, quantizer_opts, backend_info_obj=None):
        """
        Sets the default argument values used during QAIRT Converter Float Fallback.

        Parameters:
        quantizer_opts: Default initializer options using IrQuantizer object
        backend_info_obj: Backend info object
        """

        quantizer_opts.use_fallback_to_float = True
        quantizer_opts.pack_4_bit_weights = self.args.pack_4_bit_weights
        backend_name = backend_info_obj.backend_name() if backend_info_obj else ""
        if backend_name in ['HTP','GPU','AIC']:
            quantizer_opts.keep_weights_quantized = True
        quantizer_opts.backend_name = backend_name.lower()
        quantizer_opts.float_bw = 16 if backend_name not in ['LPAI'] else quantizer_opts.float_bw
        quantizer_opts.float_bias_bw = 16 if backend_name not in ['LPAI'] else quantizer_opts.float_bias_bw
        quantizer_opts.enable_qnn_quantizer = True
        quantizer_opts.disable_relu_squashing = True
        quantizer_opts.disable_dynamic_16_bit_weights = self.args.disable_dynamic_16_bit_weights
        quantizer_opts.is_qairt = self.qairt_converter
        if not quantizer_opts.disable_dynamic_16_bit_weights :
            quantizer_opts.use_dynamic_16_bit_weights = True
        if "--float_bitwidth" in sys.argv:
            quantizer_opts.float_bw = self.float_bitwidth
            quantizer_opts.float_bias_bw = self.float_bitwidth
        if "--float_bias_bitwidth" in sys.argv:
            quantizer_opts.float_bias_bw = self.float_bias_bitwidth
        quantizer_opts.quantization_overrides = str(self.args.quantization_overrides)

        return quantizer_opts

    def quantize_cpp_graph(self, graph, cpp_graph, args, backend_info_obj=None):
        class QuantizerVersion(Enum):
            NONE = 0
            V1 = 1
            APPLY_ENCODINGS = 2

        def get_quant_version(graph, args):
            def check_encodings_present(qparams):
                return any(qparam['output_encodings'] or qparam['param_encodings'] for qparam in qparams.values())

            if (not args.export_format == ExportFormatType.DLC_DEFAULT or
                not graph.quantization_params or
                not check_encodings_present(graph.quantization_params)):
                return QuantizerVersion.NONE

            external_quant_params = graph.user_quantization_overrides
            if (external_quant_params and
                "version" in external_quant_params and
                external_quant_params["version"] == "2.0.0"
                or args.use_quantize_v2):
                return QuantizerVersion.APPLY_ENCODINGS

            return QuantizerVersion.V1

        quant_version = get_quant_version(graph, args)

        # Direct mapping from EncodingVersion to QuantEncodingVersion
        encoding_version_map = {
            QuantizerVersion.NONE: ir_graph.QuantEncodingVersion.NONE,
            QuantizerVersion.V1: ir_graph.QuantEncodingVersion.V1,
            QuantizerVersion.APPLY_ENCODINGS: ir_graph.QuantEncodingVersion.V2
        }

        # Get the quant encoding version and set it in the cpp graph
        quant_encoding_version = encoding_version_map.get(quant_version, ir_graph.QuantEncodingVersion.NONE)
        # Set encoding version
        cpp_graph.set_quant_encoding_version(quant_encoding_version)

        # Currently, BF16 isn't supported in Quantizer
        # So, error out if BF16 flag is set and there are overrides.
        if quant_version != QuantizerVersion.NONE and self.float_bitwidth == 16 and self.isBfloat16:
            raise Exception("Currently, BF16 isn't supported with models having quantization overrides or QDQ nodes.")

        if quant_version == QuantizerVersion.V1:
            quantizer_opts = ir_quantizer.IrQuantizerOpts()
            qairt_float_fallback_opts = self.set_qairt_converter_float_fallback_default_args(quantizer_opts, backend_info_obj)
            quantizer = ir_quantizer.IrQuantizer(qairt_float_fallback_opts, cpp_graph)
            quantizer.quantize()
            self.applied_float_fallback = True

        elif quant_version == QuantizerVersion.APPLY_ENCODINGS:
            quantizer_opts = ir_quantizer.IrQuantizerOptsV2()
            quantizer_opts.target_backend = backend_info_obj.backend_name()
            if backend_info_obj and backend_info_obj.soc_model() != 'UNKNOWN_SDM':
                quantizer_opts.soc_model = backend_info_obj.soc_model()
            if self.args.calc_static_encodings:
                quantizer_opts.should_compute_statics = True
            if self.args.quantizer_log:
                quantizer_opts.log_file_name = self.args.quantizer_log
                # If the log level was set, use it. Otherwise, default to INFO
                if self.args.quantizer_log_level != ir_quantizer.LogLevel.NONE:
                    quantizer_opts.log_level = self.args.quantizer_log_level
                else:
                    quantizer_opts.log_level = ir_quantizer.LogLevel.INFO
            else:
                quantizer_opts.log_level = ir_quantizer.LogLevel.NONE
            with ir_quantizer.IrQuantizerV2(quantizer_opts, cpp_graph) as quantizer:
                quantizer.apply_encodings()

        return cpp_graph

    def serialize(self, graph, network_specialization: bool = False,
                  enable_tensor_deduplication: bool = True,
                  is_qairt: bool = False,
                  backend_info_obj=None):
        self.num_graphs_in_dlc += 1

        # update the remove_unused_inputs flag in args, so that we can access the update flag in GraphOptimizer
        if is_qairt:
            self.args.remove_unused_inputs = graph.remove_unused_inputs

        # TODO: pass graph as-is
        graph = self.prepare_py_graph(graph)
        cpp_graph = self.get_ir_graph(graph)
        cpp_graph = self.prepare_cpp_graph(graph, cpp_graph, network_specialization)

        # Apply Graph transforms on IrGraph
        args_dict = GraphOptimizer.ArgParser.convert_args(self.args)
        optimizer = GraphOptimizer(args_dict)
        optimizer.optimize(cpp_graph, [OptimizationStage.PostLayout])

        if is_qairt and graph.quantization_params:
            if self.args.export_format == ExportFormatType.DLC_STRIP_QUANT:
                graph.quantization_params.clear()
                cpp_graph.reset_qinfo_for_all_tensors()

        # Set zero consumers tensors as graph output for non-ONNX models for qairt-converter
        if "REMOVE_DISCONNECTED_NODES" in optimizer.optimization_pass_mode_manager.get_cpp_ir_pass_list() and self.input_network.split(".")[-1] != 'onnx':
            cpp_graph.set_zero_consumers_tensors_as_graph_output()

        if is_qairt:
            cpp_graph = self.quantize_cpp_graph(graph, cpp_graph, self.args, backend_info_obj)

        if graph.custom_datatype_tensors:
            cpp_graph.modify_io_datatype(graph.custom_datatype_tensors)

        if graph.preserve_io_datatype_passed:
            cpp_graph.modify_io_datatype(graph.preserve_datatype_tensors)

        # enable_tensor_deduplication will be True(by default) in general,
        # i.e.,Serializer will check if shared context static tensors is present in DLC or not
        # It will be False(by default) for network specialization flow until --enable_tensor_deduplication is passed
        self.dlc_serializer.serialize(graph=cpp_graph, enableTensorDeduplication=enable_tensor_deduplication)
        if graph.source_topology:
            log_info('Writing source topology to DLC...')
            add_succeeded = self.dlc_serializer.add_record_from_buffer(buffer=graph.source_topology,
                                                                       recordType=modeltools.DlcRecordType.SOURCE_TOPOLOGY)
            if not add_succeeded:
                raise Exception("Failed to write source topology")
        del cpp_graph
        log_info(code_to_message.get_progress_message("INFO_CONVERSION_SUCCESS"))

    def finish(self):
        self.dlc_serializer.finish()
        log_info(code_to_message.get_progress_message("INFO_WRITE_SUCCESS"))

    def save(self, graph, network_specialization: bool = False,
            enable_tensor_deduplication: bool = True,
            is_qairt: bool = False,
            backend_info_obj=None):
        self.initialize()
        self.serialize(graph,
                       network_specialization=network_specialization,
                       enable_tensor_deduplication=enable_tensor_deduplication,
                       is_qairt=is_qairt,
                       backend_info_obj=backend_info_obj)
        self.finish()
        if hasattr(self.args, "dump_qairt_io_config_yaml"):
            # only QNN and SNPE use the function to generate command for QAIRT
            # qairt-converter will not enter this function,
            self.dump_qairt_cmdline_io_config(graph)
