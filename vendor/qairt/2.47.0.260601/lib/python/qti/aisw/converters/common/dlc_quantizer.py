# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import sys

try:
    from qti.aisw.converters.common import ir_quantizer
    from qti.aisw.converters.common import encodings_json_serializer
    from . import ir_graph
except ImportError as ie:
    print("Failed to find necessary quantization packages:")
    print(str(ie))
    print("Please ensure that $QNN_SDK_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

try:
    from qti.aisw.converters.common import modeltools
except ImportError as ie:
    from qti.aisw.dlc_utils import modeltools

from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning, log_error
from qti.aisw.converters.common.utils.lora_utils import make_tensors_updateable
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.backend_awareness import BackendInfo
from qti.aisw.converters.common.backend_aware_configs.backend_awareness_utils import get_path_for_target_config
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.dlc_utils.snpe_dlc_utils import ModelInfo
from qti.aisw.converters.common.graph_optimizer import GraphOptimizer, OptimizationStage

import argparse
import os
import yaml


class DLCQuantizer(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(DLCQuantizer.ArgParser, self).__init__(**kwargs)
            self.add_required_argument('--input_dlc', '-i', dest='input_dlc', type=str,
                                       action=validation_utils.validate_filename_arg(must_exist=True),
                                       help='Path to the dlc container containing the model for which '
                                            'fixed-point encoding metadata should be generated. This argument is required')

            self.add_optional_argument('--output_dlc', '-o', dest='output_dlc', type=str,
                                       action=validation_utils.validate_filename_arg(must_exist=False,
                                                                                     create_missing_directory=True),
                                       help='Path at which the metadata-included quantized model container should be written.'
                                            'If this argument is omitted, the quantized model will be written at '
                                            '<unquantized_model_name>_quantized.dlc')

            self.add_optional_argument('--input_list', '-l', dest='input_list', type=str,
                                       action=validation_utils.validate_filename_arg(must_exist=True),
                                       help='Path to a file specifying the input data. This file should be a plain text '
                                            'file, containing one or more absolute file paths per line. Each path is '
                                            'expected to point to a binary file containing one input in the "raw" format, '
                                            'ready to be consumed by the quantizer without any further preprocessing. '
                                            'Multiple files per line separated by spaces indicate multiple inputs to the '
                                            'network. See documentation for more details. Must be specified for quantization. '
                                            'All subsequent quantization options are ignored when this is not provided.')

            self.add_optional_argument('--float_fallback', action='store_true', default=False,
                                       help=argparse.SUPPRESS)

            self.add_optional_argument('--enable_float_fallback', '-f', dest='float_fallback', action='store_true', default=False,
                                       help='Use this option to enable fallback to floating point (FP) instead of fixed point. \n'
                                            'This option can be paired with --float_bitwidth to indicate the bitwidth for FP (by default 32). \n'
                                            'If this option is enabled, then input list must not be provided and --ignore_quantization_overrides must not be provided.\n'
                                            'The external quantization encodings (encoding file/FakeQuant encodings) might be missing quantization parameters for some interim tensors. \n'
                                            'First it will try to fill the gaps by propagating across math-invariant functions. If the quantization params are still missing, \n'
                                            'then it will apply fallback to nodes to floating point. \n')

            self.add_optional_argument('--param_quantizer', type=str,
                                       help=argparse.SUPPRESS)

            self.add_optional_argument('--act_quantizer', type=str, default='tf',
                                       help=argparse.SUPPRESS)

            self.add_optional_argument('--algorithms', type=str, nargs='+', default=[],
                                       help=argparse.SUPPRESS)

            self.add_optional_argument('--apply_algorithms', dest='algorithms', type=str, nargs='+', default=[],
                                       help='Use this option to enable new optimization algorithms. Usage is: '
                                            '--apply_algorithms <algo_name1> ... '
                                            'The available optimization algorithms are: '
                                            '"cle" - Cross layer equalization includes a number of methods for equalizing '
                                            'weights and biases across layers in order to rectify imbalances that cause '
                                            'quantization errors. ')

            self.add_optional_argument('--bias_bitwidth', type=int, default=32,
                                       help='Use the --bias_bitwidth option to select the bitwidth to use when quantizing the '
                                            'biases, either 8 or 32 (default).')

            self.add_optional_argument('--act_bitwidth', type=int, default=8,
                                       help='Use the --act_bitwidth option to select the bitwidth to use when quantizing the '
                                            'activations, either 8 (default) or 16.')

            self.add_optional_argument('--weights_bitwidth', type=int, default=8,
                                       help='Use the --weights_bitwidth option to select the bitwidth to use when quantizing '
                                            'the weights, either 2, 4, 8 (default) or 16.')

            self.add_optional_argument('--float_bitwidth', type=int, default=32,
                                       help='Use the --float_bitwidth option to select the bitwidth to use for float tensors,'
                                            'either 32 (default) or 16.')

            self.add_optional_argument('--float_bias_bitwidth', type=int, default=0,
                                       help="Use the --float_bias_bitwidth option to select the bitwidth to use when biases "
                                            "are in float, either 32 or 16 (default '0' if not provided).")

            self.add_optional_argument('--ignore_encodings', action='store_true', default=False, help=argparse.SUPPRESS)

            self.add_optional_argument('--ignore_quantization_overrides', dest='ignore_encodings', action='store_true', default=False,
                                       help="Use only quantizer generated encodings, ignoring any user or "
                                            "model provided encodings.\n"
                                            "Note: Cannot use --ignore_quantization_overrides with "
                                            "--quantization_overrides (argument of Qairt Converter)")

            self.add_optional_argument('--use_per_channel_quantization', action='store_true', default=False,
                                       help='Use this option to enable per-channel quantization for convolution-based op weights. \n'
                                            'Note: This will only be used if built-in model Quantization-Aware Trained (QAT) encodings are not present for a given weight.')

            self.add_optional_argument('--use_per_row_quantization', action='store_true', default=False,
                                       help='Use this option to enable rowwise quantization of Matmul and FullyConnected ops.')

            self.add_optional_argument('--enable_per_row_quantized_bias', action='store_true', default=False,
                                       help='Use this option to enable rowwise quantization of bias for FullyConnected ops, when weights are per-row quantized.')

            self.add_optional_argument('--preserve_io_datatype', nargs='*', action='append', default=[],
                                       help='Use this option to preserve IO datatype. The different ways of using this option are as follows:\n'
                                            '    --preserve_io_datatype <space separated list of names of inputs and outputs of the graph>\n'
                                            'e.g.\n'
                                            '   --preserve_io_datatype input1 input2 output1 \n'
                                            'The user may choose to preserve the datatype for all the inputs and outputs of the graph.\n'
                                            '    --preserve_io_datatype\n')

            self.add_optional_argument('--use_native_input_files', action='store_true', default=False,
                                       help='Boolean flag to indicate how to read input files.\n'
                                            'If not provided, reads inputs as floats and quantizes if necessary based on quantization parameters in the model. (default)\n'
                                            'If provided, reads inputs assuming the data type to be native to the model. For ex., uint8_t.\n')

            self.add_optional_argument('--use_native_output_files', action='store_true', default=False,
                                       help='Boolean flag to indicate the data type of the output files\n'
                                            'If not provided, outputs the file as floats. (default)\n'
                                            'If provided, outputs the file that is native to the model. For ex., uint8_t.\n')

            self.add_optional_argument('--restrict_quantization_steps', type=validation_utils.two_hex, action="store",
                                       help='Specifies the number of steps to use for computing quantization encodings such that '
                                            'scale = (max - min) / number of quantization steps.\n'
                                            'The option should be passed as a space separated pair of hexadecimal string minimum and maximum values'
                                            'i.e. --restrict_quantization_steps "MIN MAX".  \n Please note that this is a hexadecimal string literal'
                                            ' and not a signed integer, to supply a negative value an explicit minus sign is required.\n'
                                            'E.g.--restrict_quantization_steps "-0x80 0x7F" indicates an example 8 bit range,\n'
                                            '    --restrict_quantization_steps "-0x8000 0x7F7F" indicates an example 16 bit range.\n'
                                            'This argument is required for 16-bit Matmul operations.\n',
                                       metavar="ENCODING_MIN, ENCODING_MAX", default=[])

            # TODO: Remove this flag once we fully support 16-bit
            self.add_optional_argument('--use_dynamic_16_bit_weights', action='store_true', default=True,
                                       help=argparse.SUPPRESS)
            self.add_optional_argument('--disable_dynamic_16_bit_weights', action='store_true', default=False,
                                       help=argparse.SUPPRESS)

            self.add_optional_argument('--pack_4_bit_weights', action='store_true', default=False,
                                       help=argparse.SUPPRESS)

            self.add_optional_argument('--keep_weights_quantized', action='store_true', default=False,
                                       help='Use this option to keep the weights quantized even when the output of the op is in floating point. '
                                            'Bias will be converted to floating point as per the output of the op. '
                                            'Required to enable wFxp_actFP configurations according to the provided bitwidth for weights and activations\n'
                                            'Note: These modes are not supported by all runtimes. Please check corresponding '
                                            'Backend OpDef supplement if these are supported')

            self.add_optional_argument('--adjust_bias_encoding', action='store_true', default=False,
                                       help= 'Use --adjust_bias_encoding option to modify bias encoding and weight encoding '
                                             'to ensure that the bias value is in the range of the bias encoding. '
                                             'This option is only applicable for per-channel quantized weights. \n '
                                             'NOTE: This may result in clipping of the weight values')

            self.add_optional_argument('--act_quantizer_calibration', type=str, default="min-max",
                                       help='Specify which quantization calibration method to use for activations\n'
                                            'supported values: min-max (default), sqnr, entropy, mse, percentile\n'
                                            'This option can be paired with --act_quantizer_schema to override the quantization\n'
                                            'schema to use for activations otherwise default schema(asymmetric) will be used\n')

            self.add_optional_argument('--param_quantizer_calibration', type=str, default="min-max",
                                       help='Specify which quantization calibration method to use for parameters\n'
                                            'supported values: min-max (default), sqnr, entropy, mse, percentile\n'
                                            'This option can be paired with --param_quantizer_schema to override the quantization\n'
                                            'schema to use for parameters otherwise default schema(asymmetric) will be used\n')

            self.add_optional_argument('--act_quantizer_schema', type=str, default="asymmetric",
                                       help='Specify which quantization schema to use for activations\n'
                                            'supported values: asymmetric (default), symmetric, unsignedsymmetric, signedasymmetric\n')

            self.add_optional_argument('--param_quantizer_schema', type=str, default="asymmetric",
                                       help='Specify which quantization schema to use for parameters\n'
                                            'supported values: asymmetric (default), symmetric, unsignedsymmetric, signedasymmetric\n')

            self.add_optional_argument('--percentile_calibration_value', type=float, default=99.99,
                                       help='Specify the percentile value to be used with Percentile calibration method\n'
                                            'The specified float value must lie within 90 and 100, default: 99.99\n')

            self.add_optional_argument("--use_aimet_quantizer",
                                       action="store_true",
                                       help='Use AIMET for Quantization instead of QNN IR quantizer',
                                       default=False)

            self.add_optional_argument('--op_package_lib', '-opl', type=str, default="",
                                       help='Use this argument to pass an op package library for quantization. '
                                            'Must be in the form <op_package_lib_path:interfaceProviderName> and'
                                            ' be separated by a comma for multiple package libs')

            self.add_optional_argument('--dump_encoding_json', action='store_true', default=False,
                                       help="Use this argument to dump encoding of all the tensors in a json file")

            self.add_optional_argument('--include_data_invariant_ops', action='store_true', help=argparse.SUPPRESS,
                                       default=False)
            self.add_optional_argument('--config_file', type=str,
                                       action=validation_utils.validate_filename_arg(must_exist=True),
                                       help=argparse.SUPPRESS)
            self.add_optional_argument('--config', '-c', dest='config_file', type=str,
                                       action=validation_utils.validate_filename_arg(must_exist=True),
                                       help='Use this argument to pass the path of the config YAML file with quantizer\n'
                                            'options ')
            self.add_optional_argument("--export_stripped_dlc",
                                       action="store_true",
                                       help='Use this argument to export a DLC which strips out data not needed for graph composition',
                                       default=False)
            self.add_optional_argument('--use_quantize_v2', action='store_true', default=False,
                                       help="Enables IrQuantizerV2 flow")
            self.add_optional_argument('--calc_static_encodings', action='store_true', default=False,
                                       help=argparse.SUPPRESS)
            self.add_optional_argument('--quantizer_log', type=str,
                                       help="Valid for use with v2.0.0 JSON schema for quantization "
                                            "overrides or when --use_quantize_v2 is provided. "
                                            "Enable logging in the quantizer, logging to the file "
                                            "<QUANTIZER_LOG>.\n"
                                            "E.g., --quantizer_log my_model_name.csv will produce the file "
                                            "my_model_name.csv. See --quantizer_log_level.")
            self.add_optional_argument('--quantizer_log_level',
                                       type=ir_quantizer.LogLevel.from_string,
                                       choices=list(ir_quantizer.LogLevel.__members__.values()),
                                       default=ir_quantizer.LogLevel.NONE,
                                       help="Sets the logging level in the quantizer. See --quantizer_log.\n"
                                            "INFO: Emits a file in the CSV format. Requires --quantizer_log "
                                            "<file_name.csv> to be set. "
                                            "Warnings and errors are emitted to the console.\n"
                                            "TRACE: Emits a file in the TXT format. Requires --quantizer_log "
                                            "<file_name.txt> to be set. "
                                            "Warnings and errors are emitted to the console.\n"
                                            "NONE: Default value. No file is emitted. "
                                            "Warnings and errors are emitted to the console.")

        @classmethod
        def validate_and_convert_args(cls, args):
            # check if legacy quantizer options are used with new quantizer options.
            if (("--param_quantizer_calibration" in sys.argv or "--act_quantizer_calibration" in sys.argv) and
                    ("--param_quantizer" in sys.argv or "--act_quantizer" in sys.argv)):
                raise Exception("Invalid combination: legacy quantizer options: --act_quantizer or --param_quantizer cannot be "
                                "combined with --act_quantizer_calibration or --param_quantizer_calibration")

            if (("--param_quantizer_schema" in sys.argv or "--act_quantizer_schema" in sys.argv) and
                    ("--param_quantizer" in sys.argv or "--act_quantizer" in sys.argv)):
                raise Exception("Invalid combination: legacy quantizer options: --act_quantizer or --param_quantizer cannot be "
                                "combined with --act_quantizer_schema or --param_quantizer_schema. "
                                "To create quantizer with different quantization schema use --act_quantizer_calibration or "
                                "--param_quantizer_calibration with --act_quantizer_schema or --param_quantizer_schema respectively")


            if "--param_quantizer_schema" in sys.argv and args.param_quantizer_schema not in ["symmetric", "asymmetric", "unsignedsymmetric", "signedasymmetric"]:
                raise Exception("Invalid param quantizer schema: ", args.param_quantizer_schema)

            if "--act_quantizer_schema" in sys.argv and args.act_quantizer_schema not in ["symmetric", "asymmetric", "unsignedsymmetric", "signedasymmetric"]:
                raise Exception("Invalid activation quantizer schema: ", args.act_quantizer_schema)

            # If percentile_calibration value is passed check if the calibration method selected is percentile.
            if ("--percentile_calibration_value" in sys.argv and
                    (args.act_quantizer_calibration != "percentile" and args.param_quantizer_calibration != "percentile")):
                raise Exception("Invalid combination: --percentile_calibration_value option should be used with "
                                "--act_quantizer_calibration percentile or --param_quantizer_calibration percentile options")

            # Throw error if an argument is provided that is not supported by AIMET Quantizer
            if "--use_aimet_quantizer" in sys.argv:
                args_not_supported_by_aimet = ["--restrict_quantization_steps", "--pack_4_bit_weights",
                                               "--use_dynamic_16_bit_weights", "--op_package_lib",
                                               "--keep_weights_quantized", "--adjust_bias_encoding"]
                args_provided_by_user = [arg for arg in sys.argv if arg[0:2] == "--"]
                args_provided_by_user_not_supported_by_aimet = [arg for arg in args_provided_by_user if arg in args_not_supported_by_aimet]
                if len(args_provided_by_user_not_supported_by_aimet) != 0:
                    raise Exception(f"AIMET Quantizer doesn't support the following options currently: "
                                    f"{args_provided_by_user_not_supported_by_aimet}")

            if '--float_fallback' in sys.argv:
                log_warning("--float_fallback flag is deprecated, use --enable_float_fallback.")
            if '--algorithms' in sys.argv:
                log_warning("--algorithms option is deprecated, use --apply_algorithms.")
            if '--ignore_encodings' in sys.argv:
                log_warning("--ignore_encodings flag is deprecated, use --ignore_quantization_overrides.")
            if '--config_file' in sys.argv:
                log_warning("--config_file option is deprecated, use --config.")

            args_dict = vars(args).copy()
            args_dict['config_file'] = parse_yaml_config(args.config_file) if args.config_file else None

            args_dict['disable_legacy_quantizer'] = True
            # If any one of legacy quantizer options is passed then enable the legacy quantizer
            if ("--param_quantizer" in sys.argv or "--act_quantizer" in sys.argv):
                args_dict['disable_legacy_quantizer'] = False

            return args_dict



    def __init__(self,
                 input_dlc,
                 output_dlc=None,
                 input_list="",
                 float_fallback=False,
                 param_quantizer="tf",
                 act_quantizer="tf",
                 algorithms=[],
                 bias_bitwidth=32,
                 act_bitwidth=8,
                 weights_bitwidth=8,
                 float_bitwidth=32,
                 float_bias_bitwidth=0,
                 ignore_encodings=False,
                 use_per_channel_quantization=False,
                 use_per_row_quantization=False,
                 enable_per_row_quantized_bias=False,
                 preserve_io_datatype=None,
                 use_native_input_files=False,
                 use_native_output_files=False,
                 restrict_quantization_steps=[],
                 use_dynamic_16_bit_weights=True,
                 disable_dynamic_16_bit_weights=False,
                 pack_4_bit_weights=False,
                 keep_weights_quantized=False,
                 adjust_bias_encoding=False,
                 act_quantizer_calibration="min-max",
                 param_quantizer_calibration="min-max",
                 act_quantizer_schema="asymmetric",
                 param_quantizer_schema="asymmetric",
                 percentile_calibration_value=99.99,
                 use_aimet_quantizer=False,
                 op_package_lib="",
                 disable_legacy_quantizer=False,
                 dump_encoding_json=False,
                 include_data_invariant_ops=False,
                 aimet_config=None,
                 backend_info_obj=None,
                 export_stripped_dlc=False,
                 use_quantize_v2=False,
                 should_compute_statics=False,
                 log_file_name=None,
                 log_level=ir_quantizer.LogLevel.NONE,
                 ):

        self.input_dlc = input_dlc

        self.aimet_config = aimet_config
        self.use_aimet_quantizer = use_aimet_quantizer
        self.export_stripped_dlc = export_stripped_dlc
        # check for the lora model
        dlc_info = ModelInfo(self.input_dlc)
        self.converter_args = dlc_info.parse_converter_command()
        self.lora_enabled = False
        self.lora_metadata_record = None
        self.quant_updatable_mode = None
        self.disable_transform_tracking = False
        if 'disable_transform_tracking' in self.converter_args and self.converter_args['disable_transform_tracking'] != None:
            self.disable_transform_tracking = self.converter_args['disable_transform_tracking']
        if 'lora_weight_list' in self.converter_args.keys() and self.converter_args['lora_weight_list'] != None:
            log_debug("LoRA enabled usecase")
            self.lora_enabled = True
        if 'quant_updatable_mode' in self.converter_args and self.converter_args['quant_updatable_mode'] != None:
            log_debug("quant_updatable_mode is found")
            self.quant_updatable_mode = self.converter_args['quant_updatable_mode']

        if not self.use_aimet_quantizer:
            # Deserialize the DLC
            self.dlc_reader = modeltools.IrDlcReader(input_dlc, disableLazyWeightLoading=True)
            self.ir_graph = self.dlc_reader.get_ir_graph()

            # store DLC metadata
            self.model_version = self.dlc_reader.custom_model_version()
            self.copyright_str = self.dlc_reader.copyright()
            self.converter_command = self.dlc_reader.converter_command()
            if backend_info_obj:
                BackendInfo.validate_dlc_metadata(self.converter_command,
                                                  backend_info_obj.backend_name(),
                                                  backend_info_obj.soc_model())

            if self.lora_enabled and self.quant_updatable_mode == "none" and \
                not self.disable_transform_tracking:
                self.lora_metadata_record = self.dlc_reader.extract_record("lora.converter.metadata", modeltools.DlcRecordType.LORA_CONVERTER_METADATA)
        else:
            self.dlc_reader = None
            self.ir_graph = None
            self.model_version = ''
            self.copyright_str = ''
            self.converter_command = ''

            # When converted model has dynamic dimensions, and if --use_aimet_quantizer is passed, error out
            if self.converter_args['input_dim'] is not None:
                for entry in self.converter_args['input_dim']:
                    if '*' in entry[1]:
                        raise Exception("AIMET quantizer does not support Graph with Dynamic Dimensions. Please try using default quantizer")

        # store the output path for the quantized DLC
        if output_dlc is None:
            filename, _ = os.path.splitext(os.path.realpath(input_dlc))
            self.output_path = filename + "_quantized.dlc"
            self.output_encoding_json_path = filename + "_quantized_encoding.json"
        else:
            self.output_path = output_dlc
            self.output_encoding_json_path = os.path.splitext(os.path.realpath(self.output_path))[0] + "_encoding.json"

        self.dump_encoding_json = dump_encoding_json
        self.include_data_invariant_ops = include_data_invariant_ops
        self.use_quantize_v2 = use_quantize_v2

        # Set Quantizer option
        self.opts = ir_quantizer.IrQuantizerOpts()

        if self.use_quantize_v2:
            self.opts_v2 = ir_quantizer.IrQuantizerOptsV2()
            self.opts_v2.should_compute_statics = should_compute_statics

            if log_file_name:
                self.opts_v2.log_file_name = log_file_name
                # If the log level was set, use it. Otherwise, default to INFO
                if log_level != ir_quantizer.LogLevel.NONE:
                    self.opts_v2.log_level = log_level
                else:
                    self.opts_v2.log_level = ir_quantizer.LogLevel.INFO
            else:
                self.opts_v2.log_level = ir_quantizer.LogLevel.NONE

            if backend_info_obj and backend_info_obj.soc_model() != 'UNKNOWN_SDM':
                self.opts_v2.soc_model = backend_info_obj.soc_model()

            # Options used during calibration in apply encodings
            if input_list:
                self.opts_v2.input_list = input_list
            self.opts_v2.param_quantizer_calibration = param_quantizer_calibration
            self.opts_v2.act_quantizer_calibration = act_quantizer_calibration
            if "--param_quantizer_schema" in sys.argv:
                self.opts_v2.param_quantizer_schema = param_quantizer_schema
            else:
                self.opts_v2.param_quantizer_schema = ""
            if "--act_quantizer_schema" in sys.argv:
                self.opts_v2.act_quantizer_schema = act_quantizer_schema
            else:
                self.opts_v2.act_quantizer_schema = ""
            self.opts_v2.percentile_calibration_value = percentile_calibration_value
            self.opts_v2.act_bw = act_bitwidth
            self.opts_v2.weight_bw = weights_bitwidth
            if "--bias_bw" in sys.argv or "--bias_bitwidth" in sys.argv:
                log_warning("Bias bitwidth cannot be used with --use_quantize_v2 or --quantization_overrides with "
                            "version 2.0.0. It will be determined based on the selected --target_backend.")
            if "--dump_encoding_json" in sys.argv or self.dump_encoding_json:
                raise RuntimeError("--dump_encoding_json cannot be used when use_quantize_v2 \
                                    is set OR when --quantization_overrides \
                                    JSON Schema version is 2.0.0")
            self.opts_v2.use_per_row_quantization = use_per_row_quantization
            self.opts_v2.use_per_channel_quantization = use_per_channel_quantization

            self.opts_v2.use_native_input_dtype = use_native_input_files
            if preserve_io_datatype:
                self.opts_v2.use_native_input_dtype = True
            self.opts_v2.use_native_output_dtype = use_native_output_files
            self.opts_v2.op_package_lib = op_package_lib
            self.opts_v2.ignore_encodings = ignore_encodings

            if restrict_quantization_steps:
                if self.opts_v2.param_quantizer_schema == "symmetric" or self.opts_v2.use_per_channel_quantization or self.opts_v2.use_per_row_quantization:
                    self.opts_v2.quantization_step_min = int(restrict_quantization_steps[0])
                    self.opts_v2.quantization_step_max = int(restrict_quantization_steps[1])
                    log_info("Restricting number of quantization steps to: min: {} - max: {}".format(self.opts_v2.quantization_step_min,
                                                                                                     self.opts_v2.quantization_step_max))
                else:
                    log_warning("Restrict_quantization_steps is only supported for --param_quantizer_schema = symmetric"
                                " or per channel/row quantization. Value will be ignored.")

            if log_file_name:
                self.opts_v2.log_file_name = log_file_name
                if self.opts_v2.log_level is ir_quantizer.LogLevel.NONE:
                    self.opts_v2.log_level = ir_quantizer.LogLevel.INFO


        if (input_list is None and not float_fallback):
            if self.aimet_config:
                self.should_quantize = True
            elif self.use_quantize_v2:
                self.should_quantize = True
            else:
                self.should_quantize = False
        elif (input_list is None and float_fallback):
            log_warning("Quantization is disabled as --enable_float_fallback flag is provided "
                        "Some Ops may fallback to float datatype")
            self.should_quantize = True
        elif (input_list is not None and float_fallback):
            raise Exception("Invalid combination: --input_list and --enable_float_fallback "
                            "cannot be provided at the same time.")
        elif (input_list is not None and self.converter_args.get('quant_updatable_mode') == 'float_only'):
            raise Exception("Invalid combination: if quant updatable mode is float only, "
                            "then input list cannot be provided")
        else:
            self.should_quantize = True
            self.opts.input_list = input_list

        self.use_fallback_to_float = float_fallback
        self.opts.disable_legacy_quantizer = False

        if not self.should_quantize:
            return

        if self.use_fallback_to_float and ignore_encodings:
            raise Exception("Cannot determine quantization encodings for any tensor. "
                            "--ignore_quantization_overrides cannot be provided with --enable_float_fallback flag")

        if percentile_calibration_value < 90 or percentile_calibration_value > 100:
            raise Exception("--percentile_calibration_value must lie with 90 and 100")

        # Set default values for act_quantizer and param_quantizer
        if not param_quantizer:
            if weights_bitwidth == 16:
                param_quantizer = "symmetric"
            else:
                param_quantizer = "tf"

        self.opts.param_quantizer = param_quantizer
        self.opts.act_quantizer = act_quantizer
        self.opts.param_quantizer_calibration = param_quantizer_calibration
        self.opts.act_quantizer_calibration = act_quantizer_calibration
        self.opts.param_quantizer_schema = param_quantizer_schema
        self.opts.act_quantizer_schema = act_quantizer_schema
        self.opts.percentile_calibration_value = percentile_calibration_value
        self.opts.algorithms = algorithms

        self.opts.bias_bw = bias_bitwidth
        self.opts.act_bw = act_bitwidth
        self.opts.weight_bw = weights_bitwidth
        self.opts.float_bw = float_bitwidth
        self.opts.float_bias_bw = float_bias_bitwidth
        self.opts.optimizations = True
        self.opts.op_package_lib = op_package_lib
        self.opts.ignore_encodings = ignore_encodings
        self.opts.use_per_row_quantization = use_per_row_quantization
        self.opts.enable_per_row_quantized_bias = enable_per_row_quantized_bias
        self.opts.use_per_channel_quantization = use_per_channel_quantization
        self.opts.use_native_input_dtype = use_native_input_files
        if preserve_io_datatype:
            self.opts.use_native_input_dtype = True
        self.opts.use_native_output_dtype = use_native_output_files
        self.opts.reset_irgraph_maps = True
        self.opts.enable_qnn_quantizer = True
        self.opts.use_dynamic_16_bit_weights = use_dynamic_16_bit_weights
        self.opts.disable_dynamic_16_bit_weights = disable_dynamic_16_bit_weights
        if disable_dynamic_16_bit_weights:
            self.opts.use_dynamic_16_bit_weights = False
            log_warning("Since --disable_dynamic_16_bit_weights flag is provided"
            " a convert from 16 to 8 would be added for dynamic 16 bit weights.")
        self.opts.is_qairt = True
        self.opts.pack_4_bit_weights = pack_4_bit_weights
        self.opts.keep_weights_quantized = keep_weights_quantized
        self.opts.adjust_bias_encoding = adjust_bias_encoding
        self.opts.disable_legacy_quantizer = disable_legacy_quantizer
        self.opts.disable_relu_squashing = True
        self.opts.use_quantize_v2 = use_quantize_v2
        self.preserve_io_datatype = preserve_io_datatype
        self.preserve_datatype_tensor_map = {}
        self.opts.use_fallback_to_float = float_fallback
        quantization_overrides_path = self.converter_args.get("quantization_overrides", "")
        self.opts.quantization_overrides = str(quantization_overrides_path)

        if backend_info_obj:
            if backend_info_obj.backend_name().lower() not in ["htp", "lpai"]:
                self.opts.backend = get_path_for_target_config(backend_info_obj.backend_name().lower())
            self.opts.backend_name = backend_info_obj.backend_name().lower()
            if self.use_quantize_v2:
                self.opts_v2.target_backend = backend_info_obj.backend_name()

        if restrict_quantization_steps:
            if self.opts.param_quantizer == "symmetric" or self.opts.use_per_channel_quantization or self.opts.use_per_row_quantization:
                self.opts.quantization_step_min = int(restrict_quantization_steps[0])
                self.opts.quantization_step_max = int(restrict_quantization_steps[1])
                log_info("Restricting number of quantization steps to: min: {} - max: {}".format(self.opts.quantization_step_min,
                                                                                                 self.opts.quantization_step_max))
            else:
                log_warning("Restrict_quantization_steps is only supported for --param_quantizer = symmetric"
                            " or per channel/row quantization. Value will be ignored.")

        self.quant_schemes = None
        if self.use_aimet_quantizer:
            self.quant_schemes = {}
            if not self.opts.disable_legacy_quantizer:
                self.quant_schemes["param_quant"] = self.opts.param_quantizer
                self.quant_schemes["act_quant"] = self.opts.act_quantizer
            else:
                self.quant_schemes["param_quant"] = {"calibration": self.opts.param_quantizer_calibration,
                                                     "schema": self.opts.param_quantizer_schema}
                self.quant_schemes["act_quant"] = {"calibration": self.opts.act_quantizer_calibration,
                                                   "schema": self.opts.act_quantizer_schema}

        self.backend_info_obj = backend_info_obj

    def set_ir_graph(self, ir_graph):
        self.ir_graph = ir_graph

    def quantize(self):
        """
        This method quantize the IR graph (inplace) generated from the DLC.
        :return: None
        """

        if not self.should_quantize:
            log_info('Skipping quantization, no input_list provided')
            return

        if self.use_aimet_quantizer:
            from qti.aisw.converters.aimet.qnn_quantsim_interface import aimet_dlc_quantizer, AimetQuantizerOpts
            backend_name = self.backend_info_obj.backend_name() if self.backend_info_obj else None
            opts = AimetQuantizerOpts(input_network=self.input_dlc,
                                      output_path=self.output_path,
                                      input_list=self.opts.input_list,
                                      quant_schemes=self.quant_schemes,
                                      float_fallback=self.use_fallback_to_float,
                                      disable_legacy_quant_scheme_opts=self.opts.disable_legacy_quantizer,
                                      algorithms=self.opts.algorithms,
                                      act_bitwidth=self.opts.float_bw if self.use_fallback_to_float else self.opts.act_bw,
                                      weights_bitwidth=self.opts.float_bw if self.use_fallback_to_float else self.opts.weight_bw,
                                      bias_bitwidth=self.opts.bias_bw,
                                      float_bias_bw=self.opts.float_bias_bw,
                                      percentile_calibration_value=self.opts.percentile_calibration_value,
                                      ignore_encodings=self.opts.ignore_encodings,
                                      use_per_channel_quantization=self.opts.use_per_channel_quantization,
                                      use_per_row_quantization=self.opts.use_per_row_quantization,
                                      use_native_input_files=self.opts.use_native_input_dtype,
                                      use_native_output_files=self.opts.use_native_output_dtype,
                                      backend_name=backend_name,
                                      config=self.aimet_config)

            self.dlc_reader = aimet_dlc_quantizer(opts)
            self.ir_graph = self.dlc_reader.get_ir_graph()
            # store DLC metadata
            self.model_version = self.dlc_reader.custom_model_version()
            self.copyright_str = self.dlc_reader.copyright()
            self.converter_command = self.dlc_reader.converter_command()
            if self.backend_info_obj:
                BackendInfo.validate_dlc_metadata(self.converter_command,
                                                  self.backend_info_obj.backend_name(),
                                                  self.backend_info_obj.soc_model())

            return

        if self.preserve_io_datatype:
            self.populate_io_src_datatype()

        for tensor in self.ir_graph.get_tensor_map().values():
            if tensor.is_quantized() or tensor.is_overridden_float():
                self.opts.float_fallback_applied = True
                if self.use_fallback_to_float:
                    self.should_quantize = False
                    log_info('Skipping Float Fallback. Float Fallback is already applied to the provided DLC.')
                    return
                else:
                    break

        # Default enocding_version is NONE
        quant_encoding_version = ir_graph.QuantEncodingVersion.NONE

        # Quantize IR graph using IrQuantizer/IrQuantizerV2
        if self.use_quantize_v2:
            # Dequantize the irgraph using quantizer_v1
            quantizer_v1 = ir_quantizer.IrQuantizer(self.opts, self.ir_graph)
            dlc_type = quantizer_v1.get_dlc_type()
            # If the dlc is already fully quantized, skip calling the quantizer
            if dlc_type == ir_quantizer.IrDlcType.DLC_FULLY_QUANTIZED and not self.opts.ignore_encodings:
                log_info('Current input dlc file is fully quantized! Skipping calibration. '
                         'To enable full calibration, use --ignore_encodings')
                return
            quantizer_v1.dequantize_graph(dlc_type)

            # Update quant encoding version to V2 if use_quantize_v2 is enabled
            quant_encoding_version = ir_graph.QuantEncodingVersion.V2
            self.ir_graph.set_quant_encoding_version(quant_encoding_version)

            # Pass float irgraph with cached encodings to quantizer_v2 (apply encodings)
            with ir_quantizer.IrQuantizerV2(self.opts_v2, self.ir_graph) as quantizer_v2:
                # If input list is provided, call calibrate to calculate encodings for tensors with missing encodings
                if self.opts_v2.input_list:
                    quantizer_v2.calibrate()
                quantizer_v2.apply_encodings()
        else:
            # Update quant encoding version to V1
            quant_encoding_version = ir_graph.QuantEncodingVersion.V1
            self.ir_graph.set_quant_encoding_version(quant_encoding_version)

            quantizer = ir_quantizer.IrQuantizer(self.opts, self.ir_graph)
            # If it is a BF16 DLC that we are trying to quantize, then error out.
            dlc_type = quantizer.get_dlc_type()
            if dlc_type == ir_quantizer.IrDlcType.DLC_BFLOAT_16:
                raise Exception("Currently, quantization isn't supported with BF16 graphs.")
            quantizer.quantize()

        if self.lora_enabled:
            self.ir_graph = make_tensors_updateable(self.ir_graph, graph_quantized=True)

        if self.preserve_io_datatype:
            self.ir_graph.modify_io_datatype(self.preserve_datatype_tensor_map)

        log_info(code_to_message.get_progress_message("Quantization completed successfully"))

        # Apply graph transforms on IrGraph. Currently there are no post quant optimizations are enabled.
        optimizer = GraphOptimizer(self.converter_args)
        optimizer.optimize(self.ir_graph, [OptimizationStage.PostQuant])

    def populate_io_src_datatype(self):
        qnn_to_np_dtype_map = {
            'QNN_DATATYPE_FLOAT_32': 'float32',
            'QNN_DATATYPE_FLOAT_16': 'float16',
            'QNN_DATATYPE_UINT_8': 'uint8',
            'QNN_DATATYPE_UINT_16': 'uint16',
            'QNN_DATATYPE_UINT_32': 'uint32',
            'QNN_DATATYPE_UINT_64': 'uint32',
            'QNN_DATATYPE_INT_8': 'int8',
            'QNN_DATATYPE_INT_16': 'int16',
            'QNN_DATATYPE_INT_32': 'int32',
            'QNN_DATATYPE_INT_64': 'int64',
            'QNN_DATATYPE_BOOL_8': 'bool_'
        }

        io_tensors = self.ir_graph.get_input_tensors_to_graph() + self.ir_graph.get_output_tensors_of_graph()
        io_tensor_names = [tensor.name() for tensor in io_tensors]

        for tensor in io_tensors:
            # Add to list only if preserve datatype is called either for all IO tensors or partial IO tensors
            if ((len(self.preserve_io_datatype[0]) == 0) or (tensor.name() in self.preserve_io_datatype[0])):
                self.preserve_datatype_tensor_map[tensor.name()] = qnn_to_np_dtype_map.get(tensor.data_type().name, "float32")

        for tensor_name in self.preserve_io_datatype[0]:
            if tensor_name not in io_tensor_names:
                log_error("Graph does not have the tensor '{}'".format(tensor_name))
                sys.exit(1)

    def save(self, quantizer_command=""):
        """
        This method saves the quantized model to the output path specifies
        during instantiation. If nothing specifies, the quantized model will
        be stored at the same location as of input dlc.
        :return: None
        """
        dlc_writer = modeltools.IrDlcSerializer(self.output_path,
                                                self.copyright_str,
                                                self.model_version,
                                                self.converter_command,
                                                quantizer_command,
                                                self.export_stripped_dlc)
        dlc_writer.initialize()
        dlc_writer.serialize(self.ir_graph)
        if self.lora_metadata_record != None:
            dlc_writer.add_record_from_buffer(self.lora_metadata_record.get_bytes(), modeltools.DlcRecordType.LORA_CONVERTER_METADATA)
        dlc_writer.finish()

        # Serialize QNNIR to ENCODING JSON
        if self.dump_encoding_json:
            self.encodings_json_serializer = encodings_json_serializer.IrEncodingsJsonSerializer(
                self.output_encoding_json_path, self.include_data_invariant_ops)
            self.encodings_json_serializer.serialize(self.ir_graph)
            encoding_json = self.encodings_json_serializer.get_graph_json()
            with open(self.output_encoding_json_path, "w") as json_file:
                json_file.write(encoding_json)
            log_info("encodings JSON saved at: %s " % self.output_encoding_json_path)

        log_info("Quantized Model saved at: %s " % self.output_path)

def parse_yaml_config(config_path: str):
    """
    This method reads the YAML config file and returns a dictionary.
    :return: Dict
    """
    with open(config_path) as configs:
        try:
            config = yaml.safe_load(configs)
        except yaml.YAMLError as e:
            logger.error('Error parsing YAML config file')
            raise RuntimeError('Error parsing YAML config file') from e

    return config
