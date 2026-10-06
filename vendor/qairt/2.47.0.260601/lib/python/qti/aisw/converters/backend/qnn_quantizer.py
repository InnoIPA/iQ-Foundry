# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import sys

try:
    from qti.aisw.converters.common import ir_quantizer
except ImportError as ie:
    print("Failed to find necessary quantization packages:")
    print(str(ie))
    print("Please ensure that $QNN_SDK_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.backend_aware_configs.backend_awareness_utils import get_path_for_target_config
import argparse
from qti.aisw.converters.common.graph_optimizer import GraphOptimizer, OptimizationStage

class QnnQuantizer(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(QnnQuantizer.ArgParser, self).__init__(**kwargs)
            q_group = self.add_argument_group(title='Quantizer Options')

            q_group.add_argument('--input_list', type=str,
                                 action=validation_utils.validate_filename_arg(must_exist=True),
                                 help='Path to a file specifying the input data. This file should be a plain text '
                                      'file, containing one or more absolute file paths per line. Each path is '
                                      'expected to point to a binary file containing one input in the "raw" format, '
                                      'ready to be consumed by the quantizer without any further preprocessing. '
                                      'Multiple files per line separated by spaces indicate multiple inputs to the '
                                      'network. See documentation for more details. Must be specified for quantization. '
                                      'All subsequent quantization options are ignored when this is not provided.')
            q_group.add_argument('--param_quantizer', type=str,
                                 help='Optional parameter to indicate the weight/bias quantizer to use. Must be followed '
                                      'by one of the following options:\n'
                                      '"tf": Uses the real min/max of the data and specified bitwidth (default).\n'
                                      '"enhanced": Uses an algorithm useful for quantizing models with long tails present '
                                      'in the weight distribution.\n'
                                      '"adjusted": Note: "adjusted" mode is deprecated.\n'
                                      '"symmetric": Ensures min and max have the same absolute values about zero. Data '
                                      'will be stored as int#_t data such that the offset is always 0.'
                                      'Note: Legacy option --param_quantizer will be deprecated, use --param_quantizer_calibration '
                                      'instead\n')
            q_group.add_argument('--act_quantizer', type=str, default='tf',
                                 help='Optional parameter to indicate the activation quantizer to use. Must be followed by '
                                      'one of the following options:\n'
                                      '"tf": Uses the real min/max of the data and specified bitwidth (default).\n'
                                      '"enhanced": Uses an algorithm useful for quantizing models with long tails present '
                                      'in the weight distribution.\n'
                                      '"adjusted": Note: "adjusted" mode is deprecated.\n'
                                      '"symmetric": Ensures min and max have the same absolute values about zero. Data '
                                      'will be stored as int#_t data such that the offset is always 0.'
                                      'Note: Legacy option --act_quantizer will be deprecated, use --act_quantizer_calibration '
                                      'instead\n')
            q_group.add_argument('--algorithms', type=str, nargs='+', default=[],
                                 help='Use this option to enable new optimization algorithms. Usage is: '
                                      '--algorithms <algo_name1> ... '
                                      'The available optimization algorithms are: '
                                      '"cle" - Cross layer equalization includes a number of methods for equalizing '
                                      'weights and biases across layers in order to rectify imbalances that cause '
                                      'quantization errors. ')
            #TODO: Remove the following deprecated arguments in future release:
            # --bias_bw, --act_bw, --weight_bw, --float_bw, --float_bias_bw
            q_group.add_argument('--bias_bitwidth', type=int, default=8, dest='bias_bitwidth',
                                 help='Use the --bias_bitwidth option to select the bitwidth to use when quantizing the '
                                      'biases, either 8 (default) or 32.')
            q_group.add_argument('--bias_bw', type=int, default=8, dest='bias_bitwidth',
                                 help='Note: --bias_bw is deprecated, use --bias_bitwidth.')

            q_group.add_argument('--act_bitwidth', type=int, default=8, dest='act_bitwidth',
                                 help='Use the --act_bitwidth option to select the bitwidth to use when quantizing the '
                                      'activations, either 8 (default) or 16.')
            q_group.add_argument('--act_bw', type=int, default=8, dest='act_bitwidth',
                                 help='Note: --act_bw is deprecated, use --act_bitwidth.')

            q_group.add_argument('--weights_bitwidth', type=int, default=8, dest='weights_bitwidth',
                                 help='Use the --weights_bitwidth option to select the bitwidth to use when quantizing '
                                      'the weights, either 4 or 8 (default).')
            q_group.add_argument('--weight_bw', type=int, default=8, dest='weights_bitwidth',
                                 help='Note: --weight_bw is deprecated, use --weights_bitwidth.')

            q_group.add_argument('--float_bitwidth', type=int, default=32, dest='float_bitwidth',
                                 help='Use the --float_bitwidth option to select the bitwidth to use for float tensors,'
                                      'either 32 (default) or 16.')
            q_group.add_argument('--float_bw', type=int, default=32, dest='float_bitwidth',
                                 help='Note: --float_bw is deprecated, use --float_bitwidth.')

            q_group.add_argument('--float_bias_bitwidth', type=int, default=0, dest='float_bias_bitwidth',
                                 help='Use the --float_bias_bitwidth option to select the bitwidth to use when biases '
                                      'are in float, either 32 or 16.')
            q_group.add_argument('--float_bias_bw', type=int, default=0, dest='float_bias_bitwidth',
                                 help='Note: --float_bias_bw is deprecated, use --float_bias_bitwidth.')

            q_group.add_argument('--ignore_encodings', action='store_true', default=False,
                                 help='Use only quantizer generated encodings, ignoring any user or model provided '
                                      'encodings.\n'
                                      'Note: Cannot use --ignore_encodings with --quantization_overrides')

            q_group.add_argument('--use_per_channel_quantization', action='store_true', default=False,
                                 help='Use this option to enable per-channel quantization for convolution-based op weights. \n'
                                      'Note: This will replace built-in model QAT encodings when used for a given weight.')

            q_group.add_argument('--use_per_row_quantization', action='store_true', default=False,
                                 help='Use this option to enable rowwise quantization of Matmul and FullyConnected ops.')

            q_group.add_argument('--enable_per_row_quantized_bias', action='store_true', default=False,
                                 help='Use this option to enable rowwise quantization of bias for FullyConnected op, when weights are per-row quantized.')

            q_group.add_argument('--float_fallback', action='store_true', default=False,
                                 help='Use this option to enable fallback to floating point (FP) instead of fixed point. \n'
                                      'This option can be paired with --float_bitwidth to indicate the bitwidth for FP (by default 32). \n'
                                      'If this option is enabled, then input list must not be provided and --ignore_encodings must not be provided.\n'
                                      'The external quantization encodings (encoding file/FakeQuant encodings) might be missing quantization parameters for some interim tensors. \n'
                                      'First it will try to fill the gaps by propagating across math-invariant functions. If the quantization params are still missing, \n'
                                      'then it will apply fallback to nodes to floating point. \n')

            q_group.add_argument('--use_native_input_files', action='store_true', default=False,
                                 help='Boolean flag to indicate how to read input files:\n'
                                      '1. float (default): reads inputs as floats and quantizes if necessary based on quantization parameters in the model.\n'
                                      '2. native:          reads inputs assuming the data type to be native to the model. For ex., uint8_t.\n')

            q_group.add_argument('--use_native_dtype', action='store_true', default=False,
                                 help='Note: This option is deprecated, use --use_native_input_files option in future.\n'
                                      'Boolean flag to indicate how to read input files:\n'
                                      '1. float (default): reads inputs as floats and quantizes if necessary based on quantization parameters in the model.\n'
                                      '2. native:          reads inputs assuming the data type to be native to the model. For ex., uint8_t.\n')

            q_group.add_argument('--use_native_output_files', action='store_true', default=False,
                                 help='Use this option to indicate the data type of the output files\n'
                                      '1. float (default): output the file as floats.\n'
                                      '2. native:          outputs the file that is native to the model. For ex., uint8_t.\n')

            q_group.add_argument('--disable_relu_squashing', action='store_true', default=False,
                                 help="Disables squashing of Relu against Convolution based ops for "
                                      "quantized models")

            q_group.add_argument('--restrict_quantization_steps', type=validation_utils.two_hex, action = "store",
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
            q_group.add_argument('--use_dynamic_16_bit_weights', action='store_true', default=False,
                                 help=argparse.SUPPRESS)

            q_group.add_argument('--pack_4_bit_weights', action='store_true', default=False,
                                 help='Store 4-bit quantized weights in packed format in a single byte i.e. two 4-bit quantized tensors can be stored in one byte')

            q_group.add_argument('--keep_weights_quantized', action='store_true', default=False,
                                 help='Use this option to keep the weights quantized even when the output of the op is in floating point. '
                                      'Bias will be converted to floating point as per the output of the op. '
                                      'Required to enable wFxp_actFP configurations according to the provided bitwidth for weights and activations\n'
                                      'Note: These modes are not supported by all runtimes. Please check corresponding Backend OpDef supplement '
                                      'if these are supported')

            q_group.add_argument('--act_quantizer_calibration', type=str, default="min-max",
                                 help='Specify which quantization calibration method to use for activations\n'
                                      'supported values: min-max (default), sqnr, entropy, mse, percentile\n'
                                      'This option can be paired with --act_quantizer_schema to override the quantization\n'
                                      'schema to use for activations otherwise default schema(asymmetric) will be used\n')

            q_group.add_argument('--param_quantizer_calibration', type=str, default="min-max",
                                 help='Specify which quantization calibration method to use for parameters\n'
                                      'supported values: min-max (default), sqnr, entropy, mse, percentile\n'
                                      'This option can be paired with --param_quantizer_schema to override the quantization\n'
                                      'schema to use for parameters otherwise default schema(asymmetric) will be used\n')

            q_group.add_argument('--act_quantizer_schema', type=str, default="asymmetric",
                                 help='Specify which quantization schema to use for activations\n'
                                      'supported values: asymmetric (default), symmetric, unsignedsymmetric, signedasymmetric\n'
                                      'This option cannot be used with legacy quantizer option --act_quantizer\n')

            q_group.add_argument('--param_quantizer_schema', type=str, default="asymmetric",
                                 help='Specify which quantization schema to use for parameters\n'
                                      'supported values: asymmetric (default), symmetric, unsignedsymmetric, signedasymmetric\n'
                                      'This option cannot be used with legacy quantizer option --param_quantizer\n')

            q_group.add_argument('--percentile_calibration_value', type=float, default=99.99,
                                 help='Specify the percentile value to be used with Percentile calibration method\n'
                                      'The specified float value must lie within 90 and 100, default: 99.99\n')

            q_group.add_argument("--use_aimet_quantizer",
                                 action="store_true",
                                 help=argparse.SUPPRESS,
                                 default=False)

            q_group.add_argument('--dump_qairt_quantizer_command', type=str,
                                 help='Use this option to dump a file which contains the equivalent Commandline \n'
                                      'input for QAIRT Quantizer\n')

            q_group.add_argument('--target_backend', type=str, help=argparse.SUPPRESS, dest='backend',
                                 choices=["HTP", "LPAI", "CPU", "GPU", "HTA", "AIC", "DSP"])
            q_group.add_argument('--use_quantize_v2', action='store_true', default=False,
                                 help=argparse.SUPPRESS)
            q_group.add_argument('--calc_static_encodings', action='store_true', default=False,
                                 help=argparse.SUPPRESS)
            q_group.add_argument('--quantizer_log', type=str,
                                 help="Enable logging in quantizer v2, logging to the file "
                                      "<QUANTIZER_LOG>.\n"
                                      "E.g., --quantizer_log my_model_name.csv will produce the file "
                                      "my_model_name.csv. See --quantizer_log_level.")
            q_group.add_argument('--quantizer_log_level',
                                 type=ir_quantizer.LogLevel.from_string,
                                 choices=list(ir_quantizer.LogLevel.__members__.values()),
                                 default=ir_quantizer.LogLevel.NONE,
                                 help="Sets the logging level in quantizer v2.\n"
                                      "INFO: Emits a file in the CSV format. Requires --quantizer_log "
                                      "<file_name.csv> to be set. "
                                      "Warnings and errors are emitted to the console.\n"
                                      "TRACE: Emits a file in the TXT format. Requires --quantizer_log "
                                      "<file_name.txt> to be set. "
                                      "Warnings and errors are emitted to the console.\n"
                                      "NONE: Default value. No file is emitted. "
                                      "Warnings and errors are emitted to the console.")

    def __init__(self, args):
        self.args = args
        self.ir_graph_reader = None
        self.opts = ir_quantizer.IrQuantizerOpts()
        self.opts_v2 = None
        if (args.input_list is None and not args.float_fallback):
            self.should_quantize = False
            self.use_fallback_to_float = False
            if args.use_quantize_v2:
                self.should_quantize = True
        elif (args.input_list is None and args.float_fallback):
            log_warning("QNN Quantization is disabled as --float_fallback flag is provided "
                        "Some Ops may fallback to float datatype")
            self.should_quantize = True
            self.use_fallback_to_float = True
        elif (args.input_list is not None and args.float_fallback):
            raise Exception("Invalid combination: --input_list and --float_fallback "
                            "cannot be provided at the same time.")
        else:
            self.should_quantize = True
            self.opts.input_list = args.input_list
            self.use_fallback_to_float = False

        self.opts.disable_legacy_quantizer = False

        if not self.should_quantize:
            return

        # Error out if BF16 graph is triggered with Quantizer
        is_bfloat16 = (args.float_bitwidth == "bf16") or (args.float_bw == "bf16")
        if is_bfloat16 and self.should_quantize:
            raise Exception("Currently, quantization isn't supported with BF16 graphs.")

        # TODO: Resolve dependency on quantization_overrides which is defined in different file
        if args.ignore_encodings and args.quantization_overrides:
            raise Exception("Invalid combination: --quantization_overrides and "
                            "--ignore_encodings cannot be provided at the same time.")

        if args.use_native_dtype:
            log_warning("--use_native_dtype option is deprecated, use --use_native_input_files option in future.")

        if '--bias_bw' in sys.argv:
            log_warning("--bias_bw option is deprecated, use --bias_bitwidth.")
            if '--bias_bitwidth' in sys.argv:
                raise Exception("Invalid combination: --bias_bw and --bias_bitwidth "
                                "cannot be provided at the same time.")

        if '--act_bw' in sys.argv:
            log_warning("--act_bw option is deprecated, use --act_bitwidth.")
            if '--act_bitwidth' in sys.argv:
                raise Exception("Invalid combination: --act_bw and --act_bitwidth "
                                "cannot be provided at the same time.")

        if '--weight_bw' in sys.argv:
            log_warning("--weight_bw option is deprecated, use --weights_bitwidth.")
            if '--weights_bitwidth' in sys.argv:
                raise Exception("Invalid combination: --weight_bw and --weights_bitwidth "
                                "cannot be provided at the same time.")

        if '--float_bw' in sys.argv:
            log_warning("--float_bw option is deprecated, use --float_bitwidth.")
            if '--float_bitwidth' in sys.argv:
                raise Exception("Invalid combination: --float_bw and --float_bitwidth "
                                "cannot be provided at the same time.")

        if '--float_bias_bw' in sys.argv:
            log_warning("--float_bias_bw option is deprecated, use --float_bias_bitwidth.")
            if '--float_bias_bitwidth' in sys.argv:
                raise Exception("Invalid combination: --float_bias_bw and --float_bias_bitwidth "
                                "cannot be provided at the same time.")

        if args.float_fallback and args.ignore_encodings:
            raise Exception("Cannot determine quantization encodings for any tensor. "
                            "--ignore_encodings cannot be provided with --float_fallback flag")

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

        if args.percentile_calibration_value < 90 or args.percentile_calibration_value > 100:
            raise Exception("--percentile_calibration_value must lie with 90 and 100")

        # Throw error if an argument is provided that is not supported by AIMET Quantizer
        if "--use_aimet_quantizer" in sys.argv:
            args_not_supported_by_aimet = ["--restrict_quantization_steps", "--pack_4_bit_weights",
                                           "--use_dynamic_16_bit_weights", "--op_package_lib",
                                           "--keep_weights_quantized"]
            args_provided_by_user = [arg for arg in sys.argv if arg[0:2]=="--"]
            args_provided_by_user_not_supported_by_aimet = [arg for arg in args_provided_by_user if arg in args_not_supported_by_aimet]
            if len(args_provided_by_user_not_supported_by_aimet)!=0:
                raise Exception(f"AIMET Quantizer doesn't support the following options currently: "
                                f"{args_provided_by_user_not_supported_by_aimet}")

        # Set default values for act_quantizer and param_quantizer
        if not args.param_quantizer:
            if args.weights_bitwidth == 16:
                args.param_quantizer = "symmetric"
            else:
                args.param_quantizer = "tf"

        # If any one of new quantizer options is passed skip using the legacy quantizer(quantizer methods with
        # fixed quantization scheme) options.
        if ("--param_quantizer_calibration" in sys.argv or "--act_quantizer_calibration" in sys.argv or
                "--param_quantizer_schema" in sys.argv or "--act_quantizer_schema" in sys.argv):
            self.opts.disable_legacy_quantizer = True

        if "--target_backend" in sys.argv and args.backend not in ["HTP", "LPAI", "CPU", "GPU", "HTA", "AIC", "DSP"]:
            raise Exception("Invalid backend info: ", args.backend)
        elif "--target_backend" in sys.argv:
            self.opts.backend_name = args.backend.lower()

        self.opts.param_quantizer = args.param_quantizer
        self.opts.act_quantizer = args.act_quantizer
        self.opts.param_quantizer_calibration = args.param_quantizer_calibration
        self.opts.act_quantizer_calibration = args.act_quantizer_calibration
        self.opts.param_quantizer_schema = args.param_quantizer_schema
        self.opts.act_quantizer_schema = args.act_quantizer_schema
        self.opts.percentile_calibration_value = args.percentile_calibration_value
        self.opts.algorithms = args.algorithms
        self.opts.bias_bw = args.bias_bitwidth
        self.opts.act_bw = args.act_bitwidth
        self.opts.weight_bw = args.weights_bitwidth
        self.opts.float_bw = int(args.float_bitwidth)
        self.opts.float_bias_bw = args.float_bias_bitwidth
        self.opts.optimizations = True
        self.opts.op_package_lib = args.op_package_lib
        self.opts.ignore_encodings = args.ignore_encodings
        self.opts.use_per_row_quantization = args.use_per_row_quantization
        self.opts.enable_per_row_quantized_bias = args.enable_per_row_quantized_bias
        self.opts.use_per_channel_quantization = args.use_per_channel_quantization
        self.opts.use_native_input_dtype = args.use_native_input_files or args.use_native_dtype
        self.opts.use_native_output_dtype = args.use_native_output_files
        self.opts.reset_irgraph_maps = True
        self.opts.enable_qnn_quantizer = True
        self.opts.disable_relu_squashing = args.disable_relu_squashing or self.use_fallback_to_float
        self.opts.pack_4_bit_weights = args.pack_4_bit_weights
        self.opts.keep_weights_quantized = args.keep_weights_quantized
        self.opts.use_fallback_to_float = args.float_fallback
        self.opts.use_quantize_v2 = args.use_quantize_v2
        self.opts.quantization_overrides = str(args.quantization_overrides)

        if self.opts.backend_name == "lpai":
            self.opts.use_dynamic_16_bit_weights = True
        else:
            self.opts.use_dynamic_16_bit_weights = args.use_dynamic_16_bit_weights

        if '--dump_qairt_quantizer_command' in sys.argv:
            self.opts.dump_qairt_quantizer_command = args.dump_qairt_quantizer_command

        if args.restrict_quantization_steps:
            if self.opts.param_quantizer == "symmetric" or self.opts.use_per_channel_quantization or self.opts.use_per_row_quantization:
                self.opts.quantization_step_min = args.restrict_quantization_steps[0]
                self.opts.quantization_step_max = args.restrict_quantization_steps[1]
                log_info("Restricting number of quantization steps to: min: {} - max: {}".format(self.opts.quantization_step_min,
                                                                                                 self.opts.quantization_step_max))
            else:
                log_warning("Restrict_quantization_steps is only supported for --param_quantizer = symmetric"
                            " or per channel/row quantization. Value will be ignored.")

        self.quant_schemes = None
        if self.args.use_aimet_quantizer:
            self.quant_schemes = {}
            if not self.opts.disable_legacy_quantizer:
                self.quant_schemes["param_quant"] = self.opts.param_quantizer
                self.quant_schemes["act_quant"] = self.opts.act_quantizer
            else:
                self.quant_schemes["param_quant"] = {"calibration": self.opts.param_quantizer_calibration,
                                                     "schema": self.opts.param_quantizer_schema}
                self.quant_schemes["act_quant"] = {"calibration": self.opts.act_quantizer_calibration,
                                                   "schema": self.opts.act_quantizer_schema}

    def get_opts(self):
        return self.opts

    def quantize(self, ir_graph, converter_backend, enc_v2=False):
        self.graph = ir_graph
        self.converter_backend = converter_backend

        if not self.should_quantize and not enc_v2:
            log_info('Skipping quantization, no input_list provided')
            return

        if not converter_backend.op_package_lib:
            if self.opts.input_list and converter_backend.custom_op_config_paths:
                log_warning('OP_PACKAGE_LIB_NOT_FOUND: Custom op configs were provided with no '
                            'custom op package libraries. '
                            'Note: Custom op packages may be required to '
                            'correctly quantize custom ops')

        if self.args.use_aimet_quantizer:
            from qti.aisw.converters.aimet.qnn_quantsim_interface import aimet_quantizer, AimetQuantizerOpts
            opts = AimetQuantizerOpts(input_network=self.args.input_network,
                                      output_path=self.args.output_path,
                                      input_list=self.args.input_list,
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
                                      use_native_output_files=self.opts.use_native_output_dtype)
            self.ir_graph_reader = aimet_quantizer(ir_graph, opts)
            return

        if enc_v2 or (hasattr(self.args, 'use_quantize_v2') and self.args.use_quantize_v2):
            self.opts_v2 = ir_quantizer.IrQuantizerOptsV2()
            self.opts_v2.should_compute_statics = self.args.calc_static_encodings
            if self.args.quantizer_log:
                self.opts_v2.log_file_name = self.args.quantizer_log
                # If the log level was set, use it. Otherwise, default to INFO
                if self.args.quantizer_log_level != ir_quantizer.LogLevel.NONE:
                    self.opts_v2.log_level = self.args.quantizer_log_level
                else:
                    self.opts_v2.log_level = ir_quantizer.LogLevel.INFO
            else:
                self.opts_v2.log_level = ir_quantizer.LogLevel.NONE
            if "--target_backend" in sys.argv:
                self.opts_v2.target_backend = self.args.backend
            if "--dump_encoding_json" in sys.argv:
                raise RuntimeError("--dump_encoding_json cannot be used when use_quantize_v2 \
                                    is set OR when --quantization_overrides \
                                    JSON Schema version is 2.0.0")

            # Options used during calibration in apply encodings
            if self.args.input_list:
                self.opts_v2.input_list = self.args.input_list
                self.opts_v2.param_quantizer_calibration = self.args.param_quantizer_calibration
                self.opts_v2.act_quantizer_calibration = self.args.act_quantizer_calibration
                if "--param_quantizer_schema" in sys.argv:
                    self.opts_v2.param_quantizer_schema = self.args.param_quantizer_schema
                else:
                    self.opts_v2.param_quantizer_schema = ""
                if "--act_quantizer_schema" in sys.argv:
                    self.opts_v2.act_quantizer_schema = self.args.act_quantizer_schema
                else:
                    self.opts_v2.act_quantizer_schema = ""
                self.opts_v2.percentile_calibration_value = self.args.percentile_calibration_value
                self.opts_v2.act_bw = self.args.act_bitwidth
                self.opts_v2.weight_bw = self.args.weights_bitwidth
                if "--bias_bw" in sys.argv or "--bias_bitwidth" in sys.argv:
                    log_warning("Bias bitwidth cannot be used with --use_quantize_v2 or --quantization_overrides with "
                                "version 2.0.0. It will be determined based on the selected --target_backend.")
                self.opts_v2.use_per_row_quantization = self.args.use_per_row_quantization
                self.opts_v2.use_per_channel_quantization = self.args.use_per_channel_quantization

                self.opts_v2.use_native_input_dtype = self.opts.use_native_input_dtype
                self.opts_v2.use_native_output_dtype = self.opts.use_native_output_dtype
                self.opts_v2.op_package_lib = self.args.op_package_lib
                self.opts_v2.ignore_encodings = self.args.ignore_encodings

                if self.args.restrict_quantization_steps:
                    if self.opts_v2.param_quantizer_schema == "symmetric" or self.opts_v2.use_per_channel_quantization or self.opts_v2.use_per_row_quantization:
                        self.opts_v2.quantization_step_min = self.args.restrict_quantization_steps[0]
                        self.opts_v2.quantization_step_max = self.args.restrict_quantization_steps[1]
                        log_info("Restricting number of quantization steps to: min: {} - max: {}".format(self.opts_v2.quantization_step_min,
                                                                                                         self.opts_v2.quantization_step_max))
                    else:
                        log_warning("Restrict_quantization_steps is only supported for --param_quantizer_schema = symmetric"
                                    " or per channel/row quantization. Value will be ignored.")

            with ir_quantizer.IrQuantizerV2(self.opts_v2, ir_graph) as quantizer:
                # If input list is provided, call calibrate to calculate encodings for tensors with missing encodings
                if self.opts_v2.input_list:
                    quantizer.calibrate()
                quantizer.apply_encodings()
        else:
            # Quantize and store as float as QNN CPU BE only supports float data
            quantizer = ir_quantizer.IrQuantizer(self.get_opts(), ir_graph)
            quantizer.run_algorithms(self.opts.algorithms)
            quantizer.quantize_params(True)  # True indicates that it should be stored as floats
            if self.use_fallback_to_float:
                quantizer.fallback_to_float()
            else:
                quantizer.generate_activations()
            # Quantize "for real"
            quantizer.quantize_params(False)  # False indicates it should be stored as normal quantized data
            converter_backend.c_ir_graph = ir_graph

            quantizer.mixed_precision_processing()
        # Apply Graph transforms on IrGraph
        args_dict = GraphOptimizer.ArgParser.convert_args(self.args)
        optimizer = GraphOptimizer(args_dict)
        optimizer.optimize(ir_graph, [OptimizationStage.PostQuant])

    def construct_model(self, modelgen_backend, modelgen_interface, context, graph_configs_info, num_graph_configs_info):
        model = self.converter_backend.construct_model(self.graph, modelgen_backend, modelgen_interface,context,
                                                       graph_configs_info, num_graph_configs_info)
        self.tensor_map = self.converter_backend.get_tensor_map()
        return model

    def get_tensor_map(self):
        return self.tensor_map
