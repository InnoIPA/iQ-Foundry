# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


import sys
import yaml
import argparse
import traceback
from abc import abstractmethod, ABC
from qti.aisw.converters.common import graph_optimizer
import qti.aisw.converters.common.converter_ir.op_graph as op_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrder
from qti.aisw.converters.common.converter_ir.op_policies import ConversionNamePolicy
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils.validation_utils import *
from qti.aisw.converters.common.common_base import ConverterBase
from qti.aisw.converters.common.utils.framework_utils import FrameworkSummary

class ConverterFrontend(ConverterBase, ABC):
    class ArgParser(ConverterBase.ArgParser):
        def __init__(self, **kwargs):
            super(ConverterFrontend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--out_node', '--out_name', type=str, dest='out_names', action='append', default=[],
                                       help="Name of the graph\'s output Tensor Names. Multiple output names "
                                            "should be provided separately like: \n"
                                            "    --out_name out_1 --out_name out_2")
            self.add_optional_argument('--input_type', "-t", nargs=2, action='append',
                                       help='Type of data expected by each input op/layer. Type for each input '
                                            'is |default| if not specified. For example: "data" image.Note that '
                                            'the quotes should always be included in order to handle special '
                                            'characters, spaces,etc. For multiple inputs specify multiple '
                                            '--input_type on the command line.\n'
                                            'Eg: \n'
                                            '   --input_type "data1" image --input_type "data2" opaque \n'
                                            'These options get used by DSP runtime and following descriptions '
                                            'state how input will be handled for each option.\n'
                                            'Image: \n'
                                            'Input is float between 0-255 and the input\'s mean is 0.0f '
                                            'and the input\'s max is 255.0f. We will cast the float to uint8ts '
                                            'and pass the uint8ts to the DSP. \n'
                                            'Default: \n'
                                            'Pass the input as floats to the dsp directly and the DSP '
                                            'will quantize it.\n'
                                            'Opaque: \n'
                                            'Assumes input is float because the consumer layer(i.e next '
                                            'layer) requires it as float, therefore it won\'t be quantized.\n'
                                            'Choices supported:\n   ' + '\n   '.join(op_graph.InputType.
                                                                                     get_supported_types()),
                                       metavar=('INPUT_NAME', 'INPUT_TYPE'), default=[])

            self.add_optional_argument('--input_dtype', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DTYPE'), default=[],
                                       help="The names and datatype of the network input layers specified "
                                            "in the format [input_name datatype], "
                                            "for example: \n"
                                            "    'data' 'float32'\n"
                                            "Default is float32 if not specified\n"
                                            "Note that the quotes should always be included in order to handle"
                                            "special characters, spaces, etc. \n"
                                            "For multiple inputs specify multiple --input_dtype on the command "
                                            "line like: \n"
                                            "    --input_dtype 'data1' 'float32' --input_dtype 'data2' 'float32'")

            self.add_optional_argument('--input_encoding', "-e", nargs='+', action='append',
                                       help='Usage: '
                                            '    --input_encoding "INPUT_NAME" INPUT_ENCODING_IN [INPUT_ENCODING_OUT]\n'
                                            'Input encoding of the network inputs. Default is bgr. \n'
                                            'e.g.\n'
                                            '   --input_encoding "data" rgba \n'
                                            'Quotes must wrap the input node name to handle special characters, \n'
                                            'spaces, etc. To specify encodings for multiple inputs, invoke \n'
                                            '--input_encoding for each one. \n'
                                            'e.g.\n'
                                            '    --input_encoding "data1" rgba --input_encoding "data2" other\n'
                                            'Optionally, an output encoding may be specified for an input node by \n'
                                            'providing a second encoding. The default output encoding is bgr.\n'
                                            'e.g. \n'
                                            '    --input_encoding "data3" rgba rgb \n'
                                            'Input encoding types:\n '
                                            '    image color encodings: bgr,rgb, nv21, nv12, ... \n'
                                            '    time_series: for inputs of rnn models; \n'
                                            '    other: not available above or is unknown. \n'
                                            'Supported encodings:\n   ' + '\n   '.join(op_graph.InputEncodings.
                                                                                       get_supported_encodings()),
                                       metavar="\b", default=[])

            self.add_optional_argument('--input_layout', "-l", nargs=2, action='append',
                                       help='Layout of each input tensor. If not specified, it will use the default\n'
                                            'based on the Source Framework, shape of input and input encoding.\n'
                                            'Accepted values are-\n'
                                            '    ' + ', '.join(op_graph.InputLayout.get_supported_types()) + '\n'
                                            'N = Batch, C = Channels, D = Depth, H = Height, W = Width, F = Feature, T = Time\n'
                                            'NDHWC/NCDHW used for 5d inputs\n'
                                            'NHWC/NCHW used for 4d image-like inputs\n'
                                            'NFC/NCF used for inputs to Conv1D or other 1D ops\n'
                                            'NTF/TNF used for inputs with time steps like the ones used for LSTM op\n'
                                            'NF used for 2D inputs, like the inputs to Dense/FullyConnected layers\n'
                                            'NC used for 2D inputs with 1 for batch and other for Channels (rarely used) \n'
                                            'F used for 1D inputs, e.g. Bias tensor\n'
                                            'NONTRIVIAL for everything else'
                                            'For multiple inputs specify multiple --input_layout on the command line.\n'
                                            'Eg: \n'
                                            '    --input_layout "data1" NCHW --input_layout "data2" NCHW \n',
                                       metavar=('INPUT_NAME', 'INPUT_LAYOUT'), default=[])

            self.add_optional_argument('--custom_io', type=str, default="",
                                       help='Use this option to specify a yaml file for custom IO.')
            self.add_optional_argument('--preserve_io', nargs='*', action='append', default=[],
                                       help='Use this option to preserve IO layout and datatype. The different ways of using this option are as follows:\n'
                                            '    --preserve_io layout <space separated list of names of inputs and outputs of the graph>\n'
                                            '    --preserve_io datatype <space separated list of names of inputs and outputs of the graph>\n'
                                            'In this case, user should also specify the string - layout or datatype in the command to indicate that converter needs to\n'
                                            'preserve the layout or datatype. e.g.\n'
                                            '   --preserve_io layout input1 input2 output1 \n'
                                            '   --preserve_io datatype input1 input2 output1 \n'
                                            'Optionally, the user may choose to preserve the layout and/or datatype for all the inputs and outputs of the graph.\n'
                                            'This can be done in the following two ways:\n'
                                            '    --preserve_io layout\n'
                                            '    --preserve_io datatype\n'
                                            'Additionally, the user may choose to preserve both layout and datatypes for all IO tensors by just passing the option as follows:\n'
                                            '    --preserve_io\n'
                                            'Note: Only one of the above usages are allowed at a time.\n'
                                            'Note: --custom_io gets higher precedence than --preserve_io.\n')
            self.add_optional_argument("--dump_qairt_io_config_yaml", type=str, nargs='?', const='io_config.yaml', default="",
                                       help='Use this option to dump a yaml file which contains the equivalent I/O configurations of '
                                            'QAIRT Converter along with the QAIRT Converter Command and can be passed to QAIRT Converter '
                                            'using the option --io_config.\n')
            self.add_optional_argument('--enable_framework_trace', default=False, action="store_true",
                                       help='Use this option to enable converter to trace the op/tensor change information.\n'
                                            'Currently framework op trace is supported only for ONNX converter.')

            q_group = self.add_argument_group(title='Quantizer Options')
            q_group.add_argument('--quantization_overrides', type=str, default="",
                                 help='Use this option to specify a json file with parameters to use for '
                                      'quantization. These will override any quantization data carried from conversion '
                                      '(eg TF fake quantization) or calculated during the normal quantization process. '
                                      'Format defined as per AIMET specification.')
            q_group.add_argument('--keep_quant_nodes', default=False, action="store_true",
                                 help='Use this option to keep activation quantization nodes in the graph rather than '
                                      'stripping them.')
            q_group.add_argument('--use_quantize_v2', default=False, action="store_true", help=argparse.SUPPRESS)

    class ArgParserv2(ConverterBase.ArgParserv2):
        def __init__(self, **kwargs):
            super(ConverterFrontend.ArgParserv2, self).__init__(**kwargs)
            self.add_optional_argument('--source_model_input_shape', '-s', dest='input_dim', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DIM'),
                                       help="The name and dimension of all the input buffers to the network specified in"
                                            " the format [input_name comma-separated-dimensions],\n"
                                            "for example: --source_model_input_shape 'data' 1,224,224,3. \n"
                                            "Note that the quotes should always be included in order to handle special\n"
                                            "characters, spaces, etc.\n"
                                            "NOTE: Required for TensorFlow and PyTorch. Optional for Onnx and Tflite\n"
                                            "In case of Onnx, this feature works only with Onnx 1.6.0 and above")
            self.add_optional_argument('--desired_input_shape', '-d', dest='input_dim', nargs=2, action='append', help=argparse.SUPPRESS)

            self.add_optional_argument('--out_tensor_node', '--out_tensor_name', type=str, dest='out_names', action='append', default=[],
                                       help="Name of the graph\'s output Tensor Names. Multiple output names "
                                            "should be provided separately like: \n"
                                            "    --out_tensor_name out_1 --out_tensor_name out_2\n"
                                            "NOTE: Required for TensorFlow. Optional for Onnx, Tflite and PyTorch\n")

            self.add_optional_argument('--source_model_input_datatype', dest='input_dtype', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DTYPE'), default=[],
                                       help="The names and datatype of the network input layers specified "
                                            "in the format [input_name datatype], "
                                            "for example: \n"
                                            "    'data' 'float32'\n"
                                            "Default is float32 if not specified\n"
                                            "Note that the quotes should always be included in order to handle"
                                            "special characters, spaces, etc. \n"
                                            "For multiple inputs specify multiple --source_model_input_datatype on the "
                                            "command line like: \n"
                                            "    --source_model_input_datatype 'data1' 'float32' --source_model_input_datatype 'data2' 'float32'")

            self.add_optional_argument('--source_model_input_layout', dest='input_layout', nargs=2, action='append',
                                       help='Layout of each input tensor. If not specified, it will use the default '
                                            'based on the Source Framework, shape of input and input encoding.\n'
                                            'Accepted values are-\n'
                                            '    ' + ', '.join(op_graph.InputLayout.get_supported_types_v2()) + '\n'
                                            'N = Batch, C = Channels, D = Depth, H = Height, W = Width, F = Feature,\n'
                                            'T = Time, I = Input, O = Output\n'
                                            'NDHWC/NCDHW used for 5d inputs\n'
                                            'NHWC/NCHW used for 4d image-like inputs\n'
                                            'HWIO/IOHW used for Weights of Conv Ops\n'
                                            'NFC/NCF used for inputs to Conv1D or other 1D ops\n'
                                            'NTF/TNF used for inputs with time steps like the ones used for LSTM op\n'
                                            'NF used for 2D inputs, like the inputs to Dense/FullyConnected layers\n'
                                            'NC used for 2D inputs with 1 for batch and other for Channels (rarely used) \n'
                                            'F used for 1D inputs, e.g. Bias tensor\n'
                                            'For multiple inputs specify multiple --source_model_input_layout on the command line.\n'
                                            'Eg: \n'
                                            '    --source_model_input_layout "data1" NCHW --source_model_input_layout "data2" NCHW \n',
                                       metavar=('INPUT_NAME', 'INPUT_LAYOUT'), default=[])
            self.add_optional_argument('--desired_input_layout', dest='desired_io_layout', nargs=2, action='append',
                                       help='Desired Layout of each input tensor. If not specified, it will use the default '
                                            'based on the Source Framework, shape of input and input encoding.\n'
                                            'Accepted values are-\n'
                                            '    ' + ', '.join(op_graph.InputLayout.get_supported_types_v2()) + '\n'
                                            'N = Batch, C = Channels, D = Depth, H = Height, W = Width, F = Feature,\n'
                                            'T = Time, I = Input, O = Output\n'
                                            'NDHWC/NCDHW used for 5d inputs\n'
                                            'NHWC/NCHW used for 4d image-like inputs\n'
                                            'HWIO/IOHW used for Weights of Conv Ops\n'
                                            'NFC/NCF used for inputs to Conv1D or other 1D ops\n'
                                            'NTF/TNF used for inputs with time steps like the ones used for LSTM op\n'
                                            'NF used for 2D inputs, like the inputs to Dense/FullyConnected layers\n'
                                            'NC used for 2D inputs with 1 for batch and other for Channels (rarely used) \n'
                                            'F used for 1D inputs, e.g. Bias tensor\n'
                                            'For multiple inputs specify multiple --desired_input_layout on the command line.\n'
                                            'Eg: \n'
                                            '    --desired_input_layout "data1" NCHW --desired_input_layout "data2" NCHW \n',
                                       metavar=('INPUT_NAME', 'DESIRED_INPUT_LAYOUT'), default=[])
            self.add_optional_argument('--source_model_output_layout', dest='output_layout', nargs=2, action='append',
                                       help='Layout of each output tensor. If not specified, it will use the default '
                                            'based on the Source Framework, shape of input and input encoding.\n'
                                            'Accepted values are-\n'
                                            '    ' + ', '.join(op_graph.InputLayout.get_supported_types_v2()) + '\n'
                                            'N = Batch, C = Channels, D = Depth, H = Height, W = Width, F = Feature, T = Time\n'
                                            'NDHWC/NCDHW used for 5d inputs\n'
                                            'NHWC/NCHW used for 4d image-like inputs\n'
                                            'NFC/NCF used for inputs to Conv1D or other 1D ops\n'
                                            'NTF/TNF used for inputs with time steps like the ones used for LSTM op\n'
                                            'NF used for 2D inputs, like the inputs to Dense/FullyConnected layers\n'
                                            'NC used for 2D inputs with 1 for batch and other for Channels (rarely used) \n'
                                            'F used for 1D inputs, e.g. Bias tensor\n'
                                            'For multiple inputs specify multiple --source_model_output_layout on the command line.\n'
                                            'Eg: \n'
                                            '    --source_model_output_layout "data1" NCHW --source_model_output_layout "data2" NCHW \n',
                                       metavar=('OUTPUT_NAME', 'OUTPUT_LAYOUT'), default=[])
            self.add_optional_argument('--desired_output_layout', dest='desired_io_layout', nargs=2, action='append',
                                       help='Desired Layout of each output tensor. If not specified, it will use the default '
                                            'based on the Source Framework.\n'
                                            'Accepted values are-\n'
                                            '    ' + ', '.join(op_graph.InputLayout.get_supported_types_v2()) + '\n'
                                            'N = Batch, C = Channels, D = Depth, H = Height, W = Width, F = Feature, T = Time\n'
                                            'NDHWC/NCDHW used for 5d outputs\n'
                                            'NHWC/NCHW used for 4d image-like outputs\n'
                                            'NFC/NCF used for outputs to Conv1D or other 1D ops\n'
                                            'NTF/TNF used for outputs with time steps like the ones used for LSTM op\n'
                                            'NF used for 2D outputs, like the outputs to Dense/FullyConnected layers\n'
                                            'NC used for 2D outputs with 1 for batch and other for Channels (rarely used) \n'
                                            'F used for 1D outputs, e.g. Bias tensor\n'
                                            'For multiple outputs specify multiple --desired_output_layout on the command line.\n'
                                            'Eg: \n'
                                            '    --desired_output_layout "data1" NCHW --desired_output_layout "data2" NCHW \n',
                                       metavar=('OUTPUT_NAME', 'DESIRED_OUTPUT_LAYOUT'), default=[])
            self.add_optional_argument('--desired_input_color_encoding', "-e", dest='input_encoding', nargs='+', action='append',
                                       help='Usage: '
                                            '    --input_color_encoding "INPUT_NAME" INPUT_ENCODING_IN [INPUT_ENCODING_OUT]\n'
                                            'Input encoding of the network inputs. Default is bgr. \n'
                                            'e.g.\n'
                                            '   --input_color_encoding "data" rgba \n'
                                            'Quotes must wrap the input node name to handle special characters, \n'
                                            'spaces, etc. To specify encodings for multiple inputs, invoke \n'
                                            '--input_color_encoding for each one. \n'
                                            'e.g.\n'
                                            '    --input_color_encoding "data1" rgba --input_color_encoding "data2" other\n'
                                            'Optionally, an output encoding may be specified for an input node by \n'
                                            'providing a second encoding. The default output encoding is bgr.\n'
                                            'e.g. \n'
                                            '    --input_color_encoding "data3" rgba rgb \n'
                                            'Input encoding types:\n '
                                            '    image color encodings: bgr,rgb, nv21, nv12, ... \n'
                                            '    time_series: for inputs of rnn models; \n'
                                            '    other: not available above or is unknown. \n'
                                            'Supported encodings:\n   ' + '\n   '.join(op_graph.InputEncodings.
                                                                                       get_supported_encodings_v2()),
                                       metavar="\b", default=[])

            self.add_optional_argument('--preserve_io_datatype', nargs='*', action='append', default=[],
                                       help='Use this option to preserve IO datatype. The different ways of using this option are as follows:\n'
                                            '    --preserve_io_datatype <space separated list of names of inputs and outputs of the graph>\n'
                                            'e.g.\n'
                                            '    --preserve_io_datatype input1 input2 output1 \n'
                                            'The user may choose to preserve the datatype for all the inputs and outputs of the graph.\n'
                                            '    --preserve_io_datatype\n'
                                            'Note: --config gets higher precedence than --preserve_io_datatype.\n')
            self.add_optional_argument('--dump_io_config_template', type=str, default="", help=argparse.SUPPRESS)
            self.add_optional_argument('--dump_config_template', dest='dump_io_config_template', type=str, default="",
                                       help='Dumps the yaml template for I/O configuration. This file can '
                                            'be edited as per the custom requirements and passed using the option --config'
                                            'Use this option to specify a yaml file to which the IO config template is dumped.')
            self.add_optional_argument('--io_config', type=str, default="", help=argparse.SUPPRESS)
            self.add_optional_argument('--config', dest='io_config', type=str, default="",
                                       help='Use this option to specify a yaml file for input and output options.')
            self.add_optional_argument("--dry_run", type=str, nargs='?', const='info', default=None,
                                       help='Evaluates the model without actually converting any ops, and '
                                            'returns unsupported ops/attributes as well as unused inputs and/or '
                                            'outputs if any.')
            self.add_optional_argument('--keep_io_legacy_qnn_layout', '--keep_io_legacy_snpe_layout', dest='disable_preserve_io',
                                       action="store_true", default=False, help=argparse.SUPPRESS)
            self.add_optional_argument('--enable_framework_trace', default=False, action="store_true",
                                       help='Use this option to enable converter to trace the op/tensor change information.\n'
                                            'Currently framework op trace is supported only for ONNX converter.')
            self.add_optional_argument('--remove_unused_inputs', default=False, action="store_true",
                                       help="Use this option to remove the disconnected graph input nodes after the conversion")

            q_group = self.add_argument_group(title='Quantizer Options')
            q_group.add_argument('--quantization_overrides', '-q', dest='quantization_overrides', type=str, default="",
                                 help='Use this option to specify a json file with parameters to use for '
                                      'quantization. These will override any quantization data carried from conversion '
                                      '(eg TF fake quantization) or calculated during the normal quantization process. '
                                      'Format defined as per AIMET specification.')
            q_group.add_argument('--keep_quant_nodes', default=False, action="store_true",
                                 help=argparse.SUPPRESS)
            q_group.add_argument('--pack_4_bit_weights', action='store_true', default=False,
                                 help=argparse.SUPPRESS)
            q_group.add_argument('--disable_dynamic_16_bit_weights', action='store_true', default=False, help=argparse.SUPPRESS)

    def __init__(self, args,
                 naming_policy=ConversionNamePolicy(),
                 shape_inference_policy=None,
                 axis_order=AxisOrder(),
                 **kwargs):
        super(ConverterFrontend, self).__init__(args, **kwargs)

        self.dump_io_config_template = ''
        self.source_output_layouts = []
        self.desired_io_layouts = []
        self.output_names = args.out_names
        self.remove_unused_inputs = True

        if 'qairt_converter_api' in args:
            self.remove_unused_inputs = args.remove_unused_inputs

        if self.qairt_converter:
            self.source_output_layouts = args.output_layout
            self.desired_io_layouts = args.desired_io_layout
            self.dump_io_config_template = args.dump_io_config_template
            self.remove_unused_inputs = args.remove_unused_inputs
            if '--desired_input_shape' or '-d' in sys.argv:
                log_warning("--desired_input_shape and -d are deprecated. Use --source_model_input_shape or -s for achieving this functionality")
            if '--io_config' in sys.argv:
                log_warning("--io_config option is deprecated, use --config")
            if '--dump_io_config_template' in sys.argv:
                log_warning("--dump_io_config_template option is deprecated, use --dump_config_template")
            if '--keep_int64_inputs' in sys.argv:
                log_warning("--keep_int64_inputs flag is deprecated, use --preserve_io_datatype for achieving same functionality")
            if '--onnx_no_simplification' in sys.argv:
                log_warning("--onnx_no_simplification option is deprecated, use --onnx_skip_simplification or -oss")
            if '--onnx_batch' in sys.argv:
                log_warning("--onnx_batch option is deprecated, use --onnx_override_batch")
            if '--tf_no_optimization' in sys.argv:
                log_warning("--tf_no_optimization option is deprecated, use --tf_disable_optimization")
            if '--onnx_defer_loading' in sys.argv:
                log_warning("--onnx_defer_loading flag is enabled by default, no longer need to pass this option as it will be deprecated."
                            "To disable it, use --onnx_disable_defer_loading")

        user_custom_io = {}
        if hasattr(args, 'user_custom_io'):
            user_custom_io = args.user_custom_io
        elif args.custom_io:
            with open(args.custom_io) as fp:
                user_custom_io = yaml.safe_load(fp)

        for input_encoding in args.input_encoding:
            if len(input_encoding) not in [2, 3]:
                raise ValueError('Received incorrect number of input encodings for input {}. Got {}, expected \n'
                                 'one input encoding and one (optional) output encoding per graph input in the \n'
                                 'following format: \n'
                                 '    --input_encoding "INPUT_NAME" INPUT_ENCODING_IN [INPUT_ENCODING_OUT] \n'
                                 .format(input_encoding[0], len(input_encoding) - 1))

        lora_tensor_names = []
        if hasattr(args, "lora_weight_list") and args.lora_weight_list != None:
            lora_tensor_names = get_lora_tensor_names_from_file(args.lora_weight_list)
            if not lora_tensor_names:
                raise ValueError('lora_weight_list cannot be empty for quant_updatable_mode=none')

        self.graph = op_graph.IROpGraph(naming_policy, shape_inference_policy,
                                        args.input_type, args.input_dtype, args.input_encoding,
                                        axis_order,
                                        input_layouts=args.input_layout,
                                        source_output_layouts=self.source_output_layouts,
                                        desired_io_layouts = self.desired_io_layouts,
                                        quantization_overrides=args.quantization_overrides,
                                        user_custom_io=user_custom_io,
                                        preserve_io=args.preserve_io,
                                        keep_quant_nodes=args.keep_quant_nodes,
                                        output_nodes=self.output_names,
                                        keep_int64_inputs=self.keep_int64_inputs,
                                        enable_framework_trace=args.enable_framework_trace,
                                        qairt_converter=self.qairt_converter,
                                        remove_unused_inputs=self.remove_unused_inputs,
                                        use_quantize_v2=args.use_quantize_v2,
                                        lora_tensor_names=lora_tensor_names)

        graph_optimizer.OptimizationPassModeManager.get_instance().set_qairt_converter(self.qairt_converter)
        graph_optimizer.OptimizationPassModeManager.get_instance().set_args(args)

        self.graph.preserve_io_datatype = ''
        if hasattr(args, 'preserve_io_datatype'):
            self.graph.preserve_io_datatype = args.preserve_io_datatype

    def summarize_model(self, model_summary: FrameworkSummary) -> FrameworkSummary:
        """
        Summarize the model contained in the ConverterFrontend class or its derived class.
        Summary will provide information about model's input-output name, shape, dtype and
        various nodes present in the model.

        :param FrameworkSummary model_summary: Model summary object containing
            information about model.
        :return FrameworkSummary: Model summary object returned for further
            processing, post printing the contents on console.
        """
        summary_str = str(model_summary)
        log_info(summary_str)
        return model_summary

    def dump_io_config_yaml_template(self):
        yaml_data = []
        i_str = "# For complete graph or subgraph conversion\n"
        i_str += "Converted Graph:\n"
        i_str += "  - Input Tensors:\n"
        i_str += "  - Output Tensors:\n\n"

        i_str += "# Input tensors specified in this section should be subset of subgraph (if specified)\n"
        i_str += "Input Tensor Configuration:\n"
        i_str += "  # Input 1\n"
        i_str += "  - Name:\n"
        i_str += "    Src Model Parameters:\n"
        i_str += "        DataType:\n"
        i_str += "        Layout:\n"
        i_str += "        Shape:\n"
        i_str += "    Desired Model Parameters:\n"
        i_str += "        DataType:\n"
        i_str += "        Layout:\n"
        i_str += "        Color Conversion:\n"
        i_str += "        QuantParams:\n          Scale:\n          Offset:\n\n"
        yaml_data.append(i_str)

        o_str = "Output Tensor Configuration:\n"
        o_str += "  # Output 1\n"
        o_str += "  - Name:\n"
        o_str += "    Src Model Parameters:\n"
        o_str += "        DataType:\n"
        o_str += "        Layout:\n"
        o_str += "    Desired Model Parameters:\n"
        o_str += "        DataType:\n"
        o_str += "        Layout:\n"
        o_str += "        QuantParams:\n          Scale:\n          Offset:\n\n"
        yaml_data.append(o_str)

        f = open(self.dump_io_config_template, 'w')
        f.write('\n'.join(yaml_data))
        log_info("Dumped IO config template in file %s" % (self.dump_io_config_template))
        f.close()

    @abstractmethod
    def convert(self):
        """
        Convert the input framework model to IROpGraph: to be overridden by each framework
        """
        pass

