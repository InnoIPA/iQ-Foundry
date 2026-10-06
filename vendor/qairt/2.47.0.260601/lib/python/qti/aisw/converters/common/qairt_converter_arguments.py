# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import argparse
import ast
import yaml
from enum import Enum

from qti.aisw.converters.common.converter_base import ConverterFrontend
from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.utils.converter_utils import log_warning
from qti.aisw.converters.common import ir_quantizer

optimized_nms_enabled = False

if optimized_nms_enabled:
    from qti.aisw.converters.onnx.optimizations import nms_utils

class ExperimentalFeature(Enum):
    """Enumeration of all experimental features."""

class QairtConverterFrontendArgParser(ConverterFrontend.ArgParserv2):
    def __init__(self, **kwargs):
        super(QairtConverterFrontendArgParser, self).__init__(conflict_handler='resolve', **kwargs)

        lora_group = self.add_argument_group(title='LoRA Converter Options')
        lora_group.add_argument('--lora_weight_list', type=str, default=None,
                                action=validation_utils.validate_filename_arg(must_exist=False),
                                help='Path to a file specifying a list of tensor names that should be updateable.')
        lora_group.add_argument('--skip_validation', action='store_true', default=False,
                                help=argparse.SUPPRESS)
        lora_group.add_argument("--quant_updatable_mode",
                                type=str,
                                choices=["none", "adapter_only", "all", "float_only"],
                                help="Specify whether/for which tensors the quantization encodings change " \
                                     "across use-cases. In none mode, no quantization encodings are updatable. " \
                                     "In adapter_only mode quantization encodings for " \
                                     "only lora/adapter branch (Conv->Mul->Conv) change across use-case, "
                                     "the base branch quantization encodings remain the same. " \
                                     "In all mode, all quantization encodings are updatable." \
                                     "In float_only mode, both the base model and all use‑case models will be in float and "
                                     " no quantization encodings will be present for any use-cases")
        lora_group.add_argument('--disable_transform_tracking', action='store_true', default=False,
                                help=argparse.SUPPRESS)
        lora_group.add_argument('--transforms_metadata', type=str, default=None,
                                action=validation_utils.validate_filename_arg(must_exist=True),
                                help="Path to a JSON file for storing transformation metadata. " \
                                     "This file contains pre-existing transforms added by TransformManager, and converter transforms " \
                                     "will be appended to it.")

        onnx_group = self.add_argument_group(title='Onnx Converter Options')
        onnx_group.add_argument('--onnx_no_simplification', dest='no_simplification', action='store_true', default=False,
                                         help=argparse.SUPPRESS)
        onnx_group.add_argument('--onnx_skip_simplification', '-oss', dest='no_simplification', action='store_true', default=False,
                                help="Do not attempt to simplify the model automatically. This may prevent some models from \n"
                                     "properly converting  when sequences of unsupported static operations are present.")
        onnx_group.add_argument('--onnx_batch', dest='batch', type=int, default=None,
                                help=argparse.SUPPRESS)
        onnx_group.add_argument('--onnx_override_batch', dest='batch', type=int, default=None,
                                         help="The batch dimension override. This will take the first dimension of all "
                                              "inputs and treat it as a batch dim, overriding it with the value provided "
                                              "here. For example:\n"
                                              "--onnx_override_batch 6\n"
                                              "will result in a shape change from [1,3,224,224] to [6,3,224,224].\n"
                                              "If there are inputs without batch dim this should not be used and each input "
                                              "should be overridden independently using -d option for input dimension overrides.")
        onnx_group.add_argument('--onnx_define_symbol', dest='define_symbol', nargs=2, action='append',
                                         metavar=('SYMBOL_NAME', 'VALUE'),
                                         help="This option allows overriding specific input dimension symbols. For instance you "
                                              "might see input shapes specified with variables such as :\n"
                                              "data: [1,3,height,width]\n"
                                              "To override these simply pass the option as:\n"
                                              "--onnx_define_symbol height 224 --onnx_define_symbol width 448\n"
                                              "which results in dimensions that look like:\n"
                                              "data: [1,3,224,448]")
        #TODO: Remove --onnx_defer_loading flag once it is deprecated in the future
        onnx_group.add_argument('--onnx_defer_loading', action='store_true', default=False, dest='defer_loading',
                                       help=argparse.SUPPRESS)
        onnx_group.add_argument('--onnx_disable_defer_loading', action='store_true', default=False, dest='disable_defer_loading',
                                       help=argparse.SUPPRESS)
        onnx_group.add_argument("--onnx_validate_models", dest='validate_models', action="store_true",
                                help="Validate the original ONNX model against optimized ONNX model.\n"
                                     "Constant inputs with all value 1s will be generated and will be used \n"
                                     "by both models and their outputs are checked against each other.\n"
                                     "The %% average error and 90th percentile of output differences will be calculated for this.\n"
                                     "Note: Usage of this flag will incur extra time due to inference of the models.")
        onnx_group.add_argument('--onnx_summary', action='store_true', dest='onnx_summary', default=False,
                                    help="Summarize the original onnx model and optimized onnx model.\n" \
                                        "Summary will print the model information such as number of parameters,\n" \
                                        "number of operators and their count, input-output tensor name, shape and dtypes.")
        onnx_group.add_argument('--onnx_perform_sequence_construct_optimizer', dest='perform_sequence_construct_optimizer',
                                         action='store_true', default=False,
                                         help="This option allows optimization on SequenceConstruct Op.\n"
                                              "When SequenceConstruct op is one of the outputs of the graph, "
                                              "it removes SequenceConstruct op and makes its inputs as graph outputs "
                                              "to replace the original output of SequenceConstruct.")

        onnx_group.add_argument('--enable_tensor_deduplication', action='store_true', dest='enable_tensor_deduplication',
                                        help=argparse.SUPPRESS, default=False)
        onnx_group.add_argument('--dump_inferred_model', action='store_true', default=False,
                                         help=argparse.SUPPRESS)
        onnx_group.add_argument('--dump_value_info', action='store_true', default=False,
                                         help=argparse.SUPPRESS)
        # hidden flag for onnx relay converter
        onnx_group.add_argument('--use_onnx_relay', action='store_true', default=False,
                                         help=argparse.SUPPRESS)
        onnx_group.add_argument('--dump_relay', type=str, default=None,
                                         help=argparse.SUPPRESS)


        if optimized_nms_enabled:
            nms_group = self.add_argument_group(
                title='Host/Device NMS Optimization Options')
            nms_group.add_argument("--onnx_nms_type",
                                    choices=nms_utils.NMSType.get_nms_types(),
                                    type=str.upper,
                                    help="This flag enables the pass that modifies object detection model\n" \
                                        "such that the anchor box processing and NonMaxSuppression part of the\n" \
                                        "model is efficiently executed. Host nms will partition the graph at\n" \
                                        "feature extractors and ABP+NMS part will be executed by supported HostNMS\n" \
                                        "library. Device nms will modify the existing graph to execute the standard\n" \
                                        "anchor box processing and NMS part on device as a part of graph.\n" \
                                        "Please check the supplemental backend XML for the targeted backend.\n" \
                                        "This feature is only supported for onnx models.")
            nms_group.add_argument("--onnx_nms_arch_type",
                                    choices=nms_utils.ModelArchType.get_supported_models(),
                                    type=str.upper,
                                    help="Type of the model architecture. E.g. YOLOV5, MV1SSD etc.\n" \
                                        "Based on the architecture provided required feature extractors\n" \
                                        "will be identified and graph will be updated as per nms_type flag.\n" \
                                        "This flag should be used along with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_max_boxes",
                                    type=int,
                                    help="The total number of boxes the model should return across all classes after\n" \
                                        "NMS optimization. This flag should be used along with --nms_type to enable\n" \
                                        "NMS optimization.")
            nms_group.add_argument("--onnx_nms_max_boxes_per_class",
                                    type=int,
                                    help="The total number of boxes to be filtered out per class during NMS\n" \
                                        "computation of the model. This flag should be used along with\n" \
                                        "--nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_iou_threshold",
                                    type=float,
                                    help="IoU (Intersection over Union) threshold to be used during NMS\n" \
                                        "computation of the model. Its value should be in [0, 1] interval.\n" \
                                        "This flag should be used along with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_score_threshold",
                                    type=float,
                                    help="Box probability score threshold to be used during NMS computation\n" \
                                        "of the model. Its value should be in [0, 1] interval. This flag \n" \
                                        "should be used along with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_num_classes",
                                    type=int,
                                    help="Number of classes that the model is predicting, including background class.\n" \
                                        "This flag should be used along with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_class_specific_nms",
                                    action="store_true",
                                    default=False,
                                    help="Perform class specific NMS in NMS computation. This flag should\n" \
                                        "be used along with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_background_class_idx",
                                    type=int,
                                    default=None,
                                    help="Class index of background class. It will be used during decoding\n" \
                                        "the predictions of the model. This flag should be used along\n" \
                                        "with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_map_coco_80_to_90",
                                    action="store_true",
                                    default=False,
                                    help="Maps the coco 80 class outputs into 90 class outputs. This will\n" \
                                        "be helpful while evaluating the mAP of model trained on COCO dataset.\n" \
                                        "This flag is applicable only for Host NMS. This flag should\n" \
                                        "be used along with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_anchor_data",
                                    type=str,
                                    default=None,
                                    help="Path of the anchor data used during anchor box processing.\n" \
                                        "The file should contain all the anchors in float32 datatype only.\n" \
                                        "This is required for anchor based object detection models only.\n" \
                                        "For Yolo category of models, the anchors shall be of shape \n" \
                                        "[num_anchors, 2] in the width-height order.\n" \
                                        "For SSD category of models, the anchors shall be of shape \n" \
                                        "[num_anchors, 4] in the yxyx order.\n" \
                                        "This flag should be used along with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_boxes_format",
                                    type=str,
                                    default="xywh",
                                    choices=["xywh", "yxhw"],
                                    help="Format of raw boxes obtained from feature extractor. This flag \n" \
                                        "is applicable for SSD category of models. This flag should be \n" \
                                        "used along with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_scale_xy",
                                    type=float,
                                    default=1.0,
                                    help="Scaling of x and y predictions while obtaining processed boxes.\n" \
                                        "This flag is applicable for SSD based models. This flag should be \n" \
                                        "used along with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_scale_wh",
                                    type=float,
                                    default=1.0,
                                    help="Scaling of w and h predictions while obtaining processed boxes.\n" \
                                        "This flag is applicable for SSD based models. This flag should be \n" \
                                        "used along with --nms_type to enable NMS optimization.")
            nms_group.add_argument("--onnx_nms_scores_activation",
                                    type=str,
                                    default="Sigmoid",
                                    choices=["Sigmoid", "Softmax"],
                                    help="Activation function to be applied to raw scores\n" \
                                        "prediction branch of the model. This flag is applicable for\n" \
                                        "SSD based models. This flag should be used along\n" \
                                        "with --nms_type to enable NMS optimization.")


        tf_group = self.add_argument_group(title='TensorFlow Converter Options')
        # add command-line options custom to tensorflow converter
        tf_group.add_argument('--tf_no_optimization', dest='no_optimization', action='store_true', default=False,
                                       help=argparse.SUPPRESS)
        tf_group.add_argument('--tf_batch', dest='batch', type=int, default=None,
                                help=argparse.SUPPRESS)
        tf_group.add_argument('--tf_override_batch', dest='batch', type=int, default=None,
                                help="The batch dimension override. This will take the first dimension of all "
                                     "inputs and treat it as a batch dim, overriding it with the value provided "
                                     "here. For example:\n"
                                     "--tf_override_batch 6\n"
                                     "will result in a shape change from [1,224,224,3] to [6,224,224,3].\n"
                                     "If there are inputs without batch dim this should not be used and each input "
                                     "should be overridden independently using -s option for input dimension overrides.")
        tf_group.add_argument('--tf_disable_optimization', dest='no_optimization', action='store_true', default=False,
                                       help="Do not attempt to optimize the model automatically.")
        tf_group.add_argument("--tf_show_unconsumed_nodes", dest='show_unconsumed_nodes', action="store_true",
                                       help="Displays a list of unconsumed nodes, if there any are found. Nodes"
                                            "which are unconsumed do not violate the structural fidelity of the"
                                            "generated graph.",
                                       default=False)
        tf_group.add_argument("--tf_saved_model_tag", dest='saved_model_tag', type=str, action='store',
                                       help="Specify the tag to seletet a MetaGraph from savedmodel. ex: "
                                            "--saved_model_tag serve. Default value will be 'serve' when it "
                                            "is not assigned.",
                                       default="serve")
        tf_group.add_argument("--tf_saved_model_signature_key", dest='saved_model_signature_key', type=str, action='store',
                                       help="Specify signature key to select input and output of the model. "
                                            "ex: --tf_saved_model_signature_key serving_default. Default value "
                                            "will be 'serving_default' when it is not assigned",
                                       default="serving_default")
        tf_group.add_argument("--tf_validate_models", dest='validate_models', action="store_true",
                                       help="Validate the original TF model against optimized TF model.\n"
                                            "Constant inputs with all value 1s will be generated and will be used \n"
                                            "by both models and their outputs are checked against each other.\n"
                                            "The %% average error and 90th percentile of output differences will be calculated for this.\n"
                                            "Note: Usage of this flag will incur extra time due to inference of the models.")
        onnx_group.add_argument('--tf_summary', action='store_true', dest='tf_summary', default=False,
                                    help="Summarize the original TF model and optimized TF model.\n" \
                                        "Summary will print the model information such as number of parameters,\n" \
                                        "number of operators and their count, input-output tensor name, shape and dtypes.")

        # TODO: remove once QNN supports known LSTM variants completely (such as multiple time-steps)
        # Added as a workaround to match lstm nodes as low-level ops
        tf_group.add_argument("--disable_match_lstms", action='store_true', default=False,
                                       help=argparse.SUPPRESS)

        tflite_group = self.add_argument_group(title='Tflite Converter Options')
        tflite_group.add_argument('--tflite_signature_name', dest='signature_name', type=str, default="",
                                   help='Use this option to specify a specific Subgraph signature to convert')
        tflite_group.add_argument('--partial_graph_input_name', action='append',
                                   help=argparse.SUPPRESS)
        tflite_group.add_argument('--dump_relay', type=str, default=None,
                                   help=argparse.SUPPRESS)

        pytorch_group = self.add_argument_group(title='PyTorch Converter Options')
        pytorch_group.add_argument('--pytorch_custom_op_lib', type=str, default="",
                                    help=argparse.SUPPRESS)
        pytorch_group.add_argument('--dump_relay', type=str, default=None,
                                   help=argparse.SUPPRESS)

        pytorch_group.add_argument('--dump_exported_onnx', action='store_true', default=False,
                                   help="Dump the exported Onnx model from input Torchscript model")

        # Hidden flag to override config.json generated by GGUF Builder
        self.add_optional_argument('--gguf_config', type=str, default=None, help=argparse.SUPPRESS)

        # A general flag can take multiple arguments,
        # which is used to create a delivery mechanism for all experimental feature.
        self.add_optional_argument(
            "--enable_experimental_feature",
            type=str,
            action="append",
            default=[],
            help=argparse.SUPPRESS,
        )
        # Hidden flag to enable legacy Axis-Tracking (Layout_Transform_v1)
        self.add_optional_argument(
            "--enable_Layout_Transform_v1",
            action='store_true',
            default=False,
            help=argparse.SUPPRESS,
        )
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
        self.add_optional_argument('--use_quantize_v2', action='store_true', default=False,
                                   help=argparse.SUPPRESS)

# Convert argsv2 (from qairt_converter which accepts i/o yaml) to argsv1 (used by SNPE/QNN)
def convert_args_v2_to_v1(args):
    args_dict = vars(args)

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

    if args.io_config:
        f = open(args.io_config)
        io_config_dict = yaml.safe_load(f)

        input_layout_dict = {}
        output_layout_dict = {}

        if 'Converted Graph' in io_config_dict:
            for i in range(len(io_config_dict['Converted Graph'])):
                for key, val in io_config_dict['Converted Graph'][i].items():
                    if key == 'Output Tensors' and val is not None:
                        for buffer_name in val:
                            output_names.append(str(buffer_name))

        if not args_dict['out_names']:
            args_dict['out_names'] = output_names

        if 'Input Tensor Configuration' in io_config_dict and io_config_dict['Input Tensor Configuration']:
            for i in range(len(io_config_dict['Input Tensor Configuration'])):
                for key, val in io_config_dict['Input Tensor Configuration'][i].items():
                    if key == 'Name':
                        if val is not None:
                            name = str(val)
                    elif key == 'Src Model Parameters':
                        if 'DataType' in val and val['DataType']:
                            input_dtype.append([name, val['DataType']])
                        if 'Layout' in val and val['Layout']:
                            input_layout.append([name, val['Layout']])
                            input_layout_dict[name] = val['Layout']
                        if 'Shape' in val and val['Shape']:
                            if input_dims is None:
                                input_dims = []

                            # for cases when user passes a shape with one dimension
                            # e.g. Shape: 1
                            if isinstance(val["Shape"], int):
                                val["Shape"] = "(" + str(val['Shape']) + ",)"

                            dim = ast.literal_eval(val['Shape'])

                            # for cases when user passes a shape with one dimension
                            # e.g. Shape: (1)
                            if isinstance(dim, int):
                                dim = (dim,)

                            if type(dim[0]) is tuple:
                                network_specialization = True
                            input_dims.append([name, dim])
                    elif key == 'Desired Model Parameters':
                        if 'Shape' in val and val['Shape']:
                            if input_dims is None:
                                input_dims = []

                            # for cases when user passes a shape with one dimension
                            # e.g. Shape: 1
                            if isinstance(val["Shape"], int):
                                val["Shape"] = "(" + str(val['Shape']) + ",)"

                            dim = ast.literal_eval(val['Shape'])

                            # for cases when user passes a shape with one dimension
                            # e.g. Shape: (1)
                            if isinstance(dim, int):
                                dim = (dim,)

                            if type(dim[0]) is tuple:
                                network_specialization = True
                            input_dims.append([name, dim])
                            log_warning("Shape Configuration is expected in 'Src Model Parameters' section of IO "
                                        "Config file. Please refer to the latest IO Config file format")

                        custom_io_options = dict()
                        custom_io_options['IOName'] = name
                        if 'DataType' in val and val['DataType']:
                            custom_io_options['Datatype'] = val['DataType']
                        if 'Layout' in val and val['Layout']:
                            custom_io_options['Layout'] = dict()
                            custom_io_options['Layout']['Custom'] = val['Layout']
                            # Get the model layout corresponding to the custom layout for current input
                            if name in input_layout_dict:
                                custom_io_options['Layout']['Model'] = input_layout_dict[name]
                        # if any of the quant params are provided
                        if 'QuantParams' in val and (val['QuantParams']['Scale'] or val['QuantParams']['Offset']):
                            custom_io_options['QuantParam'] = val['QuantParams']
                            custom_io_options['QuantParam']['Type'] = 'QNN_DEFINITION_DEFINED'
                        # optional IO tensors
                        custom_io_options['Optional'] = val.get('Optional', False)
                        if len(custom_io_options) > 1:
                            user_custom_io.append(custom_io_options)
                        if 'Color Conversion' in val and val['Color Conversion']:
                            input_encoding.append([name, val['Color Conversion']])

        if 'Output Tensor Configuration' in io_config_dict and io_config_dict['Output Tensor Configuration']:
            for i in range(len(io_config_dict['Output Tensor Configuration'])):
                name = ""
                for key, val in io_config_dict['Output Tensor Configuration'][i].items():
                    if key == 'Name':
                        if val is not None:
                            if (not args_dict['out_names'] or
                                    (args_dict['out_names'] and str(val) in args_dict['out_names'])):
                                name = str(val)
                    elif key == 'Src Model Parameters' and name:
                        if 'Layout' in val and val['Layout']:
                            output_layout_dict[name] = val['Layout']
                    elif key == 'Desired Model Parameters' and name:
                        custom_io_options = dict()
                        custom_io_options['IOName'] = name
                        if 'Layout' in val and val['Layout']:
                            custom_io_options['Layout'] = dict()
                            custom_io_options['Layout']['Custom'] = val['Layout']
                            # Get the model layout corresponding to the custom layout for current output
                            if name in output_layout_dict:
                                custom_io_options['Layout']['Model'] = output_layout_dict[name]
                        if 'DataType' in val and val['DataType']:
                            custom_io_options['Datatype'] = val['DataType']
                        # if any of the quant params are provided
                        if 'QuantParams' in val and (val['QuantParams']['Scale'] or val['QuantParams']['Offset']):
                            custom_io_options['QuantParam'] = val['QuantParams']
                            custom_io_options['QuantParam']['Type'] = 'QNN_DEFINITION_DEFINED'
                        # optional IO tensors
                        custom_io_options['Optional'] = val.get('Optional', False)
                        if len(custom_io_options) > 1:
                            user_custom_io.append(custom_io_options)

    if args_dict['desired_io_layout']:
        source_io_layout_dict = dict(args_dict['input_layout'] + args_dict['output_layout'])
        input_dtype_dict = dict(args_dict['input_dtype'])

        for buffer_name, desired_layout in args_dict['desired_io_layout']:
            data_already_exist = False
            data_index = -1
            for entry in user_custom_io:
                if buffer_name in entry['IOName']:
                    data_already_exist = True
                    data_index = user_custom_io.index(entry)
                    break

            if not data_already_exist:
                custom_io_option = dict()
                custom_io_option['IOName'] = buffer_name
                custom_io_option['Layout'] = dict()
                if buffer_name in source_io_layout_dict:
                    custom_io_option['Layout']['Model'] = source_io_layout_dict[buffer_name]
                custom_io_option['Layout']['Custom'] = desired_layout
                if buffer_name in input_dtype_dict:
                    custom_io_option['Datatype'] = input_dtype_dict[buffer_name]
                user_custom_io.append(custom_io_option)

            else:
                entry = user_custom_io[data_index]
                if 'Layout' not in entry:
                    user_custom_io[data_index]['Layout'] = dict()
                if buffer_name in source_io_layout_dict:
                    user_custom_io[data_index]['Layout']['Model'] = source_io_layout_dict[buffer_name]
                user_custom_io[data_index]['Layout']['Custom'] = desired_layout
                if buffer_name in input_dtype_dict:
                    user_custom_io[data_index]['Datatype'] = input_dtype_dict[buffer_name]

    # update following args only if they were not provided on the commandline
    if not args_dict['input_dim']:
        # convert name:str, dim:tuple to name:str, dim:str if network specialization is disabled
        if input_dims and not network_specialization:
            for i in range(len(input_dims)):
                # convert tuple of dimension to comma separated string
                if type(input_dims[i][1]) is tuple:
                    input_dims[i][1] = ','.join(map(str, input_dims[i][1]))
                # remove whitespaces if any from string of dimension
                elif isinstance(input_dims[i][1], str):
                    input_dims[i][1] = input_dims[i][1].replace(" ", "")

        args_dict["input_dim"] = input_dims

    if not args_dict['input_layout']:
        args_dict['input_layout'] = input_layout
    if not args_dict['input_dtype']:
        args_dict['input_dtype'] = input_dtype
    if not args_dict['input_encoding']:
        args_dict['input_encoding'] = input_encoding

    # following arguments will be unused
    args_dict['input_type'] = []
    args_dict['dump_custom_io_config_template'] = ""
    args_dict['user_custom_io'] = user_custom_io

    # populate preserve_io_arg with [['layout']] to apply it to all inputs/outputs
    args_dict['preserve_io'] = [['layout']]

    if args.perform_layout_transformation:
        # If layout-transform is used, we don't need to populate preserve_io_arg with [['layout']]
        # because preserve_io is layout-transform's default behavior.
        args_dict['preserve_io'] = [[]]

    if args.disable_preserve_io:
        args_dict['preserve_io'] = []

    if args.preserve_io_datatype:
        args_dict['preserve_io'].append(['datatype'] + args.preserve_io_datatype[0])

    return argparse.Namespace(**args_dict)
