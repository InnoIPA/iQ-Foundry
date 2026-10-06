# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


error_codes_to_messages = {
    # //=============================================================================
    # //                 TENSORFLOW CONVERTER ERROR CODES
    # //=============================================================================

    # start of the batchnorm errors
    'ERROR_TF_GENERAL_ABSTRACT_CLASS_MUST_BE_INHERITED': "{} not implemented for {}. This abstract class must be inherited.",
    'ERROR_TF_BATCHNORM_GLOBALNORMALIZATION_INPUT': "Cannot resolve BatchNorm layer {} due to BatchNormWithGlobalNormalization node not having at least 4 const inputs (mean, variance, beta, scale).",

    # start of the concat errors
    'ERROR_TF_CONCAT_INPUT': "Concatenation layer {} requires at least two inputs.",

    # start of the conv errors
    'ERROR_TF_CONV_RESOLVE_DILATION': "Cannot resolve convolution layer due to missing dilation for operation: {}",
    'ERROR_TF_CONV_RESOLVE_PADDING': "Cannot resolve convolution layer due to padding tensor of size {}, needs to be of size (2,2) for operation: {}",
    'ERROR_TF_CONV_RESOLVE_CROP': "Cannot resolve convolution layer due to crop tensor of size {}, needs to be of size (2,2) for operation: {}",

    # crop_and_resize error
    'ERROR_TF_CROPANDRESIZE_INTERPOLATION_UNKNOWN': "Unsupported interpolation method {} for Crop_And_Resize layer {}",
    'ERROR_TF_RESOLVE_CROP_AND_RESIZE_NUM_BOXES': "Crop_And_Resize layer {} expects num_boxes to be a list of size=1.",
    'ERROR_TF_RESOLVE_CROP_AND_RESIZE_SIZE_NOT_CONST': "Crop_And_Resize layer {} only Const crop_size.",
    'ERROR_TF_RESOLVE_CROP_AND_RESIZE_SIZE': "Crop_And_Resize layer {} expected crop_size tensor of 2 elements. Got {}",

    # start of extract glimpse errors
    'ERROR_TF_RESOLVE_EXTRACT_GLIMPSE_SIZE': "Cannot resolve extract glimpse layer {}, glimpse size must be exactly 2 values.",

    # start of the fullyconnected errors
    'ERROR_TF_MATMUL_RESOLVE_BIAS': "Cannot resolve fully connected layer {} due to missing biases for BiasAdd operation: {}",

    # start of the image projective transform errors
    'ERROR_TF_RESOLVE_IMAGE_PROJECTIVE_TRANSFORM_INTERPOLATION': "Cannot resolve image projective transform layer {}, unknown interpolation mode {}",

    # start of the l2_normalize errors
    'ERROR_TF_L2NORM_AXIS_TYPE': "Cannot resolve l2 normalization layer {}, incorrect axis type. Should be an integer.",
    'ERROR_TF_L2NORM_RESOLVE_EPSILON': "Cannot resolve L2Norm layer {} due to missing epsilon value.",
    'ERROR_TF_L2NORM_RESOLVE_AXIS': "Cannot resolve L2Norm layer {} due to missing axis value.",

    # start of the prelu errors
    'ERROR_TF_RESOLVE_PRELU_COEFF': "Cannot resolve PReLu layer {} due to missing coefficient values.",

    # start of the depth_to_space errors
    'ERROR_TF_DEPTH_TO_SPACE_DATA_FORMAT': "Unsupported data format for layer {}. Supported formats:{}, Got:{}",

    # start of the pad errors
    'ERROR_TF_PAD_INVALID_PADDINGS': "Unexpected padding tensor for the pad layer {}. expected: {}, actual: {}",
    'ERROR_TF_PAD_CONSTANT_NOT_SCALAR': "Expected scalar pad value for the pad layer {}.",
    'ERROR_TF_PAD_MODE_UNKNOWN': "Got unsupported padding mode {} for layer {}.",

    # start of the strided slice errors
    'ERROR_TF_STRIDED_SLICE_SHAPE_MISMATCH': "Expected begin, end, and strides tensors to be of same shape for StrideSlice layer {}.",
    'ERROR_TF_STRIDED_SLICE_UNSUPPORTED_MASKS': "ellipsis_mask and new_axis_mask are not supported for StrideSlice layer {}.",

    # start of the permute errors
    'ERROR_TF_PERMUTE_INVALID_ORDER_TENSOR': "Invalid order tensor {} for permute layer {}.",

    # identity_n
    'ERROR_TF_IDENTITY_N_DIFF_IN_OUT': "The num. of in/out is not matched for IdentityN {}. The input names: {} and output names: {}.",

    # nms and gather
    'ERROR_TF_NMS_BOXES_SHAPE': "Shape for boxes tensor must be 2 for NMS layer {}. Got{}",

    # space_to_depth
    'ERROR_TF_SPACE_TO_DEPTH_DATA_FORMAT': "Unsupported data format for space_to_depth layer {}. Supported formats:{}, Got:{}",

    # caffe ssd
    'ERROR_TF_CAFFE_SSD_REQUIRES_2_INPUTS': 'Ssd layer {} Detection Output expects 2 input layers: boxes and classes.',
    'ERROR_TF_CAFFE_SSD_UNKNOWN_INPUT': 'Ssd layer {} Detection Output expects boxes and classes as inputs. Got {}.',

    # MultiClass NMS error
    'ERROR_TF_NMS_GATHER_INVALID_AXIS': 'Expect Gather axis to be 0 when followed by NMS layer {}. Got {}',

    # start of the converter errors
    'ERROR_TF_ADD_N_NUM_OF_INPUTS': "Expected two or more inputs for AddN operation: {}, converter cannot resolve at least two inputs",
    'ERROR_TF_INPUT_IDX_OUT_OF_BOUND_FOR_OP': "Index out of bound. Num. of inputs {}, got idx {}, OP name {}",
    'ERROR_TF_UNABLE_TO_RESOLVE_GRAPH_INPUT_DIMS': "Unable to resolve output dimensions for input node {}.",
    'ERROR_TF_NO_INPUT_TO_CREATE_LAYER': "INTERNAL: No builder found to create layer from descriptor: {}",
    'ERROR_TF_UNABLE_TO_DETERMINE_SCOPE_FOR_OP': "Unable to determine scope_name for operation {} to resolve graph. Looking for {}",

    # start of the loader errors
    'ERROR_TF_INPUT_NODE_SHAPE_DIMS_MISMATCH': "One input shape must be specified for every input node.",
    'ERROR_TF_INVALID_INPUT_DIMS': "Invalid input dimensions: {}",
    'ERROR_TF_GRAPH_FILE_DOES_NOT_EXIST': "Graph file does not exist {}",
    'ERROR_TF_GRAPH_PATH_CANT_BE_RECOGNIZED': "The path is not a valid TF model {}",
    'ERROR_TF_NODES_NOT_FOUND_IN_GRAPH': "No nodes found in graph.",
    'ERROR_TF_CANNOT_IMPORT_GRAPH_FROM_META': "Failed to import graph from meta: {}",
    'ERROR_TF_GRAPH_META_EMPTY': "Graph meta is empty.",
    'ERROR_TF_NODE_NOT_FOUND_IN_GRAPH': "Node not found in graph. Node name: {}",
    'ERROR_TF_SIGNATURES_EMPTY_IN_SAVEDMODEL': "Signature keys are empty in SavedModel {}",

    # start of the util errors
    'ERROR_TF_OPERATION_NOT_FOUND': "Operation with type {} not found within {}",
    'ERROR_TF_INPUT_OPERATION_NOT_FOUND': "Input operation not found for {}",
    'ERROR_TF_MULTIPLE_NODES_FOUND': "Expected single node with type {}",
    'ERROR_TF_INPUT_DOES_NOT_MATCH_COUNT': "Operation ({}) inputs do not match expected count: {} vs {}",
    'ERROR_TF_INPUT_DOES_NOT_MATCH_TYPES': "Operation ({}) inputs do not match expected types: {} vs {}",
    'ERROR_TF_LAYER_INPUT_COUNT_ERROR': "Layer {} expects {} input(s), actual {} for layer {}",
    'ERROR_TF_LAYER_NO_INPUT_FOUND': "{} layer {} requires at least one input layer.",
    'ERROR_TF_FALLBACK_TO_ONDEMAND_EVALUATION': "Unable to resolve operation output shapes in single pass. "
                                                "Using on-demand evaluation!",
    'ERROR_TF_SSD_ANCHOR_INPUT_MISSING': 'Unable to resolve box encoding anchor input later for ssdnms layer {}.',
    'ERROR_TF_SSD_NMS_REQUIRES_2_INPUTS': 'Multi Class Non Max Suppression expects 2 input layers for ssdnms layer {}.',
    'ERROR_TF_SSD_NMS_REQUIRES_SINGLE_INPUT_TENSOR': 'Multi Class Non Max Suppression expects single input tensor for ssdnms layer {}.',
    'ERROR_TF_SSD_NMS_CAN_NOT_RESOLVE_SCORE_THRESHOLD': 'Unable to resolve score threshold for Multi Class Non Max Suppression layer {}.',
    'ERROR_TF_SSD_NMS_CAN_NOT_RESOLVE_IOU': 'Unable to resolve IOU threshold for Multi Class Non Max Suppression layer {}.',
    'ERROR_TF_SSD_CAFFE_CAN_NOT_RESOLVE_SCORE_THRESHOLD': 'Unable to resolve score threshold for Tensorflow based Caffe SSD layer {}.',
    'ERROR_TF_SSD_CAFFE_CAN_NOT_RESOLVE_IOU_THRESHOLD': 'Unable to resolve IOU threshold for Tensorflow based Caffe SSD layer {}.',
    'ERROR_TF_SSD_CAFFE_CAN_NOT_RESOLVE_KEEP_TOP_K': 'Unable to resolve keep top_k for Tensorflow based Caffe SSD layer {}.',
    'ERROR_TF_SSD_CAFFE_CAN_NOT_RESOLVE_NMS_TOP_K': 'Unable to resolve nms top_k for Tensorflow based Caffe SSD layer {}.',
    'ERROR_TF_SSD_CAFFE_CAN_NOT_RESOLVE_OUTPUT_DIM': 'Unable to resolve output dim for Tensorflow based Caffe SSD layer {}. '
                                                     'Last dimension must be 6 or 7 to represent [label, confidence, x_min, y_min, x_max, y_max]'
                                                     'or [image_batch, label, confidence, x_min, y_min, x_max, y_max] respectively.',


    # //=============================================================================
    # //                 ONNX CONVERTER ERROR CODES
    # //=============================================================================
    "ERROR_ASYMMETRIC_PADS_VALUES": "Asymmetric pads values is not supported",
    "ERROR_ATTRIBUTE_WRONG_TYPE": "Node {}: requested to extract parameter {} with type {}, but stored as type {}",
    "ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS": "kernel_shape parameter differs from weights shape",
    "ERROR_WEIGHTS_MISSING_KEY": "Expected a static initializer for value {}",
    "ERROR_ONNX_NOT_FOUND": "Error loading onnx, Message: {}. PYTHONPATH: {}",
    "ERROR_PAD_UNSUPPORTED_MODE": "Unsupported mode {} set in Pad layer",
    "ERROR_NUMBER_OF_PADS_UNSUPPORTED": "Node {}: Converter only support pads for 2D images for spatial values Height and Width. Got {}.",


    # //=============================================================================
    # //                 IR GENERAL CONVERTER ERROR CODES
    # //=============================================================================
    "ERROR_TIMESERIES_UNEXPECTED_RANK": "Input {} of format \"time_series\" must have shape of 3. Input has unexpected rank of {}",
    "ERROR_UNSUPPORTED_INPUT_TYPE": "Unsupported input type passed. Please provide one of this options: {} ",
    "ERROR_UNSUPPORTED_INPUT_ENCODING": "Unsupported input encoding passed {}.  Please provide one of this options: {} ",
    "ERROR_UNSUPPORTED_INPUT_LAYOUT": "Unsupported input layout passed {}. Please provide one of this options: {} ",
    "ERROR_UNSUPPORTED_QUANT_PARAM": "Unsupported quantization param passed. Please provide one of this options: {} ",
    "ERROR_LAYER_NOT_FOUND_IN_QUANT_PARAM": "Requested layer {} not found in ir graph's qparams dictionary.",
    "ERROR_OPERATION_INPUTS_NOT_BROADCASTABLE": "Op {} inputs ({}, {}) must be broadcastable. Got shapes {}, {}.",


    # //=============================================================================
    # //                 CUSTOM OP GENERAL ERROR CODES
    # //=============================================================================
    "ERROR_MISSING_ATTRIBUTE": "Required custom op attribute: {} is not present in op: {}",
    "ERROR_CANNOT_CREATE_CUSTOM_OP": "Please provide either the framework node or the framework model to create a valid Custom Op",
    "ERROR_CUSTOM_OP_PARAM_NO_DATA": "Parameter: {} has no data in src_op {}",
    "ERROR_CANNOT_INGEST_STATIC_INPUT": " Attempted to ingest static input data but static input "
                                        "{} has no associated data, please include this input as a param",


    # //=============================================================================
    # //                 IR AXISTRACKING CONVERTER ERROR CODES
    # //=============================================================================
    "ERROR_INPUT_DATA_ORDER_UNEXPECTED": "op expected input {} in one of {} order, got: {}",
    "ERROR_FC_WRONG_INPUT_SIZE": "FC Node {}: input size expected by weights ({}) does not match input size of buffer "
                                 "({}). Note: weights are assumed to have shape (input_size, output_size) when "
                                 "converted from training framework format to IR",
    "ERROR_BATCHNORM_DIM_UNSUPPORTED": "Only 4D,3D and 2D inputs to batchnorm is supported. Got {} for {}",
    "ERROR_INSTANCE_NORM_DIM_UNSUPPORTED": "Only 4D,3D and 2D inputs to instance norm is supported. Got {} for {}",
    "ERROR_MATMUL_UNEXPECTED_INPUT_ORDER": "Matmul op got unexpected input data order {}",
    "ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER": "Reshape op got unexpected input data order {}",
    "ERROR_CONVOLUTION_UNEXPECTED_INPUT_ORDER": "Convolution op {} got unexpected input data order {}",
    "ERROR_TRANPOSE_CONV_UNEXPECTED_INPUT_ORDER": "TransposeConv op {} got unexpected input data order {}",


    # //=============================================================================
    # //                 IR OPTIMIZATIONS ERROR CODES
    # //=============================================================================
    "ERROR_UNKNOWN_MATCHING_CRITERIA": "Ops optimization: Expected {} for matching criteria of buffers, Got {}",
    "ERROR_UNKNOWN_BUFFER_CRITERIA": "Op type {} optimization: Expected {} or int(Index) for buffer criteria, Got {}",
    "ERROR_BUFFER_CRITERIA_INDEX": "Op type {} optimization: Index {} for buffer list length {} not valid. Please adjust optimization criteria",
    "ERROR_BUFFER_CRITERIA_ALL": "Op type {} optimization: There should only be one expected buffer provided when Buffer criteria is 'ALL'. Got {}",
    "ERROR_UNKNOWN_OP_TYPE(S)_FOUND": "Not all requested op_type(s) '{}' for matching sequence  supported. Please verify that each op_type exists in op_adapter.py",
    'ERROR_INVALID_PRIORBOX_VARIANCES': "All Priorbox layer variances must be equal. Got {} for layer {}, vs {} for layer {}",
    'ERROR_DETECTIONOUT_UNKNOWN_INPUTS': "Unknown input for detectionout layer. Got {}",
    'ERROR_UNSUPPORTED_COLOR_TRANSFORMATION': "Unsupported color transformation {} to {}. Support includes [ARGB|RGBA|NV21|NV12]To[RGB|BGR] transformations.",
    'ERROR_INVALID_COLOR_TRANSFORM_INPUT_SHAPE': "Invalid output shape {} of inputOp {} for requested color transformation. Expected {}",


}

warning_codes_to_messages = {
    # //=============================================================================
    # //                 TENSORFLOW CONVERTER WARNING CODES
    # //=============================================================================

    # start of the converter warning messages
    'WARNING_TF_OP_NOT_SUPPORTED': "Operation ({}) of type ({}) is not supported by converter.",
    'WARNING_UNSUPPORTED_OPS_FOUND': "Some Operations are not supported by converter. Use option "
                                     "--show_unconsumed_nodes to see the list of operations.",
    'WARNING_TF_LAYER_NOT_CONSUMED': "Layer ({}) of type ({}) is not consumed by converter.",
    'WARNING_UNCONSUMED_LAYERS': "The TF Graph has been disconnected due to some unsupported operations. Use option "
                                 "--show_unconsumed_nodes to see the list of disconnected layers.",
    'WARNING_TF_GROUP_CONV_RESOLVE': "Cannot resolve group convolution layer due to some incorrect parameters.",
    'WARNING_TF_USE_FIRST_META_GRAPH': "There is no matched tags for savedmodel [{}], try to get first MetaGraph.",
    'WARNING_TF_USE_FIRST_SIGNATURE_KEY': "There is no match signature key [{}], try to use first key [{}]",
    'WARNING_TF_MODEL_VERSION_DOES_NOT_MATCHED': "The model version [{}] is not matched with your environment [{}].",

    # //=============================================================================
    # //                 ONNX CONVERTER WARNING CODES
    # //=============================================================================
    "WARNING_OP_VERSION_NOT_SUPPORTED_BY_ONNX": "Version mismatch issue found. Model has [Operator: {}, Version: {}] but it is "
                                                "not found in ONNX installation(max supported is: {}). Please fix model or ONNX installation as "
                                                "appropriate to avoid possible error(s).",
    "WARNING_OPSET_VERSION": "Warning multiple opset versions specified, using highest.",
    "WARNING_UNSUPPORTED_ATTRIBUTE": "Unsupported attribute: {} found in {} Op with input: {}. This attribute may be ignored or cause an error during model conversion.",

    # //=============================================================================
    # //                 IR COMMON CONVERTER WARNING CODES
    # //=============================================================================
    "WARNING_OP_NOT_SUPPORTED": "Operation {} Not Supported.",
    "WARNING_OP_NOT_SUPPORTED_BY_SCHEMA": "Operation {} Not Supported by ONNX schema.",
    "WARNING_OP_VERSION_NOT_SUPPORTED": "Operation {} Not Supported. "
                                        "Expected operator version: {}, instead got version: {}"
}

debug_codes_to_messages = {
    # //=============================================================================
    # //                 GENERAL CONVERTER DEBUGGUING CODES
    # //=============================================================================
    "DEBUG_DOWNCAST_TENSOR": "Downcasting tensor dtype from {} to {} on node {}",
    'DEBUG_EXTRACT_BIAS': "Bias extracted for source node {} with shape {}",
    'DEBUG_EXTRACT_WEIGHTS': "Weights extracted for source node {} with shape {}",

    # //=============================================================================
    # //                 TENSORFLOW CONVERTER DEBUGGUING CODES
    # //=============================================================================

    # start of the util debugging messages
    'DEBUG_TF_SCOPE_PRINT': "Scope({})",
    'DEBUG_TF_OP_NAME_TYPE_PRINT': "\tOperation({}) [{}])",

    # //=============================================================================
    # //                 ONNX CONVERTER DEBUGGING CODES
    # //=============================================================================
    "DEBUG_CONVERTING_NODE": "Attempting to convert node {} with type {}",
    "DEBUG_CONSTANT_PRUNED": "Constant op {} consumed by weight layer, pruning from network",
    "DEBUG_RETRIEVE_WEIGHTS": "Retrieving weights {}",
    "DEBUG_STATIC_OP": "Node {} with static input(s) is resolved as Constant Op and interpreted during conversion",

    # //=============================================================================
    # //               CUSTOM OP DEBUGGING CODES
    # //=============================================================================
    "DEBUG_CUSTOM_OP_NOT_FOUND": "Custom op of type: {} was not found in framework model: {}",
    "DEBUG_CUSTOM_OPS_FOUND": "Identified {} custom ops matching op type: {}",

    # //=============================================================================
    # //                 IR AXISTRACKING CONVERTER DEBUGGING CODES
    # //=============================================================================
    "DEBUG_AXES_TRANSFORMATION_ENTRY": "Node {}: axes_to_spatial_first_order",
    "DEBUG_AXES_TRANSFORMATION_INPUT_SIZE": "Input buffer {}: shape {}",

    # //=============================================================================
    # //                 IR OPTIMIZATIONS DEBUGGING CODES
    # //=============================================================================
    "DEBUG_BATCHNORM_SQUASH": "Squashed Batchnorm:{} into {}:{}",
    "DEBUG_CONCAT_FOLD": "Folded Concat:{} into Concat:{}",
    "DEBUG_CHANNEL_SHUFFLE_REPLACE": "Replaced [Reshape:{}, Permute:{}, Reshape:{}] with ChannelShuffle:{}",
    "DEBUG_ELEMENTWISEBINARY_CHAIN": "Chaining op ({}, type: {}) into binary elementwise ops by splitting inputs: {}",
    "DEBUG_SQUASH_INTO_NN_NODE": "Squashed {} node {} into {} node {}",
    "DEBUG_DETECTIONOUT_FOLDING": "Squashed concat of priorboxes:{} into {}",
    "DEBUG_DETECTIONOUT_CAFFE_TO_TF_STYLE": "Matching DetectionOut caffe style to TF + postprocess for:{}",
    "DEBUG_BOXDECODER_SQUASH": "Squashed Box Decoder(Ssd) Node:{} into proceeding NMS node:{}",
    "DEBUG_COLOR_TRANSFORM_EXTRACTION": "Color Transform extraction from network InputOp complete. Updated network "
                                        "input {} will expect shape {} and encoding {}",
}

progress_codes_to_messages = {

    # //=============================================================================
    # //                 TENSORFLOW CONVERTER INFO CODES
    # //=============================================================================

    # start of the converter info messages
    'INFO_ALL_BUILDING_NETWORK':
        """
    ==============================================================
    Building Network
    ==============================================================""",

    'INFO_INPUT_OUTPUT_FROM_SAVEDMODEL' : "Input nodes : {}, Output nodes : {}",
    'INFO_TF_BUILDING_INPUT_LAYER': "Building layer (INPUT) with node: {}, shape {}",
    'INFO_ALL_BUILDING_LAYER_W_NODES': "Building layer ({}) with nodes: {}",
    'INFO_TF_CHANGE_NODE_NAME': "Change node name: {} with name {}",

    # //=============================================================================
    # //                 ONNX CONVERTER INFO CODES
    # //=============================================================================
    "INFO_STATIC_RESHAPE": "Applying static reshape to {}: new name {} new shape {}",

    # //=============================================================================
    # //                 IR COMMON CONVERTER INFO CODES
    # //=============================================================================
    "INFO_CONVERSION_SUCCESS": "Conversion completed successfully"
}


def _wrapper_(error_code, message_table):
    try:
        message = message_table[error_code]
    except KeyError:
        message = ""

    def _formatter_(*args):
        if message.count('{}') == len(args):
            return "{}: {}".format(error_code, message.format(*[str(arg) for arg in args]))
        else:
            return "{}: N/A".format(error_code)
    if message.count('{}') > 0:
        return _formatter_
    else:
        return '{}: {}'.format(error_code, message)


def get_error_message(error_code):
    return _wrapper_(error_code, error_codes_to_messages)


def get_warning_message(error_code):
    return _wrapper_(error_code, warning_codes_to_messages)


def get_debugging_message(error_code):
    return _wrapper_(error_code, debug_codes_to_messages)


def get_progress_message(error_code):
    return _wrapper_(error_code, progress_codes_to_messages)
