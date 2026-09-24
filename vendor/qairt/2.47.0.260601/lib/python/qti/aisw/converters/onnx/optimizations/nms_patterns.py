# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

patterns = {
    "yolo": [
        # P1-yolov3-onnx, tiny-yolov3-onnx
        [
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Transpose", "ANY")]),
            ),
            (
                "Transpose",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
            ),
            (
                "Reshape",
                ("MATCH_BUFS_AT_INDEX", [("Transpose", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Slice", "ANY")]),
            ),
        ],
        # P1.1-yolov4-onnx
        [
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Transpose", "ANY")]),
            ),
            (
                "Transpose",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
            ),
            (
                "Reshape",
                ("MATCH_BUFS_AT_INDEX", [("Transpose", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Split", 0)]),
            ),
        ],
        # FIXME: Can p1 and p1.1 be merged.
        # P1.2-yolov5-ultralytics, yolov7
        [
            (
                "Conv",
                (),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
            ),
            (
                "Reshape",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Transpose", 0)]),
            ),
            (
                "Transpose",
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Sigmoid", 0)]),
            ),
        ],
        # P2-yolov2-onnx
        [
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("Concat", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
            ),
            (
                "LeakyRelu",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
            ),
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Output", 0)]),
            ),
        ],
        # P2-tiny-yolov2-onnx
        [
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
            ),
            (
                "LeakyRelu",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
            ),
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Output", 0)]),
            ),
        ],
        # FIXME: Can above p2 both be merged?
        # P3-yolov2-keras, tiny-yolov3-keras
        [
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("Concat", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
            ),
            (
                "LeakyRelu",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
            ),
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Transpose", 0)]),
            ),
            (
                "Transpose",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Output", 0)]),
            ),
        ],
        # P4-tiny-yolov2-keras, yolov3-keras, tiny-yolov3-keras, yolov4-keras
        [
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
            ),
            (
                "LeakyRelu",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
            ),
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("LeakyRelu", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Transpose", 0)]),
            ),
            (
                "Transpose",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Output", 0)]),
            ),
        ],
        # FIXME: Can p3 and p4 be merged?
        # P5-yolox-1
        [
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("Mul", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Concat", 0)]),
            ),
            (
                "Concat",
                ("MATCH_BUFS_AT_INDEX", [("Conv", "ANY")]),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
            ),
        ],
        # P5-yolox-2
        [
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("Mul", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Sigmoid", 0)]),
            ),
            (
                "Sigmoid",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Concat", 0)]),
            ),
            (
                "Concat",
                ("MATCH_BUFS_AT_INDEX", [("Sigmoid", "ANY")]),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
            ),
        ],
    ],
    "ssd": [
        # ssd-mv1-a, ssd-mv1-mlperf, ssd-vgg-lufficc, ssd_inception_v2_coco_2018_01_28-a
        [
            (
                "Conv",
                (),
                ("MATCH_BUFS_AT_INDEX", [("Transpose", "ANY")]),
            ),
            (
                "Transpose",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
            ),
            (
                "Reshape",
                ("MATCH_BUFS_AT_INDEX", [("Transpose", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Concat", 0)]),
            ),
            (
                "Concat",
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                (),
            ),
        ],
        # ssd-mv1-b
        [
            (
                "Conv",
                ("MATCH_BUFS_AT_INDEX", [("Clip", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
            ),
            (
                "Reshape",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
            ),
            (
                "Reshape",
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Concat", 0)]),
            ),
            (
                "Concat",
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                (),
            ),
        ],
        # ssd-resnet, ssd_inception_v2_coco_2018_01_28-b
        [
            (
                "Conv",
                (),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
            ),
            (
                "Reshape",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Concat", 0)]),
            ),
            (
                "Concat",
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                (),
            ),
        ],
        # retinanet
        [
            (
                "Conv",
                (),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
            ),
            (
                "Reshape",
                ("MATCH_BUFS_AT_INDEX", [("Conv", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Transpose", 0)]),
            ),
            (
                "Transpose",
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
            ),
            (
                "Reshape",
                ("MATCH_BUFS_AT_INDEX", [("Transpose", 0)]),
                ("MATCH_BUFS_AT_INDEX", [("Concat", 0)]),
            ),
            (
                "Concat",
                ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                (),
            ),
        ],
    ],
}
