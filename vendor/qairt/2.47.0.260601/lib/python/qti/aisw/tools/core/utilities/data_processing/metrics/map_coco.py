# =============================================================================
#
# Copyright (c) 2018-2021 cTuning foundation
#
# Copyright (c) cTuning foundation <admin@cTuning.org>
# All rights reserved

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the cTuning foundation
#       nor the names of its contributors may be used to endorse
#       or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
##############################################################################
##############################################################################
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# License: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/LICENSE
# Source: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/ba50a82dce412df97f088c572d86d7977753bf74/lib/dataset/coco.py
###############################################################################
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import gc
import json
import os
import tempfile
from typing import Dict, List, Literal, Optional

import numpy as np
from qti.aisw.tools.core.utilities.data_processing import Representation
from qti.aisw.tools.core.utilities.data_processing.metrics.base import Metric
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class_map = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]


###################################################################################################
# Bounding box rectangle specified by top-left point (x, y), width(w) and height (h) in pixels.
###################################################################################################


class box:
    """Represents a rectangular region on an image."""

    def __init__(self, x: float, y: float, w: float, h: float):
        """Initializes a Box instance.

        Args:
            x (float): The x-coordinate of the box's center.
            y (float): The y-coordinate of the box's center.
            w (float): The width of the box.
            h (float): The height of the box.
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h


###################################################################################################
# detection result element contains a class id, score and a bounding box
###################################################################################################


class det_box:
    """Represents a detected bounding box.

    Attributes:
        object_id (int): The ID of the object that this detection is for.
        category_id (int): The ID of the category that this detection belongs to.
        confidence (float): The confidence score of this detection.
        bounding_box (box): The bounding box coordinates of the detected object.
    """

    def __init__(self, fid: int, cid: int, score: float, bbox: box):
        """Initializes a Box instance.

        Args:
            fid (int): The ID of the object that this detection is for.
            cid (int): The ID of the category that this detection belongs to.
            score (float): The confidence score of this detection.
            bbox (box): The bounding box coordinates of the detected object.
        """
        self.fid = fid
        self.cid = cid
        self.score = score
        self.bbox = bbox


class MAP_COCOMetric(Metric):
    """Class to compute metric for COCO dataset"""

    def __init__(
        self,
        map_80_to_90: bool = False,
        seg_map: bool = False,
        keypoint_map: bool = False,
        dataset_type: Literal["coco", "openimages"] = "coco",
    ):
        """Initializes the MAP_COCOMetric intance.

        Args:
            map_80_to_90 (bool): Calculates the mean Average Precision if set to True. Defaults to False.
            seg_map (bool):  Whether to include segmentation maps in the evaluation. Defaults to False.
            keypoint_map (bool): Whether to include keypoint maps in the evaluation. Defaults to False.
            dataset_type (Literal) : Specifies the type of dataset being used which can be either "coco"
                                    or "openimages". Defaults to 'coco'.
        """
        pycocotools = Helper.safe_import_package("pycocotools", "2.0.7")  # noqa: F841
        self.map_80_to_90 = map_80_to_90
        self.map_list = ["bbox"]
        self.seg_map = seg_map
        self.keypoint_map = keypoint_map
        self.dataset_type = dataset_type
        self.res_array = []
        self.processed_image_ids = []
        self.validate()

        if self.seg_map:
            self.map_list.append("segm")
        if self.keypoint_map:
            self.map_list.append("keypoints")

    def validate(self):
        """Validate the MAP_COCOMetric parameters provided"""
        if not isinstance(self.map_80_to_90, bool):
            raise ValueError("map_80_to_90 must be a boolean.")

        if not isinstance(self.seg_map, bool):
            raise ValueError("seg_map must be a boolean.")

        if not isinstance(self.keypoint_map, bool):
            raise ValueError("keypoint_map must be a boolean.")

        if self.dataset_type not in ["coco", "openimages"]:
            raise ValueError('dataset_type must be either "coco" or "openimages".')

    def calculate(self, input_sample: Representation) -> None:
        """Processes the input sample to calculate detection metrics and appends the results to the res_array.

        Args:
            input_sample (Representation): The input data containing data and annotations.

        Returns:
            None (appends results to self.res_array)
        """
        data = input_sample.data
        self.annotations_file = input_sample.annotation.data
        if not hasattr(self, "targets_gt"):
            if self.dataset_type == "openimages":
                self.images_gt = {}
                self.targets_gt = {}
                with open(input_sample.annotation.data, "r") as f:
                    gt_dict = json.load(f)
                    for dic in gt_dict["images"]:
                        self.images_gt[dic["id"]] = dic["file_name"].split(".jpg")[0]
                    for dic in gt_dict["annotations"]:
                        self.targets_gt[self.images_gt[dic["image_id"]]] = dic["image_id"]
        frow = data[0].strip().split(",")
        row = frow[1:]
        filepath = os.path.basename(frow[0])
        short_name, _ = os.path.splitext(filepath)

        if self.dataset_type == "openimages":
            fid = self.targets_gt[short_name]
        else:
            fid = int(short_name.split("_")[-1])

        self.processed_image_ids.append(fid)
        t = 0
        n = int(row[t])
        t += 1
        if self.seg_map:
            image_mask = data[1]
            assert len(image_mask) == n, "length of masks and detections should match"

        for i in range(n):
            cid = int(row[t])
            if self.map_80_to_90:
                cid = class_map[cid + 1]
            t += 1
            score = float(row[t])
            t += 1
            bbox = box(float(row[t]), float(row[t + 1]), float(row[t + 2]), float(row[t + 3]))
            t += 4
            if self.seg_map:
                mask = image_mask[i]
                mask["counts"] = mask["counts"].decode("utf-8")
                res = self.detection_to_coco_object(det_box(fid, cid, score, bbox), mask=mask)
            elif self.keypoint_map:
                res = (
                    np.concatenate(
                        [
                            np.array(row[t : t + 34], dtype=np.float32).reshape(-1, 2),
                            np.ones((17, 1), dtype=np.float32),
                        ],
                        axis=1,
                    )
                    .reshape(51)
                    .tolist()
                )
                t += 34
                res = self.detection_to_coco_object(det_box(fid, cid, score, bbox), keypoints=res)
            else:
                res = self.detection_to_coco_object(det_box(fid, cid, score, bbox))

            self.res_array.append(res)

        return

    def finalize(self) -> Dict[str, float]:
        """Calculate the mean Average Precision (mAP) metrics for the COCO dataset.

        Returns:
            dict: Dictionary containing the calculated mAP and mAP_50 metrics for each map type.
        """
        num_results = len(self.res_array)
        with tempfile.NamedTemporaryFile(
            suffix=".json", prefix="coco_results_", mode="w", delete=False
        ) as results_json_file:
            json_file_path = results_json_file.name
            results_json_file.write(json.dumps(self.res_array, indent=6, sort_keys=False))

        del self.res_array  # This is no longer needed and just occupies ram memory [idle]
        gc.collect()  # forces python to perform garbage collection
        results = {}
        for map_type in self.map_list:
            results[map_type + "_mAP"] = 0.0
            results[map_type + "_mAP_50"] = 0.0

            if num_results:
                mAP, mAP_50, _, _ = self.evaluate(
                    self.processed_image_ids, json_file_path, self.annotations_file, map_type
                )
                results[map_type + "_mAP"] = mAP
                results[map_type + "_mAP_50"] = mAP_50

        return results

    def detection_to_coco_object(
        self, det: det_box, mask: Optional[dict] = None, keypoints: Optional[list] = None
    ):
        """Returns result object in COCO format."""
        det_box = det.bbox
        results = {
            "image_id": det.fid,
            "category_id": det.cid,
            "bbox": [det_box.x, det_box.y, det_box.w, det_box.h],
            "score": det.score,
        }
        if mask is not None:
            results["segmentation"] = mask
        if keypoints is not None:
            results["keypoints"] = keypoints

        return results

    @staticmethod
    def evaluate(
        image_ids_list: List[int],
        results_dir: str,
        annotations_file: str,
        map_type: Literal["keypoints", "bbox", "segm"],
    ):
        """Calculate COCO metrics via evaluator from pycocotool package.
        MSCOCO evaluation protocol: http://cocodataset.org/#detections-eval

        This method uses original COCO json-file annotations
        and results of detection converted into json file too.
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # builtins.print = pl_print
        cocoGt = COCO(annotations_file)
        cocoDt = cocoGt.loadRes(results_dir)
        cocoEval = COCOeval(cocoGt, cocoDt, iouType=map_type)
        cocoEval.params.imgIds = image_ids_list
        cocoEval.params.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        if map_type == "keypoints":
            cocoEval.params.useSegm = None

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        # builtins.print = orig_prin_fn

        if map_type == "bbox" or map_type == "segm":
            # These are the same names as object returned by CocoDetectionEvaluator has
            all_metrics = {
                "DetectionBoxes_Precision/mAP": cocoEval.stats[0],
                "DetectionBoxes_Precision/mAP@.50IOU": cocoEval.stats[1],
                "DetectionBoxes_Precision/mAP@.75IOU": cocoEval.stats[2],
                "DetectionBoxes_Precision/mAP (small)": cocoEval.stats[3],
                "DetectionBoxes_Precision/mAP (medium)": cocoEval.stats[4],
                "DetectionBoxes_Precision/mAP (large)": cocoEval.stats[5],
                "DetectionBoxes_Recall/AR@1": cocoEval.stats[6],
                "DetectionBoxes_Recall/AR@10": cocoEval.stats[7],
                "DetectionBoxes_Recall/AR@100": cocoEval.stats[8],
                "DetectionBoxes_Recall/AR@100 (small)": cocoEval.stats[9],
                "DetectionBoxes_Recall/AR@100 (medium)": cocoEval.stats[10],
                "DetectionBoxes_Recall/AR@100 (large)": cocoEval.stats[11],
            }

            mAP = all_metrics["DetectionBoxes_Precision/mAP"]
            mAP_50 = all_metrics["DetectionBoxes_Precision/mAP@.50IOU"]
            recall = all_metrics["DetectionBoxes_Recall/AR@100"]

            return mAP, mAP_50, recall, all_metrics

        elif map_type == "keypoints":
            metrics_list = [
                "AP",
                "Ap(.5)",
                "AP(.75)",
                "AP(M)",
                "AP(L)",
                "AR",
                "AR(.5)",
                "AR(.75)",
                "AR(M)",
                "AR(L)",
            ]
            res_info_str = ""
            for ind, name in enumerate(metrics_list):
                res_info_str += "{}: {:.12f}\n".format(name, cocoEval.stats[ind])

            return cocoEval.stats[0], cocoEval.stats[1], cocoEval.stats[5], res_info_str
        else:
            raise ValueError("Invalid map_type:{}".format(map_type))
