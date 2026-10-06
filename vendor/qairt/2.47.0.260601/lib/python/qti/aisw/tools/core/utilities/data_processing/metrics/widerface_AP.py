##############################################################################
# MIT License
# Copyright (c) 2019
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################
# WiderFace evaluation code
# author: wondervictor
# mail: tianhengcheng@gmail.com
# copyright@wondervictor
#
# License : https://github.com/biubug6/Pytorch_Retinaface/blob/master/LICENSE.MIT
# Source : https://github.com/biubug6/Pytorch_Retinaface/blob/master/widerface_evaluate/evaluation.py
#
##############################################################################
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os

import numpy as np
from qti.aisw.tools.core.utilities.data_processing import Annotation, Representation
from qti.aisw.tools.core.utilities.data_processing.metrics import Metric
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class WiderFaceAPMetric(Metric):
    """Compute Average Precision (AP) metric for WIDER FACE dataset."""

    def __init__(self, iou_threshold: float = 0.4):
        """Initialize the AP calculator.

        Args:
            iou_threshold (float): IoU threshold for evaluation. Defaults to 0.4.
        """
        self.iou_threshold = iou_threshold
        self._predictions = {}
        self.validate()

    def validate(self):
        """Validate the WiderFaceAPMetric parameters provided"""
        if not isinstance(self.iou_threshold, float):
            raise ValueError(f"iou_threshold must be a float but provided: {self.iou_threshold}")
        scipy = Helper.safe_import_package("scipy")  # noqa: F841
        if scipy is None:
            raise ImportError("Scipy module is required for to load the annotation .mat files")

    def validate_input(self, input_sample: Representation) -> Representation:
        """Validates the input representation by checking if it contains exactly three elements.

        Args:
            input_sample (Representation): The input representation to be validated.

        Returns:
            Representation: The validated input representation.

        Raises:
            RuntimeError: If the length of the data is not 3.
        """
        # Check if the length of the data list is 3
        if len(input_sample.data) != 3:
            raise RuntimeError(
                "WiderfaceAP metric expects data to have a list of 3 items. i.e."
                " image_id, event_name, bbox predictions"
            )
        if not isinstance(input_sample.annotation, Annotation):
            raise RuntimeError("WiderfaceAP metric expects annotation field to be set.")
        return input_sample

    @Metric.validate_input_output
    def calculate(self, model_outputs: Representation) -> None:
        """Organize the predictions in a dictionary event-wise.

        Args:
            model_outputs (Representation): Model outputs containing data and annotations.
        """
        # organizing the predictions in dictionary event wise as shown below
        ###
        # boxes dict --| eve0  -- image predictions
        # --| eve1  -- image predictions
        # --| ...
        # --| eve61 -- image predictions
        #########################################################################
        # Assuming 'data' is a list of tuples (image_id, event_name, predictions)
        data = model_outputs.data
        image_id = data[0]
        event_name = data[1]
        predictions = data[2]

        if image_id not in self._predictions.keys():
            # If the image ID is new, create an empty dictionary for it
            self._predictions[image_id] = {}
            # predictions are list of tuples (x1, y1, x2, y2)
            self._predictions[image_id][event_name] = predictions
        else:
            # If the image ID already exists, append new event_name and predictions to it

            self._predictions[image_id][event_name] = predictions
        if not hasattr(self, "annotations_file"):
            # Set annotations file path if not set yet
            self.annotations_file = model_outputs.annotation.data

    def finalize(self) -> dict:
        """Compute Average Precision (AP) metric for WIDER FACE dataset.

        Returns:
            dict: Dictionary containing AP values for Easy, Medium and Hard cases.
        """
        result = {}
        if len(self._predictions) > 0:
            # Load ground truth boxes from annotations file
            facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(
                self.annotations_file
            )

            # Get number of events and thresholds for precision-recall curve
            event_num = len(event_list)
            thresh_num = 1000

            # Initialize AP values for each difficulty level
            setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
            aps = [0 for _ in (setting_gts)]  # initialize to zero

            # Compute AP for each difficulty level
            for _idx, gt_list in enumerate(setting_gts):
                # Different setting (Easy, Medium, Hard)
                count_face = 0
                pr_curve = np.zeros((thresh_num, 2)).astype("float")

                # Iterate over events and images
                for i in range(event_num):
                    event_name = str(event_list[i][0][0])
                    if event_name not in self._predictions:
                        continue

                    img_list = file_list[i][0]
                    pred_list = self._predictions[event_name]
                    sub_gt_list = gt_list[i][0]
                    gt_bbx_list = facebox_list[i][0]

                    # Iterate over images and process ground truth boxes
                    for j in range(len(img_list)):
                        if str(img_list[j][0][0]) not in pred_list:
                            continue

                        pred_info = pred_list[str(img_list[j][0][0])]
                        gt_boxes = gt_bbx_list[j][0].astype("float")
                        keep_index = sub_gt_list[j][0]

                        # Count number of ground truth boxes
                        count_face += len(keep_index)

                        if len(gt_boxes) == 0 or len(pred_info) == 0:
                            continue

                        ignore = np.zeros(gt_boxes.shape[0])
                        if len(keep_index) != 0:
                            ignore[keep_index - 1] = 1

                        # Evaluate precision-recall curve for current image
                        pred_recall, proposal_list = image_eval(
                            pred_info, gt_boxes, ignore, self.iou_threshold
                        )
                        _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                        # Accumulate precision-recall information across images
                        pr_curve += _img_pr_info

                # Compute final precision-recall curve for current difficulty level
                pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

                # Extract propose and recall values from precision-recall curve
                propose = pr_curve[:, 0]
                recall = pr_curve[:, 1]

                # Compute average precision using VOC AP metric
                ap = voc_ap(recall, propose)  # compute the average precision under the curve.
                aps[_idx] = ap
            result = {"Easy_Val_AP": aps[0], "Medium_Val_AP": aps[1], "Hard_Val_AP": aps[2]}
        return result


######################################################################
# Reading the events, hard, medium and easy ground truth boxes
######################################################################
def get_gt_boxes(gt_dir: str) -> tuple:
    """Loads the ground truth boxes for hard, medium and easy faces from the provided directory.

    Args:
        gt_dir (str): Path to the directory containing the WIDER Face annotation files.

    Returns:
        A tuple of six lists:
            - facebox_list: Ground truth bounding boxes for all faces
            - event_list: Event list associated with each file
            - file_list: File names in the dataset
            - hard_gt_list: Ground truth bounding boxes for hard faces
            - medium_gt_list: Ground truth bounding boxes for medium faces
            - easy_gt_list: Ground truth bounding boxes for easy faces
    """
    if not os.path.exists(gt_dir):
        raise ValueError("Path to the directory containing the WIDER Face annotation files doesn't exist")
    mat_file_names = [
        "wider_face_val.mat",
        "wider_hard_val.mat",
        "wider_medium_val.mat",
        "wider_easy_val.mat",
    ]
    for mat_file_name in mat_file_names:
        if not os.path.exists(os.path.join(gt_dir, mat_file_name)):
            raise ValueError(f"{mat_file_name} WIDER Face annotation file is missing and is required")
    scipy = Helper.safe_import_package("scipy")
    gt_mat = scipy.io.loadmat(os.path.join(gt_dir, "wider_face_val.mat"))
    hard_mat = scipy.io.loadmat(os.path.join(gt_dir, "wider_hard_val.mat"))
    medium_mat = scipy.io.loadmat(os.path.join(gt_dir, "wider_medium_val.mat"))
    easy_mat = scipy.io.loadmat(os.path.join(gt_dir, "wider_easy_val.mat"))

    # Load face bounding boxes
    facebox_list = gt_mat["face_bbx_list"]
    # Load event information
    event_list = gt_mat["event_list"]
    # Load file information
    file_list = gt_mat["file_list"]
    # Load ground truth bounding boxes for hard faces
    hard_gt_list = hard_mat["gt_list"]
    # Load ground truth bounding boxes for medium faces
    medium_gt_list = medium_mat["gt_list"]
    # Load ground truth bounding boxes for easy faces
    easy_gt_list = easy_mat["gt_list"]
    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


######################################################################
# Compute the precision and recall for an image.
######################################################################
def image_eval(
    predictions: np.ndarray, ground_truths: np.ndarray, ignore_mask: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Single image evaluation.

    Args:
        predictions (np.ndarray): Nx5 array of predicted bounding boxes.
            Each row represents a box with format [x1, y1, x2, y2, confidence].
        ground_truths (np.ndarray): Nx4 array of ground truth bounding boxes.
            Each row represents a box with format [x1, y1, x2, y2].
        ignore_mask (np.ndarray): Array of shape (N,) indicating whether each
            predicted box should be ignored during evaluation. 0 indicates the box
            should not be ignored, and non-zero values indicate it should.
        iou_threshold (float): IOU threshold for determining true positives.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays of shape (N,)
            representing precision and recall at each proposal index.
    """
    _pred = predictions.copy()
    _gt = ground_truths.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])
    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]
    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_threshold:
            if ignore_mask[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


######################################################################
# Compute the pr curve for single image
######################################################################
def img_pr_info(
    thresh_num: int, pred_info: np.ndarray, proposal_list: np.ndarray, pred_recall: np.ndarray
) -> np.ndarray:
    """Calculate precision-recall information for a set of threshold values.

    Args:
        thresh_num (int): Number of threshold values to consider.
        pred_info (np.ndarray): Array containing prediction information.
            Each row represents a predicted object and contains the following
            fields: x-coordinate, y-coordinate, width, height, confidence score.
        proposal_list (np.ndarray): list of proposals for each predicted object.
            1 indicates that the proposal is valid.
        pred_recall (np.ndarray): Array containing recall values corresponding
            to each predicted object.

    Returns:
        np.ndarray: Precision-recall information for each threshold value. Each row
            contains two elements: precision and recall value at the given threshold.
    """
    pr_info = np.zeros((thresh_num, 2)).astype("float")
    for t in range(thresh_num):
        # Calculate current threshold value (confidence score)
        thresh = 1 - (t + 1) / thresh_num
        # Get indices of predicted objects with confidence scores above the threshold
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            continue
        # Consider last valid predicted object for current threshold
        r_index = r_index[-1]
        # Get indices of proposals corresponding to the considered predicted object
        p_index = np.where(proposal_list[: r_index + 1] == 1)[0]
        pr_info[t, 0] = len(p_index)
        pr_info[t, 1] = pred_recall[r_index]
    return pr_info


######################################################################
# Compute the precision-recall curve for easy/medium/hard cases
######################################################################
def dataset_pr_info(thresh_num: int, pr_curve: np.ndarray, count_face: int) -> np.ndarray:
    """Calculate the precision-recall info for a given dataset.

    This function takes in a precision-recall curve and the total number of faces in the dataset.
    It returns an array where each row represents the precision at a certain recall level
    and the corresponding fraction of faces that have been detected up to that point.

    Args:
        thresh_num (int): The number of thresholds in the precision-recall curve.
        pr_curve (np.ndarray): A 2D array where the first column is the precision,
                                            the second column is the recall, and the third column
                                            is the count of faces at each threshold.
        count_face (int): The total number of faces in the dataset.

    Returns:
        np.ndarray: An array with shape (num_thresholds, 2) where each row contains the precision
                    and the fraction of faces that have been detected up to a certain recall level.
    """
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


######################################################################
# Compute the box overlap between groundtruth and predicted boxes.
######################################################################
def bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Calculate Overlap between two sets of bounding boxes.

    Args:
        boxes (np.ndarray): Array of shape (N, 4), where each row contains the coordinates
         of a bounding box in the format [x1, y1, x2, y2].
        query_boxes (np.ndarray): Array of shape (K, 4), where each row contains the coordinates
         of a bounding box in the format [x1, y1, x2, y2].

    Returns:
        np.ndarray: Array of shape (N, K), where element at position [n, k] is
         the Overlap between boxes[n] and query_boxes[k].
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=float)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
            if iw <= 0 or ih <= 0:
                continue
            ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
            overlaps[n, k] = iw * ih / ua
    return overlaps


######################################################################
# Compute the average precision under the curve.
######################################################################
def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Compute the Average Precision (AP) for a given set of recall and precision values.

    Args:
        rec (np.ndarray): Array of recall values, sorted in ascending order.
        prec (np.ndarray): Array of precision values, corresponding to the recall values.

    Returns:
        float: The Average Precision (AP) value.

    Notes:
        This function modifies the input arrays in-place, so it is not suitable for
    using with immutable data structures like tuples or strings.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
