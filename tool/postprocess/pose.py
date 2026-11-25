# Copyright 2026 Innodisk Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pose (keypoint) post-processing shared by every runtime, `test` and `eval`.

The exported YOLO26-pose graph returns three raw tensors (see TASK_SPECS["pose"]):
  boxes  [1, 4, 8400]      ltrb distances in stride units, as for detection
  scores [1, C, 8400]      class logits (C = 1 for the official person model)
  kpts   [1, 3 * K, 8400]  per keypoint (x, y, visibility), x / y relative to the anchor

Everything after that (anchor decode, thresholds, NMS, mapping back to the original image,
drawing, the label text) is here, so the runtimes only have to produce those tensors.
The detection decoders in detect.py are reused as they are and left untouched.
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tool.postprocess.detect import (
        _normalize_output_layout,
        _should_run_nms,
        build_anchor_centers_8400,
        classwise_nms_xyxy,
        ltrb_grid_to_xyxy_pixels,
        sigmoid_clip,
        sort_and_cap,
    )
except ModuleNotFoundError:  # pushed to the board next to the runners as postprocess/
    from postprocess.detect import (
        _normalize_output_layout,
        _should_run_nms,
        build_anchor_centers_8400,
        classwise_nms_xyxy,
        ltrb_grid_to_xyxy_pixels,
        sigmoid_clip,
        sort_and_cap,
    )

# Standard 17-point COCO skeleton, drawn when the model has 17 keypoints.
COCO_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
)
SKELETON_COLOR = (255, 128, 0)
KEYPOINT_COLOR = (0, 255, 0)
# Keypoints below this visibility are not drawn (the label file still lists them).
DRAW_VISIBILITY = 0.5


def _channels_first(tensor, anchors: int, name: str):
    """A per-anchor output as [C, N] (accepts [1, C, N] and [1, N, C])."""
    return _normalize_output_layout(tensor, tensor_name=name, anchor_count=anchors)[0]


def _select_detections(boxes, scores, *, conf_thres, iou_thres, topk, max_det, flow, o2o_nms):
    """Threshold, top-k, box decode and NMS -> (anchor_index, xyxy, conf, class_id, pre_nms).

    Boxes are ltrb distances (the yolov26 export). `anchor_index` points into the 8400 anchors
    so the caller can pick the matching keypoints. xyxy is in the 640x640 input space.
    """
    box_raw = _normalize_output_layout(boxes, tensor_name="box", channel_sizes=(4,))[0]
    anchors = int(box_raw.shape[1])
    prob = sigmoid_clip(_channels_first(scores, anchors, "class").astype(np.float32))
    class_ids = np.argmax(prob, axis=0).astype(np.int32)
    conf = np.max(prob, axis=0).astype(np.float32)

    empty = (
        np.zeros((0,), dtype=np.int32),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
    )
    keep_idx = np.nonzero(conf >= float(conf_thres))[0]
    if keep_idx.size == 0:
        return (*empty, 0)
    conf, class_ids = conf[keep_idx], class_ids[keep_idx]
    pre_nms = int(conf.shape[0])

    if topk > 0 and conf.shape[0] > topk:
        order = np.argsort(conf)[-topk:][::-1]
        keep_idx, conf, class_ids = keep_idx[order], conf[order], class_ids[order]

    centers, strides = build_anchor_centers_8400()
    xyxy = ltrb_grid_to_xyxy_pixels(
        box_raw.astype(np.float32)[:, keep_idx], centers[keep_idx], strides[keep_idx]
    )
    if _should_run_nms(flow, iou_thres, o2o_nms):
        order = classwise_nms_xyxy(
            xyxy=xyxy, scores=conf, class_ids=class_ids, conf_thres=float(conf_thres),
            iou_thres=float(iou_thres), max_det=int(max_det),
        )
    else:
        order = sort_and_cap(conf, int(max_det))
    order = np.asarray(order, dtype=np.int32)
    if order.size == 0:
        return (*empty, pre_nms)
    return keep_idx[order], xyxy[order], conf[order], class_ids[order], pre_nms


def decode_keypoints(kpts_raw, centers_xy, strides):
    """Raw kpts [3K, N] -> [N, K, 3] of (x_px, y_px, visibility) in the 640x640 input space."""
    channels, count = kpts_raw.shape
    if channels % 3 != 0:
        raise RuntimeError(f"kpts channel count {channels} is not a multiple of 3")
    k = channels // 3
    raw = kpts_raw.astype(np.float32).reshape(k, 3, count)
    x = raw[:, 0, :] * strides[None, :] + centers_xy[None, :, 0]
    y = raw[:, 1, :] * strides[None, :] + centers_xy[None, :, 1]
    vis = sigmoid_clip(raw[:, 2, :])
    return np.stack([x, y, vis], axis=-1).transpose(1, 0, 2).astype(np.float32)


def postprocess_pose(
    boxes,
    scores,
    kpts,
    *,
    conf_thres: float,
    iou_thres: float,
    topk: int,
    max_det: int,
    flow: str,
    o2o_nms: bool = False,
):
    """Raw tensors -> (xyxy [N,4], scores [N], class_ids [N], keypoints [N,K,3], pre_nms_count).

    Boxes and keypoints are in the 640x640 letterboxed input space.
    """
    anchor_idx, xyxy, conf, class_ids, pre_nms = _select_detections(
        boxes, scores, conf_thres=conf_thres, iou_thres=iou_thres, topk=topk,
        max_det=max_det, flow=flow, o2o_nms=o2o_nms,
    )
    kpt_raw = _channels_first(kpts, 8400, "kpts")
    if anchor_idx.size == 0:
        empty_kpts = np.zeros((0, kpt_raw.shape[0] // 3, 3), dtype=np.float32)
        return xyxy, conf, class_ids, empty_kpts, pre_nms

    # Decode keypoints only for the detections that survived.
    centers, strides = build_anchor_centers_8400()
    points = decode_keypoints(
        kpt_raw[:, anchor_idx], centers[anchor_idx], strides[anchor_idx]
    )
    return xyxy, conf, class_ids, points, pre_nms


def to_original_image(xyxy, keypoints, ratio, pad_w, pad_h, width, height):
    """Map boxes and keypoints from the letterboxed input back to the original image."""
    box = xyxy.astype(np.float32).copy()
    box[:, [0, 2]] = np.clip((box[:, [0, 2]] - pad_w) / ratio, 0.0, width - 1.0)
    box[:, [1, 3]] = np.clip((box[:, [1, 3]] - pad_h) / ratio, 0.0, height - 1.0)
    pts = keypoints.astype(np.float32).copy()
    # Keypoints are clipped to the image itself, as Ultralytics does.
    pts[:, :, 0] = np.clip((pts[:, :, 0] - pad_w) / ratio, 0.0, float(width))
    pts[:, :, 1] = np.clip((pts[:, :, 1] - pad_h) / ratio, 0.0, float(height))
    return box, pts


def format_label_line(class_id, box, keypoints, score, width, height) -> str:
    """One detection as a label line: class xc yc w h (x y visibility)... score, all normalized.

    Same layout as the Ultralytics pose label files, with the score appended.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    fields = [
        str(int(class_id)),
        f"{(x1 + x2) / 2.0 / width:.6f}",
        f"{(y1 + y2) / 2.0 / height:.6f}",
        f"{(x2 - x1) / width:.6f}",
        f"{(y2 - y1) / height:.6f}",
    ]
    for x, y, v in keypoints:
        fields += [f"{float(x) / width:.6f}", f"{float(y) / height:.6f}", f"{float(v):.6f}"]
    fields.append(f"{float(score):.6f}")
    return " ".join(fields)


def draw_pose(image_bgr, box, keypoints, label, color):
    """Draw one person: box, label, skeleton (17 points only) and visible keypoints."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
    y_text = max(0, y1 - 5)
    cv2.putText(image_bgr, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    if keypoints.shape[0] == 17:
        for a, b in COCO_SKELETON:
            if keypoints[a, 2] >= DRAW_VISIBILITY and keypoints[b, 2] >= DRAW_VISIBILITY:
                cv2.line(
                    image_bgr,
                    (int(keypoints[a, 0]), int(keypoints[a, 1])),
                    (int(keypoints[b, 0]), int(keypoints[b, 1])),
                    SKELETON_COLOR, 2, cv2.LINE_AA,
                )
    for x, y, v in keypoints:
        if v >= DRAW_VISIBILITY:
            cv2.circle(image_bgr, (int(x), int(y)), 3, KEYPOINT_COLOR, -1, cv2.LINE_AA)


# Fixed, readable box colors (BGR); pose has few classes and a fixed palette keeps runs repeatable.
_PALETTE = ((0, 200, 255), (255, 120, 0), (0, 220, 120), (200, 0, 200), (60, 60, 255))


def save_pose_outputs(
    output_dir,
    image_name,
    image_bgr,
    ratio,
    pad_w,
    pad_h,
    xyxy_640,
    scores,
    class_ids,
    keypoints_640,
    classes,
) -> int:
    """Write the annotated image and the label .txt for one image; returns the number written.

    Detections whose box is smaller than 1 px after mapping back are skipped, as for detection.
    """
    import os

    h0, w0 = image_bgr.shape[:2]
    boxes, points = to_original_image(xyxy_640, keypoints_640, ratio, pad_w, pad_h, w0, h0)
    out_img = image_bgr.copy()
    lines = []
    for i in range(scores.shape[0]):
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue
        cid = int(class_ids[i])
        name = classes[cid] if 0 <= cid < len(classes) else str(cid)
        draw_pose(
            out_img, boxes[i], points[i], f"{name}:{float(scores[i]):.2f}",
            _PALETTE[cid % len(_PALETTE)],
        )
        lines.append(format_label_line(cid, boxes[i], points[i], scores[i], w0, h0))

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(image_name)[0]
    cv2.imwrite(os.path.join(output_dir, image_name), out_img)
    with open(os.path.join(output_dir, stem + ".txt"), "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    return len(lines)
