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
"""Instance-segmentation post-processing shared by every runtime, `test` and `eval`.

The exported YOLO26-seg graph returns four raw tensors (see TASK_SPECS["segment"]):
  boxes      [1, 4, 8400]        ltrb distances in stride units, as for detection
  scores     [1, C, 8400]        class logits
  mask_coef  [1, M, 8400]        per-anchor mask coefficients (M = 32)
  proto      [1, M, 160, 160]    mask prototypes (some converters write it NHWC)

A mask is `coef @ proto` for each kept detection, then it follows the Ultralytics steps: crop to
the box at prototype resolution, upsample to the 640x640 input, threshold at 0, undo the
letterbox and resize to the original image.
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

INPUT_SIZE = 640
MASK_ALPHA = 0.45
_PALETTE = (
    (0, 200, 255), (255, 120, 0), (0, 220, 120), (200, 0, 200), (60, 60, 255),
    (255, 200, 0), (120, 255, 60), (255, 60, 160),
)


def _channels_first(tensor, anchors: int, name: str):
    """A per-anchor output as [C, N] (accepts [1, C, N] and [1, N, C])."""
    return _normalize_output_layout(tensor, tensor_name=name, anchor_count=anchors)[0]


def _select_detections(boxes, scores, *, conf_thres, iou_thres, topk, max_det, flow, o2o_nms):
    """Threshold, top-k, box decode and NMS -> (anchor_index, xyxy, conf, class_id, pre_nms).

    Boxes are ltrb distances (the yolov26 export). `anchor_index` points into the 8400 anchors
    so the caller can pick the matching mask coefficients. xyxy is in the 640x640 input space.
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


def normalize_proto(proto, mask_dim: int):
    """Prototype masks as [M, H, W], accepting [1, M, H, W] and [1, H, W, M]."""
    arr = np.asarray(proto)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected proto tensor shape: {tuple(proto.shape)}")
    if arr.shape[0] == mask_dim:
        return arr.astype(np.float32)
    if arr.shape[2] == mask_dim:
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)
    raise RuntimeError(
        f"proto tensor {tuple(proto.shape)} has no axis of size {mask_dim} (mask coefficients)"
    )


def postprocess_segment(
    boxes,
    scores,
    mask_coef,
    proto,
    *,
    conf_thres: float,
    iou_thres: float,
    topk: int,
    max_det: int,
    flow: str,
    o2o_nms: bool = False,
):
    """Raw tensors -> (xyxy [N,4], scores [N], class_ids [N], coefs [N,M], proto [M,H,W], pre_nms).

    Boxes are in the 640x640 letterboxed input space. Masks are built later (build_masks) so a
    caller that only needs boxes does not pay for them.
    """
    anchor_idx, xyxy, conf, class_ids, pre_nms = _select_detections(
        boxes, scores, conf_thres=conf_thres, iou_thres=iou_thres, topk=topk,
        max_det=max_det, flow=flow, o2o_nms=o2o_nms,
    )
    coef_raw = _channels_first(mask_coef, 8400, "mask_coef")
    proto_mhw = normalize_proto(proto, int(coef_raw.shape[0]))
    coefs = coef_raw.astype(np.float32)[:, anchor_idx].T
    return xyxy, conf, class_ids, coefs, proto_mhw, pre_nms


def _unletterbox_slices(pad_w: float, pad_h: float):
    """Region of the 640x640 input that holds the image (mirrors letterbox_bgr's padding)."""
    top = int(round(pad_h - 0.1))
    bottom = INPUT_SIZE - int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = INPUT_SIZE - int(round(pad_w + 0.1))
    return slice(top, bottom), slice(left, right)


def build_masks(coefs, proto_mhw, xyxy_640, pad_w, pad_h, width, height):
    """Binary masks [N, height, width] (uint8) for the original image."""
    count = coefs.shape[0]
    if count == 0:
        return np.zeros((0, height, width), dtype=np.uint8)
    mask_dim, mh, mw = proto_mhw.shape
    logits = (coefs @ proto_mhw.reshape(mask_dim, -1)).reshape(count, mh, mw)

    # Crop to each box at prototype resolution, as Ultralytics' crop_mask does: box edges are
    # rounded to whole prototype pixels for a few detections (predict) and left exact for many
    # (validation).
    ratio_x, ratio_y = mw / INPUT_SIZE, mh / INPUT_SIZE
    scaled = xyxy_640 * np.array([ratio_x, ratio_y, ratio_x, ratio_y], dtype=np.float32)
    if count < 50:
        scaled = np.round(scaled)
    x1, y1, x2, y2 = (scaled[:, k][:, None, None] for k in range(4))
    cols = np.arange(mw, dtype=np.float32)[None, None, :]
    rows = np.arange(mh, dtype=np.float32)[None, :, None]
    logits = logits * ((cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2))

    sy, sx = _unletterbox_slices(pad_w, pad_h)
    out = np.zeros((count, height, width), dtype=np.uint8)
    for i in range(count):
        full = cv2.resize(logits[i], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        binary = (full > 0.0).astype(np.float32)[sy, sx]
        resized = cv2.resize(binary, (width, height), interpolation=cv2.INTER_LINEAR)
        out[i] = (resized > 0.5).astype(np.uint8)
    return out


def boxes_to_original(xyxy, ratio, pad_w, pad_h, width, height):
    box = xyxy.astype(np.float32).copy()
    box[:, [0, 2]] = np.clip((box[:, [0, 2]] - pad_w) / ratio, 0.0, width - 1.0)
    box[:, [1, 3]] = np.clip((box[:, [1, 3]] - pad_h) / ratio, 0.0, height - 1.0)
    return box


def mask_to_polygon(mask):
    """Outline of the largest blob of a binary mask as [[x, y], ...] (empty if none)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float32)


def format_label_line(class_id, polygon, score, width, height) -> str:
    """class x1 y1 x2 y2 ... score, coordinates normalized (the Ultralytics segment layout)."""
    fields = [str(int(class_id))]
    for x, y in polygon:
        fields += [f"{float(x) / width:.6f}", f"{float(y) / height:.6f}"]
    fields.append(f"{float(score):.6f}")
    return " ".join(fields)


def save_segment_outputs(
    output_dir,
    image_name,
    image_bgr,
    ratio,
    pad_w,
    pad_h,
    xyxy_640,
    scores,
    class_ids,
    coefs,
    proto_mhw,
    classes,
) -> int:
    """Write the annotated image and the label .txt for one image; returns the number written."""
    import os

    h0, w0 = image_bgr.shape[:2]
    masks = build_masks(coefs, proto_mhw, xyxy_640, pad_w, pad_h, w0, h0)
    boxes = boxes_to_original(xyxy_640, ratio, pad_w, pad_h, w0, h0)
    out_img = image_bgr.copy()
    overlay = image_bgr.copy()
    lines = []
    drawn = []
    for i in range(scores.shape[0]):
        if not masks[i].any():  # like Ultralytics, drop detections without a mask
            continue
        cid = int(class_ids[i])
        color = _PALETTE[cid % len(_PALETTE)]
        overlay[masks[i] > 0] = color
        drawn.append(i)
        lines.append(format_label_line(cid, mask_to_polygon(masks[i]), scores[i], w0, h0))
    if drawn:
        out_img = cv2.addWeighted(overlay, MASK_ALPHA, out_img, 1.0 - MASK_ALPHA, 0)
    for i in drawn:
        cid = int(class_ids[i])
        color = _PALETTE[cid % len(_PALETTE)]
        x1, y1, x2, y2 = (int(v) for v in boxes[i])
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        name = classes[cid] if 0 <= cid < len(classes) else str(cid)
        label = f"{name}:{float(scores[i]):.2f}"
        y_text = max(0, y1 - 5)
        cv2.putText(out_img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out_img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(image_name)[0]
    cv2.imwrite(os.path.join(output_dir, image_name), out_img)
    with open(os.path.join(output_dir, stem + ".txt"), "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    return len(lines)
