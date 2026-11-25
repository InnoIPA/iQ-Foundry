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
"""Object-detection geometry, box decode and NMS helpers shared by the LiteRT and ONNX runners.

These used to be copied into tool/inference_tflite.py and tool/onnx_inference.py. They were moved
here unchanged (same names, same behaviour) so a fix or a new task only has to be made once.

numpy and cv2 are imported with a guard so that importing this module never fails by itself; the
callers (inference_*.py) report missing runtime dependencies with their own messages.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

IMG_W = 640
IMG_H = 640
REG_MAX = 16


def _extract_box_layout(shape: Sequence[int]) -> tuple[int, int] | None:
    shp = tuple(int(x) for x in shape)
    if len(shp) != 3 or shp[0] != 1:
        return None
    if shp[1] in (4, 64):
        return int(shp[1]), int(shp[2])
    if shp[2] in (4, 64):
        return int(shp[2]), int(shp[1])
    return None


def _class_dim_for_anchor_count(shape: Sequence[int], anchor_count: int) -> int | None:
    shp = tuple(int(x) for x in shape)
    if len(shp) != 3 or shp[0] != 1:
        return None
    if int(shp[2]) == anchor_count:
        return int(shp[1])
    if int(shp[1]) == anchor_count:
        return int(shp[2])
    return None


def _output_order_key(name: str | None, position: int) -> tuple[int, int]:
    """Graph output order; converted models name their outputs output_0, output_1."""
    if name and name.startswith("output_") and name[len("output_") :].isdigit():
        return int(name[len("output_") :]), position
    return position, position


def _select_box_and_class_outputs(
    outputs: Sequence[tuple[str | None, Sequence[int]]],
    expected_box_channels: int | None = None,
    anchor_count: int = 8400,
) -> tuple[int, int] | None:
    """Return (box_index, class_index) into `outputs`, or None.

    A class tensor with 4 or 64 classes has the same shape as a box tensor, so
    shape alone cannot tell them apart. Candidates are narrowed by the box
    channel count expected for --type (when known), and any remaining tie is
    broken by output order: every iQ-Foundry export returns (boxes, scores).
    """
    channels: dict[int, int] = {}
    for idx, (_, shape) in enumerate(outputs):
        shp = tuple(int(x) for x in shape)
        if len(shp) != 3 or shp[0] != 1:
            continue
        if shp[2] == anchor_count:
            channels[idx] = shp[1]
        elif shp[1] == anchor_count:
            channels[idx] = shp[2]
    box_candidates = [idx for idx, ch in channels.items() if ch == expected_box_channels]
    if not box_candidates:
        # Unknown --type, or a model of another family: keep the shape rule so the
        # caller's --type mismatch check can still report it clearly.
        box_candidates = [idx for idx, ch in channels.items() if ch in (4, 64)]
    if not box_candidates:
        return None
    order = {idx: _output_order_key(outputs[idx][0], idx) for idx in channels}
    box_idx = min(box_candidates, key=order.__getitem__)
    cls_candidates = [idx for idx in channels if idx != box_idx]
    if not cls_candidates:
        return None
    return box_idx, min(cls_candidates, key=order.__getitem__)


def _normalize_output_layout(
    tensor: np.ndarray,
    *,
    tensor_name: str,
    channel_sizes: tuple[int, ...] | None = None,
    anchor_count: int | None = None,
) -> np.ndarray:
    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
        raise RuntimeError(
            f"Unexpected {tensor_name} tensor rank/shape: "
            f"{tuple(int(x) for x in tensor.shape)}"
        )
    if channel_sizes is not None:
        if int(tensor.shape[1]) in channel_sizes:
            return tensor
        if int(tensor.shape[2]) in channel_sizes:
            return np.transpose(tensor, (0, 2, 1))
        raise RuntimeError(
            f"Unexpected {tensor_name} tensor shape: "
            f"{tuple(int(x) for x in tensor.shape)}"
        )
    if anchor_count is not None:
        if int(tensor.shape[2]) == anchor_count:
            return tensor
        if int(tensor.shape[1]) == anchor_count:
            return np.transpose(tensor, (0, 2, 1))
        raise RuntimeError(
            f"Unexpected {tensor_name} tensor shape: "
            f"{tuple(int(x) for x in tensor.shape)}"
        )
    raise ValueError("Either channel_sizes or anchor_count must be provided.")


def sigmoid_clip(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def letterbox_bgr(
    image_bgr: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[float, float, float]]:
    h0, w0 = image_bgr.shape[:2]
    new_h, new_w = new_shape

    ratio = min(new_h / h0, new_w / w0)
    resized_w = int(round(w0 * ratio))
    resized_h = int(round(h0 * ratio))

    dw = (new_w - resized_w) / 2.0
    dh = (new_h - resized_h) / 2.0

    resized = cv2.resize(
        image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, (float(ratio), float(dw), float(dh))


@lru_cache(maxsize=1)
def build_anchor_centers_8400() -> tuple[np.ndarray, np.ndarray]:
    centers = []
    strides = []
    for grid, stride in ((80, 8.0), (40, 16.0), (20, 32.0)):
        ys, xs = np.meshgrid(
            np.arange(grid, dtype=np.float32),
            np.arange(grid, dtype=np.float32),
            indexing="ij",
        )
        ax = (xs + 0.5) * stride
        ay = (ys + 0.5) * stride
        centers.append(np.stack([ax, ay], axis=-1).reshape(-1, 2))
        strides.append(np.full((grid * grid,), stride, dtype=np.float32))
    return np.concatenate(centers, axis=0), np.concatenate(strides, axis=0)


def dfl_decode_to_ltrb_pixels(box64x_n: np.ndarray, strides: np.ndarray) -> np.ndarray:
    num_anchors = box64x_n.shape[1]
    reg = box64x_n.reshape(4, REG_MAX, num_anchors).astype(np.float32)
    reg = reg - reg.max(axis=1, keepdims=True)
    exp = np.exp(reg)
    prob = exp / exp.sum(axis=1, keepdims=True)
    proj = np.arange(REG_MAX, dtype=np.float32).reshape(1, REG_MAX, 1)
    dist_bins = (prob * proj).sum(axis=1)
    return dist_bins * strides.reshape(1, num_anchors)


def dfl_to_xyxy_pixels(
    box64x_n: np.ndarray, centers_xy: np.ndarray, strides: np.ndarray
) -> np.ndarray:
    ltrb = dfl_decode_to_ltrb_pixels(box64x_n, strides)
    left, top, right, bottom = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
    ax, ay = centers_xy[:, 0], centers_xy[:, 1]

    x1 = ax - left
    y1 = ay - top
    x2 = ax + right
    y2 = ay + bottom

    xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0.0, IMG_W)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0.0, IMG_H)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0.0, IMG_W)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0.0, IMG_H)
    return xyxy


def ltrb_grid_to_xyxy_pixels(
    box4x_n: np.ndarray, centers_xy: np.ndarray, strides: np.ndarray
) -> np.ndarray:
    box = np.maximum(box4x_n.astype(np.float32).T, 0.0)
    left, top, right, bottom = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    ax, ay = centers_xy[:, 0], centers_xy[:, 1]

    x1 = ax - left * strides
    y1 = ay - top * strides
    x2 = ax + right * strides
    y2 = ay + bottom * strides

    xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0.0, IMG_W)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0.0, IMG_H)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0.0, IMG_W)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0.0, IMG_H)
    return xyxy


def classwise_nms_xyxy(
    xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> list[int]:
    kept: list[int] = []

    for cid in np.unique(class_ids):
        idx = np.where(class_ids == cid)[0]
        if idx.size == 0:
            continue
        idx = idx[scores[idx] >= conf_thres]
        if idx.size == 0:
            continue

        tlwh = []
        cls_scores = scores[idx].tolist()
        for x1, y1, x2, y2 in xyxy[idx].astype(np.float32):
            tlwh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

        keep_local = cv2.dnn.NMSBoxes(tlwh, cls_scores, conf_thres, iou_thres)
        if len(keep_local) == 0:
            continue

        if isinstance(keep_local, np.ndarray):
            keep_local = keep_local.flatten().astype(int).tolist()
        else:
            keep_local = [int(i) for i in keep_local]

        kept.extend(idx[keep_local].tolist())

    kept = sorted(kept, key=lambda i: float(scores[i]), reverse=True)
    if max_det > 0 and len(kept) > max_det:
        kept = kept[:max_det]
    return kept


def sort_and_cap(scores: np.ndarray, max_det: int) -> list[int]:
    keep = np.argsort(scores)[::-1].astype(int).tolist()
    if max_det > 0 and len(keep) > max_det:
        keep = keep[:max_det]
    return keep


def _should_run_nms(flow: str, iou_thres: float, o2o_nms: bool) -> bool:
    if flow in ("o2m", "default"):
        return iou_thres > 0
    if flow == "o2o":
        return bool(o2o_nms and iou_thres > 0)
    raise ValueError(f"Unsupported flow: {flow}")
