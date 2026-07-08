#!/usr/bin/env python3
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
"""Shared end-to-end YOLO TFLite inference core with selectable postprocess flow."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

cv2 = None
np = None
tf = None
yaml = None

IMG_W = 640
IMG_H = 640
REG_MAX = 16
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
IQ9_ARCH_ALIASES = {"aarch64", "arm64"}
EXPECTED_BOX_CHANNELS = {
    "yolov10": 64,
    "yolov11": 64,
    "yolov26": 4,
}


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


def _resolve_raw_output_details(output_details: Sequence[dict]) -> tuple[dict, dict]:
    box_candidates: list[tuple[dict, int]] = []
    cls_candidates: list[tuple[dict, int]] = []

    for od in output_details:
        shp = tuple(int(x) for x in od["shape"])
        box_layout = _extract_box_layout(shp)
        if box_layout is not None:
            box_channels, anchor_count = box_layout
            if anchor_count == 8400:
                box_candidates.append((od, box_channels))
            continue

        class_dim = _class_dim_for_anchor_count(shp, anchor_count=8400)
        if class_dim is not None:
            cls_candidates.append((od, class_dim))

    if len(box_candidates) != 1:
        output_shapes = [tuple(int(x) for x in od["shape"]) for od in output_details]
        raise RuntimeError(
            "Expected exactly one box output with shape "
            f"[1,4/64,8400] or [1,8400,4/64], got {output_shapes}"
        )

    box_od, _ = box_candidates[0]
    anchor_count = _extract_box_layout(box_od["shape"])[1]
    matching_cls = [
        (od, class_dim)
        for od, class_dim in cls_candidates
        if _class_dim_for_anchor_count(od["shape"], anchor_count) is not None
    ]
    if not matching_cls:
        output_shapes = [tuple(int(x) for x in od["shape"]) for od in output_details]
        raise RuntimeError(
            "Expected class output [1,C,8400] or [1,8400,C], "
            f"got {output_shapes}"
        )

    cls_od, _ = sorted(matching_cls, key=lambda pair: pair[1], reverse=True)[0]
    return box_od, cls_od


def _extract_class_count_from_output_details(output_details: Sequence[dict]) -> int:
    box_od, cls_od = _resolve_raw_output_details(output_details)
    anchor_count = _extract_box_layout(box_od["shape"])[1]
    class_count = _class_dim_for_anchor_count(cls_od["shape"], anchor_count)
    if class_count is None or int(class_count) <= 0:
        raise RuntimeError(
            "Could not determine class count from output details: "
            f"{[tuple(int(x) for x in od['shape']) for od in output_details]}"
        )
    return int(class_count)


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


def _ensure_runtime_deps() -> None:
    global cv2, np, tf, yaml
    if cv2 is None:
        import cv2 as _cv2

        cv2 = _cv2
    if np is None:
        import numpy as _np

        np = _np
    if tf is None:
        import tensorflow as _tf

        tf = _tf
    if yaml is None:
        import yaml as _yaml

        yaml = _yaml


def sigmoid_clip(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def dequant(arr: np.ndarray, scale: float, zp: int) -> np.ndarray:
    if scale and scale != 0:
        return (arr.astype(np.float32) - float(zp)) * float(scale)
    return arr.astype(np.float32)


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


def confidence_to_quantized_logit_threshold(
    conf_thres: float, scale: float, zp: int, dtype: np.dtype
) -> int:
    info = np.iinfo(dtype)
    conf = float(conf_thres)
    if conf <= 0.0:
        return int(info.min)
    if conf >= 1.0:
        return int(info.max) + 1
    if scale <= 0.0:
        return int(info.min)

    logit_thr = float(np.log(conf / (1.0 - conf)))
    return int(np.ceil(logit_thr / float(scale) + float(zp)))


def load_class_names(yaml_path: str) -> list[str]:
    _ensure_runtime_deps()
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]
    if isinstance(names, dict):
        ordered = []
        for key, value in names.items():
            try:
                idx = int(key)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "YAML 'names' dict keys must be integer-like class ids."
                ) from exc
            ordered.append((idx, value))
        ordered.sort(key=lambda pair: pair[0])
        indices = [idx for idx, _ in ordered]
        if len(set(indices)) != len(indices):
            raise RuntimeError(
                "YAML 'names' dict contains duplicate class ids after integer parsing."
            )
        expected = list(range(len(indices)))
        if indices != expected:
            raise RuntimeError(
                "YAML 'names' dict class ids must be contiguous and start at 0. "
                f"Expected {expected}, got {indices}."
            )
        return [str(value) for _, value in ordered]
    return [str(x) for x in names]


def resolve_effective_flow(requested_flow: str, default_flow: str) -> str:
    if requested_flow == "auto":
        return default_flow
    return requested_flow


def _empty_postprocess_result(
    pre_nms_count: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
        pre_nms_count,
    )


def _select_candidates_with_int8_prefilter(
    cls_head: np.ndarray,
    cls_dtype: np.dtype,
    conf_thres: float,
    c_scale: float,
    c_zp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    max_q = np.max(cls_head, axis=0)
    q_thr = confidence_to_quantized_logit_threshold(
        conf_thres=conf_thres,
        scale=float(c_scale),
        zp=int(c_zp),
        dtype=cls_dtype,
    )
    keep = max_q.astype(np.int16) >= int(q_thr)
    if not np.any(keep):
        return None

    keep_idx = np.nonzero(keep)[0]
    cls_logits_k = dequant(cls_head[:, keep_idx], c_scale, c_zp)
    cls_prob_k = sigmoid_clip(cls_logits_k)
    class_ids_k = np.argmax(cls_prob_k, axis=0).astype(np.int32)
    scores_k = np.max(cls_prob_k, axis=0).astype(np.float32)

    score_keep = scores_k >= float(conf_thres)
    if not np.any(score_keep):
        return None
    return keep_idx[score_keep], class_ids_k[score_keep], scores_k[score_keep]


def _select_candidates_from_probabilities(
    cls_head: np.ndarray,
    conf_thres: float,
    c_scale: float,
    c_zp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    cls_logits = dequant(cls_head, c_scale, c_zp)
    cls_prob = sigmoid_clip(cls_logits)
    class_ids_all = np.argmax(cls_prob, axis=0).astype(np.int32)
    scores_all = np.max(cls_prob, axis=0).astype(np.float32)

    keep = scores_all >= float(conf_thres)
    if not np.any(keep):
        return None

    keep_idx = np.nonzero(keep)[0]
    return keep_idx, class_ids_all[keep_idx], scores_all[keep_idx]


def _apply_topk_limit(
    keep_idx: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if topk > 0 and scores.shape[0] > topk:
        order = np.argsort(scores)[-topk:][::-1]
        return keep_idx[order], class_ids[order], scores[order]
    return keep_idx, class_ids, scores


def _should_run_nms(flow: str, iou_thres: float, o2o_nms: bool) -> bool:
    if flow in ("o2m", "default"):
        return iou_thres > 0
    if flow == "o2o":
        return bool(o2o_nms and iou_thres > 0)
    raise ValueError(f"Unsupported flow: {flow}")


class EndToEndInference:
    def __init__(self, args: argparse.Namespace, default_flow: str):
        self.args = args
        self.default_flow = default_flow
        self.flow = resolve_effective_flow(args.postprocess_flow, default_flow)

        self.classes = load_class_names(args.yaml)
        self.class_colors = self._build_class_colors(len(self.classes))

        with open(args.model, "rb") as f:
            model_content = f.read()

        delegates = []
        if not args.no_qnn:
            delegate = tf.lite.experimental.load_delegate(
                args.qnn_lib, options={"backend_type": args.backend}
            )
            delegates = [delegate]
            print("QNN delegate enabled.")
        else:
            print("CPU only.")

        self.interpreter = tf.lite.Interpreter(
            model_content=model_content, experimental_delegates=delegates
        )
        self.interpreter.allocate_tensors()

        self.inp = self.interpreter.get_input_details()[0]
        self.outs = self.interpreter.get_output_details()

        self.input_dtype = np.dtype(self.inp["dtype"])
        self.input_is_int8 = self.input_dtype == np.dtype(np.int8)
        self.in_scale, self.in_zp = self.inp.get("quantization", (0.0, 0))

        self.box_out, self.cls_out = self._resolve_outputs()
        box_ch = _extract_box_layout(self.box_out["shape"])[0]
        self.box_mode = "dfl64" if box_ch == 64 else "ltrb4"

        self.centers_all, self.strides_all = build_anchor_centers_8400()

        print("Flow:", self.flow)
        print(
            "Input:",
            self.inp["shape"],
            self.inp["dtype"],
            "quant",
            self.inp.get("quantization"),
        )
        print(
            "Box out:",
            self.box_out["shape"],
            self.box_out["dtype"],
            "quant",
            self.box_out.get("quantization"),
        )
        print(
            "Cls out:",
            self.cls_out["shape"],
            self.cls_out["dtype"],
            "quant",
            self.cls_out.get("quantization"),
        )

    @staticmethod
    def _build_class_colors(num_classes: int) -> list[tuple[int, int, int]]:
        n = max(1, num_classes)
        rng = np.random.default_rng()
        hues = rng.random(n) * 179.0
        sats = rng.uniform(170.0, 255.0, n)
        vals = rng.uniform(190.0, 255.0, n)
        hsv = np.stack([hues, sats, vals], axis=1).astype(np.uint8).reshape(-1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
        return [(int(c[0]), int(c[1]), int(c[2])) for c in bgr]

    def _class_color(self, class_id: int) -> tuple[int, int, int]:
        if not self.class_colors:
            return (0, 255, 0)
        return self.class_colors[int(class_id) % len(self.class_colors)]

    def _resolve_outputs(self):
        return _resolve_raw_output_details(self.outs)

    def _build_input_tensor(self, padded_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.dtype(np.int8):
            if (
                abs(self.in_zp + 128.0) < 1e-3
                and abs(self.in_scale - (1.0 / 255.0)) < 1e-4
            ):
                return (rgb.astype(np.int16) - 128).astype(np.int8)[None, ...]
            x = rgb.astype(np.float32) / 255.0
            q = np.round(x / float(self.in_scale) + float(self.in_zp))
            return np.clip(q, -128, 127).astype(np.int8)[None, ...]

        if self.input_dtype == np.dtype(np.uint8):
            return rgb.astype(np.uint8)[None, ...]

        if np.issubdtype(self.input_dtype, np.floating):
            return (rgb.astype(np.float32) / 255.0).astype(self.input_dtype)[None, ...]

        raise RuntimeError(f"Unsupported TFLite input dtype: {self.input_dtype}")

    def _postprocess(self, box_raw: np.ndarray, cls_raw: np.ndarray):
        b_scale, b_zp = self.box_out.get("quantization", (0.0, 0))
        c_scale, c_zp = self.cls_out.get("quantization", (0.0, 0))
        cls_head = cls_raw[0]
        box_head = box_raw[0]

        use_int8_prefilter = (
            (not self.args.disable_int8_prefilter)
            and cls_raw.dtype == np.int8
            and c_scale is not None
            and float(c_scale) > 0.0
        )

        if use_int8_prefilter:
            selected = _select_candidates_with_int8_prefilter(
                cls_head=cls_head,
                cls_dtype=cls_raw.dtype,
                conf_thres=float(self.args.conf_thres),
                c_scale=float(c_scale),
                c_zp=int(c_zp),
            )
        else:
            selected = _select_candidates_from_probabilities(
                cls_head=cls_head,
                conf_thres=float(self.args.conf_thres),
                c_scale=float(c_scale),
                c_zp=int(c_zp),
            )

        if selected is None:
            return _empty_postprocess_result()

        keep_idx, class_ids_k, scores_k = _apply_topk_limit(
            *selected,
            topk=int(self.args.topk),
        )

        pre_nms_count = int(scores_k.shape[0])
        boxes_k = dequant(box_head[:, keep_idx], b_scale, b_zp)
        centers_k = self.centers_all[keep_idx]
        strides_k = self.strides_all[keep_idx]

        if self.box_mode == "dfl64":
            xyxy = dfl_to_xyxy_pixels(boxes_k, centers_k, strides_k)
        else:
            xyxy = ltrb_grid_to_xyxy_pixels(boxes_k, centers_k, strides_k)

        if _should_run_nms(self.flow, self.args.iou_thres, self.args.o2o_nms):
            keep2 = classwise_nms_xyxy(
                xyxy=xyxy,
                scores=scores_k,
                class_ids=class_ids_k,
                conf_thres=float(self.args.conf_thres),
                iou_thres=float(self.args.iou_thres),
                max_det=int(self.args.max_det),
            )
        else:
            keep2 = sort_and_cap(scores_k, int(self.args.max_det))

        keep2 = np.asarray(keep2, dtype=np.int32)
        if keep2.size == 0:
            return _empty_postprocess_result(pre_nms_count)

        return xyxy[keep2], scores_k[keep2], class_ids_k[keep2], pre_nms_count

    def _save_outputs(
        self,
        image_name: str,
        image_bgr: np.ndarray,
        ratio: float,
        pad_w: float,
        pad_h: float,
        xyxy_640: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> int:
        h0, w0 = image_bgr.shape[:2]
        out_img = image_bgr.copy()
        lines = []
        written = 0

        for i in range(scores.shape[0]):
            x1, y1, x2, y2 = map(float, xyxy_640[i])
            sc = float(scores[i])
            cid = int(class_ids[i])

            x1o = clamp((x1 - pad_w) / ratio, 0.0, w0 - 1.0)
            y1o = clamp((y1 - pad_h) / ratio, 0.0, h0 - 1.0)
            x2o = clamp((x2 - pad_w) / ratio, 0.0, w0 - 1.0)
            y2o = clamp((y2 - pad_h) / ratio, 0.0, h0 - 1.0)

            if (x2o - x1o) < 1.0 or (y2o - y1o) < 1.0:
                continue

            box_color = self._class_color(cid)
            cv2.rectangle(
                out_img, (int(x1o), int(y1o)), (int(x2o), int(y2o)), box_color, 2
            )
            cls_name = self.classes[cid] if 0 <= cid < len(self.classes) else str(cid)
            label = f"{cls_name}:{sc:.2f}"
            cv2.putText(
                out_img,
                label,
                (int(x1o), int(max(0.0, y1o - 5.0))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out_img,
                label,
                (int(x1o), int(max(0.0, y1o - 5.0))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            xc = (x1o + x2o) / 2.0
            yc = (y1o + y2o) / 2.0
            bw = x2o - x1o
            bh = y2o - y1o
            yolo_line = (
                f"{cid} {xc / w0:.6f} {yc / h0:.6f} "
                f"{bw / w0:.6f} {bh / h0:.6f} {sc:.6f}"
            )
            lines.append(yolo_line)
            written += 1

        os.makedirs(self.args.output_dir, exist_ok=True)
        out_img_path = os.path.join(self.args.output_dir, image_name)
        base = os.path.splitext(image_name)[0]
        out_txt_path = os.path.join(self.args.output_dir, base + ".txt")

        cv2.imwrite(out_img_path, out_img)
        with open(out_txt_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        return written

    def run(self) -> None:
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(
            os.path.join(self.args.output_dir, "classes.txt"), "w", encoding="utf-8"
        ) as f:
            for i, name in enumerate(self.classes):
                f.write(f"{i} {name}\n")

        image_files = sorted(
            [
                p
                for p in os.listdir(self.args.img_dir)
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
        )
        if not image_files:
            raise RuntimeError(f"No images found in {self.args.img_dir}")

        processed = 0
        total_time_s = 0.0
        invoke_time_s = 0.0

        for fn in image_files:
            path = os.path.join(self.args.img_dir, fn)
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                print("WARN read fail:", path)
                continue

            t0 = time.perf_counter()
            padded_bgr, (ratio, pad_w, pad_h) = letterbox_bgr(img_bgr, (IMG_H, IMG_W))
            inp_tensor = self._build_input_tensor(padded_bgr)

            self.interpreter.set_tensor(self.inp["index"], inp_tensor)
            t_invoke0 = time.perf_counter()
            self.interpreter.invoke()
            t_invoke1 = time.perf_counter()

            box_raw = _normalize_output_layout(
                self.interpreter.get_tensor(self.box_out["index"]),
                tensor_name="box",
                channel_sizes=(4, 64),
            )
            cls_raw = _normalize_output_layout(
                self.interpreter.get_tensor(self.cls_out["index"]),
                tensor_name="class",
                anchor_count=int(box_raw.shape[2]),
            )

            xyxy, scores, class_ids, pre_nms = self._postprocess(box_raw, cls_raw)
            written = self._save_outputs(
                image_name=fn,
                image_bgr=img_bgr,
                ratio=ratio,
                pad_w=pad_w,
                pad_h=pad_h,
                xyxy_640=xyxy,
                scores=scores,
                class_ids=class_ids,
            )
            t1 = time.perf_counter()

            processed += 1
            total_time_s += t1 - t0
            invoke_time_s += t_invoke1 - t_invoke0
            print(f"{fn}: flow={self.flow} preNMS={pre_nms} kept={written}")

        if processed > 0:
            avg_total_ms = (total_time_s / processed) * 1000.0
            avg_invoke_ms = (invoke_time_s / processed) * 1000.0
            print("=== Inference Timing Summary ===")
            print(f"processed={processed}")
            print(f"avg_total_inference_ms={avg_total_ms:.3f}")
            print(f"avg_model_invoke_ms={avg_invoke_ms:.3f}")


def run_script(
    script_name: str,
    default_flow: str,
    default_conf_thres: float,
    default_iou_thres: float,
    default_topk: int,
    default_max_det: int,
) -> None:
    parser = argparse.ArgumentParser(prog=script_name)
    parser.add_argument("--model", required=True, help="Path to TFLite model")
    parser.add_argument(
        "--yaml",
        default="/root/workspace/sanoop/coco.yaml",
        help="YAML with class names",
    )
    parser.add_argument("--img-dir", default="./image")
    parser.add_argument("--output-dir", default="./output")

    parser.add_argument("--conf-thres", type=float, default=default_conf_thres)
    parser.add_argument("--iou-thres", type=float, default=default_iou_thres)
    parser.add_argument("--topk", type=int, default=default_topk)
    parser.add_argument("--max-det", type=int, default=default_max_det)

    parser.add_argument(
        "--postprocess-flow", choices=["auto", "default", "o2o", "o2m"], default="auto"
    )
    parser.add_argument(
        "--o2o-nms", action="store_true", help="If flow is o2o, enable class-wise NMS"
    )
    parser.add_argument(
        "--disable-int8-prefilter",
        action="store_true",
        help=(
            "Disable int8 class prefilter and use the baseline "
            "full dequant+sigmoid class path."
        ),
    )

    parser.add_argument("--no-qnn", action="store_true")
    parser.add_argument("--qnn-lib", default="/usr/lib/libQnnTFLiteDelegate.so")
    parser.add_argument("--backend", default="htp")

    args = parser.parse_args()
    run_inference(
        model_path=args.model,
        yaml_path=args.yaml,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        model_type=None,
        default_flow=default_flow,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        topk=args.topk,
        max_det=args.max_det,
        postprocess_flow=args.postprocess_flow,
        o2o_nms=args.o2o_nms,
        disable_int8_prefilter=args.disable_int8_prefilter,
        no_qnn=args.no_qnn,
        qnn_lib=args.qnn_lib,
        backend=args.backend,
    )


def collect_image_files(img_dir: str) -> list[Path]:
    image_dir = Path(img_dir)
    if not image_dir.is_dir():
        raise RuntimeError(f"Image directory not found: {img_dir}")
    files = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
    )
    if not files:
        raise RuntimeError(f"No images found in {img_dir}")
    return files


def _validate_inference_inputs(model_path: str, yaml_path: str, img_dir: str) -> None:
    if not os.path.isfile(model_path):
        raise RuntimeError(f"Model not found: {model_path}")
    if not os.path.isfile(yaml_path):
        raise RuntimeError(f"YAML not found: {yaml_path}")
    collect_image_files(img_dir)


def _extract_box_channel_count(model_path: str) -> int:
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    box_od, _ = _resolve_raw_output_details(interp.get_output_details())
    return _extract_box_layout(box_od["shape"])[0]


def _extract_class_count(model_path: str) -> int:
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return _extract_class_count_from_output_details(interp.get_output_details())


def validate_model_type_compatibility(model_path: str, model_type: str) -> None:
    if model_type not in EXPECTED_BOX_CHANNELS:
        raise RuntimeError(
            f"Unsupported model_type for compatibility validation: {model_type}"
        )
    _ensure_runtime_deps()
    expected = EXPECTED_BOX_CHANNELS[model_type]
    actual = _extract_box_channel_count(model_path=model_path)
    if actual != expected:
        raise RuntimeError(
            f"Output shape mismatch: --type {model_type} expects "
            f"box channels={expected}, but model outputs channels={actual}. "
            "Check that --type and --model match."
        )


def validate_class_name_count_compatibility(model_path: str, yaml_path: str) -> None:
    _ensure_runtime_deps()
    yaml_class_count = len(load_class_names(yaml_path))
    model_class_count = _extract_class_count(model_path=model_path)
    if yaml_class_count != model_class_count:
        raise RuntimeError(
            "Class count mismatch: "
            f"YAML defines {yaml_class_count} classes, "
            f"but model outputs {model_class_count}. "
            "Check that --yaml matches --model."
        )


def validate_test_model_inputs(
    model_path: str, yaml_path: str, model_type: str | None = None
) -> None:
    if not os.path.isfile(model_path):
        raise RuntimeError(f"Model not found: {model_path}")
    if not os.path.isfile(yaml_path):
        raise RuntimeError(f"YAML not found: {yaml_path}")
    if model_type is not None:
        validate_model_type_compatibility(model_path=model_path, model_type=model_type)
    validate_class_name_count_compatibility(model_path=model_path, yaml_path=yaml_path)


def _has_meaningful_outputs(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    for p in output_dir.iterdir():
        if not p.is_file():
            continue
        if p.name == "classes.txt":
            continue
        return True
    return False


def _commit_output_dir(staging_dir: Path, final_output_dir: Path) -> None:
    if final_output_dir.exists():
        if not final_output_dir.is_dir():
            raise RuntimeError(
                f"Output path exists and is not a directory: {final_output_dir}"
            )
        raise RuntimeError(
            "Output directory already exists: "
            f"{final_output_dir}. Remove it or choose a different output path."
        )
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging_dir), str(final_output_dir))


def run_inference(
    model_path: str,
    yaml_path: str,
    img_dir: str,
    output_dir: str,
    model_type: str | None,
    default_flow: str,
    conf_thres: float,
    iou_thres: float,
    topk: int,
    max_det: int,
    postprocess_flow: str = "auto",
    o2o_nms: bool = False,
    disable_int8_prefilter: bool = False,
    no_qnn: bool = False,
    qnn_lib: str = "/usr/lib/libQnnTFLiteDelegate.so",
    backend: str = "htp",
) -> None:
    _ensure_runtime_deps()
    _validate_inference_inputs(
        model_path=model_path, yaml_path=yaml_path, img_dir=img_dir
    )
    validate_test_model_inputs(
        model_path=model_path,
        yaml_path=yaml_path,
        model_type=model_type,
    )
    final_output_dir = Path(output_dir).expanduser().resolve()
    staging_tmp = tempfile.TemporaryDirectory(prefix="yolo_local_output_")
    staging_output_dir = Path(staging_tmp.name) / "output"
    staging_output_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        model=model_path,
        yaml=yaml_path,
        img_dir=img_dir,
        output_dir=str(staging_output_dir),
        conf_thres=float(conf_thres),
        iou_thres=float(iou_thres),
        topk=int(topk),
        max_det=int(max_det),
        postprocess_flow=postprocess_flow,
        o2o_nms=bool(o2o_nms),
        disable_int8_prefilter=bool(disable_int8_prefilter),
        no_qnn=bool(no_qnn),
        qnn_lib=qnn_lib,
        backend=backend,
    )
    try:
        runner = EndToEndInference(args=args, default_flow=default_flow)
        runner.run()
        if not _has_meaningful_outputs(staging_output_dir):
            raise RuntimeError(
                "Inference finished but no output artifacts were produced."
            )
        _commit_output_dir(staging_output_dir, final_output_dir)
    finally:
        staging_tmp.cleanup()


def _is_iq9_native_runtime() -> bool:
    return platform.machine().lower() in IQ9_ARCH_ALIASES


def _prepare_image_input(
    image_dir: str | None, image_path: str | None
) -> tuple[str, tempfile.TemporaryDirectory | None]:
    if bool(image_dir) == bool(image_path):
        raise RuntimeError("Specify exactly one of image_dir or image_path")

    if image_dir:
        collect_image_files(image_dir)
        return image_dir, None

    img = Path(image_path).expanduser().resolve()
    if not img.is_file():
        raise RuntimeError(f"Image not found: {img}")

    tmp_obj = tempfile.TemporaryDirectory(prefix="yolo_single_image_")
    dst = Path(tmp_obj.name) / img.name
    shutil.copy2(str(img), str(dst))
    return tmp_obj.name, tmp_obj


def _adb_prefix(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def _run_cmd(cmd: Sequence[str]) -> None:
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(list(cmd), check=True)


def _adb_shell(serial: str | None, command: str) -> None:
    _run_cmd(_adb_prefix(serial) + ["shell", command])


def _adb_push(serial: str | None, src: str, dst: str) -> None:
    _run_cmd(_adb_prefix(serial) + ["push", src, dst])


def _adb_pull(serial: str | None, src: str, dst: str) -> None:
    _run_cmd(_adb_prefix(serial) + ["pull", src, dst])


def run_test_inference_local(
    model_path: str,
    yaml_path: str,
    output_dir: str,
    model_type: str,
    default_flow: str,
    conf_thres: float,
    iou_thres: float,
    topk: int,
    max_det: int,
    image_dir: str | None = None,
    image_path: str | None = None,
    postprocess_flow: str = "auto",
    o2o_nms: bool = False,
    disable_int8_prefilter: bool = False,
    no_qnn: bool = False,
    qnn_lib: str = "/usr/lib/libQnnTFLiteDelegate.so",
    backend: str = "htp",
    enforce_iq9_native: bool = True,
) -> None:
    if enforce_iq9_native and not _is_iq9_native_runtime():
        raise RuntimeError(
            "test mode without --adb is IQ9-native only. "
            f"Current architecture is '{platform.machine()}'. "
            "Use --adb from x86 host or run the same command directly on IQ9."
        )

    prepared_dir, tmp_obj = _prepare_image_input(
        image_dir=image_dir, image_path=image_path
    )
    try:
        run_inference(
            model_path=model_path,
            yaml_path=yaml_path,
            img_dir=prepared_dir,
            output_dir=output_dir,
            model_type=model_type,
            default_flow=default_flow,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            topk=topk,
            max_det=max_det,
            postprocess_flow=postprocess_flow,
            o2o_nms=o2o_nms,
            disable_int8_prefilter=disable_int8_prefilter,
            no_qnn=no_qnn,
            qnn_lib=qnn_lib,
            backend=backend,
        )
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()


def run_test_inference_adb(
    model_path: str,
    yaml_path: str,
    output_dir: str,
    model_type: str,
    default_flow: str,
    conf_thres: float,
    iou_thres: float,
    topk: int,
    max_det: int,
    image_dir: str | None = None,
    image_path: str | None = None,
    postprocess_flow: str = "auto",
    o2o_nms: bool = False,
    disable_int8_prefilter: bool = False,
    adb_serial: str | None = None,
    remote_workdir: str = "/data/local/tmp/yolo_test",
    no_qnn: bool = False,
    qnn_lib: str = "/usr/lib/libQnnTFLiteDelegate.so",
    backend: str = "htp",
) -> None:
    _ensure_runtime_deps()
    try:
        from tool.adb_runtime_bootstrap import ensure_adb_runtime_venv
    except ModuleNotFoundError:
        from adb_runtime_bootstrap import ensure_adb_runtime_venv

    validate_test_model_inputs(
        model_path=model_path,
        yaml_path=yaml_path,
        model_type=model_type,
    )
    prepared_dir, tmp_obj = _prepare_image_input(
        image_dir=image_dir, image_path=image_path
    )
    final_output_dir = Path(output_dir).expanduser().resolve()
    pull_tmp = tempfile.TemporaryDirectory(prefix="yolo_adb_pull_")
    pull_output_dir = Path(pull_tmp.name) / "output"
    pull_output_dir.mkdir(parents=True, exist_ok=True)

    remote_root = remote_workdir.rstrip("/")
    if not remote_root:
        remote_root = "/data/local/tmp/yolo_test"
    remote_run_dir = f"{remote_root}/test_run_{os.getpid()}_{int(time.time() * 1000)}"
    remote_model = f"{remote_run_dir}/{Path(model_path).name}"
    remote_yaml = f"{remote_run_dir}/{Path(yaml_path).name}"
    remote_script = f"{remote_run_dir}/inference_tflite.py"
    remote_img_dir = f"{remote_run_dir}/images"
    remote_output_dir = f"{remote_run_dir}/output"

    local_model = str(Path(model_path).expanduser().resolve())
    local_yaml = str(Path(yaml_path).expanduser().resolve())
    local_script = str(Path(__file__).resolve())
    local_target_requirements = str(
        Path(__file__).resolve().parent.parent / "requirements" / "target.txt"
    )
    remote_python = ensure_adb_runtime_venv(
        adb_serial=adb_serial,
        local_requirements_path=local_target_requirements,
    )

    try:
        _adb_shell(
            adb_serial,
            (
                f"rm -rf {shlex.quote(remote_run_dir)} && "
                f"mkdir -p {shlex.quote(remote_img_dir)}"
            ),
        )
        _adb_push(adb_serial, local_model, remote_model)
        _adb_push(adb_serial, local_yaml, remote_yaml)
        _adb_push(adb_serial, local_script, remote_script)

        for image in collect_image_files(prepared_dir):
            _adb_push(adb_serial, str(image), f"{remote_img_dir}/{image.name}")

        remote_cmd = [
            remote_python,
            remote_script,
            "--model",
            remote_model,
            "--yaml",
            remote_yaml,
            "--img-dir",
            remote_img_dir,
            "--output-dir",
            remote_output_dir,
            "--default-flow",
            default_flow,
            "--conf-thres",
            str(conf_thres),
            "--iou-thres",
            str(iou_thres),
            "--topk",
            str(topk),
            "--max-det",
            str(max_det),
            "--postprocess-flow",
            postprocess_flow,
            "--qnn-lib",
            qnn_lib,
            "--backend",
            backend,
        ]
        if o2o_nms:
            remote_cmd.append("--o2o-nms")
        if disable_int8_prefilter:
            remote_cmd.append("--disable-int8-prefilter")
        if no_qnn:
            remote_cmd.append("--no-qnn")

        _adb_shell(adb_serial, " ".join(shlex.quote(c) for c in remote_cmd))
        _adb_pull(adb_serial, f"{remote_output_dir}/.", str(pull_output_dir))
        if not _has_meaningful_outputs(pull_output_dir):
            raise RuntimeError(
                "Remote inference finished but no output artifacts were produced."
            )
        _commit_output_dir(pull_output_dir, final_output_dir)
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()
        pull_tmp.cleanup()
        try:
            _adb_shell(adb_serial, f"rm -rf {shlex.quote(remote_run_dir)}")
        except subprocess.CalledProcessError as cleanup_exc:
            print(f"[warn] remote cleanup failed: {cleanup_exc}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="inference_tflite.py")
    parser.add_argument("--model", required=True, help="Path to TFLite model")
    parser.add_argument("--yaml", required=True, help="YAML with class names")
    parser.add_argument("--img-dir", required=True, help="Directory containing images")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for annotated images/txt"
    )
    parser.add_argument(
        "--default-flow", choices=["default", "o2o", "o2m"], default="o2m"
    )
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument(
        "--postprocess-flow", choices=["auto", "default", "o2o", "o2m"], default="auto"
    )
    parser.add_argument("--o2o-nms", action="store_true")
    parser.add_argument("--disable-int8-prefilter", action="store_true")
    parser.add_argument("--no-qnn", action="store_true")
    parser.add_argument("--qnn-lib", default="/usr/lib/libQnnTFLiteDelegate.so")
    parser.add_argument("--backend", default="htp")

    args = parser.parse_args()
    run_inference(
        model_path=args.model,
        yaml_path=args.yaml,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        model_type=None,
        default_flow=args.default_flow,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        topk=args.topk,
        max_det=args.max_det,
        postprocess_flow=args.postprocess_flow,
        o2o_nms=args.o2o_nms,
        disable_int8_prefilter=args.disable_int8_prefilter,
        no_qnn=args.no_qnn,
        qnn_lib=args.qnn_lib,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
