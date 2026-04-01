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
import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tool.adb_runtime_bootstrap import ensure_adb_runtime_venv
except ModuleNotFoundError:
    from adb_runtime_bootstrap import ensure_adb_runtime_venv

DEFAULT_REMOTE_RUNNER_LOCAL = str(
    Path(__file__).resolve().parent / "remote_tflite_raw_runner.py"
)
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class ImageRec:
    image_id: int
    file_name: str
    path: str
    width: int
    height: int


@dataclass
class ResolvedAnnotations:
    source_path: Path
    eval_ann_path: Path
    format_name: str
    temp_dir_obj: tempfile.TemporaryDirectory | None = None

    def cleanup(self) -> None:
        if self.temp_dir_obj is not None:
            self.temp_dir_obj.cleanup()
            self.temp_dir_obj = None


def load_coco_images(ann_path: Path, images_dir: Path) -> list[ImageRec]:
    data = json.loads(ann_path.read_text())
    out = []
    for im in data["images"]:
        fp = images_dir / im["file_name"]
        if not fp.exists():
            raise FileNotFoundError(fp)
        out.append(
            ImageRec(
                int(im["id"]),
                im["file_name"],
                str(fp),
                int(im["width"]),
                int(im["height"]),
            )
        )
    return out


def load_fp_model_class_names(model_path: Path) -> list[str]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = model.names
    if isinstance(names, dict):
        ordered = sorted((int(k), str(v)) for k, v in names.items())
        expected = list(range(len(ordered)))
        indices = [idx for idx, _ in ordered]
        if indices != expected:
            raise ValueError(
                f"FP model class ids must be contiguous and start at 0. Got {indices}."
            )
        return [name for _, name in ordered]
    return [str(x) for x in names]


def collect_custom_dataset_images(images_dir: Path) -> list[ImageRec]:
    image_files = sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    out = []
    for idx, image_path in enumerate(image_files, start=1):
        with Image.open(image_path) as im:
            width, height = im.size
        out.append(
            ImageRec(
                image_id=idx,
                file_name=image_path.name,
                path=str(image_path),
                width=width,
                height=height,
            )
        )
    return out


def clip_xyxy_to_image(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float]:
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return x1, y1, x2, y2


def xyxy_to_coco_bbox(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> list[float]:
    x1, y1, x2, y2 = clip_xyxy_to_image(x1, y1, x2, y2, width, height)
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    if bbox_w <= 0.0 or bbox_h <= 0.0:
        raise ValueError(
            f"Invalid bounding box after clipping: {(x1, y1, x2, y2)} "
            f"for image size {(width, height)}"
        )
    return [x1, y1, bbox_w, bbox_h]


def parse_yolo_txt_annotations(
    label_path: Path, rec: ImageRec, class_names: list[str], start_ann_id: int
) -> tuple[list[dict], int]:
    if not label_path.exists():
        return [], start_ann_id

    annotations = []
    ann_id = start_ann_id
    lines = [
        line.strip() for line in label_path.read_text().splitlines() if line.strip()
    ]
    for line_no, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"Expected 5 values in {label_path}:{line_no}, got {len(parts)}"
            )
        try:
            class_id = int(parts[0])
            cx, cy, bw, bh = (float(parts[i]) for i in range(1, 5))
        except ValueError as exc:
            raise ValueError(
                f"Invalid YOLO annotation in {label_path}:{line_no}"
            ) from exc
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(
                f"YOLO class id {class_id} in {label_path}:{line_no} is out of range "
                f"for {len(class_names)} FP model classes"
            )
        x1 = (cx - bw / 2.0) * rec.width
        y1 = (cy - bh / 2.0) * rec.height
        x2 = (cx + bw / 2.0) * rec.width
        y2 = (cy + bh / 2.0) * rec.height
        bbox = xyxy_to_coco_bbox(x1, y1, x2, y2, rec.width, rec.height)
        annotations.append(
            {
                "id": ann_id,
                "image_id": rec.image_id,
                "category_id": class_id + 1,
                "bbox": bbox,
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
            }
        )
        ann_id += 1
    return annotations, ann_id


def parse_voc_xml_annotations(
    xml_path: Path, rec: ImageRec, class_names: list[str], start_ann_id: int
) -> tuple[list[dict], int]:
    if not xml_path.exists():
        return [], start_ann_id

    tree = ET.parse(xml_path)
    root = tree.getroot()
    class_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}

    annotations = []
    ann_id = start_ann_id
    for obj in root.findall("object"):
        label = obj.findtext("name")
        if not label:
            raise ValueError(f"Missing object name in {xml_path}")
        if label not in class_to_id:
            raise ValueError(
                f"XML label '{label}' in {xml_path} is not present in FP model classes "
                f"{class_names}"
            )
        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise ValueError(f"Missing bndbox in {xml_path}")
        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid VOC box values in {xml_path}") from exc
        bbox = xyxy_to_coco_bbox(xmin, ymin, xmax, ymax, rec.width, rec.height)
        annotations.append(
            {
                "id": ann_id,
                "image_id": rec.image_id,
                "category_id": class_to_id[label],
                "bbox": bbox,
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
            }
        )
        ann_id += 1
    return annotations, ann_id


def collect_yolo_txt_class_ids(annotations_dir: Path) -> set[int]:
    class_ids: set[int] = set()
    for label_path in sorted(annotations_dir.glob("*.txt")):
        lines = [
            line.strip() for line in label_path.read_text().splitlines() if line.strip()
        ]
        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Expected 5 values in {label_path}:{line_no}, got {len(parts)}"
                )
            try:
                class_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid YOLO annotation in {label_path}:{line_no}"
                ) from exc
            if class_id < 0:
                raise ValueError(
                    f"YOLO class id {class_id} in {label_path}:{line_no} must be >= 0"
                )
            class_ids.add(class_id)
    return class_ids


def collect_voc_xml_labels(annotations_dir: Path) -> set[str]:
    labels: set[str] = set()
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            label = obj.findtext("name")
            if not label:
                raise ValueError(f"Missing object name in {xml_path}")
            labels.add(label)
    return labels


def validate_custom_annotation_class_compatibility(
    annotations_dir: Path,
    fp_model_path: Path,
    format_name: str,
) -> None:
    class_names = load_fp_model_class_names(fp_model_path)
    fp_class_count = len(class_names)

    if format_name == "yolo_txt":
        class_ids = collect_yolo_txt_class_ids(annotations_dir)
        if not class_ids:
            return
        observed_min = min(class_ids)
        observed_max = max(class_ids)
        if observed_max >= fp_class_count:
            raise RuntimeError(
                "Custom annotation class mismatch: observed YOLO class ids span "
                f"{observed_min}..{observed_max}, "
                f"but FP model defines {fp_class_count} classes."
            )
        return

    if format_name == "voc_xml":
        labels = sorted(collect_voc_xml_labels(annotations_dir))
        if not labels:
            return
        label_set = set(labels)
        model_label_set = set(class_names)
        if not label_set.issubset(model_label_set):
            preview = ", ".join(labels[:5])
            if len(labels) > 5:
                preview += ", ..."
            raise RuntimeError(
                "Custom annotation class mismatch: observed VOC labels "
                f"[{preview}] ({len(labels)} classes), "
                f"but FP model defines classes {class_names[:5]}"
                f"{'...' if fp_class_count > 5 else ''}."
            )
        return

    raise ValueError(f"Unsupported custom annotation format: {format_name}")


def normalize_custom_annotations_to_coco(
    annotations_dir: Path,
    images_dir: Path,
    model_path: Path,
    format_name: str,
) -> ResolvedAnnotations:
    class_names = load_fp_model_class_names(model_path)
    image_records = collect_custom_dataset_images(images_dir)
    categories = [
        {"id": idx + 1, "name": name, "supercategory": "none"}
        for idx, name in enumerate(class_names)
    ]

    images = []
    annotations = []
    ann_id = 1
    for rec in image_records:
        images.append(
            {
                "id": rec.image_id,
                "file_name": rec.file_name,
                "width": rec.width,
                "height": rec.height,
            }
        )
        stem = Path(rec.file_name).stem
        if format_name == "yolo_txt":
            parsed, ann_id = parse_yolo_txt_annotations(
                annotations_dir / f"{stem}.txt", rec, class_names, ann_id
            )
        elif format_name == "voc_xml":
            parsed, ann_id = parse_voc_xml_annotations(
                annotations_dir / f"{stem}.xml", rec, class_names, ann_id
            )
        else:
            raise ValueError(f"Unsupported custom annotation format: {format_name}")
        annotations.extend(parsed)

    tmpdir_obj = tempfile.TemporaryDirectory()
    out_path = Path(tmpdir_obj.name) / "normalized_annotations.coco.json"
    out_path.write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": categories,
            }
        )
    )
    print(
        f"[info] normalized custom {format_name} annotations from {annotations_dir} "
        f"to temporary COCO JSON {out_path}"
    )
    return ResolvedAnnotations(
        source_path=annotations_dir,
        eval_ann_path=out_path,
        format_name=format_name,
        temp_dir_obj=tmpdir_obj,
    )


def resolve_annotations_for_map(
    annotations_path: Path, images_dir: Path, fp_model_path: Path
) -> ResolvedAnnotations:
    if annotations_path.is_file():
        if annotations_path.suffix.lower() != ".json":
            raise ValueError(
                "mAP --annotations file must be a COCO .json file, or use a directory "
                "of custom .txt/.xml annotations"
            )
        return ResolvedAnnotations(
            source_path=annotations_path,
            eval_ann_path=annotations_path,
            format_name="coco",
        )

    if not annotations_path.is_dir():
        raise FileNotFoundError(annotations_path)

    txt_files = sorted(annotations_path.glob("*.txt"))
    xml_files = sorted(annotations_path.glob("*.xml"))
    if txt_files and xml_files:
        print(
            "[warn] found both YOLO .txt and VOC .xml annotations in "
            f"{annotations_path}; "
            "using .txt files and ignoring .xml"
        )
        validate_custom_annotation_class_compatibility(
            annotations_path, fp_model_path, "voc_xml"
        )
        validate_custom_annotation_class_compatibility(
            annotations_path, fp_model_path, "yolo_txt"
        )
        return normalize_custom_annotations_to_coco(
            annotations_path, images_dir, fp_model_path, "yolo_txt"
        )
    if txt_files:
        print(f"[info] detected YOLO .txt annotations in {annotations_path}")
        validate_custom_annotation_class_compatibility(
            annotations_path, fp_model_path, "yolo_txt"
        )
        return normalize_custom_annotations_to_coco(
            annotations_path, images_dir, fp_model_path, "yolo_txt"
        )
    if xml_files:
        print(f"[info] detected VOC .xml annotations in {annotations_path}")
        validate_custom_annotation_class_compatibility(
            annotations_path, fp_model_path, "voc_xml"
        )
        return normalize_custom_annotations_to_coco(
            annotations_path, images_dir, fp_model_path, "voc_xml"
        )
    raise ValueError(
        f"No supported annotations found in {annotations_path}. "
        "Expected a COCO .json file or a directory containing .txt/.xml labels."
    )


def make_anchors_8400(imgsz: int = 640) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    strides = []
    for s in (8, 16, 32):
        h = w = imgsz // s
        ys, xs = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        cx = (xs.reshape(-1) + 0.5) * s
        cy = (ys.reshape(-1) + 0.5) * s
        centers.append(np.stack([cx, cy], axis=1))
        strides.append(np.full((h * w,), s, dtype=np.float32))
    return np.concatenate(centers, axis=0), np.concatenate(strides, axis=0)


ANCHOR_CENTERS_8400, STRIDES_8400 = make_anchors_8400(640)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(im: Image.Image, new_shape=(640, 640), color=(114, 114, 114)):
    w0, h0 = im.size
    nw, nh = new_shape
    r = min(nw / w0, nh / h0)
    resized = im.resize((int(round(w0 * r)), int(round(h0 * r))), Image.BILINEAR)
    canvas = Image.new("RGB", (nw, nh), color)
    padw = (nw - resized.size[0]) / 2
    padh = (nh - resized.size[1]) / 2
    canvas.paste(resized, (int(round(padw - 0.1)), int(round(padh - 0.1))))
    return np.asarray(canvas), r, (padw, padh)


def preprocess_for_pt(image_path: str, imgsz: int = 640):
    import torch

    im = Image.open(image_path).convert("RGB")
    arr, ratio, (padw, padh) = letterbox(im, (imgsz, imgsz))
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return x, ratio, padw, padh, im.size


def preprocess_for_tflite(image_path: str, input_detail: dict):
    im = Image.open(image_path).convert("RGB")
    shape = input_detail["shape"]
    h, w = int(shape[1]), int(shape[2])
    arr, ratio, (padw, padh) = letterbox(im, (w, h))
    dtype = input_detail["dtype"]

    if dtype == np.float32:
        x = arr.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        x = arr.astype(np.uint8)
    elif dtype == np.int8:
        scale, zp = input_detail["quantization"]
        if scale == 0:
            raise ValueError("Invalid int8 input scale=0")
        x_f = arr.astype(np.float32) / 255.0
        x = np.clip(np.round(x_f / scale + zp), -128, 127).astype(np.int8)
    else:
        raise TypeError(f"Unsupported TFLite input dtype: {dtype}")

    return x[None], ratio, padw, padh, im.size


def adb_prefix(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    return cmd


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def adb_shell(serial: str | None, command: str) -> None:
    run_cmd(adb_prefix(serial) + ["shell", command])


def adb_push(serial: str | None, src: str, dst: str) -> None:
    run_cmd(adb_prefix(serial) + ["push", src, dst])


def adb_pull(serial: str | None, src: str, dst: str) -> None:
    run_cmd(adb_prefix(serial) + ["pull", src, dst])


def maybe_dequant(arr: np.ndarray, detail: dict) -> np.ndarray:
    scale, zp = detail.get("quantization", (0.0, 0))
    if arr.dtype in (np.int8, np.uint8) and scale not in (0, 0.0):
        return (arr.astype(np.float32) - zp) * scale
    return arr.astype(np.float32)


def prepare_shared_int8_inputs(
    images: list[ImageRec],
    input_detail: dict,
    adb_serial: str | None,
    remote_input_dir: str,
):
    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = Path(tmpdir_obj.name)
    local_input_dir = tmpdir / "inputs"
    local_input_dir.mkdir(parents=True, exist_ok=True)

    meta = {}

    for rec in images:
        x, ratio, padw, padh, orig_size = preprocess_for_tflite(rec.path, input_detail)
        stem = str(rec.image_id)
        np.save(local_input_dir / f"{stem}.npy", x)
        meta[rec.image_id] = {
            "ratio": ratio,
            "padw": padw,
            "padh": padh,
            "orig_size": orig_size,
        }

    adb_shell(
        adb_serial,
        f"rm -rf {shlex.quote(remote_input_dir)} && "
        f"mkdir -p {shlex.quote(remote_input_dir)}",
    )

    for f in sorted(local_input_dir.glob("*.npy")):
        adb_push(adb_serial, str(f), f"{remote_input_dir}/{f.name}")

    adb_shell(adb_serial, f"ls -l {shlex.quote(remote_input_dir)}")

    return tmpdir_obj, meta


def _find_boxes_scores(
    outputs: list[np.ndarray], details: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    box_candidates: list[tuple[np.ndarray, int]] = []
    cls_candidates: list[np.ndarray] = []
    for arr, det in zip(outputs, details, strict=True):
        x = maybe_dequant(arr, det)
        shp = tuple(int(v) for v in x.shape)
        if len(shp) != 3 or shp[0] != 1:
            continue

        if shp[1] in (4, 64):
            box_candidates.append((x, 2))
        elif shp[2] in (4, 64):
            box_candidates.append((x, 1))
        else:
            cls_candidates.append(x)

    if len(box_candidates) != 1:
        shapes = [tuple(o.shape) for o in outputs]
        raise RuntimeError(
            f"Could not uniquely identify raw box tensor. Shapes={shapes}"
        )

    boxes, anchor_axis = box_candidates[0]
    anchor_count = int(boxes.shape[anchor_axis])
    matching_cls = [
        s
        for s in cls_candidates
        if int(s.shape[1]) == anchor_count or int(s.shape[2]) == anchor_count
    ]
    if not matching_cls:
        shapes = [tuple(o.shape) for o in outputs]
        raise RuntimeError(
            f"Could not locate raw boxes/scores tensors. Shapes={shapes}"
        )

    def class_dim(arr: np.ndarray) -> int:
        if int(arr.shape[1]) == anchor_count:
            return int(arr.shape[2])
        return int(arr.shape[1])

    scores = sorted(matching_cls, key=class_dim, reverse=True)[0]
    return boxes, scores


def _extract_box_layout(shape: tuple[int, ...]) -> tuple[int, int] | None:
    if len(shape) != 3 or int(shape[0]) != 1:
        return None
    if int(shape[1]) in (4, 64):
        return int(shape[1]), int(shape[2])
    if int(shape[2]) in (4, 64):
        return int(shape[2]), int(shape[1])
    return None


def _class_dim_for_anchor_count(
    shape: tuple[int, ...], anchor_count: int
) -> int | None:
    if len(shape) != 3 or int(shape[0]) != 1:
        return None
    if int(shape[2]) == anchor_count:
        return int(shape[1])
    if int(shape[1]) == anchor_count:
        return int(shape[2])
    return None


def _extract_tflite_class_count(model_path: Path) -> int:
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter

    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    output_details = interp.get_output_details()

    box_candidates = []
    cls_candidates = []
    for detail in output_details:
        shape = tuple(int(v) for v in detail["shape"])
        box_layout = _extract_box_layout(shape)
        if box_layout is not None:
            _, anchor_count = box_layout
            if anchor_count == 8400:
                box_candidates.append(detail)
            continue
        cls_dim = _class_dim_for_anchor_count(shape, anchor_count=8400)
        if cls_dim is not None:
            cls_candidates.append((detail, cls_dim))

    if len(box_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify TFLite box output for class-count validation. "
            f"Shapes={[tuple(int(v) for v in d['shape']) for d in output_details]}"
        )

    anchor_count = _extract_box_layout(
        tuple(int(v) for v in box_candidates[0]["shape"])
    )[1]
    matching_cls = [
        cls_dim
        for detail, cls_dim in cls_candidates
        if _class_dim_for_anchor_count(
            tuple(int(v) for v in detail["shape"]), anchor_count
        )
        is not None
    ]
    if not matching_cls:
        raise RuntimeError(
            "Could not identify TFLite class output for class-count validation. "
            f"Shapes={[tuple(int(v) for v in d['shape']) for d in output_details]}"
        )

    return int(max(matching_cls))


def normalize_raw_pair(
    boxes: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Output to [C,N]
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)

    if boxes.ndim != 2 or scores.ndim != 2:
        raise RuntimeError(
            f"Unexpected raw tensor ranks: boxes={boxes.shape}, scores={scores.shape}"
        )

    # boxes
    if boxes.shape[0] in (4, 64):
        b = boxes
    elif boxes.shape[1] in (4, 64):
        b = boxes.T
    else:
        raise RuntimeError(f"Unrecognized boxes shape: {boxes.shape}")

    # scores (dynamic class count; only anchor dimension must match boxes)
    if scores.shape[1] == b.shape[1]:
        s = scores
    elif scores.shape[0] == b.shape[1]:
        s = scores.T
    else:
        raise RuntimeError(
            f"Unrecognized scores shape: {scores.shape} for anchors={b.shape[1]}"
        )

    if b.shape[1] != s.shape[1]:
        raise RuntimeError(
            f"Boxes/scores anchor count mismatch: {b.shape} vs {s.shape}"
        )
    return b, s


def decode_dfl16(boxes_c_n: np.ndarray) -> np.ndarray:
    # [64, N] -> [4, N] distances in stride units
    n = boxes_c_n.shape[1]
    x = boxes_c_n.reshape(4, 16, n)
    x = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(x)
    p = p / np.sum(p, axis=1, keepdims=True)
    bins = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
    dist = np.sum(p * bins, axis=1)
    return dist.astype(np.float32)


def dist_to_xyxy(
    dist4_n: np.ndarray, centers_xy: np.ndarray, strides: np.ndarray
) -> np.ndarray:
    left = dist4_n[0] * strides
    top = dist4_n[1] * strides
    right = dist4_n[2] * strides
    bottom = dist4_n[3] * strides
    cx = centers_xy[:, 0]
    cy = centers_xy[:, 1]
    out = np.stack([cx - left, cy - top, cx + right, cy + bottom], axis=1)
    return out.astype(np.float32)


def ensure_probs(scores_c_n: np.ndarray) -> np.ndarray:
    mn, mx = float(scores_c_n.min()), float(scores_c_n.max())
    if mn < 0.0 or mx > 1.0:
        return sigmoid(scores_c_n)
    return scores_c_n


def nms_class_aware(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thr: float = 0.7,
) -> np.ndarray:
    keep_all = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        b = boxes[idx]
        s = scores[idx]
        order = np.argsort(-s)
        keep_local = []
        while order.size > 0:
            i = order[0]
            keep_local.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(b[i, 0], b[rest, 0])
            yy1 = np.maximum(b[i, 1], b[rest, 1])
            xx2 = np.minimum(b[i, 2], b[rest, 2])
            yy2 = np.minimum(b[i, 3], b[rest, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            area_r = (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1])
            union = np.maximum(area_i + area_r - inter, 1e-9)
            iou = inter / union
            order = rest[iou <= iou_thr]
        keep_all.extend(idx[np.array(keep_local, dtype=np.int64)].tolist())
    return np.array(sorted(keep_all), dtype=np.int64)


def undo_letterbox_xyxy(
    boxes: np.ndarray,
    ratio: float,
    padw: float,
    padh: float,
    orig_size: tuple[int, int],
) -> np.ndarray:
    ow, oh = orig_size
    out = boxes.copy()
    out[:, [0, 2]] -= padw
    out[:, [1, 3]] -= padh
    out[:, :4] /= ratio
    out[:, 0] = np.clip(out[:, 0], 0, ow)
    out[:, 1] = np.clip(out[:, 1], 0, oh)
    out[:, 2] = np.clip(out[:, 2], 0, ow)
    out[:, 3] = np.clip(out[:, 3], 0, oh)
    return out


def postprocess_raw(
    boxes_raw: np.ndarray,
    scores_raw: np.ndarray,
    decoder: str,
    ratio: float,
    padw: float,
    padh: float,
    orig_size: tuple[int, int],
    conf_thr: float,
    nms_thr: float,
    max_det: int,
) -> np.ndarray:
    boxes_c_n, scores_c_n = normalize_raw_pair(boxes_raw, scores_raw)
    probs = ensure_probs(scores_c_n)

    if decoder == "dfl16":
        dist4_n = decode_dfl16(boxes_c_n)
    elif decoder == "ltrb4":
        if boxes_c_n.shape[0] != 4:
            raise RuntimeError(f"ltrb4 decoder expected [4,N], got {boxes_c_n.shape}")
        dist4_n = boxes_c_n.astype(np.float32)
    else:
        raise ValueError(f"Unknown decoder: {decoder}")

    if dist4_n.shape[1] != 8400:
        raise RuntimeError(f"Expected 8400 anchors, got {dist4_n.shape[1]}")

    boxes_xyxy = dist_to_xyxy(dist4_n, ANCHOR_CENTERS_8400, STRIDES_8400)
    cls_ids = np.argmax(probs, axis=0)
    cls_scores = probs[cls_ids, np.arange(probs.shape[1])]
    keep = cls_scores >= conf_thr
    if not np.any(keep):
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xyxy = boxes_xyxy[keep]
    cls_scores = cls_scores[keep]
    cls_ids = cls_ids[keep].astype(np.float32)
    boxes_xyxy = undo_letterbox_xyxy(boxes_xyxy, ratio, padw, padh, orig_size)

    keep_nms = nms_class_aware(
        boxes_xyxy, cls_scores, cls_ids.astype(np.int64), iou_thr=nms_thr
    )
    boxes_xyxy = boxes_xyxy[keep_nms]
    cls_scores = cls_scores[keep_nms]
    cls_ids = cls_ids[keep_nms]

    order = np.argsort(-cls_scores)[:max_det]
    dets = np.concatenate(
        [boxes_xyxy[order], cls_scores[order, None], cls_ids[order, None]], axis=1
    )
    return dets.astype(np.float32)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    return out


class PTRawModel:
    def __init__(self, model_path: str, family: str, head: str):
        from ultralytics import YOLO

        self.family = family
        self.head = head
        self.yolo = YOLO(model_path)
        self.core = self.yolo.model
        self.core.eval()

    def infer_raw(
        self, image_path: str
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, tuple[int, int]]:
        import torch

        x, ratio, padw, padh, orig_size = preprocess_for_pt(image_path)
        with torch.no_grad():
            y = self.core(x)

        if isinstance(y, (tuple, list)) and len(y) >= 2 and isinstance(y[1], dict):
            y_dict = y[1]
        elif isinstance(y, dict):
            y_dict = y
        else:
            raise RuntimeError(
                f"Unsupported FP output structure for family={self.family}: "
                f"type={type(y)}. "
                "This may indicate an incompatible --type and --fp-model pair."
            )

        if self.family in ("yolo10", "yolo26"):
            branch = "one2one" if self.head == "o2o" else "one2many"
            if branch not in y_dict or not isinstance(y_dict[branch], dict):
                raise RuntimeError(
                    f"FP model/type mismatch: expected branch '{branch}' "
                    f"for family={self.family}, "
                    f"but got keys={list(y_dict.keys())}"
                )
            if "boxes" not in y_dict[branch] or "scores" not in y_dict[branch]:
                raise RuntimeError(
                    f"FP model/type mismatch: branch '{branch}' missing boxes/scores. "
                    f"Available keys={list(y_dict[branch].keys())}"
                )
            boxes = y_dict[branch]["boxes"].detach().cpu().numpy()
            scores = y_dict[branch]["scores"].detach().cpu().numpy()
        elif self.family == "yolo11":
            if "boxes" not in y_dict or "scores" not in y_dict:
                keys = list(y_dict.keys())
                hint = ""
                if "one2many" in y_dict or "one2one" in y_dict:
                    hint = (
                        " Detected one2many/one2one keys; "
                        "this FP model looks like YOLO10/YOLO26, not YOLO11."
                    )
                raise RuntimeError(
                    "FP model/type mismatch: expected YOLO11 keys "
                    f"['boxes', 'scores'], got keys={keys}.{hint}"
                )
            boxes = y_dict["boxes"].detach().cpu().numpy()
            scores = y_dict["scores"].detach().cpu().numpy()
        else:
            raise ValueError(self.family)
        return boxes, scores, ratio, padw, padh, orig_size


class TFLiteRawModel:
    def __init__(self, model_path: str):
        try:
            from tflite_runtime.interpreter import Interpreter
        except Exception:
            from tensorflow.lite.python.interpreter import Interpreter
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.input_detail = self.interp.get_input_details()[0]
        self.output_details = self.interp.get_output_details()

    def infer_raw(
        self, image_path: str
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, tuple[int, int]]:
        x, ratio, padw, padh, orig_size = preprocess_for_tflite(
            image_path, self.input_detail
        )
        self.interp.set_tensor(self.input_detail["index"], x)
        self.interp.invoke()
        outputs = [self.interp.get_tensor(d["index"]) for d in self.output_details]
        boxes, scores = _find_boxes_scores(outputs, self.output_details)
        return boxes, scores, ratio, padw, padh, orig_size


class ADBTFLiteRawModel:
    def __init__(
        self,
        model_path: str,
        adb_serial: str | None,
        remote_workdir: str,
        remote_runner: str,
        remote_python: str,
        qnn_lib: str,
        backend: str,
        no_qnn: bool,
        shared_remote_input_dir: str,
        shared_meta: dict,
    ):
        self.model_path = model_path
        self.adb_serial = adb_serial
        self.remote_workdir = remote_workdir
        self.remote_runner = remote_runner
        self.remote_python = remote_python
        self.qnn_lib = qnn_lib
        self.backend = backend
        self.no_qnn = no_qnn
        self.shared_remote_input_dir = shared_remote_input_dir
        self.shared_meta = shared_meta

        self.remote_model = f"{self.remote_workdir}/{Path(model_path).name}"
        self.remote_output_dir = f"{self.remote_workdir}/outputs"

        adb_shell(self.adb_serial, f"mkdir -p {shlex.quote(self.remote_workdir)}")
        adb_push(self.adb_serial, self.model_path, self.remote_model)

        self.local_output_dir = None

    def prepare_batch(self, images: list[ImageRec]) -> None:
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = self._tmpdir_obj.name
        local_output_dir = Path(tmpdir) / "outputs"
        local_output_dir.mkdir(parents=True, exist_ok=True)
        self.local_output_dir = local_output_dir

        adb_shell(
            self.adb_serial,
            f"rm -rf {shlex.quote(self.remote_output_dir)} && "
            f"mkdir -p {shlex.quote(self.remote_output_dir)}",
        )

        cmd = [
            self.remote_python,
            self.remote_runner,
            "--model",
            self.remote_model,
            "--input-dir",
            self.shared_remote_input_dir,
            "--output-dir",
            self.remote_output_dir,
            "--qnn-lib",
            self.qnn_lib,
            "--backend",
            self.backend,
        ]
        if self.no_qnn:
            cmd.append("--no-qnn")

        adb_shell(self.adb_serial, " ".join(shlex.quote(c) for c in cmd))

        adb_pull(self.adb_serial, self.remote_output_dir, str(local_output_dir.parent))

    def infer_raw(self, image_path: str):
        raise RuntimeError(
            "ADBTFLiteRawModel now uses prepare_batch() + get_result_for_image()"
        )

    def get_result_for_image(
        self, rec: ImageRec
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, tuple[int, int]]:
        if self.local_output_dir is None:
            raise RuntimeError(
                "prepare_batch(images) must be called before get_result_for_image(rec)"
            )

        stem = str(rec.image_id)
        boxes_path = self.local_output_dir / f"{stem}_boxes.npy"
        scores_path = self.local_output_dir / f"{stem}_scores.npy"

        if not boxes_path.exists():
            raise FileNotFoundError(boxes_path)
        if not scores_path.exists():
            raise FileNotFoundError(scores_path)

        boxes = np.load(boxes_path)
        scores = np.load(scores_path)

        meta = self.shared_meta[rec.image_id]
        return (
            boxes,
            scores,
            meta["ratio"],
            meta["padw"],
            meta["padh"],
            meta["orig_size"],
        )

    def cleanup(self):
        if hasattr(self, "_tmpdir_obj"):
            self._tmpdir_obj.cleanup()


def coco_category_ids(ann_path: Path) -> list[int]:
    data = json.loads(ann_path.read_text())
    cats = sorted(data["categories"], key=lambda x: x["id"])
    return [int(c["id"]) for c in cats]


def dets_to_coco_json(
    dets: np.ndarray, image_id: int, cat_ids: list[int]
) -> list[dict]:
    if dets.shape[0] == 0:
        return []
    boxes = xyxy_to_xywh(dets[:, :4])
    out = []
    for i in range(dets.shape[0]):
        cls = int(dets[i, 5])
        if cls < 0 or cls >= len(cat_ids):
            raise RuntimeError(
                "Prediction class id out of range for evaluation categories: "
                f"class_id={cls}, categories={len(cat_ids)}. "
                "Check FP/INT model class counts and annotation categories."
            )
        out.append(
            {
                "image_id": int(image_id),
                "category_id": int(cat_ids[cls]),
                "bbox": [float(v) for v in boxes[i].tolist()],
                "score": float(dets[i, 4]),
            }
        )
    return out


def eval_map50(
    ann_path: Path, pred_json: Path, image_ids: list[int], max_det: int = 100
) -> float:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(ann_path))
    preds = json.loads(pred_json.read_text())
    if not preds:
        return 0.0
    coco_dt = coco_gt.loadRes(preds)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = list(image_ids)  # <-- critical fix
    ev.params.iouThrs = np.array([0.5], dtype=np.float32)
    ev.params.maxDets = [1, 10, max(100, int(max_det))]
    ev.evaluate()
    ev.accumulate()

    precision = ev.eval["precision"]  # [T,R,K,A,M]
    p = precision[0, :, :, 0, -1]
    p = p[p > -1]
    return float(p.mean()) if p.size else 0.0


def validate_eval_class_count_compatibility(
    fp_model_path: Path,
    int_model_path: Path,
    cat_ids: list[int],
) -> None:
    fp_class_count = len(load_fp_model_class_names(fp_model_path))
    int_class_count = _extract_tflite_class_count(int_model_path)
    annotation_class_count = len(cat_ids)

    if annotation_class_count != fp_class_count:
        raise RuntimeError(
            "Annotation category count mismatch: "
            f"annotations define {annotation_class_count} categories, "
            f"but FP model defines {fp_class_count} classes."
        )
    if int_class_count != fp_class_count:
        raise RuntimeError(
            "FP/INT class count mismatch: "
            f"FP model defines {fp_class_count} classes, "
            f"but INT model outputs {int_class_count}."
        )


def build_model_runner(cfg: dict, args, shared_int8=None):
    if cfg["backend"] == "pt":
        return PTRawModel(cfg["path"], cfg["family"], cfg["head"])

    if cfg["backend"] == "tflite":
        if args.int8_on_device:
            if shared_int8 is None:
                raise RuntimeError(
                    "shared_int8 cache is required for on-device INT8 mode"
                )
            print("INT8 inference mode: IQ9 via adb")
            return ADBTFLiteRawModel(
                model_path=cfg["path"],
                adb_serial=args.adb_serial,
                remote_workdir=args.remote_workdir,
                remote_runner=args.remote_runner_remote,
                remote_python=args.remote_python,
                qnn_lib=args.qnn_lib,
                backend=args.backend,
                no_qnn=args.no_qnn,
                shared_remote_input_dir=shared_int8["remote_input_dir"],
                shared_meta=shared_int8["meta"],
            )
        return TFLiteRawModel(cfg["path"])

    raise ValueError(cfg["backend"])


def _format_run_location(model_key: str, cfg: dict, is_adb_batch: bool) -> str:
    runtime = "IQ9 (adb)" if is_adb_batch else "x86 host"
    quant = cfg.get("quant", "fp32")
    return (
        f"[RUN-LOCATION] model={model_key} runtime={runtime} "
        f"backend={cfg['backend']} quant={quant}"
    )


def evaluate_one_model(
    model_key: str,
    cfg: dict,
    images: list[ImageRec],
    cat_ids: list[int],
    ann_path: Path,
    outdir: Path | None,
    conf_thr: float,
    nms_thr: float,
    max_det: int,
    args,
    shared_int8=None,
    save_pred_json: bool = True,
) -> dict:
    runner = build_model_runner(cfg, args, shared_int8=shared_int8)
    all_preds = []

    is_adb_batch = isinstance(runner, ADBTFLiteRawModel)
    print(_format_run_location(model_key, cfg, is_adb_batch))

    if is_adb_batch:
        print(f"[{model_key}] preparing batch on host + IQ9 ...")
        runner.prepare_batch(images)

    try:
        for idx, rec in enumerate(images, 1):
            if is_adb_batch:
                boxes, scores, ratio, padw, padh, orig_size = (
                    runner.get_result_for_image(rec)
                )
            else:
                boxes, scores, ratio, padw, padh, orig_size = runner.infer_raw(rec.path)

            dets = postprocess_raw(
                boxes,
                scores,
                cfg["decoder"],
                ratio,
                padw,
                padh,
                orig_size,
                conf_thr=conf_thr,
                nms_thr=nms_thr,
                max_det=max_det,
            )
            all_preds.extend(dets_to_coco_json(dets, rec.image_id, cat_ids))

            if idx % 100 == 0:
                print(f"[{model_key}] processed {idx}/{len(images)} images")

    finally:
        if is_adb_batch:
            runner.cleanup()

    out_json_str = None
    if save_pred_json:
        if outdir is None:
            raise ValueError("outdir is required when save_pred_json=True")
        out_json = outdir / f"{model_key}_predictions.json"
        out_json.write_text(json.dumps(all_preds))
        map50 = eval_map50(
            ann_path,
            out_json,
            [rec.image_id for rec in images],
            max_det=max_det,
        )
        out_json_str = str(out_json)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_json = Path(tmp_dir) / f"{model_key}_predictions.json"
            tmp_json.write_text(json.dumps(all_preds))
            map50 = eval_map50(
                ann_path,
                tmp_json,
                [rec.image_id for rec in images],
                max_det=max_det,
            )

    return {
        "model_key": model_key,
        "family": cfg["family"],
        "backend": cfg["backend"],
        "head": cfg["head"],
        "quant": cfg.get("quant", "fp32"),
        "decoder": cfg["decoder"],
        "path": cfg["path"],
        "map50": map50,
        "num_predictions": len(all_preds),
        "pred_json": out_json_str,
    }


def model_decoder_for_type(model_type: str) -> str:
    if model_type in ("yolov10", "yolov11"):
        return "dfl16"
    if model_type == "yolov26":
        return "ltrb4"
    raise ValueError(f"Unsupported model type: {model_type}")


def fp_head_to_cfg_head(model_type: str, fp_head: str) -> str:
    if model_type == "yolov11":
        return "default"
    if fp_head == "one2many":
        return "o2m"
    if fp_head == "one2one":
        return "o2o"
    raise ValueError(f"Unsupported fp_head: {fp_head}")


def make_runtime_args(
    adb_serial: str | None,
    remote_workdir: str,
    remote_runner_remote: str,
    remote_python: str,
    qnn_lib: str,
    backend: str,
    no_qnn: bool,
):
    return argparse.Namespace(
        int8_on_device=True,
        adb_serial=adb_serial,
        remote_workdir=remote_workdir,
        remote_runner_remote=remote_runner_remote,
        remote_python=remote_python,
        qnn_lib=qnn_lib,
        backend=backend,
        no_qnn=no_qnn,
    )


def _resolve_remote_base_dir(remote_workdir: str) -> str:
    remote_root = remote_workdir.rstrip("/")
    if not remote_root:
        return "/data/local/tmp/yolo_map_eval"
    return remote_root


def make_map_remote_run_layout(
    remote_workdir: str,
    int_model_path: Path,
    remote_runner_remote: str,
) -> dict[str, str]:
    remote_root = _resolve_remote_base_dir(remote_workdir)
    remote_runner_name = Path(remote_runner_remote).name
    run_id = f"map_run_{os.getpid()}_{int(time.time() * 1000)}"
    remote_run_dir = f"{remote_root}/{run_id}"
    return {
        "remote_root": remote_root,
        "remote_run_dir": remote_run_dir,
        "remote_model": f"{remote_run_dir}/{int_model_path.name}",
        "remote_runner": f"{remote_run_dir}/{remote_runner_name}",
        "remote_input_dir": f"{remote_run_dir}/inputs",
    }


def make_report_text(
    model_type: str,
    fp_head: str,
    annotations: Path,
    images: Path,
    image_count: int,
    fp_model: str,
    int_model: str,
    fp_map50: float,
    int_map50: float,
    abs_delta: float,
    pct_delta: float | None,
    trend: str,
) -> str:
    if pct_delta is None:
        pct_delta_str = "N/A (FP mAP@0.5 is 0.0)"
    else:
        pct_delta_str = f"{pct_delta:+.2f}%"

    lines = [
        "FP vs INT mAP@0.5 Pair Evaluation",
        f"model_type: {model_type}",
        f"fp_head: {fp_head}",
        f"annotations: {annotations}",
        f"images: {images}",
        f"num_images: {image_count}",
        f"fp_model: {fp_model}",
        f"int_model: {int_model}",
        f"fp_map50: {fp_map50:.6f}",
        f"int_map50: {int_map50:.6f}",
        f"abs_delta_int_minus_fp: {abs_delta:+.6f}",
        f"pct_delta_vs_fp: {pct_delta_str}",
        f"trend: {trend}",
    ]
    return "\n".join(lines) + "\n"


def _validate_pair_eval_inputs(
    model_type: str,
    fp_head: str,
    ann_path: Path,
    img_dir: Path,
    fp_model_path: Path,
    int_model_path: Path,
    remote_runner_local_path: Path,
) -> str:
    if model_type not in ("yolov10", "yolov11", "yolov26"):
        raise ValueError(f"Unsupported model_type: {model_type}")
    if model_type == "yolov11":
        return "default"
    if fp_head not in ("one2many", "one2one"):
        raise ValueError("fp_head must be one of: one2many, one2one")
    for path in (ann_path, img_dir, fp_model_path, int_model_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not remote_runner_local_path.exists():
        raise FileNotFoundError(f"Remote runner not found: {remote_runner_local_path}")
    return fp_head


def _build_model_cfgs(
    model_type: str,
    fp_head: str,
    fp_model_path: Path,
    int_model_path: Path,
) -> tuple[dict, dict]:
    decoder = model_decoder_for_type(model_type)
    family = model_type.replace("v", "")
    fp_cfg = {
        "backend": "pt",
        "family": family,
        "head": fp_head_to_cfg_head(model_type, fp_head),
        "decoder": decoder,
        "path": str(fp_model_path),
    }
    int_cfg = {
        "backend": "tflite",
        "family": family,
        "head": "default",
        "quant": "int8",
        "decoder": decoder,
        "path": str(int_model_path),
    }
    return fp_cfg, int_cfg


def _load_tflite_input_detail(int_model_path: Path) -> dict:
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter

    interp = Interpreter(model_path=str(int_model_path))
    interp.allocate_tensors()
    return interp.get_input_details()[0]


def _compute_delta_summary(
    fp_map50: float, int_map50: float
) -> tuple[float, float | None, str]:
    abs_delta = int_map50 - fp_map50
    if abs_delta > 0:
        trend = "increase"
    elif abs_delta < 0:
        trend = "decrease"
    else:
        trend = "no change"

    pct_delta = None
    if fp_map50 != 0.0:
        pct_delta = (abs_delta / fp_map50) * 100.0
    return abs_delta, pct_delta, trend


def run_fp_int_pair_map_eval(
    model_type: str,
    fp_model: str,
    int_model: str,
    annotations: str,
    images: str,
    output_text: str,
    conf: float = 0.25,
    nms: float = 0.7,
    max_det: int = 300,
    max_images: int | None = None,
    fp_head: str = "one2many",
    adb_serial: str | None = None,
    remote_workdir: str = "/data/local/tmp/yolo_map_eval",
    remote_runner_local: str = DEFAULT_REMOTE_RUNNER_LOCAL,
    remote_runner_remote: str = (
        "/data/local/tmp/yolo_map_eval/remote_tflite_raw_runner.py"
    ),
    qnn_lib: str = "/usr/lib/libQnnTFLiteDelegate.so",
    backend: str = "htp",
    no_qnn: bool = False,
) -> dict:
    ann_path = Path(annotations)
    img_dir = Path(images)
    fp_model_path = Path(fp_model)
    int_model_path = Path(int_model)
    remote_runner_local_path = Path(remote_runner_local)
    fp_head = _validate_pair_eval_inputs(
        model_type=model_type,
        fp_head=fp_head,
        ann_path=ann_path,
        img_dir=img_dir,
        fp_model_path=fp_model_path,
        int_model_path=int_model_path,
        remote_runner_local_path=remote_runner_local_path,
    )
    resolved_annotations = resolve_annotations_for_map(
        ann_path, img_dir, fp_model_path
    )

    local_target_requirements = str(
        Path(__file__).resolve().parent.parent / "requirements" / "target.txt"
    )
    remote_python = ensure_adb_runtime_venv(
        adb_serial=adb_serial,
        local_requirements_path=local_target_requirements,
    )
    remote_layout = make_map_remote_run_layout(
        remote_workdir=remote_workdir,
        int_model_path=int_model_path,
        remote_runner_remote=remote_runner_remote,
    )

    runtime_args = make_runtime_args(
        adb_serial=adb_serial,
        remote_workdir=remote_layout["remote_run_dir"],
        remote_runner_remote=remote_layout["remote_runner"],
        remote_python=remote_python,
        qnn_lib=qnn_lib,
        backend=backend,
        no_qnn=no_qnn,
    )

    adb_shell(adb_serial, f"mkdir -p {shlex.quote(remote_layout['remote_run_dir'])}")
    adb_push(
        adb_serial,
        str(remote_runner_local_path),
        remote_layout["remote_runner"],
    )
    try:
        eval_ann_path = resolved_annotations.eval_ann_path
        imgs = load_coco_images(eval_ann_path, img_dir)
        if max_images is not None:
            imgs = imgs[:max_images]
        if not imgs:
            raise ValueError(
                "No images selected for evaluation. "
                "Check --images path and --max-images value."
            )

        cat_ids = coco_category_ids(eval_ann_path)
        validate_eval_class_count_compatibility(
            fp_model_path=fp_model_path,
            int_model_path=int_model_path,
            cat_ids=cat_ids,
        )
        fp_cfg, int_cfg = _build_model_cfgs(
            model_type=model_type,
            fp_head=fp_head,
            fp_model_path=fp_model_path,
            int_model_path=int_model_path,
        )
        input_detail = _load_tflite_input_detail(int_model_path)

        shared_inputs_obj, shared_meta = prepare_shared_int8_inputs(
            images=imgs,
            input_detail=input_detail,
            adb_serial=adb_serial,
            remote_input_dir=remote_layout["remote_input_dir"],
        )
        shared_int8 = {
            "meta": shared_meta,
            "remote_input_dir": remote_layout["remote_input_dir"],
        }

        try:
            fp_row = evaluate_one_model(
                model_key=f"{model_type}_fp",
                cfg=fp_cfg,
                images=imgs,
                cat_ids=cat_ids,
                ann_path=eval_ann_path,
                outdir=None,
                conf_thr=conf,
                nms_thr=nms,
                max_det=max_det,
                args=runtime_args,
                shared_int8=None,
                save_pred_json=False,
            )
            int_row = evaluate_one_model(
                model_key=f"{model_type}_int",
                cfg=int_cfg,
                images=imgs,
                cat_ids=cat_ids,
                ann_path=eval_ann_path,
                outdir=None,
                conf_thr=conf,
                nms_thr=nms,
                max_det=max_det,
                args=runtime_args,
                shared_int8=shared_int8,
                save_pred_json=False,
            )
        finally:
            shared_inputs_obj.cleanup()
    finally:
        resolved_annotations.cleanup()
        try:
            adb_shell(
                adb_serial,
                f"rm -rf {shlex.quote(remote_layout['remote_run_dir'])}",
            )
        except subprocess.CalledProcessError as cleanup_exc:
            print(f"[warn] remote cleanup failed: {cleanup_exc}")

    fp_map50 = float(fp_row["map50"])
    int_map50 = float(int_row["map50"])
    abs_delta, pct_delta, trend = _compute_delta_summary(fp_map50, int_map50)

    report_text = make_report_text(
        model_type=model_type,
        fp_head=fp_head,
        annotations=ann_path,
        images=img_dir,
        image_count=len(imgs),
        fp_model=str(fp_model_path),
        int_model=str(int_model_path),
        fp_map50=fp_map50,
        int_map50=int_map50,
        abs_delta=abs_delta,
        pct_delta=pct_delta,
        trend=trend,
    )

    print(report_text, end="")
    output_path = Path(output_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text)

    return {
        "model_type": model_type,
        "fp_head": fp_head,
        "fp_map50": fp_map50,
        "int_map50": int_map50,
        "abs_delta": abs_delta,
        "pct_delta": pct_delta,
        "trend": trend,
        "output_text": str(output_path),
    }
