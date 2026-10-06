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
"""Check mAP annotations/images against what iQ-Foundry's mAP mode accepts, and reformat them.

    prepare_map_data.py inspect --annotations A --images I --model M.pt
    prepare_map_data.py convert --annotations A --images I --model M.pt [--out DIR]
    prepare_map_data.py class-yaml --model M.pt [--out FILE.yaml]

`inspect` changes nothing. It prints VERDICT: READY (use the data as it is) or
VERDICT: NEEDS_REFORMAT (with the reasons), or VERDICT: NEEDS_INPUT (a question for the user,
e.g. unknown class names or several dataset splits).

`class-yaml` writes the class-names YAML that test mode needs (--yaml), taken from the model
itself, for users who do not have one.

`convert` writes a NEW folder (default <repo>/out/iqf-assistant/map_data/<timestamp>/) with
annotations_coco.json, conversion_report.md and, only when some images must be re-encoded,
an images/ copy. The user's files are never modified. The result is checked with the same
loaders mAP mode uses (tool/test_map.py) before it is reported as usable.

Accepted inputs: COCO json, YOLO txt (incl. the Ultralytics images/ + labels/ + data.yaml layout,
segmentation polygons and a trailing confidence column), Pascal VOC xml, LabelMe json and
CVAT-for-images xml. Images may be in nested folders; .webp/.tif/.tiff/.jfif are converted to .png.

On the host only python3 is needed: the work runs inside the iqf Docker image (PIL, ultralytics),
started with the user's uid so new files belong to the user. Inputs are mounted read-only.

Exit codes: 0 READY / CONVERTED, 3 NEEDS_REFORMAT, 4 NEEDS_INPUT, 1 error / unsupported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

EXIT_OK, EXIT_ERROR, EXIT_REFORMAT, EXIT_INPUT = 0, 1, 3, 4

SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")  # tool/test_map.py:71
CONVERTIBLE_IMAGE_SUFFIXES = (".webp", ".tif", ".tiff", ".jfif")
ALL_IMAGE_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES + CONVERTIBLE_IMAGE_SUFFIXES
NAMES_FILE_CANDIDATES = (
    "data.yaml",
    "dataset.yaml",
    "data.yml",
    "classes.txt",
    "obj.names",
    "classes.names",
)
LABEL_DIR_NAMES = {"labels", "annotations", "label", "ann", "anns"}
IMAGE_DIR_NAMES = {"images", "jpegimages", "imgs", "image"}
SPLIT_NAMES = ("train", "val", "valid", "validation", "test")
NON_LABEL_TXT = {
    "classes.txt",
    "readme.txt",
    "obj.names",
    "train.txt",
    "val.txt",
    "test.txt",
}


class UserInputNeeded(Exception):
    """The data can be converted, but only after the user answers a question."""


class Unsupported(Exception):
    """The annotation format is not one this tool understands."""


def default_repo_root() -> Path:
    # <repo>/.agents/skills/iqf-assistant/scripts/prepare_map_data.py
    return Path(__file__).resolve().parents[4]


# ----------------------------
# Small helpers
# ----------------------------


def norm_name(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(name).strip().casefold())


def match_key(rel: Path, strip_dirs: set[str]) -> str:
    """Key used to pair a label file with an image: relative path without the extension and
    without folder names like labels/ or images/, so labels/val/a.txt pairs with images/val/a.jpg."""
    parts = list(rel.with_suffix("").parts)
    kept = [p for p in parts[:-1] if p.casefold() not in strip_dirs]
    return "/".join(kept + [parts[-1]]).casefold()


def load_names_file(path: Path) -> list[str]:
    """Class names from a YOLO data.yaml (names: list or {0: a, 1: b}) or a classes.txt."""
    text = path.read_text(errors="replace")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # available in the iqf image

            data = yaml.safe_load(text) or {}
        except ImportError:
            data = {"names": _parse_simple_yaml_names(text)}
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            ordered = sorted((int(k), str(v)) for k, v in names.items())
            if [k for k, _ in ordered] != list(range(len(ordered))):
                raise ValueError(f"class ids in {path} must start at 0 with no gaps")
            return [v for _, v in ordered]
        if isinstance(names, list):
            return [str(n) for n in names]
        raise ValueError(f"{path} has no 'names:' list")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_simple_yaml_names(text: str) -> list[str] | dict[int, str]:
    inline = re.search(r"^names:\s*\[(.*)\]\s*$", text, re.MULTILINE)
    if inline:
        return [p.strip().strip("'\"") for p in inline.group(1).split(",") if p.strip()]
    out: dict[int, str] = {}
    listed: list[str] = []
    block = re.search(r"^names:\s*$((?:\n[ \t]+.*)*)", text, re.MULTILINE)
    for line in (block.group(1) if block else "").splitlines():
        line = line.strip()
        m = re.match(r"^(\d+)\s*:\s*(.+)$", line)
        if m:
            out[int(m.group(1))] = m.group(2).strip().strip("'\"")
        elif line.startswith("- "):
            listed.append(line[2:].strip().strip("'\""))
    return out or listed


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


# ----------------------------
# Data model
# ----------------------------


@dataclass
class ImageEntry:
    path: Path  # absolute
    rel: Path  # relative to the images root
    width: int = 0
    height: int = 0


@dataclass
class Box:
    label: object  # str (class name) or int (YOLO class id)
    x1: float
    y1: float
    x2: float
    y2: float
    normalized: bool = False  # YOLO: coordinates are 0-1 of the image size


@dataclass
class Dataset:
    format_name: str
    annotations: Path
    images_root: Path
    images: list[ImageEntry] = field(default_factory=list)
    boxes: dict[str, list[Box]] = field(
        default_factory=dict
    )  # image rel (posix) -> boxes
    dataset_names: list[str] | None = None  # YOLO class order, if known
    names_source: Path | None = None
    coco: dict | None = None  # original COCO json (format coco)
    notes: list[str] = field(
        default_factory=list
    )  # what is different from mAP's format
    warnings: list[str] = field(default_factory=list)
    unmatched_labels: list[str] = field(default_factory=list)
    skipped_shapes: int = 0
    nested_images: bool = False
    convert_images: bool = False


# ----------------------------
# Scanning
# ----------------------------


def collect_images(images_root: Path, split: str | None) -> list[ImageEntry]:
    entries = []
    for p in sorted(images_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in ALL_IMAGE_SUFFIXES:
            continue
        rel = p.relative_to(images_root)
        if split and split.casefold() not in (
            part.casefold() for part in rel.parts[:-1]
        ):
            continue
        entries.append(ImageEntry(path=p, rel=rel))
    return entries


def detect_splits(paths: list[Path]) -> list[str]:
    found = set()
    for rel in paths:
        for part in rel.parts[:-1]:
            if part.casefold() in SPLIT_NAMES:
                found.add(part)
    return sorted(found)


def find_names_file(annotations: Path, images_root: Path) -> Path | None:
    roots = []
    for base in (
        annotations if annotations.is_dir() else annotations.parent,
        images_root,
    ):
        roots += [base, base.parent, base.parent.parent]
    for root in roots:
        for name in NAMES_FILE_CANDIDATES:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def _xml_root_tag(path: Path) -> str:
    try:
        return ET.parse(path).getroot().tag
    except ET.ParseError:
        return ""


def _json_kind(path: Path) -> str:
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (ValueError, OSError):
        return ""
    if isinstance(data, dict) and {"images", "annotations"} <= data.keys():
        return "coco"
    if isinstance(data, dict) and "shapes" in data:
        return "labelme"
    return ""


def detect_format(annotations: Path) -> tuple[str, Path]:
    """Return (format, path). For single-file formats the path is that file."""
    if annotations.is_file():
        suffix = annotations.suffix.lower()
        if suffix == ".json":
            kind = _json_kind(annotations)
            if kind:
                return kind, annotations
            raise Unsupported(
                f"{annotations.name} is a .json file but not COCO or LabelMe"
            )
        if suffix == ".xml":
            tag = _xml_root_tag(annotations)
            if tag == "annotations":
                return "cvat", annotations
            if tag == "annotation":
                return "voc", annotations.parent
        raise Unsupported(
            f"{annotations.name}: unsupported annotation file. Use COCO .json, CVAT .xml, "
            "or a folder of YOLO .txt / VOC .xml / LabelMe .json files"
        )
    if not annotations.is_dir():
        raise FileNotFoundError(annotations)

    files = [p for p in annotations.rglob("*") if p.is_file()]
    jsons = [p for p in files if p.suffix.lower() == ".json"]
    xmls = [p for p in files if p.suffix.lower() == ".xml"]
    txts = [
        p
        for p in files
        if p.suffix.lower() == ".txt" and p.name.casefold() not in NON_LABEL_TXT
    ]

    coco_files = [p for p in jsons if _json_kind(p) == "coco"]
    if coco_files:
        if len(coco_files) > 1:
            names = ", ".join(str(p.relative_to(annotations)) for p in coco_files[:5])
            raise UserInputNeeded(
                f"several COCO files found ({names}); which one should be used?"
            )
        return "coco", coco_files[0]
    counts = {
        "labelme": sum(1 for p in jsons[:50] if _json_kind(p) == "labelme"),
        "voc": sum(1 for p in xmls[:50] if _xml_root_tag(p) == "annotation"),
        "yolo": len(txts),
    }
    cvat = [p for p in xmls if _xml_root_tag(p) == "annotations"]
    if cvat and not counts["voc"]:
        return "cvat", cvat[0]
    best = max(counts, key=lambda k: counts[k])
    if counts[best] == 0:
        raise Unsupported(
            f"no annotation files found in {annotations}. Expected COCO .json, YOLO .txt, "
            "VOC .xml, LabelMe .json or CVAT .xml"
        )
    return best, annotations


def _label_files(root: Path, suffix: str, split: str | None) -> list[Path]:
    out = []
    for p in sorted(root.rglob(f"*{suffix}")):
        if not p.is_file() or p.name.casefold() in NON_LABEL_TXT:
            continue
        rel = p.relative_to(root)
        if split and split.casefold() not in (
            part.casefold() for part in rel.parts[:-1]
        ):
            continue
        out.append(p)
    return out


class ImageIndex:
    def __init__(self, images: list[ImageEntry]):
        self.by_key: dict[str, ImageEntry] = {}
        self.by_stem: dict[str, list[ImageEntry]] = {}
        self.by_name: dict[str, list[ImageEntry]] = {}
        for im in images:
            self.by_key[match_key(im.rel, IMAGE_DIR_NAMES)] = im
            self.by_stem.setdefault(im.rel.stem.casefold(), []).append(im)
            self.by_name.setdefault(im.rel.name.casefold(), []).append(im)

    def find(self, label_rel: Path, hint_name: str | None = None) -> ImageEntry | None:
        if hint_name:
            hits = self.by_name.get(
                Path(hint_name.replace("\\", "/")).name.casefold(), []
            )
            if len(hits) == 1:
                return hits[0]
        hit = self.by_key.get(match_key(label_rel, LABEL_DIR_NAMES))
        if hit:
            return hit
        hits = self.by_stem.get(label_rel.stem.casefold(), [])
        return hits[0] if len(hits) == 1 else None


def _parse_yolo_file(path: Path) -> tuple[list[Box], list[str], int]:
    boxes, notes, skipped = [], [], 0
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            skipped += 1
            continue
        if len(values) == 4:
            cx, cy, w, h = values
        elif len(values) == 5:  # trailing confidence column (prediction export)
            cx, cy, w, h = values[:4]
            notes.append("confidence column")
        elif len(values) >= 6 and len(values) % 2 == 0:  # segmentation polygon
            xs, ys = values[0::2], values[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            notes.append("polygon")
        else:
            skipped += 1
            continue
        boxes.append(
            Box(cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, normalized=True)
        )
    return boxes, notes, skipped


def _parse_voc_file(path: Path) -> tuple[list[Box], str | None]:
    root = ET.parse(path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if not name or bb is None:
            continue
        try:
            coords = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        boxes.append(Box(name, *coords))
    return boxes, root.findtext("filename")


def _parse_labelme_file(path: Path) -> tuple[list[Box], str | None, int]:
    data = json.loads(path.read_text(errors="replace"))
    boxes, skipped = [], 0
    for shape in data.get("shapes", []):
        points = shape.get("points") or []
        kind = shape.get("shape_type") or "polygon"
        label = str(shape.get("label", "")).strip()
        if kind in ("rectangle", "polygon") and len(points) >= 2 and label:
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            boxes.append(Box(label, min(xs), min(ys), max(xs), max(ys)))
        else:
            skipped += 1
    return boxes, data.get("imagePath"), skipped


def scan(
    annotations: Path,
    images_root: Path,
    names_file: Path | None = None,
    split: str | None = None,
) -> Dataset:
    fmt, ann_path = detect_format(annotations)
    all_images = collect_images(images_root, None)
    if not all_images:
        raise ValueError(f"no images found in {images_root}")
    splits = detect_splits([im.rel for im in all_images])
    if len(splits) > 1 and not split and fmt != "coco":
        raise UserInputNeeded(
            f"the images folder has several dataset parts ({', '.join(splits)}); which one should be "
            "used for the accuracy check? (usually 'val')"
        )
    images = collect_images(images_root, split) if split else all_images
    if not images:
        raise ValueError(f"no images found in {images_root} for split '{split}'")

    ds = Dataset(
        format_name=fmt, annotations=ann_path, images_root=images_root, images=images
    )
    ds.nested_images = any(len(im.rel.parts) > 1 for im in images)
    # COCO images are opened by name with PIL, so any readable format works there. The YOLO/VOC
    # loader only lists .jpg/.jpeg/.png/.bmp, so other formats are re-encoded for those.
    ds.convert_images = fmt != "coco" and any(
        im.path.suffix.lower() in CONVERTIBLE_IMAGE_SUFFIXES for im in images
    )
    if ds.convert_images:
        n = sum(
            1 for im in images if im.path.suffix.lower() in CONVERTIBLE_IMAGE_SUFFIXES
        )
        ds.notes.append(
            f"{n} image(s) are .webp/.tif/.jfif, which mAP mode does not read; they are converted to .png copies"
        )
    index = ImageIndex(images)

    if fmt == "coco":
        ds.coco = json.loads(ann_path.read_text(errors="replace"))
        return ds

    if fmt == "yolo":
        names_file = names_file or find_names_file(annotations, images_root)
        if names_file:
            ds.dataset_names = load_names_file(names_file)
            ds.names_source = names_file
        label_files = _label_files(ann_path, ".txt", split)
        styles: set[str] = set()
        for lf in label_files:
            boxes, notes, skipped = _parse_yolo_file(lf)
            styles.update(notes)
            ds.skipped_shapes += skipped
            _attach(ds, index, lf.relative_to(ann_path), boxes, None)
        if "polygon" in styles:
            ds.notes.append(
                "some YOLO labels are segmentation polygons (outlines); they are turned into boxes"
            )
        if "confidence column" in styles:
            ds.notes.append(
                "some YOLO labels have a 6th (confidence) column; it is removed"
            )
    elif fmt == "voc":
        for lf in _label_files(ann_path, ".xml", split):
            boxes, hint = _parse_voc_file(lf)
            _attach(ds, index, lf.relative_to(ann_path), boxes, hint)
    elif fmt == "labelme":
        for lf in _label_files(ann_path, ".json", split):
            boxes, hint, skipped = _parse_labelme_file(lf)
            ds.skipped_shapes += skipped
            _attach(ds, index, lf.relative_to(ann_path), boxes, hint)
        ds.notes.append(
            "labels are in LabelMe format, which mAP mode does not read; they are rewritten as one COCO file"
        )
    elif fmt == "cvat":
        root = ET.parse(ann_path).getroot()
        for node in root.iter("image"):
            name = node.get("name") or ""
            boxes = []
            for box in node.findall("box"):
                coords = [float(box.get(k, 0)) for k in ("xtl", "ytl", "xbr", "ybr")]
                boxes.append(Box(box.get("label", ""), *coords))
            for poly in node.findall("polygon"):
                pts = [
                    tuple(map(float, p.split(",")))
                    for p in (poly.get("points") or "").split(";")
                    if p
                ]
                if pts:
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    boxes.append(
                        Box(poly.get("label", ""), min(xs), min(ys), max(xs), max(ys))
                    )
            ds.skipped_shapes += len(node.findall("points")) + len(
                node.findall("polyline")
            )
            _attach(ds, index, Path(name), boxes, name)
        ds.notes.append(
            "labels are in CVAT format, which mAP mode does not read; they are rewritten as one COCO file"
        )

    if ds.skipped_shapes:
        ds.warnings.append(
            f"{ds.skipped_shapes} label(s) could not be used (not a box/polygon, or unreadable) and were skipped"
        )
    return ds


def _attach(
    ds: Dataset, index: ImageIndex, label_rel: Path, boxes: list[Box], hint: str | None
) -> None:
    im = index.find(label_rel, hint)
    if im is None:
        ds.unmatched_labels.append(str(label_rel))
        return
    ds.boxes.setdefault(im.rel.as_posix(), []).extend(boxes)


# ----------------------------
# Class mapping
# ----------------------------


def parse_class_map(text: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in (text or "").split(","):
        if "=" in item:
            src, dst = item.split("=", 1)
            out[src.strip()] = dst.strip()
    return out


def resolve_class_mapping(
    dataset_labels: list[str],
    model_names: list[str],
    user_map: dict[str, str],
    drop_unknown: bool,
) -> tuple[dict[str, int | None], list[str], list[str]]:
    """Map every dataset class name to a model class index (None = drop).

    Returns (mapping, notes, unknown). Exact names map directly; names that differ only in
    case, spaces, '_' or '-' are mapped and noted; user_map entries (value 'drop' drops a class)
    win over both. Anything left is unknown unless drop_unknown is set."""
    exact = {name: i for i, name in enumerate(model_names)}
    loose: dict[str, int] = {}
    for i, name in enumerate(model_names):
        loose.setdefault(norm_name(name), i)
    user_norm = {norm_name(k): v for k, v in user_map.items()}

    mapping: dict[str, int | None] = {}
    notes, unknown, respelled = [], [], []
    for label in dataset_labels:
        target = user_map.get(label, user_norm.get(norm_name(label)))
        if target is not None:
            if target.casefold() == "drop":
                mapping[label] = None
                notes.append(f"class '{label}' is left out (as you asked)")
                continue
            idx = exact.get(target, loose.get(norm_name(target)))
            if idx is None:
                raise ValueError(
                    f"--class-map target '{target}' is not a class of the model"
                )
            mapping[label] = idx
            notes.append(
                f"class '{label}' is counted as model class '{model_names[idx]}' (as you asked)"
            )
        elif label in exact:
            mapping[label] = exact[label]
        elif norm_name(label) in loose:
            idx = loose[norm_name(label)]
            mapping[label] = idx
            respelled.append(f"'{label}'->'{model_names[idx]}'")
        elif drop_unknown:
            mapping[label] = None
            notes.append(f"class '{label}' is not a model class and is left out")
        else:
            unknown.append(label)
    if respelled:
        shown = ", ".join(respelled[:6]) + (", ..." if len(respelled) > 6 else "")
        notes.insert(
            0,
            f"{len(respelled)} class name(s) differ from the model's only in capitals, spaces, "
            f"'_' or '-' and are matched to the model's names ({shown})",
        )
    return mapping, notes, unknown


def suggestions_for(label: str, model_names: list[str]) -> list[str]:
    import difflib

    return difflib.get_close_matches(label, model_names, n=3, cutoff=0.5)


# ----------------------------
# Building the COCO file
# ----------------------------


@dataclass
class BuildResult:
    coco: dict
    images_root: Path  # root that file_name values are relative to
    copy_plan: list[
        tuple[Path, Path]
    ]  # (source, destination) when images must be copied
    notes: list[str]
    warnings: list[str]
    stats: dict


def _dataset_labels(ds: Dataset) -> list[str]:
    if ds.format_name == "coco":
        return [str(c.get("name", "")) for c in ds.coco.get("categories", [])]
    if ds.format_name == "yolo":
        return list(ds.dataset_names or [])
    seen: dict[str, None] = {}
    for boxes in ds.boxes.values():
        for b in boxes:
            seen.setdefault(str(b.label), None)
    return list(seen)


def plan_classes(
    ds: Dataset, model_names: list[str], user_map: dict[str, str], drop_unknown: bool
):
    """Return (label -> model index | None) or raise UserInputNeeded for unknown classes."""
    if ds.format_name == "yolo" and not ds.dataset_names:
        # No names file: YOLO ids are taken in the model's own class order, as mAP mode does.
        return None, []
    labels = _dataset_labels(ds)
    mapping, notes, unknown = resolve_class_mapping(
        labels, model_names, user_map, drop_unknown
    )
    if unknown:
        lines = []
        for label in unknown:
            close = suggestions_for(label, model_names)
            hint = f" (did you mean: {', '.join(close)}?)" if close else ""
            lines.append(f"'{label}'{hint}")
        raise UserInputNeeded(
            "these dataset classes are not classes of the model: "
            + "; ".join(lines)
            + '. Map them with --class-map "name=model_name" or leave them out with '
            "--drop-unknown-classes"
        )
    if ds.format_name == "yolo" and ds.dataset_names != list(model_names):
        notes.insert(
            0,
            f"the dataset class order ({ds.names_source.name}) differs from the model's; class ids are re-numbered by name",
        )
    return mapping, notes


def build_coco(
    ds: Dataset,
    model_names: list[str],
    user_map: dict[str, str] | None = None,
    drop_unknown: bool = False,
    out_images_dir: Path | None = None,
) -> BuildResult:
    mapping, class_notes = plan_classes(ds, model_names, user_map or {}, drop_unknown)
    notes = list(ds.notes) + class_notes
    warnings = list(ds.warnings)
    categories = [
        {"id": i + 1, "name": n, "supercategory": "none"}
        for i, n in enumerate(model_names)
    ]

    if ds.format_name == "coco":
        return _rebuild_coco(ds, model_names, mapping, notes, warnings, categories)

    copy_plan: list[tuple[Path, Path]] = []
    images_root = ds.images_root
    if ds.convert_images:
        images_root = out_images_dir or ds.images_root
    coco_images, coco_anns = [], []
    dropped_boxes = bad_class = 0
    for image_id, im in enumerate(
        sorted(ds.images, key=lambda e: e.rel.as_posix()), start=1
    ):
        if not im.width:
            im.width, im.height = image_size(im.path)
        file_name = im.rel.as_posix()
        if ds.convert_images:
            target_rel = (
                im.rel.with_suffix(".png")
                if im.path.suffix.lower() in CONVERTIBLE_IMAGE_SUFFIXES
                else im.rel
            )
            file_name = target_rel.as_posix()
            copy_plan.append((im.path, images_root / target_rel))
        coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": im.width,
                "height": im.height,
            }
        )
        for b in ds.boxes.get(im.rel.as_posix(), []):
            if isinstance(b.label, int):  # YOLO class id
                names = model_names if mapping is None else ds.dataset_names
                if not 0 <= b.label < len(names):
                    bad_class += 1
                    continue
                cls = b.label if mapping is None else mapping.get(names[b.label])
            else:
                cls = (mapping or {}).get(str(b.label))
            if cls is None:  # class left out
                dropped_boxes += 1
                continue
            x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
            if b.normalized:
                x1, x2, y1, y2 = (
                    x1 * im.width,
                    x2 * im.width,
                    y1 * im.height,
                    y2 * im.height,
                )
            x1, x2 = sorted((min(max(x1, 0.0), im.width), min(max(x2, 0.0), im.width)))
            y1, y2 = sorted(
                (min(max(y1, 0.0), im.height), min(max(y2, 0.0), im.height))
            )
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                dropped_boxes += 1
                continue
            coco_anns.append(
                {
                    "id": len(coco_anns) + 1,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
                    "area": round(w * h, 3),
                    "iscrowd": 0,
                }
            )

    if bad_class:
        raise UserInputNeeded(
            f"{bad_class} YOLO label(s) use a class number the model does not have "
            f"(the model has {len(model_names)} classes, numbered 0-{len(model_names) - 1}). "
            "Is there a data.yaml or classes.txt with the dataset's class names? Pass it with --dataset-names"
        )
    if dropped_boxes:
        warnings.append(
            f"{dropped_boxes} box(es) were left out (class left out, or zero size after clipping to the image)"
        )
    labelled = sum(1 for im in ds.images if ds.boxes.get(im.rel.as_posix()))
    if ds.unmatched_labels:
        warnings.append(
            f"{len(ds.unmatched_labels)} label file(s) have no matching image and were ignored "
            f"(e.g. {', '.join(ds.unmatched_labels[:3])})"
        )
    unlabelled = len(ds.images) - labelled
    if unlabelled:
        warnings.append(
            f"{unlabelled} image(s) have no labels; they are evaluated as images with no objects, "
            "exactly as mAP mode does"
        )
    stats = {
        "images": len(coco_images),
        "labelled_images": labelled,
        "boxes": len(coco_anns),
    }
    coco = {
        "info": {"description": "made by iqf-assistant prepare_map_data.py"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_anns,
        "categories": categories,
    }
    return BuildResult(coco, images_root, copy_plan, notes, warnings, stats)


def _rebuild_coco(ds, model_names, mapping, notes, warnings, categories) -> BuildResult:
    src = ds.coco
    index = ImageIndex(ds.images)
    by_rel = {im.rel.as_posix().casefold(): im for im in ds.images}
    cat_name = {c.get("id"): str(c.get("name", "")) for c in src.get("categories", [])}
    keep_images: dict[object, tuple[ImageEntry, int, int]] = {}
    missing = fixed_names = filled_sizes = 0
    for img in src.get("images", []):
        file_name = str(img.get("file_name", ""))
        entry = by_rel.get(file_name.replace("\\", "/").casefold())
        if entry is None:
            entry = index.find(Path(file_name), file_name)
            if entry is not None:
                fixed_names += 1
        if entry is None:
            missing += 1
            continue
        width, height = img.get("width"), img.get("height")
        if not width or not height:
            width, height = image_size(entry.path)
            filled_sizes += 1
        keep_images[img.get("id")] = (entry, int(width), int(height))

    if missing:
        warnings.append(
            f"{missing} image(s) listed in the COCO file were not found in the images folder and were left out"
        )
    if fixed_names:
        notes.append(
            f"{fixed_names} image path(s) in the COCO file did not match the images folder; they were matched by file name"
        )
    if filled_sizes:
        notes.append(
            f"{filled_sizes} image(s) had no width/height in the COCO file; they were read from the images"
        )
    if not keep_images:
        raise ValueError(
            "none of the images listed in the COCO file were found in the images folder"
        )

    # Keep the original image order: mAP mode's --max-images takes the first N images.
    new_ids, coco_images = {}, []
    for new_id, (old_id, (entry, w, h)) in enumerate(keep_images.items(), start=1):
        new_ids[old_id] = new_id
        coco_images.append(
            {"id": new_id, "file_name": entry.rel.as_posix(), "width": w, "height": h}
        )

    anns, dropped = [], 0
    for ann in src.get("annotations", []):
        if ann.get("image_id") not in new_ids:
            continue
        cls = mapping.get(cat_name.get(ann.get("category_id"), "")) if mapping else None
        bbox = ann.get("bbox") or []
        if cls is None or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            dropped += 1
            continue
        anns.append(
            {
                "id": len(anns) + 1,
                "image_id": new_ids[ann["image_id"]],
                "category_id": cls + 1,
                "bbox": [float(v) for v in bbox],
                "area": float(ann.get("area") or bbox[2] * bbox[3]),
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
        )
    if dropped:
        warnings.append(
            f"{dropped} box(es) were left out (class left out, or an empty box)"
        )
    stats = {
        "images": len(coco_images),
        "labelled_images": len({a["image_id"] for a in anns}),
        "boxes": len(anns),
    }
    coco = {
        "info": src.get("info", {}),
        "licenses": src.get("licenses", []),
        "images": coco_images,
        "annotations": anns,
        "categories": categories,
    }
    return BuildResult(coco, ds.images_root, [], notes, warnings, stats)


# ----------------------------
# Checks against iQ-Foundry's own loaders
# ----------------------------


def load_model_names(model: Path) -> list[str]:
    from tool.test_map import load_fp_model_class_names

    return load_fp_model_class_names(model)


def official_check(annotations: Path, images: Path, model: Path) -> None:
    """Run the same loading steps mAP mode runs before evaluation (tool/test_map.py)."""
    from tool.test_map import (
        build_model_class_to_coco_category_id_map,
        load_coco_images,
        load_fp_model_class_names,
        resolve_annotations_for_map,
    )

    resolved = resolve_annotations_for_map(annotations, images, model)
    try:
        if not load_coco_images(resolved.eval_ann_path, images):
            raise ValueError("no images selected for evaluation")
        build_model_class_to_coco_category_id_map(
            resolved.eval_ann_path, load_fp_model_class_names(model)
        )
    finally:
        resolved.cleanup()


def silent_problems(ds: Dataset, model_names: list[str]) -> list[str]:
    """Problems mAP mode would NOT report, but that make the result wrong or incomplete."""
    problems = []
    if ds.format_name in ("yolo", "voc") and ds.nested_images:
        problems.append(
            "images are in sub-folders; mAP mode only reads images directly inside the images folder"
        )
    if ds.format_name in ("yolo", "voc") and ds.annotations.is_dir():
        nested = [
            p
            for p in ds.annotations.rglob(
                "*.txt" if ds.format_name == "yolo" else "*.xml"
            )
            if p.parent != ds.annotations
        ]
        if nested:
            problems.append(
                "label files are in sub-folders; mAP mode only reads labels directly inside the labels folder"
            )
    if (
        ds.format_name == "yolo"
        and ds.dataset_names
        and ds.dataset_names != list(model_names)
    ):
        problems.append(
            f"the dataset class list ({ds.names_source.name if ds.names_source else 'names file'}) is in a different "
            "order than the model's classes; mAP mode would read class numbers in the model's order and give a wrong score"
        )
    if ds.unmatched_labels:
        problems.append(
            f"{len(ds.unmatched_labels)} label file(s) do not match any image name"
        )
    if ds.convert_images:
        problems.append("some images are .webp/.tif/.jfif, which mAP mode skips")
    return problems


# ----------------------------
# Commands (run inside the iqf image)
# ----------------------------


def _print_kv(key: str, value: object) -> None:
    print(f"{key}: {value}")


def inner_inspect(args) -> int:
    ann, images, model = Path(args.annotations), Path(args.images), Path(args.model)
    model_names = load_model_names(model)
    _print_kv(
        "MODEL_CLASSES",
        f"{len(model_names)} ({', '.join(model_names[:8])}{', ...' if len(model_names) > 8 else ''})",
    )
    try:
        ds = scan(
            ann,
            images,
            Path(args.dataset_names) if args.dataset_names else None,
            args.split,
        )
    except UserInputNeeded as exc:
        _print_kv("VERDICT", "NEEDS_INPUT")
        _print_kv("QUESTION", exc)
        return EXIT_INPUT
    _print_kv("FORMAT", ds.format_name)
    _print_kv(
        "IMAGES",
        f"{len(ds.images)} found{' (in sub-folders)' if ds.nested_images else ''}",
    )
    if ds.names_source:
        _print_kv("DATASET_CLASSES", f"{len(ds.dataset_names)} from {ds.names_source}")

    problems = silent_problems(ds, model_names)
    official_error = None
    # A COCO file found inside a folder is used directly (mAP mode needs the file itself).
    check_ann = ds.annotations if ds.format_name == "coco" else ann
    if ds.format_name in ("coco", "yolo", "voc"):
        try:
            official_check(check_ann, images, model)
        except Exception as exc:  # any loader error means mAP mode would stop
            official_error = f"{type(exc).__name__}: {exc}"
    else:
        problems.insert(0, f"{ds.format_name} labels are not a format mAP mode reads")

    if official_error is None and not problems:
        _print_kv("VERDICT", "READY")
        _print_kv("USE_ANNOTATIONS", check_ann)
        _print_kv("USE_IMAGES", images)
        return EXIT_OK

    # Will a conversion work? Plan it (no files written) to surface questions now.
    try:
        result = build_coco(
            ds,
            model_names,
            parse_class_map(args.class_map),
            args.drop_unknown_classes,
            out_images_dir=Path("/nonexistent/images"),
        )
    except UserInputNeeded as exc:
        _print_kv("VERDICT", "NEEDS_INPUT")
        for p in ([official_error] if official_error else []) + problems:
            _print_kv("REASON", p)
        _print_kv("QUESTION", exc)
        return EXIT_INPUT
    _print_kv("VERDICT", "NEEDS_REFORMAT")
    if official_error:
        _print_kv("REASON", f"mAP mode would stop with: {official_error}")
    for p in problems:
        _print_kv("REASON", p)
    for n in result.notes:
        _print_kv("WILL_DO", n)
    for w in result.warnings:
        _print_kv("NOTE", w)
    _print_kv(
        "AFTER_CONVERT",
        f"{result.stats['images']} images, {result.stats['boxes']} boxes",
    )
    return EXIT_REFORMAT


def inner_convert(args) -> int:
    ann, images, model, out = (
        Path(args.annotations),
        Path(args.images),
        Path(args.model),
        Path(args.out),
    )
    model_names = load_model_names(model)
    try:
        ds = scan(
            ann,
            images,
            Path(args.dataset_names) if args.dataset_names else None,
            args.split,
        )
        result = build_coco(
            ds,
            model_names,
            parse_class_map(args.class_map),
            args.drop_unknown_classes,
            out_images_dir=out / "images",
        )
    except UserInputNeeded as exc:
        _print_kv("VERDICT", "NEEDS_INPUT")
        _print_kv("QUESTION", exc)
        return EXIT_INPUT
    write_outputs(ds, result, out, model_names)
    out_json = out / "annotations_coco.json"
    try:
        official_check(out_json, result.images_root, model)
    except Exception as exc:
        _print_kv("VERDICT", "FAILED_SELF_CHECK")
        _print_kv("ERROR", f"{type(exc).__name__}: {exc}")
        return EXIT_ERROR
    _print_kv("VERDICT", "CONVERTED")
    _print_kv("USE_ANNOTATIONS", out_json)
    _print_kv("USE_IMAGES", result.images_root)
    _print_kv("REPORT", out / "conversion_report.md")
    _print_kv(
        "SUMMARY",
        f"{result.stats['images']} images ({result.stats['labelled_images']} with labels), {result.stats['boxes']} boxes",
    )
    return EXIT_OK


def inner_class_yaml(args) -> int:
    """Write a test-mode class-names YAML (names: list) from the model's own classes."""
    model_names = load_model_names(Path(args.model))
    out = Path(args.out)
    lines = [f"# class names of {Path(args.model).name}, made by iqf-assistant"]
    lines.append(f"nc: {len(model_names)}")
    lines.append("names:")
    lines += [f"  - {json.dumps(name)}" for name in model_names]
    out.write_text("\n".join(lines) + "\n")
    _print_kv("VERDICT", "WRITTEN")
    _print_kv("USE_YAML", out)
    _print_kv(
        "CLASSES",
        f"{len(model_names)} ({', '.join(model_names[:8])}"
        f"{', ...' if len(model_names) > 8 else ''})",
    )
    return EXIT_OK


def write_outputs(
    ds: Dataset, result: BuildResult, out: Path, model_names: list[str]
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for src, dst in result.copy_plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() in CONVERTIBLE_IMAGE_SUFFIXES:
            from PIL import Image

            with Image.open(src) as im:
                im.convert("RGB").save(dst, format="PNG")
        else:
            shutil.copy2(src, dst)
    (out / "annotations_coco.json").write_text(json.dumps(result.coco))
    lines = [
        "# mAP data conversion report",
        "",
        f"- Created: {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}",
        f"- Original labels: `{ds.annotations}` (format: {ds.format_name})",
        f"- Original images: `{ds.images_root}`",
        "- The original files were **not** changed.",
        "",
        "## Why a conversion was needed and what was done",
        "",
        *[
            f"- {n}"
            for n in (
                result.notes
                or [
                    "labels were rewritten as one COCO file that mAP mode reads directly"
                ]
            )
        ],
        "",
        "## Notes",
        "",
        *[f"- {w}" for w in (result.warnings or ["none"])],
        "",
        "## Result",
        "",
        f"- Labels for mAP mode: `{out / 'annotations_coco.json'}`",
        f"- Images for mAP mode: `{result.images_root}`",
        f"- {result.stats['images']} images, {result.stats['labelled_images']} with labels, {result.stats['boxes']} boxes",
        f"- Classes (from the model, {len(model_names)}): {', '.join(model_names)}",
        "",
    ]
    (out / "conversion_report.md").write_text("\n".join(lines))


# ----------------------------
# Host side: start the work inside the iqf Docker image
# ----------------------------


def host_path(value: str, repo_root: Path) -> Path:
    value = value.strip().strip('"').strip("'")
    mapper = repo_root / "docker"
    sys.path.insert(0, str(mapper))
    try:
        from iqf_path_mapper import normalize_host_path  # handles C:\ and \\wsl$ paths

        return normalize_host_path(value)
    except ImportError:
        return Path(value).expanduser().resolve()
    finally:
        sys.path.pop(0)


def resolve_image(repo_root: Path, override: str | None) -> str:
    if override:
        return override
    sys.path.insert(0, str(repo_root / "docker"))
    try:
        from iqf_path_mapper import resolve_image_name

        return resolve_image_name(None)
    except ImportError:
        return os.environ.get("IQF_DOCKER_IMAGE") or "innodiskorg/iqf:latest"
    finally:
        sys.path.pop(0)


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def host_main(args) -> int:
    repo_root = Path(args.repo).resolve() if args.repo else default_repo_root()
    paths = {"model": host_path(args.model, repo_root)}
    for key in ("annotations", "images", "dataset_names"):
        if getattr(args, key, None):
            paths[key] = host_path(getattr(args, key), repo_root)
    for key, p in paths.items():
        if not p.exists():
            print(f"[error] {key} path not found: {p}")
            return EXIT_ERROR
        if ":" in str(p):
            print(f"[error] {key} path contains ':' which Docker cannot mount: {p}")
            return EXIT_ERROR
    if "images" in paths and not paths["images"].is_dir():
        print(f"[error] images must be a folder: {paths['images']}")
        return EXIT_ERROR

    out = None
    if args.cmd == "class-yaml":
        out = (
            host_path(args.out, repo_root)
            if args.out
            else repo_root
            / "out"
            / "iqf-assistant"
            / "class_names"
            / f"{paths['model'].stem}_{datetime.now():%Y%m%d_%H%M%S}.yaml"
        )
        if out.exists():
            print(f"[error] file already exists, choose a new name: {out}")
            return EXIT_ERROR
        out.parent.mkdir(parents=True, exist_ok=True)
    elif args.cmd == "convert":
        out = (
            host_path(args.out, repo_root)
            if args.out
            else (
                repo_root
                / "out"
                / "iqf-assistant"
                / "map_data"
                / f"{datetime.now():%Y%m%d_%H%M%S}"
            )
        )
        if out.exists():
            print(f"[error] output folder already exists, choose a new one: {out}")
            return EXIT_ERROR
        for key in ("annotations", "images"):
            base = paths[key] if paths[key].is_dir() else paths[key].parent
            if is_inside(out, base):
                print(
                    f"[error] the output folder must not be inside the {key} folder (your files stay untouched): {out}"
                )
                return EXIT_ERROR
        out.mkdir(parents=True)

    if args.no_docker:
        inner = argparse.Namespace(**vars(args))
        for key, p in paths.items():
            setattr(inner, key, str(p))
        inner.out = str(out) if out else None
        sys.path.insert(0, str(repo_root))
        return INNER_COMMANDS[args.cmd](inner)

    if not shutil.which("docker"):
        print("[error] docker not found; run the setup check first")
        return EXIT_ERROR
    image = resolve_image(repo_root, args.image)
    script = Path(__file__).resolve()
    mounts: list[str] = []

    def mount(p: Path, writable: bool = False) -> None:
        mounts.extend(["-v", f"{p}:{p}{'' if writable else ':ro'}"])

    mount(repo_root)
    for key, p in paths.items():
        if not is_inside(p, repo_root):
            mount(p)
        # YOLO names files are found next to the labels/images (up to two folders up).
        if key in ("annotations", "images") and not args.dataset_names:
            base = p if p.is_dir() else p.parent
            for parent in (base.parent, base.parent.parent):
                for name in NAMES_FILE_CANDIDATES:
                    candidate = parent / name
                    if (
                        candidate.is_file()
                        and not is_inside(candidate, repo_root)
                        and ":" not in str(candidate)
                    ):
                        mount(candidate)
    if not is_inside(script, repo_root):
        mount(script)
    if out:
        mount(out.parent if args.cmd == "class-yaml" else out, writable=True)

    inner_argv = [args.cmd, "--inside-container"]
    for key, p in paths.items():
        inner_argv += [f"--{key.replace('_', '-')}", str(p)]
    if getattr(args, "split", None):
        inner_argv += ["--split", args.split]
    if getattr(args, "class_map", None):
        inner_argv += ["--class-map", args.class_map]
    if getattr(args, "drop_unknown_classes", False):
        inner_argv.append("--drop-unknown-classes")
    if out:
        inner_argv += ["--out", str(out)]

    docker_argv = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-e",
        "YOLO_CONFIG_DIR=/tmp/ultralytics",
        "-e",
        "YOLO_VERBOSE=False",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        f"PYTHONPATH={repo_root}",
        "-w",
        str(repo_root),
        *mounts,
        "--entrypoint",
        "python3",
        image,
        str(script),
        *inner_argv,
    ]
    print(
        f"[info] checking the data inside the Docker image {image} (your files are mounted read-only)"
    )
    sys.stdout.flush()
    code = subprocess.run(docker_argv, stdin=subprocess.DEVNULL, check=False).returncode
    if out and code != EXIT_OK and args.cmd == "convert":
        # Leave nothing half-written behind when the conversion did not finish.
        shutil.rmtree(out, ignore_errors=True)
    return code


INNER_COMMANDS = {
    "inspect": inner_inspect,
    "convert": inner_convert,
    "class-yaml": inner_class_yaml,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check and reformat mAP data for iQ-Foundry."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "convert"):
        p = sub.add_parser(name)
        p.add_argument(
            "--annotations",
            required=True,
            help="labels: a file (COCO json, CVAT xml) or a folder",
        )
        p.add_argument("--images", required=True, help="folder with the images")
        p.add_argument(
            "--model",
            required=True,
            help="the reference .pt model (its class names are used)",
        )
        p.add_argument(
            "--dataset-names",
            help="data.yaml or classes.txt with the dataset's class names (YOLO)",
        )
        p.add_argument(
            "--split", help="dataset part to use when there are several, e.g. val"
        )
        p.add_argument(
            "--class-map", help='rename classes: "dataset_name=model_name,other=drop"'
        )
        p.add_argument(
            "--drop-unknown-classes",
            action="store_true",
            help="leave out classes the model does not have",
        )
        p.add_argument(
            "--repo", help="iQ-Foundry folder (default: the one containing this skill)"
        )
        p.add_argument("--image", help="Docker image (default: same as ./docker/iqf)")
        p.add_argument(
            "--no-docker",
            action="store_true",
            help="run on the host (needs PIL and ultralytics)",
        )
        p.add_argument(
            "--inside-container", action="store_true", help=argparse.SUPPRESS
        )
        if name == "convert":
            p.add_argument(
                "--out",
                help="new folder for the converted data (default: out/iqf-assistant/map_data/<time>)",
            )
    p = sub.add_parser("class-yaml", help="write a class-names YAML for test mode")
    p.add_argument("--model", required=True, help="the .pt model")
    p.add_argument(
        "--out", help="YAML file to write (default: out/iqf-assistant/class_names/)"
    )
    p.add_argument(
        "--repo", help="iQ-Foundry folder (default: the one containing this skill)"
    )
    p.add_argument("--image", help="Docker image (default: same as ./docker/iqf)")
    p.add_argument("--no-docker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        if args.inside_container:
            return INNER_COMMANDS[args.cmd](args)
        return host_main(args)
    except Unsupported as exc:
        print("VERDICT: UNSUPPORTED")
        print(f"ERROR: {exc}")
        return EXIT_ERROR
    except PermissionError as exc:
        # Usually out/ was first created by Docker (as root) on the host.
        print("VERDICT: ERROR")
        print(
            f"ERROR: no permission to write {exc.filename}. If it is under out/, ask the user "
            "to run in their own terminal: sudo chown -R $USER <iQ-Foundry folder>/out"
        )
        return EXIT_ERROR
    except (ValueError, FileNotFoundError, ET.ParseError) as exc:
        print("VERDICT: ERROR")
        print(f"ERROR: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
