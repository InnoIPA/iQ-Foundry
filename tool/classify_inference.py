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
"""`test` and `eval` for classification (resnet50), on the target over adb.

The host resizes + center-crops every image (tool/postprocess/classify.py), pushes the tensors
to the board, the board runs the converted model, and the host reads the logits back and does
softmax / top-5 / drawing / accuracy. The board side is the same batch runners the YOLO models
use (remote_tflite_raw_runner.py, onnx_inference.py batch-raw, qnn-net-run); only the way the
input is made and the logits are read differs, so nothing here is shared with the detectors.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tool.adb_runtime_bootstrap import ensure_adb_runtime_venv, push_postprocess_package
from tool.onnx_inference import (
    DEFAULT_ORT_QNN_WHEEL,
    load_onnx_model_metadata,
    resolve_onnx_model_artifact,
)
from tool.postprocess.classify import TOP_K, preprocess_classify, save_classify_outputs, top_k
from tool.postprocess.outputs import load_avg_invoke_ms
from tool.qairt_inference import load_qairt_model_metadata
from tool.test_map import (
    SUPPORTED_IMAGE_SUFFIXES,
    ImageRec,
    _load_tflite_input_detail,
    adb_push,
    adb_shell,
    build_model_runner,
    make_map_remote_run_layout,
    make_runtime_args,
)

TOOL_DIR = Path(__file__).resolve().parent
# Board-side script per runtime that has one (qairt uses the board's own qnn-net-run).
REMOTE_RUNNERS = {
    "litert": TOOL_DIR / "remote_tflite_raw_runner.py",
    "onnx": TOOL_DIR / "onnx_inference.py",
}
BACKENDS = {"litert": "tflite", "onnx": "onnx", "qairt": "qairt"}


# ---------------------------------------------------------------- inputs and labels
def load_class_names(yaml_path: str) -> list[str]:
    """`names:` of a class YAML (a list, or {index: name}), in model output order."""
    import yaml

    with open(yaml_path, encoding="utf-8") as handle:
        names = yaml.safe_load(handle)["names"]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=int)]
    return [str(name) for name in names]


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def collect_labeled_images(images_dir: Path, class_names: list[str], max_images: int | None):
    """[(image path, class id)] from images_dir/<class name>/*, spread evenly over the classes."""
    folders = sorted(p for p in images_dir.iterdir() if p.is_dir())
    unknown = [p.name for p in folders if p.name not in class_names]
    if not folders or unknown:
        raise ValueError(
            f"--images must hold one folder per class, named as in --yaml; "
            f"{'no class folders in ' + str(images_dir) if not folders else 'not in the yaml: ' + ', '.join(unknown[:5])}"
        )
    per_class = [(class_names.index(p.name), list_images(p)) for p in folders]
    picked, depth = [], 0
    while len(picked) < (max_images or float("inf")):
        round_ = [(path[depth], cid) for cid, path in per_class if depth < len(path)]
        if not round_:
            break
        picked += round_[: (max_images - len(picked)) if max_images else None]
        depth += 1
    return picked


def input_size(runtime: str, model_path: str, precision: str) -> int:
    """Side of the square input the converted model expects."""
    if runtime == "litert":
        return int(_load_tflite_input_detail(Path(model_path))["shape"][1])
    if runtime == "qairt":
        return int(load_qairt_model_metadata(model_path, "resnet50", precision, "classify").input_shape[1])
    meta = load_onnx_model_metadata(model_path, "resnet50", precision, task="classify")
    return int(meta.input_shape[1 if meta.input_layout == "nhwc" else 2])


def _board_tensor(x01: np.ndarray, runtime: str, precision: str, meta) -> np.ndarray:
    """The [0, 1] NHWC image as the converted model's own input dtype / layout."""
    if runtime == "qairt":
        return x01  # qnn-net-run converts the float32 input to the graph's type
    if runtime == "litert":
        dtype = meta["dtype"]
        if dtype == np.float32:
            return x01
        if dtype == np.uint8:
            return np.round(x01 * 255).astype(np.uint8)
        scale, zero_point = meta["quantization"]
        return np.clip(np.round(x01 / scale + zero_point), -128, 127).astype(np.int8)
    tensor = np.round(x01 * 255).astype(np.uint16) * 257 if precision == "w8a16" else x01
    if meta.input_layout == "nchw":
        tensor = np.transpose(tensor, (0, 3, 1, 2))
    return tensor.astype(meta.input_dtype, copy=False)


# ---------------------------------------------------------------- running on the board
def board_logits(
    *,
    image_paths: list[Path],
    model_path: str,
    runtime: str,
    precision: str,
    adb_serial: str | None,
    remote_workdir: str,
    qnn_lib: str,
    backend: str,
    no_qnn: bool,
) -> tuple[list[np.ndarray], float | None]:
    """Logits of the converted model for each image, run on the board over adb, and the
    mean model invoke time in ms (None if the runtime did not report it)."""
    artifact = resolve_onnx_model_artifact(model_path) if runtime == "onnx" else None
    model_file = artifact.model_path if artifact else model_path
    runner = None
    stage = tempfile.TemporaryDirectory(prefix="classify_inputs_")
    layout = make_map_remote_run_layout(
        remote_workdir, Path(model_file), str(REMOTE_RUNNERS.get(runtime, ""))
    )
    try:
        size = input_size(runtime, model_file, precision)
        if runtime == "litert":
            tensor_meta = _load_tflite_input_detail(Path(model_file))
        elif runtime == "onnx":
            tensor_meta = load_onnx_model_metadata(model_file, "resnet50", precision, task="classify")
        else:
            tensor_meta = None
        recs = []
        for index, path in enumerate(image_paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image: {path}")
            tensor = _board_tensor(preprocess_classify(image, size), runtime, precision, tensor_meta)
            if runtime == "qairt":
                tensor.tofile(os.path.join(stage.name, f"{index}.raw"))
            else:
                np.save(os.path.join(stage.name, f"{index}.npy"), tensor)
            recs.append(ImageRec(index, path.name, str(path), image.shape[1], image.shape[0]))

        remote_python = None
        if runtime != "qairt":
            remote_python = ensure_adb_runtime_venv(
                adb_serial=adb_serial,
                local_requirements_path=str(TOOL_DIR.parent / "requirements" / "target.txt"),
                local_ort_qnn_wheel_path=DEFAULT_ORT_QNN_WHEEL if runtime == "onnx" else None,
            )
        adb_shell(adb_serial, f"mkdir -p {shlex.quote(layout['remote_run_dir'])}")
        if runtime in REMOTE_RUNNERS:
            adb_push(adb_serial, str(REMOTE_RUNNERS[runtime]), layout["remote_runner"])
            push_postprocess_package(adb_serial, layout["remote_run_dir"])
        remote_inputs = layout["remote_input_dir"]
        adb_shell(adb_serial, f"rm -rf {shlex.quote(remote_inputs)} && mkdir -p {shlex.quote(remote_inputs)}")
        adb_push(adb_serial, f"{stage.name}/.", remote_inputs)

        runner = build_model_runner(
            {
                "backend": BACKENDS[runtime],
                "model_type": "resnet50",
                "quant": precision,
                "path": model_file,
                "task": "classify",
            },
            make_runtime_args(
                adb_serial=adb_serial,
                remote_workdir=layout["remote_run_dir"],
                remote_runner_remote=layout["remote_runner"],
                remote_python=remote_python,
                qnn_lib=qnn_lib,
                backend=backend,
                no_qnn=no_qnn,
            ),
            {
                "remote_input_dir": remote_inputs,
                "meta": {rec.image_id: {"ratio": 1.0, "padw": 0, "padh": 0, "orig_size": (0, 0)} for rec in recs},
            },
        )
        runner.prepare_batch(recs)
        if runtime == "qairt":
            seconds = runner.last_invoke_time_s
            avg_invoke_ms = seconds * 1000.0 if seconds is not None else None
        else:
            avg_invoke_ms = load_avg_invoke_ms(runner.local_output_dir)
        logits = [runner.get_result_for_image(rec)[0]["logits"].reshape(-1) for rec in recs]
        return logits, avg_invoke_ms
    finally:
        if runner is not None:
            runner.cleanup()
        stage.cleanup()
        if artifact is not None:
            artifact.cleanup()
        try:
            adb_shell(adb_serial, f"rm -rf {shlex.quote(layout['remote_run_dir'])}")
        except subprocess.CalledProcessError as exc:
            print(f"[warn] remote cleanup failed: {exc}")


# ---------------------------------------------------------------- test
def run_classify_test(
    *,
    model_path: str,
    yaml_path: str,
    output_dir: str,
    runtime: str,
    precision: str,
    image_dir: str | None,
    image_path: str | None,
    adb: bool,
    adb_serial: str | None,
    remote_workdir: str,
    qnn_lib: str,
    backend: str,
    no_qnn: bool,
) -> None:
    if not adb:
        raise ValueError("resnet50 test runs on the board over adb: add --adb")
    class_names = load_class_names(yaml_path)
    images = [Path(image_path)] if image_path else list_images(Path(image_dir))
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    all_logits, avg_invoke_ms = board_logits(
        image_paths=images,
        model_path=model_path,
        runtime=runtime,
        precision=precision,
        adb_serial=adb_serial,
        remote_workdir=remote_workdir,
        qnn_lib=qnn_lib,
        backend=backend,
        no_qnn=no_qnn,
    )
    if len(all_logits[0]) != len(class_names):
        raise ValueError(
            f"Class count mismatch: the model outputs {len(all_logits[0])} classes but --yaml "
            f"defines {len(class_names)}. Check that --yaml matches --model."
        )
    for path, logits in zip(images, all_logits):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        save_classify_outputs(output_dir, path.name, image, logits, class_names)
        ids, probs = top_k(logits, 1)
        print(f"{path.name}: top1={class_names[ids[0]]} ({float(probs[0]):.3f})")
    print(f"avg_model_invoke_ms={'n/a' if avg_invoke_ms is None else f'{avg_invoke_ms:.3f}'}")


# ---------------------------------------------------------------- eval
def reference_logits(model_path: str, image_paths: list[Path], size: int) -> list[np.ndarray]:
    """The original torchvision model on the host, fed the same preprocessing."""
    import torch

    from yolo_models.resnet50 import ResNet50LogitsWrapper, load_resnet50

    model = ResNet50LogitsWrapper(load_resnet50(model_path)).eval()
    result = []
    with torch.no_grad():
        for path in image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image: {path}")
            x = torch.from_numpy(preprocess_classify(image, size))
            result.append(model(x).numpy().reshape(-1))
    return result


def accuracy(logits_list: list[np.ndarray], labels: list[int]) -> tuple[float, float]:
    """(top-1, top-5) over the images."""
    hits1 = hits5 = 0
    for logits, label in zip(logits_list, labels):
        ids, _ = top_k(logits, TOP_K)
        hits1 += int(ids[0] == label)
        hits5 += int(label in ids)
    return hits1 / len(labels), hits5 / len(labels)


def run_classify_eval(
    *,
    reference_model: str,
    converted_model: str,
    yaml_path: str,
    images: str,
    output_text: str,
    runtime: str,
    precision: str,
    max_images: int | None,
    adb_serial: str | None,
    remote_workdir: str,
    qnn_lib: str,
    backend: str,
    no_qnn: bool,
) -> dict:
    class_names = load_class_names(yaml_path)
    labeled = collect_labeled_images(Path(images), class_names, max_images)
    paths, labels = [p for p, _ in labeled], [c for _, c in labeled]

    converted, _ = board_logits(
        image_paths=paths,
        model_path=converted_model,
        runtime=runtime,
        precision=precision,
        adb_serial=adb_serial,
        remote_workdir=remote_workdir,
        qnn_lib=qnn_lib,
        backend=backend,
        no_qnn=no_qnn,
    )
    if len(converted[0]) != len(class_names):
        raise ValueError(
            f"Class count mismatch: the converted model outputs {len(converted[0])} classes but "
            f"--yaml defines {len(class_names)}. Check that --yaml matches the model."
        )
    size = input_size(
        runtime,
        resolve_onnx_model_artifact(converted_model).model_path if runtime == "onnx" else converted_model,
        precision,
    )
    reference = reference_logits(reference_model, paths, size)

    ref1, ref5 = accuracy(reference, labels)
    conv1, conv5 = accuracy(converted, labels)
    agreement = sum(
        int(top_k(r, 1)[0][0] == top_k(c, 1)[0][0]) for r, c in zip(reference, converted)
    ) / len(labels)
    report = (
        "Reference vs Converted Top-1 / Top-5 Accuracy Evaluation\n"
        "task: classify\n"
        "model_type: resnet50\n"
        f"runtime: {runtime}\n"
        f"precision: {precision}\n"
        f"yaml: {yaml_path}\n"
        f"images: {images}\n"
        f"num_images: {len(labels)}\n"
        f"reference_model: {reference_model}\n"
        f"converted_model: {converted_model}\n"
        f"reference_top1: {ref1:.6f}\n"
        f"converted_top1: {conv1:.6f}\n"
        f"abs_delta_top1_converted_minus_reference: {conv1 - ref1:+.6f}\n"
        f"reference_top5: {ref5:.6f}\n"
        f"converted_top5: {conv5:.6f}\n"
        f"abs_delta_top5_converted_minus_reference: {conv5 - ref5:+.6f}\n"
        f"top1_agreement_reference_vs_converted: {agreement:.6f}\n"
    )
    print(report, end="")
    Path(output_text).parent.mkdir(parents=True, exist_ok=True)
    Path(output_text).write_text(report)
    return {
        "reference_top1": ref1,
        "converted_top1": conv1,
        "reference_top5": ref5,
        "converted_top5": conv5,
        "top1_agreement": agreement,
    }
