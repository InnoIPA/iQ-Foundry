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
"""Every rule for the command-line options, in one place.

Which values --type / --task / --runtime / --precision accept and which combinations of them are
valid is all here, and check_command() checks a whole command against it in one call. Both the
host-side Docker wrapper (docker/iqf, through docker/iqf_path_mapper.py) and the in-container
backend (cli.py) call it, so the two always agree.

It runs on plain host python3, so it must stay standard library only. What a model's outputs look
like (needed at inference time, also on the target) is tool/postprocess/specs.py, not here.
"""

from __future__ import annotations

MODE_CHOICES = ("qc", "eval", "test")
RUNTIME_CHOICES = ("litert", "onnx", "qairt")
PRECISION_CHOICES = ("fp32", "int8", "w8a16", "fp16")
SUPPORTED_RUNTIME_PRECISION_ROWS = (
    ("litert", "int8", "Existing LiteRT/TFLite INT8 path"),
    ("litert", "fp32", "LiteRT/TFLite FP32 path"),
    ("onnx", "fp32", "ONNX Runtime FP32 path"),
    ("onnx", "w8a16", "ONNX Runtime W8A16 path"),
    # QAIRT builds a pre-compiled HTP context binary offline. FP32 is absent because
    # HTP has no FP32 math; a float graph runs as FP16 and only FP16 finalizes on v73.
    ("qairt", "int8", "QAIRT HTP context binary W8A8 path"),
    ("qairt", "w8a16", "QAIRT HTP context binary W8A16 path"),
    ("qairt", "fp16", "QAIRT HTP context binary FP16 path"),
)
SUPPORTED_RUNTIME_PRECISION_COMBINATIONS = {
    (runtime, precision)
    for runtime, precision, _ in SUPPORTED_RUNTIME_PRECISION_ROWS
}
# Converted-artifact extension per runtime.
QC_OUTPUT_SUFFIXES = {"litert": ".tflite", "onnx": ".onnx", "qairt": ".bin"}


def qc_requires_calibration(runtime: str, precision: str) -> bool:
    # Float precisions carry no calibration step: litert/onnx fp32 and qairt fp16.
    if runtime in {"litert", "onnx"} and precision == "fp32":
        return False
    return not (runtime == "qairt" and precision == "fp16")


# Every task iQ-Foundry knows about, and the tasks each model type supports. A new model type
# starts here (plus its entry in tool/postprocess/specs.py, which a test keeps in step).
TASK_CHOICES = ("detect", "pose", "segment", "classify")
DEFAULT_TASK = "detect"
TYPE_TASKS = {
    "yolov10": ("detect",),
    "yolov11": ("detect",),
    "yolov26": ("detect", "pose", "segment"),
    "resnet50": ("classify",),
}
MODEL_TYPES = tuple(TYPE_TASKS)
SUPPORTED_TYPE_TASK_ROWS = tuple(
    (model_type, task) for model_type, tasks in TYPE_TASKS.items() for task in tasks
)

# Runtimes each task works with so far. Detection works everywhere; the others are added one
# runtime at a time as they are implemented and verified.
TASK_RUNTIMES = {
    "detect": RUNTIME_CHOICES,
    "pose": RUNTIME_CHOICES,
    "segment": RUNTIME_CHOICES,
    "classify": RUNTIME_CHOICES,
}


class UnsupportedRuntimePrecision(ValueError):
    """--runtime / --precision is not a supported pair; callers print their own matrix message."""


def check_command(model_type: str, task: str | None, runtime: str, precision: str) -> str:
    """Check a command's --type / --task / --runtime / --precision together; return the task to run.

    Checks, in order: runtime x precision, type x task, task x runtime. Raises
    UnsupportedRuntimePrecision for the first, ValueError (message, blank line, supported list)
    for the other two.
    """
    if (runtime, precision) not in SUPPORTED_RUNTIME_PRECISION_COMBINATIONS:
        raise UnsupportedRuntimePrecision(f"runtime={runtime}, precision={precision}")
    task = resolve_task(model_type, task)
    check_task_runtime(task, runtime)
    return task


def check_task_runtime(task: str, runtime: str) -> None:
    """Raise ValueError when the task is not available on this runtime yet."""
    allowed = TASK_RUNTIMES.get(task, ())
    if runtime not in allowed:
        raise ValueError(
            f"--task {task} is not available for --runtime {runtime} yet. "
            f"Available runtimes for {task}: {', '.join(allowed)}."
        )


def supported_type_task_message() -> str:
    lines = ["Supported combinations:"]
    for model_type, task in SUPPORTED_TYPE_TASK_ROWS:
        lines.append(f"  --type {model_type:<9} --task {task}")
    return "\n".join(lines)


def resolve_task(model_type: str, task: str | None) -> str:
    """Return the task to run for --type/--task, applying the defaults.

    --task may be left out: a type that supports detect runs detect (so commands written before
    --task existed keep working), and a type with a single task runs that one. Anything else
    raises ValueError with a message listing what is supported.
    """
    tasks = TYPE_TASKS[model_type]
    if task is None:
        if DEFAULT_TASK in tasks:
            return DEFAULT_TASK
        if len(tasks) == 1:
            return tasks[0]
        raise ValueError(
            f"--task is required for --type {model_type} "
            f"(supported: {', '.join(tasks)})\n\n{supported_type_task_message()}"
        )
    if task not in tasks:
        raise ValueError(
            f"--task {task} is not supported for --type {model_type}\n\n"
            f"{supported_type_task_message()}"
        )
    return task
