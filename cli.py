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
import platform
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

DEFAULT_REMOTE_RUNNER_LOCAL = str(
    Path(__file__).resolve().parent / "tool" / "remote_tflite_raw_runner.py"
)
DEFAULT_MAP_RESULTS_DIR = Path(__file__).resolve().parent / "out" / "mAP_results"
DEFAULT_QC_RESULTS_DIR = Path(__file__).resolve().parent / "out" / "model"
DEFAULT_TEST_RESULTS_DIR = Path(__file__).resolve().parent / "out" / "test"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_WHITE = "\033[97m"
ANSI_RESET = "\033[0m"
WARNING_HOLD_SECONDS = 1.2
NOTICE_HOLD_SECONDS = 3.0

MODEL_TYPES = ("yolov10", "yolov11", "yolov26")

MODE_REQUIRED_FIELDS = {
    "qc": ("model", "calib_dir"),
    "mAP": ("annotations", "fp_model", "images", "int_model"),
    "test": ("model", "yaml", "images"),
}

MODE_REQUIRED_FLAGS = {
    "qc": {
        "model": "--model",
        "calib_dir": "--calib_dir",
    },
    "mAP": {
        "annotations": "--annotations",
        "fp_model": "--fp-model",
        "images": "--images",
        "int_model": "--int-model",
    },
}


@lru_cache(maxsize=1)
def _load_model_registry():
    from yolo_models.yolov10 import YOLOV10_TEST_DEFAULTS, YoloV10Pipeline
    from yolo_models.yolov11 import YOLOV11_TEST_DEFAULTS, YoloV11Pipeline
    from yolo_models.yolov26 import YOLOV26_TEST_DEFAULTS, YoloV26Pipeline

    pipelines = {
        "yolov10": YoloV10Pipeline,
        "yolov11": YoloV11Pipeline,
        "yolov26": YoloV26Pipeline,
    }
    test_defaults = {
        "yolov10": YOLOV10_TEST_DEFAULTS,
        "yolov11": YOLOV11_TEST_DEFAULTS,
        "yolov26": YOLOV26_TEST_DEFAULTS,
    }
    return pipelines, test_defaults


def get_pipeline(model_type: str):
    pipelines, _ = _load_model_registry()
    if model_type not in pipelines:
        raise SystemExit(
            f"[error] Unsupported --type {model_type}. "
            f"Use one of: {list(MODEL_TYPES)}"
        )
    return pipelines[model_type]()


def get_test_defaults(model_type: str) -> dict:
    _, test_defaults = _load_model_registry()
    if model_type not in test_defaults:
        raise SystemExit(
            f"[error] Unsupported --type {model_type}. "
            f"Use one of: {list(MODEL_TYPES)}"
        )
    return test_defaults[model_type]


def show_warning(message: str, hold_seconds: float = WARNING_HOLD_SECONDS) -> None:
    print(f"{ANSI_YELLOW}{message}{ANSI_RESET}", file=sys.stderr, flush=True)
    if hold_seconds > 0:
        time.sleep(hold_seconds)


def show_notice(message: str, hold_seconds: float = NOTICE_HOLD_SECONDS) -> None:
    print(f"{ANSI_MAGENTA}{message}{ANSI_RESET}", file=sys.stderr, flush=True)
    if hold_seconds > 0:
        time.sleep(hold_seconds)


def color_text(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def print_error(message: str) -> None:
    print(color_text(message, ANSI_RED))


def print_success(message: str) -> None:
    print(color_text(message, ANSI_GREEN))


def print_info(message: str) -> None:
    print(color_text(message, ANSI_CYAN))


def _strip_trailing_slash(path_value: str) -> str:
    stripped = path_value.rstrip("/")
    return stripped or "/"


def _load_output_display_map() -> dict[str, str]:
    raw_value = os.environ.get("IQF_OUTPUT_DISPLAY_MAP")
    if not raw_value:
        return {}

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    return {
        key: value
        for key, value in parsed.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def resolve_output_display_path(path_value: str) -> str:
    explicit_overrides = _load_output_display_map()
    if path_value in explicit_overrides:
        return explicit_overrides[path_value]

    host_repo_root = os.environ.get("IQF_HOST_REPO_ROOT")
    container_repo_root = os.environ.get("IQF_CONTAINER_REPO_ROOT")
    if not host_repo_root or not container_repo_root:
        return path_value

    normalized_host_root = _strip_trailing_slash(host_repo_root)
    normalized_container_root = _strip_trailing_slash(container_repo_root)
    container_out_root = f"{normalized_container_root}/out"
    if path_value == container_out_root or path_value.startswith(container_out_root + "/"):
        suffix = path_value[len(container_out_root) :]
        return f"{normalized_host_root}/out{suffix}"
    return path_value


def print_boxed_summary(
    lines: list[str],
    color: str = ANSI_GREEN,
    white_line_indices: set[int] | None = None,
) -> None:
    white_line_indices = white_line_indices or set()
    width = max(len(line) for line in lines)
    border = "+" + "-" * (width + 2) + "+"
    print(color_text(border, color))
    for idx, line in enumerate(lines):
        line_color = ANSI_WHITE if idx in white_line_indices else color
        print(color_text(f"| {line.ljust(width)} |", line_color))
    print(color_text(border, color))


def print_help_title(title: str, subtitle: str | None = None) -> None:
    print(color_text(title, ANSI_WHITE))
    if subtitle:
        print(color_text(subtitle, ANSI_BLUE))


def print_help_section(title: str, color: str = ANSI_CYAN) -> None:
    print()
    print(color_text(title, color))


def print_help_lines(lines: list[str], color: str = ANSI_WHITE) -> None:
    for line in lines:
        if not line:
            print()
            continue
        print(color_text(line, color))


def print_help_command_pairs(
    pairs: list[tuple[str, str]],
    command_color: str = ANSI_WHITE,
    description_color: str = ANSI_BLUE,
) -> None:
    for command, description in pairs:
        print(color_text(f"  {command}", command_color))
        print(color_text(f"    {description}", description_color))
        print()


def print_help_example_block(lines: list[str], border_color: str = ANSI_CYAN) -> None:
    print_boxed_summary(
        lines,
        color=border_color,
        white_line_indices=set(range(len(lines))),
    )


def print_help_args(
    rows: list[tuple[str, str, str | None]],
    flag_color: str = ANSI_WHITE,
    note_color: str = ANSI_BLUE,
) -> None:
    width = max(len(flag) for flag, _, _ in rows) if rows else 0
    for flag, meaning, note in rows:
        print(
            f"{color_text('  ' + flag.ljust(width), flag_color)}  "
            f"{color_text(meaning, ANSI_WHITE)}"
        )
        if note:
            print(color_text(f"    {note}", note_color))


def _wrapper_help_style_enabled() -> bool:
    return os.environ.get("IQF_HELP_COMMAND_STYLE") == "wrapper"


def _help_usage_command() -> str:
    if _wrapper_help_style_enabled():
        return (
            "./docker/iqf run {qc,mAP,test} --type {yolov10,yolov11,yolov26} "
            "[known path flags] [backend options]"
        )
    return (
        "python3 cli.py --type {yolov10,yolov11,yolov26} "
        "--mode {qc,mAP,test} [options]"
    )


def _help_qc_command() -> str:
    if _wrapper_help_style_enabled():
        return (
            "./docker/iqf run qc --type yolov26 --model model.pt "
            "--calib_dir calib/"
        )
    return (
        "python3 cli.py --type yolov26 --mode qc --model model.pt "
        "--calib_dir calib/"
    )


def _help_map_command() -> str:
    if _wrapper_help_style_enabled():
        return (
            "./docker/iqf run mAP --type yolov26 --images val/ "
            "--annotations ann.json --fp-model ref.pt "
            "--int-model compiled.tflite"
        )
    return (
        "python3 cli.py --type yolov26 --mode mAP --images val/ "
        "--annotations ann.json --fp-model ref.pt "
        "--int-model compiled.tflite"
    )


def _help_test_image_command() -> str:
    if _wrapper_help_style_enabled():
        return (
            "./docker/iqf run test --type yolov26 --model model.tflite "
            "--image test.jpg --yaml coco.yaml"
        )
    return (
        "python3 cli.py --type yolov26 --mode test --model "
        "model.tflite --image test.jpg --yaml coco.yaml"
    )


def _help_test_images_command() -> str:
    if _wrapper_help_style_enabled():
        return (
            "./docker/iqf run test --type yolov26 --model model.tflite "
            "--images images/ --yaml coco.yaml"
        )
    return (
        "python3 cli.py --type yolov26 --mode test --model "
        "model.tflite --images images/ --yaml coco.yaml"
    )


def _help_mode_details_command(mode_name: str) -> str:
    if _wrapper_help_style_enabled():
        return f"./docker/iqf run {mode_name} --help"
    return f"python3 cli.py --mode {mode_name} --help"


def _mode_required_argument_row(mode_name: str) -> tuple[str, str, str | None] | None:
    if _wrapper_help_style_enabled():
        return None
    return (f"--mode {mode_name}", f"Select {mode_name} mode", "Required")


def _extract_help_mode(argv: list[str]) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == "--mode" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--mode="):
            return arg.split("=", 1)[1]
    return None


def _contains_option(argv: list[str], option: str) -> bool:
    for arg in argv:
        if arg == option or arg.startswith(f"{option}="):
            return True
    return False


def _help_requested(argv: list[str]) -> bool:
    return any(arg in {"-h", "--help"} for arg in argv)


def _has_bare_option(argv: list[str], option: str) -> bool:
    for idx, arg in enumerate(argv):
        if arg != option:
            continue
        if idx + 1 >= len(argv) or argv[idx + 1].startswith("--"):
            return True
    return False


def _has_option_value(argv: list[str], option: str) -> bool:
    for idx, arg in enumerate(argv):
        if arg == option and idx + 1 < len(argv):
            return True
        if arg.startswith(f"{option}="):
            return True
    return False


def render_main_help() -> None:
    print_help_title(
        "iQ-Foundry - Simplify the Workflow, Accelerate Deployment.",
        (
            "Direct backend usage for environments where the input paths are "
            "already valid locally or inside the container."
        ),
    )

    print_help_section("Usage")
    print_help_command_pairs(
        [
            (
                _help_usage_command(),
                "Run a mode with explicit paths",
            ),
        ]
    )

    print_help_section("Recommended Flow")
    print_help_lines(
        [
            "  For Docker-based interactive setup, use ./docker/iqf configure",
            "  and ./docker/iqf run. This backend does not save or translate paths.",
        ]
    )

    print_help_section("Modes")
    print_help_lines(
        [
            "  qc    Quantize and convert a computer vision model",
            "  mAP   Compare reference and compiled model accuracy",
            "  test  Run model inference on one image or a folder",
        ]
    )

    print_help_section("Quick Start")
    print_help_command_pairs(
        [
            (
                _help_qc_command(),
                "Run qc by passing the required paths directly",
            ),
            (
                _help_map_command(),
                "Run mAP by passing the required paths directly",
            ),
            (
                _help_test_image_command(),
                "Run test on one image by passing the required paths directly",
            ),
        ]
    )

    print_help_section("More Help")
    print_help_command_pairs(
        [
            (_help_mode_details_command("qc"), "Show detailed qc help"),
            (_help_mode_details_command("mAP"), "Show detailed mAP help"),
            (_help_mode_details_command("test"), "Show detailed test help"),
            (
                "./docker/iqf configure qc --type yolov26",
                "Use the Docker wrapper for interactive host-path setup",
            ),
        ]
    )


def render_qc_help() -> None:
    print_help_title(
        "QC mode - quantize and convert a computer vision model",
        "Detailed help for preparing a compiled model from a source model.",
    )

    print_help_section("Purpose", ANSI_BLUE)
    print_help_lines(
        [
            "  Use qc mode to convert a source model into a compiled model "
            "using a calibration image directory."
        ]
    )

    print_help_section("Required Arguments", ANSI_GREEN)
    qc_rows = [
        ("--type TYPE", "Model family", "Required"),
        ("--model MODEL", "Source model path", "Required"),
        ("--calib_dir DIR", "Calibration image directory", "Required"),
    ]
    mode_row = _mode_required_argument_row("qc")
    if mode_row:
        qc_rows.insert(1, mode_row)
    print_help_args(qc_rows, flag_color=ANSI_GREEN)

    print_help_section("Optional Arguments", ANSI_YELLOW)
    print_help_args(
        [
            (
                "--output OUTPUT",
                "Output path override",
                "Default directory: out/model/<type>/",
            ),
            ("--max_calib N", "Max calibration images", "Default: 200"),
            (
                "--qc-head HEAD",
                "Head override",
                "Choices: one2many, one2one; default: one2many for "
                "yolov10/yolov26",
            ),
            (
                "--qc-quant-scheme SCHEME",
                "Quantization scheme override",
                "Choices: mse, minmax; defaults: yolov10=mse, "
                "yolov11=minmax, yolov26=mse",
            ),
        ],
        flag_color=ANSI_YELLOW,
    )

    print_help_section("Advanced Arguments", ANSI_MAGENTA)
    print_help_lines(["  None."], ANSI_WHITE)

    print_help_section("Example Commands")
    print_help_lines(
        [
            f"  {_help_qc_command()}",
        ]
    )

    print_help_section("Notes", ANSI_BLUE)
    print_help_lines(
        [
            "  - yolov11 uses the default head and ignores --qc-head.",
            "  - Default quantization scheme: yolov10=mse, yolov11=minmax, "
            "yolov26=mse.",
        ]
    )


def render_map_help() -> None:
    print_help_title(
        "mAP mode - compare reference and compiled model accuracy",
        "Detailed help for pair evaluation with shared inputs and report output.",
    )

    print_help_section("Purpose", ANSI_BLUE)
    print_help_lines(
        [
            "  Use mAP mode to compare a reference model against a compiled "
            "model and write an accuracy report."
        ]
    )

    print_help_section("Required Arguments", ANSI_GREEN)
    map_rows = [
        ("--type TYPE", "Model family", "Required"),
        ("--annotations PATH", "Annotation file or directory", "Required"),
        ("--images DIR", "Image directory", "Required"),
        ("--fp-model PATH", "Reference model path", "Required"),
        ("--int-model PATH", "Compiled model path", "Required"),
    ]
    mode_row = _mode_required_argument_row("mAP")
    if mode_row:
        map_rows.insert(1, mode_row)
    print_help_args(map_rows, flag_color=ANSI_GREEN)

    print_help_section("Optional Arguments", ANSI_YELLOW)
    print_help_args(
        [
            (
                "--output_text PATH",
                "Text report path",
                "Default directory: out/mAP_results/<type>/ with generated filename",
            ),
            ("--conf CONF", "Confidence threshold", "Default: 0.25"),
            ("--nms NMS", "NMS IoU threshold", "Default: 0.7"),
            ("--max-det N", "Max detections per image", "Default: 300"),
            ("--max-images N", "Max images to process", "Default: 300"),
        ],
        flag_color=ANSI_YELLOW,
    )

    print_help_section("Advanced Arguments", ANSI_MAGENTA)
    print_help_args(
        [
            (
                "--fp-head HEAD",
                "Reference branch override",
                "Choices: one2many, one2one; default: one2many for "
                "yolov10/yolov26",
            ),
            (
                "--adb-serial SERIAL",
                "ADB device serial",
                "Default: first available device",
            ),
            (
                "--remote-workdir DIR",
                "Remote working directory",
                "Default: /data/local/tmp/yolo_map_eval",
            ),
            (
                "--remote-runner-local PATH",
                "Local remote runner path",
                "Default: tool/remote_tflite_raw_runner.py",
            ),
            (
                "--remote-runner-remote PATH",
                "Remote runner path on device",
                "Default: /data/local/tmp/yolo_map_eval/remote_tflite_raw_runner.py",
            ),
            (
                "--qnn-lib PATH",
                "Delegate library path",
                "Default: /usr/lib/libQnnTFLiteDelegate.so",
            ),
            ("--backend BACKEND", "Delegate backend", "Default: htp"),
            ("--no-qnn", "Disable delegate usage", "Default: off"),
        ],
        flag_color=ANSI_MAGENTA,
    )

    print_help_section("Example Commands")
    print_help_lines(
        [
            f"  {_help_map_command()}",
        ]
    )

    print_help_section("Notes", ANSI_BLUE)
    print_help_lines(
        [
            "  - Accepted annotations: COCO .json, YOLO .txt directories, "
            "and VOC .xml directories.",
            "  - If qc used --qc-head one2one for yolov10 or yolov26, use "
            "--fp-head one2one here as well.",
            "  - Default branch selection is one2many for yolov10 and "
            "yolov26 when --fp-head is not provided.",
        ]
    )


def render_test_help() -> None:
    print_help_title(
        "Test mode - run inference on one image or a folder",
        "Detailed help for local or adb-based inference runs.",
    )

    print_help_section("Purpose", ANSI_BLUE)
    print_help_lines(
        [
            "  Use test mode to run model inference on a single image or an "
            "image directory and write output files."
        ]
    )

    print_help_section("Required Arguments", ANSI_GREEN)
    test_rows = [
        ("--type TYPE", "Model family", "Required"),
        ("--model MODEL", "Compiled model path", "Required"),
        ("--yaml YAML", "Class names YAML", "Required"),
        (
            "--image IMAGE",
            "Single image path",
            "Use exactly one of --image or --images",
        ),
        (
            "--images DIR",
            "Image directory",
            "Use exactly one of --image or --images",
        ),
    ]
    mode_row = _mode_required_argument_row("test")
    if mode_row:
        test_rows.insert(1, mode_row)
    print_help_args(test_rows, flag_color=ANSI_GREEN)

    print_help_section("Common Optional Arguments", ANSI_YELLOW)
    print_help_args(
        [
            (
                "--output OUTPUT",
                "Output path override",
                "Default directory: out/test/<type>/ with generated folder name",
            ),
            ("--conf CONF", "Confidence threshold", "Default: 0.25"),
            ("--nms NMS", "NMS IoU threshold", "Default: 0.6"),
            ("--topk TOPK", "Top-k before NMS", "Default: 300"),
            ("--max-det N", "Max detections", "Default: 100"),
            (
                "--postprocess-flow FLOW",
                "Postprocess flow override",
                "Choices: auto, default, o2o, o2m; default: auto",
            ),
            ("--o2o-nms", "Enable class-wise NMS for o2o flow", "Default: off"),
        ],
        flag_color=ANSI_YELLOW,
    )

    print_help_section("ADB Optional Arguments", ANSI_CYAN)
    print_help_args(
        [
            ("--adb", "Run through adb", "Default: off"),
            (
                "--adb-serial SERIAL",
                "ADB device serial",
                "Default: first available device",
            ),
            (
                "--remote-workdir DIR",
                "Remote working directory",
                "Default: /data/local/tmp/yolo_map_eval",
            ),
            (
                "--qnn-lib PATH",
                "Delegate library path",
                "Default: /usr/lib/libQnnTFLiteDelegate.so",
            ),
            ("--backend BACKEND", "Delegate backend", "Default: htp"),
            ("--no-qnn", "Disable delegate usage", "Default: off"),
        ],
        flag_color=ANSI_CYAN,
    )

    print_help_section("Advanced Arguments", ANSI_MAGENTA)
    print_help_args(
        [
            (
                "--disable-int8-prefilter",
                "Disable class prefilter during postprocess",
                "Default: off",
            ),
        ],
        flag_color=ANSI_MAGENTA,
    )

    print_help_section("Example Commands")
    print_help_lines(
        [
            "  Single image",
            f"    {_help_test_image_command()}",
            "",
            "  Image directory",
            f"    {_help_test_images_command()}",
        ]
    )

    print_help_section("Notes", ANSI_BLUE)
    print_help_lines(
        [
            "  - Use exactly one of --image or --images.",
            "  - --output overrides the default output directory.",
            "  - adb-related flags are needed only when running through adb.",
            "  - Current defaults: conf=0.25, nms=0.6, topk=300, "
            "max-det=100, postprocess-flow=auto.",
        ]
    )


def render_help_for_mode(mode_name: str | None) -> None:
    if mode_name == "qc":
        render_qc_help()
        return
    if mode_name == "mAP":
        render_map_help()
        return
    if mode_name == "test":
        render_test_help()
        return
    render_main_help()


def validate_mode_requirements(args: argparse.Namespace) -> None:
    missing_fields = []
    if args.mode == "test":
        if not args.model:
            missing_fields.append("--model")
        if not args.yaml:
            missing_fields.append("--yaml")
        if not args.images and not args.image:
            missing_fields.append("exactly one of --images or --image")
    else:
        missing_fields = [
            MODE_REQUIRED_FLAGS[args.mode][field_name]
            for field_name in MODE_REQUIRED_FIELDS[args.mode]
            if not getattr(args, field_name, None)
        ]

    if missing_fields:
        missing_flags = ", ".join(missing_fields)
        raise SystemExit(
            f"[error] {args.mode} requires {missing_flags}. "
            "Provide required paths explicitly. Docker users can run:\n"
            f"  ./docker/iqf configure {args.mode} --type {args.type}"
        )


def print_mode_execution_paths(args: argparse.Namespace) -> None:
    print_info(f"[info] {args.mode} execution paths")
    if args.mode == "qc":
        for field_name in ("model", "calib_dir"):
            print_info(f"[info] {field_name}: {getattr(args, field_name)}")
        return

    if args.mode == "mAP":
        for field_name in ("annotations", "fp_model", "images", "int_model"):
            print_info(f"[info] {field_name}: {getattr(args, field_name)}")
        return

    if args.mode == "test":
        for field_name in ("model", "yaml"):
            print_info(f"[info] {field_name}: {getattr(args, field_name)}")
        if args.image:
            print_info(f"[info] image: {args.image}")
        else:
            print_info(f"[info] images: {args.images}")


def parse_args():
    argv = sys.argv[1:]
    if _contains_option(argv, "--configure"):
        raise SystemExit(
            "[error] Interactive configure mode has moved to the Docker wrapper.\n"
            "Use: ./docker/iqf configure <qc|mAP|test> --type <type>"
        )

    selected_mode = _extract_help_mode(argv)
    if not argv or _help_requested(argv):
        render_help_for_mode(selected_mode)
        raise SystemExit(0)

    if selected_mode in {"qc", "mAP", "test"} and not _has_option_value(argv, "--type"):
        render_help_for_mode(selected_mode)
        raise SystemExit(0)

    p = argparse.ArgumentParser(
        "iQ-Foundry",
        add_help=False,
        usage=(
            "cli.py --type {yolov10,yolov11,yolov26} --mode {qc,mAP,test} [options]"
        ),
    )

    common = p.add_argument_group("Global Required")
    common.add_argument("--mode", choices=["qc", "test", "mAP"], required=True)
    common.add_argument(
        "--type",
        required=True,
        choices=MODEL_TYPES,
        help="Model family",
    )
    common.add_argument("--model", help="Model path")

    shared = p.add_argument_group("Shared Optional")
    shared.add_argument("--images", help="Image directory")
    shared.add_argument("--output", help="Output path override")

    qc = p.add_argument_group("QC Args (--mode qc)")
    qc.add_argument("--calib_dir", default=None, help="Calibration image directory")
    qc.add_argument(
        "--max_calib",
        type=int,
        default=200,
        help="Max calibration images",
    )
    qc.add_argument(
        "--qc-head",
        choices=["one2many", "one2one"],
        default=None,
        help="Head override",
    )
    qc.add_argument(
        "--qc-quant-scheme",
        choices=["mse", "minmax"],
        default=None,
        help="Quantization scheme override",
    )

    map_eval = p.add_argument_group("mAP Args (--mode mAP)")
    map_eval.add_argument("--annotations", help="Annotation file or directory")
    map_eval.add_argument("--fp-model", dest="fp_model", help="Reference model path")
    map_eval.add_argument("--int-model", dest="int_model", help="Compiled model path")
    map_eval.add_argument("--output_text", help="Text report path")
    map_eval.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold",
    )
    map_eval.add_argument(
        "--fp-head",
        choices=["one2many", "one2one"],
        default=None,
        help="Reference branch override",
    )
    map_eval.add_argument(
        "--nms",
        type=float,
        default=None,
        help="NMS IoU threshold",
    )
    map_eval.add_argument(
        "--max-det",
        dest="max_det",
        type=int,
        default=None,
        help="Max detections per image",
    )
    map_eval.add_argument(
        "--max-images",
        dest="max_images",
        type=int,
        default=300,
        help="Max images to process",
    )
    adb_opts = p.add_argument_group("ADB Optional (mAP + test --adb)")
    adb_opts.add_argument(
        "--adb-serial",
        default=None,
        help="ADB device serial",
    )
    adb_opts.add_argument(
        "--remote-workdir",
        dest="remote_workdir",
        default="/data/local/tmp/yolo_map_eval",
        help="Remote working directory",
    )
    adb_opts.add_argument(
        "--remote-runner-local",
        dest="remote_runner_local",
        default=DEFAULT_REMOTE_RUNNER_LOCAL,
        help="Local remote runner path",
    )
    adb_opts.add_argument(
        "--remote-runner-remote",
        dest="remote_runner_remote",
        default="/data/local/tmp/yolo_map_eval/remote_tflite_raw_runner.py",
        help="Remote runner path on device",
    )
    adb_opts.add_argument(
        "--qnn-lib",
        dest="qnn_lib",
        default="/usr/lib/libQnnTFLiteDelegate.so",
        help="Delegate library path",
    )
    adb_opts.add_argument(
        "--backend", default="htp", help="Delegate backend"
    )
    adb_opts.add_argument(
        "--no-qnn",
        dest="no_qnn",
        action="store_true",
        help="Disable delegate usage",
    )

    test = p.add_argument_group("Test Args (--mode test)")
    test.add_argument("--image", help="Single image path")
    test.add_argument("--yaml", help="Class names YAML")
    test.add_argument(
        "--adb",
        action="store_true",
        help="Run through adb",
    )
    test.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Top-k before NMS",
    )
    test.add_argument(
        "--postprocess-flow",
        dest="postprocess_flow",
        choices=["auto", "default", "o2o", "o2m"],
        default=None,
        help="Postprocess flow override",
    )
    test.add_argument(
        "--o2o-nms",
        dest="o2o_nms",
        action="store_true",
        help="Enable class-wise NMS for o2o flow",
    )
    test.add_argument(
        "--disable-int8-prefilter",
        dest="disable_int8_prefilter",
        action="store_true",
        help="Disable class prefilter during postprocess",
    )

    return p.parse_args()


def resolve_default_fp_head(model_type: str, fp_head_override: str | None) -> str:
    if model_type == "yolov11":
        if fp_head_override is not None:
            show_warning(
                "[warn] --fp-head is not applicable for yolov11 "
                "and will be ignored (using default head)."
            )
        return "default"

    if fp_head_override is not None:
        return fp_head_override
    if model_type == "yolov10":
        return "one2many"
    if model_type == "yolov26":
        return "one2many"
    return "one2many"


def resolve_default_qc_head(model_type: str, qc_head_override: str | None) -> str:
    if model_type == "yolov11":
        if qc_head_override is not None:
            show_warning(
                "[warn] --qc-head ignored for yolov11 "
                "(yolov11 supports only the default head)"
            )
        return "default"

    if qc_head_override is not None:
        return qc_head_override

    if model_type == "yolov10":
        return "one2many"
    if model_type == "yolov26":
        return "one2many"
    return "one2many"


def resolve_default_qc_quant_scheme(
    model_type: str, qc_quant_override: str | None
) -> str:
    if qc_quant_override is not None:
        return qc_quant_override
    if model_type == "yolov11":
        return "minmax"
    return "mse"


def resolve_default_qc_output_path(
    model_type: str, output_override: str | None, quant_label: str = "int8"
) -> str:
    if output_override:
        return output_override
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(
        DEFAULT_QC_RESULTS_DIR / model_type / f"{model_type}_{quant_label}_{ts}.tflite"
    )


def resolve_default_output_text(
    model_type: str, output_text_override: str | None
) -> str:
    if output_text_override:
        return output_text_override
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(
        DEFAULT_MAP_RESULTS_DIR / model_type / f"{model_type}_mAP_result_{ts}.txt"
    )


def resolve_default_test_output_path(
    model_type: str, output_override: str | None
) -> str:
    if output_override:
        return output_override
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(DEFAULT_TEST_RESULTS_DIR / model_type / f"{model_type}_inference_{ts}")


def resolve_default_map_conf(conf_override: float | None) -> float:
    if conf_override is not None:
        return conf_override
    return 0.25


def resolve_default_map_nms(nms_override: float | None) -> float:
    if nms_override is not None:
        return nms_override
    return 0.7


def resolve_default_map_max_det(max_det_override: int | None) -> int:
    if max_det_override is not None:
        return max_det_override
    return 300


def is_iq9_runtime() -> bool:
    return platform.machine().lower() in {"aarch64", "arm64"}


def _run_qc_mode(args: argparse.Namespace) -> None:
    pipe = get_pipeline(args.type)
    if not args.model:
        raise SystemExit("[error] qc requires --model")
    if not args.calib_dir:
        raise SystemExit("[error] qc requires --calib_dir (calibration image folder)")

    effective_qc_head = resolve_default_qc_head(args.type, args.qc_head)
    effective_qc_quant_scheme = resolve_default_qc_quant_scheme(
        args.type, args.qc_quant_scheme
    )
    effective_output = resolve_default_qc_output_path(
        args.type, args.output, quant_label="int8"
    )

    Path(effective_output).parent.mkdir(parents=True, exist_ok=True)
    pipe.quantize_convert(
        model_path=args.model,
        out_tflite=effective_output,
        calib_dir=args.calib_dir,
        max_calib=args.max_calib,
        qc_head=effective_qc_head,
        qc_quant_scheme=effective_qc_quant_scheme,
    )
    print(f"[ok] wrote: {resolve_output_display_path(effective_output)}")


def _run_map_mode(args: argparse.Namespace) -> None:
    from tool.test_map import run_fp_int_pair_map_eval

    if not args.annotations or not args.images:
        raise SystemExit("[error] mAP requires --annotations and --images")
    if not args.fp_model or not args.int_model:
        raise SystemExit("[error] mAP requires --fp-model and --int-model")
    if args.type in {"yolov10", "yolov26"} and args.fp_head is None:
        show_notice(
            "[notice] Caution: if this INT8 model was generated with --qc-head "
            "one2one, run mAP with --fp-head one2one. Otherwise the FP comparison "
            "will use one2many and the result can be unfair."
        )

    effective_fp_head = resolve_default_fp_head(args.type, args.fp_head)
    effective_output_text = resolve_default_output_text(args.type, args.output_text)
    effective_conf = resolve_default_map_conf(args.conf)
    effective_nms = resolve_default_map_nms(args.nms)
    effective_max_det = resolve_default_map_max_det(args.max_det)
    try:
        run_fp_int_pair_map_eval(
            model_type=args.type,
            fp_model=args.fp_model,
            int_model=args.int_model,
            annotations=args.annotations,
            images=args.images,
            output_text=effective_output_text,
            conf=effective_conf,
            nms=effective_nms,
            max_det=effective_max_det,
            max_images=args.max_images,
            fp_head=effective_fp_head,
            adb_serial=args.adb_serial,
            remote_workdir=args.remote_workdir,
            remote_runner_local=args.remote_runner_local,
            remote_runner_remote=args.remote_runner_remote,
            qnn_lib=args.qnn_lib,
            backend=args.backend,
            no_qnn=args.no_qnn,
        )
    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        raise SystemExit(f"[error] {exc}") from exc
    print(f"[ok] wrote: {resolve_output_display_path(effective_output_text)}")


def _resolve_test_mode_options(
    args: argparse.Namespace,
) -> tuple[dict, str, float, float, int, int, str, bool]:
    defaults = get_test_defaults(args.type)
    effective_output = resolve_default_test_output_path(args.type, args.output)
    effective_conf = args.conf if args.conf is not None else defaults["conf_thres"]
    effective_nms = args.nms if args.nms is not None else defaults["iou_thres"]
    effective_topk = args.topk if args.topk is not None else defaults["topk"]
    effective_max_det = (
        args.max_det if args.max_det is not None else defaults["max_det"]
    )
    effective_postprocess_flow = args.postprocess_flow or "auto"
    effective_o2o_nms = args.o2o_nms

    if args.type == "yolov11":
        if effective_postprocess_flow in ("o2o", "o2m"):
            show_warning(
                "[warn] yolov11 supports only the default postprocess flow. "
                f"Ignoring --postprocess-flow {effective_postprocess_flow} "
                "and using default."
            )
            effective_postprocess_flow = "default"
        if effective_o2o_nms:
            show_warning(
                "[warn] --o2o-nms is not applicable for yolov11 and will be ignored."
            )
            effective_o2o_nms = False

    return (
        defaults,
        effective_output,
        effective_conf,
        effective_nms,
        effective_topk,
        effective_max_det,
        effective_postprocess_flow,
        effective_o2o_nms,
    )


def _run_test_mode(args: argparse.Namespace) -> None:
    from tool.inference import run_test_inference_adb, run_test_inference_local

    if not args.model:
        raise SystemExit("[error] test requires --model (INT .tflite)")
    if not args.yaml:
        raise SystemExit("[error] test requires --yaml")
    if bool(args.images) == bool(args.image):
        raise SystemExit("[error] test requires exactly one of --images or --image")

    show_warning(
        "[warn] Caution: make sure --type matches the quantized model family "
        "before running inference."
    )
    (
        defaults,
        effective_output,
        effective_conf,
        effective_nms,
        effective_topk,
        effective_max_det,
        effective_postprocess_flow,
        effective_o2o_nms,
    ) = _resolve_test_mode_options(args)

    runner = run_test_inference_adb if args.adb else run_test_inference_local
    runner_kwargs = {
        "model_path": args.model,
        "yaml_path": args.yaml,
        "output_dir": effective_output,
        "model_type": args.type,
        "default_flow": defaults["default_flow"],
        "conf_thres": effective_conf,
        "iou_thres": effective_nms,
        "topk": effective_topk,
        "max_det": effective_max_det,
        "image_dir": args.images,
        "image_path": args.image,
        "postprocess_flow": effective_postprocess_flow,
        "o2o_nms": effective_o2o_nms,
        "disable_int8_prefilter": args.disable_int8_prefilter,
        "no_qnn": args.no_qnn,
        "qnn_lib": args.qnn_lib,
        "backend": args.backend,
    }
    if args.adb:
        runner_kwargs.update(
            adb_serial=args.adb_serial,
            remote_workdir=args.remote_workdir,
        )
    else:
        runner_kwargs["enforce_iq9_native"] = True

    try:
        runner(**runner_kwargs)
    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        raise SystemExit(f"[error] {exc}") from exc
    print(f"[ok] wrote: {resolve_output_display_path(effective_output)}")


def main():
    args = parse_args()
    validate_mode_requirements(args)

    if is_iq9_runtime() and (args.mode != "test" or args.adb):
        show_warning(
            "[warn] IQ9 runtime supports only '--mode test' without '--adb'. "
            "Use an x86 host for qc/mAP or adb-orchestrated test runs."
        )
        raise SystemExit(1)

    print_mode_execution_paths(args)

    if args.mode == "qc":
        _run_qc_mode(args)
        return
    if args.mode == "mAP":
        _run_map_mode(args)
        return
    if args.mode == "test":
        _run_test_mode(args)
        return


if __name__ == "__main__":
    main()
