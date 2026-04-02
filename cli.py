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
import platform
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from tool.test_map import run_fp_int_pair_map_eval
from yolo_models.yolov10 import YOLOV10_TEST_DEFAULTS, YoloV10Pipeline
from yolo_models.yolov11 import YOLOV11_TEST_DEFAULTS, YoloV11Pipeline
from yolo_models.yolov26 import YOLOV26_TEST_DEFAULTS, YoloV26Pipeline

DEFAULT_REMOTE_RUNNER_LOCAL = str(
    Path(__file__).resolve().parent / "tool" / "remote_tflite_raw_runner.py"
)
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
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


PIPELINES = {
    "yolov10": YoloV10Pipeline,
    "yolov11": YoloV11Pipeline,
    "yolov26": YoloV26Pipeline,
}

TEST_DEFAULTS = {
    "yolov10": YOLOV10_TEST_DEFAULTS,
    "yolov11": YOLOV11_TEST_DEFAULTS,
    "yolov26": YOLOV26_TEST_DEFAULTS,
}

MODE_REQUIRED_FIELDS = {
    "qc": ("model", "calib_dir"),
    "mAP": ("annotations", "fp_model", "images", "int_model"),
    "test": ("model", "yaml", "images"),
}

CONFIG_TEMPLATE = {
    model_type: {
        "qc": {"calib_dir": "", "model": ""},
        "mAP": {
            "annotations": "",
            "fp_model": "",
            "images": "",
            "int_model": "",
        },
        "shared": {},
        "test": {"images": "", "model": "", "yaml": ""},
    }
    for model_type in PIPELINES
}

CONFIG_PROMPTS = {
    "qc": (
        ("model", "Enter FP model path for qc (--model)"),
        ("calib_dir", "Enter calibration image directory for qc (--calib_dir)"),
    ),
    "mAP": (
        ("annotations", "Enter annotations path for mAP (--annotations)"),
        ("fp_model", "Enter FP model path for mAP (--fp-model)"),
        ("images", "Enter image directory for mAP (--images)"),
        ("int_model", "Enter INT model path for mAP (--int-model)"),
    ),
    "test": (
        ("model", "Enter INT model path for test (--model)"),
        ("yaml", "Enter class YAML path for test (--yaml)"),
        ("images", "Enter image directory for test (--images)"),
    ),
}

CONFIG_PATH_RULES = {
    "qc": {
        "model": "file",
        "calib_dir": "dir",
    },
    "mAP": {
        "annotations": "file_or_dir",
        "fp_model": "file",
        "images": "dir",
        "int_model": "file",
    },
    "test": {
        "model": "file",
        "yaml": "file",
        "images": "dir",
    },
}


def get_pipeline(model_type: str):
    if model_type not in PIPELINES:
        raise SystemExit(
            f"[error] Unsupported --type {model_type}. "
            f"Use one of: {list(PIPELINES.keys())}"
        )
    return PIPELINES[model_type]()


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


def input_prompt(message: str) -> str:
    return input(f"{ANSI_YELLOW}{message}{ANSI_RESET}")


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


def _extract_help_mode(argv: list[str]) -> str | None:
    for option in ("--mode", "--configure"):
        for idx, arg in enumerate(argv):
            if arg == option and idx + 1 < len(argv):
                return argv[idx + 1]
            if arg.startswith(f"{option}="):
                return arg.split("=", 1)[1]
    return None


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
            "Use configure flow to save paths first, or run directly with "
            "paths in one command."
        ),
    )

    print_help_section("Usage")
    print_help_command_pairs(
        [
            (
                "python3 cli.py --type {yolov10,yolov11,yolov26} "
                "--configure {qc,mAP,test}",
                "Save the required paths for a mode into config.json",
            ),
            (
                "python3 cli.py --type {yolov10,yolov11,yolov26} "
                "--mode {qc,mAP,test} [options]",
                "Run a mode with saved paths or with paths passed directly",
            ),
        ]
    )

    print_help_section("Two Ways To Use iQ-Foundry")
    print_help_lines(
        [
            "  Configure flow",
            "    First save the required paths for a mode, then run that mode "
            "using the saved paths",
            "  Direct run",
            "    Pass the required paths directly in one command without "
            "saving them first",
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
    print_help_lines(
        [
            "  Configure flow",
        ]
    )
    print_help_command_pairs(
        [
            (
                "python3 cli.py --type yolov26 --configure qc",
                "Save the required qc paths first",
            ),
            (
                "python3 cli.py --type yolov26 --mode qc",
                "Run qc using the saved paths",
            ),
        ]
    )
    print_help_lines(["  Direct run"])
    print_help_command_pairs(
        [
            (
                "python3 cli.py --type yolov10 --mode qc --model model.pt "
                "--calib_dir calib/",
                "Run qc by passing the required paths directly",
            ),
            (
                "python3 cli.py --type yolov10 --mode mAP --images val/ "
                "--annotations ann.json --fp-model ref.pt "
                "--int-model compiled.tflite",
                "Run mAP by passing the required paths directly",
            ),
            (
                "python3 cli.py --type yolov10 --mode test --model "
                "compiled.tflite --image test.jpg --yaml coco.yaml",
                "Run test on one image by passing the required paths directly",
            ),
        ]
    )

    print_help_section("More Help")
    print_help_command_pairs(
        [
            ("python3 cli.py --mode qc --help", "Show detailed qc help"),
            ("python3 cli.py --mode mAP --help", "Show detailed mAP help"),
            ("python3 cli.py --mode test --help", "Show detailed test help"),
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
    print_help_args(
        [
            ("--type TYPE", "Model family", "Required"),
            ("--mode qc", "Select qc mode", "Required"),
            ("--model MODEL", "Source model path", "Required"),
            ("--calib_dir DIR", "Calibration image directory", "Required"),
        ],
        flag_color=ANSI_GREEN,
    )

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
            "  Configure flow",
            "    python3 cli.py --type yolov26 --configure qc",
            "    python3 cli.py --type yolov26 --mode qc",
            "",
            "  Direct run",
            "    python3 cli.py --type yolov26 --mode qc --model model.pt "
            "--calib_dir calib/",
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
    print_help_args(
        [
            ("--type TYPE", "Model family", "Required"),
            ("--mode mAP", "Select mAP mode", "Required"),
            ("--annotations PATH", "Annotation file or directory", "Required"),
            ("--images DIR", "Image directory", "Required"),
            ("--fp-model PATH", "Reference model path", "Required"),
            ("--int-model PATH", "Compiled model path", "Required"),
        ],
        flag_color=ANSI_GREEN,
    )

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
            "  Configure flow",
            "    python3 cli.py --type yolov26 --configure mAP",
            "    python3 cli.py --type yolov26 --mode mAP",
            "",
            "  Direct run",
            "    python3 cli.py --type yolov26 --mode mAP --images val/ "
            "--annotations ann.json --fp-model ref.pt "
            "--int-model compiled.tflite",
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
    print_help_args(
        [
            ("--type TYPE", "Model family", "Required"),
            ("--mode test", "Select test mode", "Required"),
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
        ],
        flag_color=ANSI_GREEN,
    )

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
            "  Configure flow",
            "    python3 cli.py --type yolov26 --configure test",
            "    python3 cli.py --type yolov26 --mode test",
            "",
            "  Direct run",
            "    Single image",
            "      python3 cli.py --type yolov26 --mode test --model "
            "model.tflite --image test.jpg --yaml coco.yaml",
            "",
            "    Image directory",
            "      python3 cli.py --type yolov26 --mode test --model "
            "model.tflite --images images/ --yaml coco.yaml",
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


def render_config_overview() -> None:
    config = load_config()
    print_help_title(
        "Saved Configure Paths",
        f"Current values from {CONFIG_PATH.name}. Use configure mode with --type and a mode name to update them.",
    )

    for model_type in PIPELINES:
        print_help_section(model_type, ANSI_CYAN)
        for mode_name in ("qc", "mAP", "test"):
            print_help_lines([f"  {mode_name}"], ANSI_YELLOW)
            rows = []
            for field_name in MODE_REQUIRED_FIELDS[mode_name]:
                value = config[model_type][mode_name].get(field_name, "").strip()
                rows.append(
                    (
                        field_name,
                        value if value else "[not set]",
                        "saved path" if value else "missing",
                    )
                )
            print_help_args(rows, flag_color=ANSI_WHITE, note_color=ANSI_BLUE)
            print()

    print_help_section("How To Configure", ANSI_GREEN)
    print_help_command_pairs(
        [
            (
                "python3 cli.py --type yolov26 --configure qc",
                "Configure the required qc paths",
            ),
            (
                "python3 cli.py --type yolov26 --configure mAP",
                "Configure the required mAP paths",
            ),
            (
                "python3 cli.py --type yolov26 --configure test",
                "Configure the required test paths",
            ),
        ]
    )


def _default_config() -> dict:
    return deepcopy(CONFIG_TEMPLATE)


def _normalize_config(raw_config: object) -> dict:
    config = _default_config()
    if not isinstance(raw_config, dict):
        return config

    for model_type, model_cfg in raw_config.items():
        if model_type not in config or not isinstance(model_cfg, dict):
            continue
        for mode_name in ("qc", "mAP", "test"):
            mode_cfg = model_cfg.get(mode_name)
            if not isinstance(mode_cfg, dict):
                continue
            for field_name in MODE_REQUIRED_FIELDS[mode_name]:
                value = mode_cfg.get(field_name)
                if isinstance(value, str):
                    config[model_type][mode_name][field_name] = value
        shared_cfg = model_cfg.get("shared")
        if isinstance(shared_cfg, dict):
            config[model_type]["shared"] = shared_cfg

    return config


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return _default_config()

    try:
        raw_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[error] Invalid JSON in {CONFIG_PATH}: {exc}") from exc

    return _normalize_config(raw_config)


def save_config(config: dict) -> None:
    normalized = _normalize_config(config)
    CONFIG_PATH.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_candidate_path(value: str) -> str:
    return str(Path(value).expanduser())


def validate_configure_path(
    mode_name: str, field_name: str, value: str
) -> tuple[bool, str]:
    if not value.strip():
        return False, "path is required"

    normalized = Path(_normalize_candidate_path(value))
    expected_kind = CONFIG_PATH_RULES[mode_name][field_name]
    if not normalized.exists():
        return False, "path does not exist"
    if expected_kind == "file" and normalized.is_dir():
        return False, "expected a file, but got a directory"
    if expected_kind == "dir" and normalized.is_file():
        return False, "expected a directory, but got a file"
    if expected_kind == "file" and not normalized.is_file():
        return False, "expected a file"
    if expected_kind == "dir" and not normalized.is_dir():
        return False, "expected a directory"
    if expected_kind == "file_or_dir" and not (
        normalized.is_file() or normalized.is_dir()
    ):
        return False, "expected an existing file or directory"

    return True, ""


def prompt_yes_no(prompt: str) -> bool:
    while True:
        value = input_prompt(
            f"{prompt} ({ANSI_GREEN}y{ANSI_YELLOW}/{ANSI_RED}n{ANSI_YELLOW}): "
        ).strip().lower()
        if value in {"yes", "y"}:
            return True
        if value in {"no", "n"}:
            return False
        print_error("[error] Please answer y/yes or n/no.")


def prompt_for_configured_path(
    mode_name: str, field_name: str, prompt: str, current_value: str
) -> str:
    if current_value:
        valid_current, current_error = validate_configure_path(
            mode_name, field_name, current_value
        )
        if valid_current:
            print_info(f"[info] Existing saved path: {current_value}")
            if prompt_yes_no("Do you wish to use this path"):
                return current_value
        else:
            show_warning(
                f"[warn] Existing saved path for {mode_name}/{field_name} is invalid: "
                f"{current_error}",
                hold_seconds=0,
            )

    while True:
        value = input_prompt(f"{prompt}: ").strip()
        valid_value, error = validate_configure_path(mode_name, field_name, value)
        if valid_value:
            return _normalize_candidate_path(value)
        print_error(f"[error] {error}")


def run_configure_mode(model_type: str, mode_name: str) -> None:
    config = load_config()
    mode_config = config[model_type][mode_name]
    run_command = f"python3 cli.py --type {model_type} --mode {mode_name}"

    print_info(
        f"[info] Updating {CONFIG_PATH.name} for --type {model_type} "
        f"and --configure {mode_name}"
    )
    for field_name, prompt in CONFIG_PROMPTS[mode_name]:
        mode_config[field_name] = prompt_for_configured_path(
            mode_name, field_name, prompt, mode_config[field_name]
        )

    save_config(config)
    summary_lines = [
        f"[ok] saved {mode_name} paths for {model_type} in {CONFIG_PATH}",
        *[
            f"{field_name}: {mode_config[field_name]}"
            for field_name in MODE_REQUIRED_FIELDS[mode_name]
        ],
        f"The {mode_name} mode is now ready for execution.",
        f"Run: {run_command}",
    ]
    print_boxed_summary(
        summary_lines,
        white_line_indices={len(summary_lines) - 2, len(summary_lines) - 1},
    )


def apply_saved_mode_paths(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config()
    mode_config = config[args.type][args.mode]
    sourced_from_config: dict[str, bool] = {}
    path_sources: dict[str, str] = {}

    for field_name in MODE_REQUIRED_FIELDS[args.mode]:
        if args.mode == "test" and field_name == "images" and args.image:
            saved_value = mode_config.get(field_name, "")
            if saved_value:
                print_info(
                    "[info] Using CLI --image and ignoring saved "
                    f"{CONFIG_PATH.name} value for "
                    f"{args.type}/{args.mode}/{field_name}."
                )
            sourced_from_config[field_name] = False
            path_sources[field_name] = "CLI"
            continue

        current_value = getattr(args, field_name, None)
        saved_value = mode_config.get(field_name, "")
        if current_value:
            if saved_value:
                print_info(
                    f"[info] Using CLI --{field_name.replace('_', '-')} "
                    "and ignoring saved "
                    f"{CONFIG_PATH.name} value for "
                    f"{args.type}/{args.mode}/{field_name}."
                )
            sourced_from_config[field_name] = False
            path_sources[field_name] = "CLI"
            continue

        if saved_value:
            setattr(args, field_name, saved_value)
            sourced_from_config[field_name] = True
            path_sources[field_name] = "config.json"
        else:
            sourced_from_config[field_name] = False
            path_sources[field_name] = ""

    args._config_sourced_fields = sourced_from_config
    args._path_sources = path_sources
    return args


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
            f"--{field_name.replace('_', '-')}"
            for field_name in MODE_REQUIRED_FIELDS[args.mode]
            if not getattr(args, field_name, None)
        ]

    if missing_fields:
        missing_flags = ", ".join(missing_fields)
        raise SystemExit(
            f"[error] {args.mode} requires {missing_flags}. "
            f"Provide them explicitly or save them with "
            f"'--type {args.type} --configure {args.mode}'."
        )


def print_mode_execution_paths(args: argparse.Namespace) -> None:
    path_sources = getattr(args, "_path_sources", {})

    print_info(f"[info] {args.mode} execution paths")
    if args.mode == "qc":
        for field_name in ("model", "calib_dir"):
            print_info(
                f"[info] {field_name}: {getattr(args, field_name)} "
                f"(source: {path_sources.get(field_name, 'CLI')})"
            )
        return

    if args.mode == "mAP":
        for field_name in ("annotations", "fp_model", "images", "int_model"):
            print_info(
                f"[info] {field_name}: {getattr(args, field_name)} "
                f"(source: {path_sources.get(field_name, 'CLI')})"
            )
        return

    if args.mode == "test":
        for field_name in ("model", "yaml"):
            print_info(
                f"[info] {field_name}: {getattr(args, field_name)} "
                f"(source: {path_sources.get(field_name, 'CLI')})"
            )
        if args.image:
            print_info(f"[info] image: {args.image} (source: CLI)")
        else:
            print_info(
                f"[info] images: {args.images} "
                f"(source: {path_sources.get('images', 'CLI')})"
            )


def parse_args():
    argv = sys.argv[1:]
    if _has_bare_option(argv, "--configure"):
        render_config_overview()
        raise SystemExit(0)

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
            "cli.py --type {yolov10,yolov11,yolov26} --mode {qc,mAP,test} [options]\n"
            "       cli.py --type {yolov10,yolov11,yolov26} --configure {qc,mAP,test}"
        ),
    )

    common = p.add_argument_group("Global Required")
    run_selector = common.add_mutually_exclusive_group(required=True)
    run_selector.add_argument(
        "--mode", choices=["qc", "test", "mAP"], help="Execution mode"
    )
    run_selector.add_argument(
        "--configure",
        choices=["qc", "test", "mAP"],
        help="Save mode-required paths into config.json",
    )
    common.add_argument(
        "--type",
        required=True,
        choices=list(PIPELINES.keys()),
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
    print(f"[ok] wrote: {effective_output}")


def _run_map_mode(args: argparse.Namespace) -> None:
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
    print(f"[ok] wrote: {effective_output_text}")


def _resolve_test_mode_options(
    args: argparse.Namespace,
) -> tuple[dict, str, float, float, int, int, str, bool]:
    defaults = TEST_DEFAULTS[args.type]
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

    config_sourced_fields = getattr(args, "_config_sourced_fields", {})
    if any(
        config_sourced_fields.get(field_name, False)
        for field_name in ("model", "yaml", "images")
    ):
        if args.image:
            raise SystemExit(
                "[error] simple test mode supports saved --images only. "
                "Use explicit CLI flags for single-image test runs."
            )
        if not args.adb:
            print_info(
                "[info] test simple path detected. Enabling adb mode "
                "automatically."
            )
            args.adb = True

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
    print(f"[ok] wrote: {effective_output}")


def main():
    args = parse_args()
    if args.configure:
        run_configure_mode(args.type, args.configure)
        return

    args = apply_saved_mode_paths(args)
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
