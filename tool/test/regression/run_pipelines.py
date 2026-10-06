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
"""Deterministic pipeline regression runner for iQ-Foundry.

The runner drives the documented host flow (``./docker/iqf run <mode> ...``) for every
selected pipeline in regression/pipelines.yaml, exactly as a person following the docs
would, and records what happened. It never edits product code, never retries a failed
stage and never changes flags after a failure: the goal is to observe, not to repair.

Subcommands (see ``--help`` of each):
    list       show all pipeline definitions
    init       create a run directory with an inputs.yaml to fill in
    hub-login  run ./qaihub_login.sh with the token from $IQF_QAI_HUB_TOKEN
    validate   check inputs, docker, QAI Hub login, adb device and wrapper dry-runs
    plan       print the exact commands `run` will execute
    run        execute the selected pipelines and write results.json + report.md
    status     show progress of a run directory
    report     regenerate report.md from results.json

Exit codes: 0 ok, 1 usage/internal error, 2 validation failed, 3 run finished with failures,
130 interrupted.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import pty
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML ships with the host requirements
    print("[error] PyYAML is required: pip install pyyaml", file=sys.stderr)
    raise SystemExit(1) from None

RUNNER_VERSION = 1
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PIPELINES_FILE = Path("regression") / "pipelines.yaml"
TEMPLATE_FILE = (
    Path(".agents")
    / "skills"
    / "iqf-pipeline-regression"
    / "templates"
    / "inputs.template.yaml"
)
DEFAULT_RUNS_DIR = Path("out") / "regression"
TOKEN_ENV = "IQF_QAI_HUB_TOKEN"
MODES = ("qc", "mAP", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_VALIDATION = 2
EXIT_FAILURES = 3
EXIT_INTERRUPTED = 130

# Stage statuses. Ordered from worst to best for the per-pipeline "overall" column.
FAIL = "FAIL"
TIMEOUT = "TIMEOUT"
ABORTED = "ABORTED"
BLOCKED = "BLOCKED"
NOT_FEASIBLE = "NOT_FEASIBLE"
PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
PASS = "PASS"
SKIPPED = "SKIPPED"
PENDING = "PENDING"
RUNNING = "RUNNING"
OVERALL_PRIORITY = (
    FAIL,
    TIMEOUT,
    ABORTED,
    BLOCKED,
    NOT_FEASIBLE,
    RUNNING,
    PENDING,
    PASS_WITH_WARNINGS,
    PASS,
    SKIPPED,
)
FAILED_STATUSES = {FAIL, TIMEOUT, ABORTED, BLOCKED, NOT_FEASIBLE}

# Flags the runner owns per mode. Users may not pass them through extra_flags because the
# runner sets them from the inputs file (paths) or they would change the harness behavior.
CONTROLLED_FLAGS = {
    "common": {"--type", "--runtime", "--precision", "--mode", "--save", "--dry-run"},
    "qc": {"--model", "--calib_dir", "--output", "--max_calib"},
    "mAP": {
        "--annotations",
        "--images",
        "--reference-model",
        "--converted-model",
        "--output_text",
        "--max-images",
        "--adb-serial",
    },
    "test": {
        "--model",
        "--yaml",
        "--image",
        "--images",
        "--output",
        "--adb",
        "--adb-serial",
    },
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"
ANSI_RESET = "\033[0m"


class RegressionError(RuntimeError):
    """A usage or configuration problem that stops the runner before any stage runs."""


# ----------------------------
# Console helpers
# ----------------------------


def _color(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{ANSI_RESET}"


def info(message: str) -> None:
    print(_color(f"[info] {message}", ANSI_CYAN), flush=True)


def ok(message: str) -> None:
    print(_color(f"[ok] {message}", ANSI_GREEN), flush=True)


def warn(message: str) -> None:
    print(_color(f"[warn] {message}", ANSI_YELLOW), flush=True)


def error(message: str) -> None:
    print(_color(f"[error] {message}", ANSI_RED), file=sys.stderr, flush=True)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_size(size: int | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


# ----------------------------
# Pipeline definitions
# ----------------------------


@dataclass(frozen=True)
class Pipeline:
    num: int
    id: str
    type: str
    runtime: str
    precision: str
    needs_calibration: bool


def load_mapper(repo_root: Path):
    """Import docker/iqf_path_mapper.py so validation follows the product's own rules."""
    docker_dir = repo_root / "docker"
    if not (docker_dir / "iqf_path_mapper.py").is_file():
        raise RegressionError(f"docker/iqf_path_mapper.py not found under {repo_root}")
    if str(docker_dir) not in sys.path:
        sys.path.insert(0, str(docker_dir))
    import iqf_path_mapper  # noqa: PLC0415 - imported from the checkout under test

    return iqf_path_mapper


def load_pipelines(path: Path) -> tuple[dict, list[Pipeline]]:
    if not path.is_file():
        raise RegressionError(f"pipeline definitions not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("schema_version") != 1:
        raise RegressionError(f"{path}: schema_version must be 1")
    for key in ("defaults", "artifact_ext", "qc_backend", "pipelines"):
        if key not in data:
            raise RegressionError(f"{path}: missing top-level key '{key}'")

    pipelines: list[Pipeline] = []
    seen: set[str] = set()
    seen_numbers: set[int] = set()
    for index, entry in enumerate(data["pipelines"]):
        missing = [
            k
            for k in ("num", "id", "type", "runtime", "precision", "needs_calibration")
            if k not in entry
        ]
        if missing:
            raise RegressionError(
                f"{path}: pipelines[{index}] is missing {', '.join(missing)}"
            )
        if not isinstance(entry["num"], int) or entry["num"] <= 0:
            raise RegressionError(
                f"{path}: pipelines[{index}].num must be a positive integer"
            )
        pipeline = Pipeline(
            num=entry["num"],
            id=str(entry["id"]),
            type=str(entry["type"]),
            runtime=str(entry["runtime"]),
            precision=str(entry["precision"]),
            needs_calibration=bool(entry["needs_calibration"]),
        )
        if pipeline.id in seen:
            raise RegressionError(f"{path}: duplicate pipeline id {pipeline.id}")
        seen.add(pipeline.id)
        if pipeline.num in seen_numbers:
            raise RegressionError(f"{path}: duplicate pipeline num {pipeline.num}")
        seen_numbers.add(pipeline.num)
        if pipeline.runtime not in data["artifact_ext"]:
            raise RegressionError(
                f"{path}: no artifact_ext for runtime {pipeline.runtime}"
            )
        if pipeline.runtime not in data["qc_backend"]:
            raise RegressionError(
                f"{path}: no qc_backend for runtime {pipeline.runtime}"
            )
        pipelines.append(pipeline)
    return data, pipelines


def cross_check_pipelines(
    pipelines: list[Pipeline], mapper
) -> tuple[list[str], list[str]]:
    """Compare pipelines.yaml with the product's supported matrix.

    Returns (errors, warnings). Errors mean the yaml asks for something the product rejects;
    warnings mean the product supports a combination the yaml does not cover.
    """
    errors: list[str] = []
    warnings: list[str] = []
    supported = set(mapper.SUPPORTED_RUNTIME_PRECISION_COMBINATIONS)
    for p in pipelines:
        if p.type not in mapper.MODEL_TYPES:
            errors.append(
                f"{p.id}: model type {p.type} is not in MODEL_TYPES {mapper.MODEL_TYPES}"
            )
        if (p.runtime, p.precision) not in supported:
            errors.append(
                f"{p.id}: {p.runtime}/{p.precision} is not a supported runtime/precision"
            )
            continue
        expected = mapper.qc_requires_calibration(p.runtime, p.precision)
        if p.needs_calibration != expected:
            errors.append(
                f"{p.id}: needs_calibration={p.needs_calibration} but the product says {expected}"
            )
    covered = {(p.type, p.runtime, p.precision) for p in pipelines}
    for model_type in mapper.MODEL_TYPES:
        for runtime, precision in sorted(supported):
            if (model_type, runtime, precision) not in covered:
                warnings.append(
                    f"coverage drift: {model_type}/{runtime}/{precision} is supported "
                    "but has no pipeline in pipelines.yaml"
                )
    return errors, warnings


# ----------------------------
# Inputs
# ----------------------------


def _as_path(value: object, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _as_flag_list(value: object, where: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(v) for v in value]
    raise RegressionError(f"{where} must be a list of strings or a single string")


@dataclass
class Inputs:
    source: Path
    repo_root: Path
    docker_image: str
    pipeline_patterns: list[str]
    modes: list[str]
    models_pt: dict[str, Path]
    calib_dir: Path | None
    max_calib: int | None
    map_annotations: Path | None
    map_images: Path | None
    map_max_images: int | None
    test_image: Path | None
    test_images: Path | None
    test_yaml: Path | None
    adb_serial: str | None
    hub_use_existing_login: bool
    prebuilt_models: dict[str, Path]
    extra_flags: dict[str, list[str]]
    pipeline_extra_flags: dict[str, dict[str, list[str]]]
    timeouts_s: dict[str, int]
    output_dir: Path

    def to_dict(self) -> dict:
        def s(p: Path | None) -> str | None:
            return str(p) if p else None

        return {
            "source": str(self.source),
            "repo_root": str(self.repo_root),
            "docker_image": self.docker_image,
            "pipelines": self.pipeline_patterns,
            "modes": self.modes,
            "models_pt": {k: str(v) for k, v in self.models_pt.items()},
            "calibration": {"dir": s(self.calib_dir), "max_calib": self.max_calib},
            "map": {
                "annotations": s(self.map_annotations),
                "images": s(self.map_images),
                "max_images": self.map_max_images,
            },
            "test": {
                "image": s(self.test_image),
                "images": s(self.test_images),
                "yaml": s(self.test_yaml),
            },
            "device": {"adb_serial": self.adb_serial},
            "qai_hub": {
                "use_existing_login": self.hub_use_existing_login,
                "token": "(never stored)",
            },
            "prebuilt_models": {k: str(v) for k, v in self.prebuilt_models.items()},
            "extra_flags": self.extra_flags,
            "pipeline_extra_flags": self.pipeline_extra_flags,
            "timeouts_s": self.timeouts_s,
            "output_dir": str(self.output_dir),
        }


def _optional_int(value: object, where: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise RegressionError(f"{where} must be an integer or null") from None
    if number <= 0:
        raise RegressionError(f"{where} must be > 0")
    return number


def load_inputs(path: Path, defaults: dict) -> Inputs:
    if not path.is_file():
        raise RegressionError(f"inputs file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RegressionError(f"{path}: top level must be a mapping")
    for key in raw:
        if "token" in str(key).lower() or "api_key" in str(key).lower():
            raise RegressionError(
                f"{path}: never store the QAI Hub token in the inputs file; "
                f"pass it through ${TOKEN_ENV} to `hub-login`"
            )
    hub_raw = raw.get("qai_hub") or {}
    if any(k for k in hub_raw if k != "use_existing_login"):
        raise RegressionError(
            f"{path}: qai_hub only accepts use_existing_login; pass the token via ${TOKEN_ENV}"
        )

    repo_root = _as_path(raw.get("repo_root"), path.parent)
    if repo_root is None:
        raise RegressionError(f"{path}: repo_root is required")
    if not (repo_root / "docker" / "iqf").is_file():
        raise RegressionError(
            f"{path}: repo_root {repo_root} does not contain docker/iqf"
        )

    docker_image = str(raw.get("docker_image") or "").strip()
    if not docker_image:
        raise RegressionError(f"{path}: docker_image is required")

    patterns = raw.get("pipelines")
    if patterns in (None, "", []):
        raise RegressionError(
            f"{path}: pipelines is required (ids, glob patterns or 'all')"
        )
    if not isinstance(patterns, list):
        patterns = [patterns]
    patterns = [str(p).strip() for p in patterns]

    modes_raw = raw.get("modes") or list(defaults.get("modes", MODES))
    if isinstance(modes_raw, str):
        modes_raw = [m.strip() for m in modes_raw.split(",")]
    unknown = [m for m in modes_raw if m not in MODES]
    if unknown:
        raise RegressionError(f"{path}: unknown modes {unknown}; use {list(MODES)}")
    modes = [m for m in MODES if m in modes_raw]

    models_pt = {
        str(k): p
        for k, v in (raw.get("models_pt") or {}).items()
        if (p := _as_path(v, repo_root)) is not None
    }
    calibration = raw.get("calibration") or {}
    map_raw = raw.get("map") or {}
    test_raw = raw.get("test") or {}
    device_raw = raw.get("device") or {}

    extra_raw = raw.get("extra_flags") or {}
    extra_flags = {
        m: _as_flag_list(extra_raw.get(m), f"extra_flags.{m}") for m in MODES
    }
    pipeline_extra: dict[str, dict[str, list[str]]] = {}
    for pid, per_mode in (raw.get("pipeline_extra_flags") or {}).items():
        pipeline_extra[str(pid)] = {
            m: _as_flag_list((per_mode or {}).get(m), f"pipeline_extra_flags.{pid}.{m}")
            for m in MODES
        }

    timeouts = {m: int(v) for m, v in (defaults.get("timeouts_s") or {}).items()}
    for m, v in (raw.get("timeouts_s") or {}).items():
        if m not in MODES:
            raise RegressionError(f"{path}: timeouts_s.{m} is not a mode")
        timeouts[m] = int(v)

    output_dir = _as_path(raw.get("output_dir"), repo_root)
    if output_dir is None:
        output_dir = path.parent.resolve()

    serial = device_raw.get("adb_serial")
    return Inputs(
        source=path.resolve(),
        repo_root=repo_root,
        docker_image=docker_image,
        pipeline_patterns=patterns,
        modes=modes,
        models_pt=models_pt,
        calib_dir=_as_path(calibration.get("dir"), repo_root),
        max_calib=_optional_int(calibration.get("max_calib"), "calibration.max_calib"),
        map_annotations=_as_path(map_raw.get("annotations"), repo_root),
        map_images=_as_path(map_raw.get("images"), repo_root),
        map_max_images=_optional_int(map_raw.get("max_images"), "map.max_images"),
        test_image=_as_path(test_raw.get("image"), repo_root),
        test_images=_as_path(test_raw.get("images"), repo_root),
        test_yaml=_as_path(test_raw.get("yaml"), repo_root),
        adb_serial=str(serial).strip() if serial not in (None, "") else None,
        hub_use_existing_login=bool(hub_raw.get("use_existing_login", True)),
        prebuilt_models={
            str(k): p
            for k, v in (raw.get("prebuilt_models") or {}).items()
            if (p := _as_path(v, repo_root)) is not None
        },
        extra_flags=extra_flags,
        pipeline_extra_flags=pipeline_extra,
        timeouts_s=timeouts,
        output_dir=output_dir,
    )


RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


def _match_selector(selector: str, pipelines: list[Pipeline]) -> list[Pipeline]:
    """One selector: a pipeline number (``5``), a number range (``1-7``), an id or a glob."""
    if selector.isdigit():
        return [p for p in pipelines if p.num == int(selector)]
    span = RANGE_RE.match(selector)
    if span:
        low, high = sorted((int(span.group(1)), int(span.group(2))))
        return [p for p in pipelines if low <= p.num <= high]
    return [p for p in pipelines if fnmatch.fnmatchcase(p.id, selector)]


def select_pipelines(
    patterns: list[str | int], pipelines: list[Pipeline]
) -> list[Pipeline]:
    selectors = [str(p).strip() for p in patterns]
    if any(s.lower() == "all" for s in selectors):
        return list(pipelines)
    selected: list[Pipeline] = []
    for selector in selectors:
        matches = _match_selector(selector, pipelines)
        if not matches:
            raise RegressionError(
                f"pipeline '{selector}' matches nothing. Use a pipeline no (1-"
                f"{max(p.num for p in pipelines)}), a range like 1-7, an id, a glob, or all; "
                "see `run_pipelines.py list`"
            )
        for match in matches:
            if match not in selected:
                selected.append(match)
    # Keep the definition order so reports are stable regardless of how they were typed.
    order = {p.id: i for i, p in enumerate(pipelines)}
    return sorted(selected, key=lambda p: order[p.id])


# ----------------------------
# Stage planning
# ----------------------------


@dataclass
class StagePlan:
    pipeline: Pipeline
    mode: str
    selected: bool
    argv: list[str]
    model_from_qc: bool
    uses_adb: bool
    artifact: Path | None  # qc output file, mAP result txt or test output dir
    skip_reason: str | None = None


def run_layout(run_dir: Path, pipeline: Pipeline, ext: str) -> dict[str, Path]:
    base = run_dir / "artifacts" / pipeline.id
    return {
        "qc_dir": base / "qc",
        "qc_artifact": base / "qc" / f"{pipeline.id}{ext}",
        "map_dir": base / "mAP",
        "map_result": base / "mAP" / "mAP_result.txt",
        "test_dir": base / "test",
        "test_output": base / "test" / "output",
        "logs": run_dir / "logs" / pipeline.id,
    }


def _extra_for(inputs: Inputs, pipeline: Pipeline, mode: str) -> list[str]:
    extra = list(inputs.extra_flags.get(mode, []))
    extra.extend(inputs.pipeline_extra_flags.get(pipeline.id, {}).get(mode, []))
    return extra


def converted_model_for(
    inputs: Inputs, pipeline: Pipeline, layout: dict[str, Path]
) -> tuple[Path, bool]:
    prebuilt = inputs.prebuilt_models.get(pipeline.id)
    if prebuilt is not None:
        return prebuilt, False
    return layout["qc_artifact"], True


def build_stage_plans(
    inputs: Inputs, config: dict, pipeline: Pipeline, run_dir: Path
) -> list[StagePlan]:
    ext = config["artifact_ext"][pipeline.runtime]
    layout = run_layout(run_dir, pipeline, ext)
    wrapper = ["./docker/iqf", "--image", inputs.docker_image, "run"]
    common = [
        "--type",
        pipeline.type,
        "--runtime",
        pipeline.runtime,
        "--precision",
        pipeline.precision,
    ]
    model_path, model_from_qc = converted_model_for(inputs, pipeline, layout)
    pt_model = inputs.models_pt.get(pipeline.type)
    plans: list[StagePlan] = []

    # qc
    qc_argv = [
        *wrapper,
        "qc",
        *common,
        "--model",
        str(pt_model) if pt_model else "<missing .pt>",
    ]
    if pipeline.needs_calibration:
        qc_argv += [
            "--calib_dir",
            str(inputs.calib_dir) if inputs.calib_dir else "<missing calib dir>",
        ]
        if inputs.max_calib:
            qc_argv += ["--max_calib", str(inputs.max_calib)]
    qc_argv += ["--output", str(layout["qc_artifact"])]
    qc_argv += _extra_for(inputs, pipeline, "qc")
    qc_skip = None
    if "qc" not in inputs.modes:
        qc_skip = "qc not selected"
    elif not model_from_qc:
        qc_skip = "prebuilt converted model supplied"
    plans.append(
        StagePlan(
            pipeline=pipeline,
            mode="qc",
            selected=qc_skip is None,
            argv=qc_argv,
            model_from_qc=False,
            uses_adb=False,
            artifact=layout["qc_artifact"],
            skip_reason=qc_skip,
        )
    )

    # mAP
    map_argv = [
        *wrapper,
        "mAP",
        *common,
        "--annotations",
        str(inputs.map_annotations)
        if inputs.map_annotations
        else "<missing annotations>",
        "--images",
        str(inputs.map_images) if inputs.map_images else "<missing mAP images>",
        "--reference-model",
        str(pt_model) if pt_model else "<missing .pt>",
        "--converted-model",
        str(model_path),
        "--output_text",
        str(layout["map_result"]),
    ]
    if inputs.map_max_images:
        map_argv += ["--max-images", str(inputs.map_max_images)]
    if inputs.adb_serial:
        map_argv += ["--adb-serial", inputs.adb_serial]
    map_argv += _extra_for(inputs, pipeline, "mAP")
    plans.append(
        StagePlan(
            pipeline=pipeline,
            mode="mAP",
            selected="mAP" in inputs.modes,
            argv=map_argv,
            model_from_qc=model_from_qc,
            uses_adb=True,
            artifact=layout["map_result"],
            skip_reason=None if "mAP" in inputs.modes else "mAP not selected",
        )
    )

    # test
    test_argv = [*wrapper, "test", *common, "--model", str(model_path)]
    test_argv += [
        "--yaml",
        str(inputs.test_yaml) if inputs.test_yaml else "<missing class yaml>",
    ]
    if inputs.test_image:
        test_argv += ["--image", str(inputs.test_image)]
    else:
        test_argv += [
            "--images",
            str(inputs.test_images) if inputs.test_images else "<missing test images>",
        ]
    test_argv += ["--output", str(layout["test_output"]), "--adb"]
    if inputs.adb_serial:
        test_argv += ["--adb-serial", inputs.adb_serial]
    test_argv += _extra_for(inputs, pipeline, "test")
    plans.append(
        StagePlan(
            pipeline=pipeline,
            mode="test",
            selected="test" in inputs.modes,
            argv=test_argv,
            model_from_qc=model_from_qc,
            uses_adb=True,
            artifact=layout["test_output"],
            skip_reason=None if "test" in inputs.modes else "test not selected",
        )
    )
    return plans


# ----------------------------
# Environment probes
# ----------------------------


def run_capture(
    argv: list[str], cwd: Path | None = None, timeout: int = 120
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, f"{argv[0]}: command not found"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s: {shlex.join(argv)}"
    return completed.returncode, strip_ansi(
        (completed.stdout or "") + (completed.stderr or "")
    )


def parse_adb_devices(output: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices") or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        entry = {"serial": parts[0], "state": parts[1]}
        for extra in parts[2:]:
            if ":" in extra:
                key, value = extra.split(":", 1)
                entry[key] = value
        devices.append(entry)
    return devices


def host_adb_kill_server() -> None:
    # docs/mAP_mode.md and docs/test_mode.md: a host adb server holding the USB interface hides
    # the device from the container, so release it before every containerized adb stage.
    if shutil.which("adb"):
        run_capture(["adb", "kill-server"], timeout=30)


def collect_environment(inputs: Inputs) -> dict:
    env: dict[str, object] = {
        "runner_version": RUNNER_VERSION,
        "host": platform.node(),
        "host_platform": platform.platform(),
        "host_python": platform.python_version(),
    }
    repo = inputs.repo_root
    _, branch = run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    _, commit = run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
    _, porcelain = run_capture(["git", "status", "--porcelain"], cwd=repo)
    env["git_branch"] = branch.strip()
    env["git_commit"] = commit.strip()
    env["git_dirty_entries"] = len(
        [line for line in porcelain.splitlines() if line.strip()]
    )
    code, image_id = run_capture(
        ["docker", "image", "inspect", inputs.docker_image, "--format", "{{.Id}}"]
    )
    env["docker_image"] = inputs.docker_image
    env["docker_image_id"] = image_id.strip() if code == 0 else None
    return env


def check_qai_hub_login(inputs: Inputs) -> tuple[bool, str]:
    """Ask QAI Hub for the target device with the saved client.ini. Read-only network call."""
    ini_dir = Path.home() / ".qai_hub"
    snippet = (
        "import qai_hub as hub; "
        "d = hub.get_devices(name='Dragonwing IQ-9075 EVK'); "
        "print('hub_devices_found=%d' % len(d))"
    )
    code, output = run_capture(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{ini_dir}:/root/.qai_hub:ro",
            "--entrypoint",
            "python3",
            inputs.docker_image,
            "-c",
            snippet,
        ],
        timeout=180,
    )
    lines = [line for line in output.strip().splitlines() if line.strip()]
    last = lines[-1] if lines else "(no output)"
    return code == 0 and "hub_devices_found=" in output, last


# ----------------------------
# Validation
# ----------------------------


def _check_path(path: Path | None, kind: str, label: str, errors: list[str]) -> bool:
    if path is None:
        errors.append(f"{label}: not provided")
        return False
    if kind == "file" and not path.is_file():
        errors.append(f"{label}: file not found: {path}")
        return False
    if kind == "dir" and not path.is_dir():
        errors.append(f"{label}: directory not found: {path}")
        return False
    if kind == "file_or_dir" and not path.exists():
        errors.append(f"{label}: path not found: {path}")
        return False
    return True


def count_images(directory: Path) -> int:
    return sum(
        1
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _controlled_flag_errors(inputs: Inputs) -> list[str]:
    errors: list[str] = []

    def check(flags: list[str], mode: str, where: str) -> None:
        forbidden = CONTROLLED_FLAGS["common"] | CONTROLLED_FLAGS[mode]
        for token in flags:
            name = token.split("=", 1)[0]
            if name in forbidden:
                errors.append(
                    f"{where}: {name} is set by the runner from the inputs file; remove it"
                )

    for mode in MODES:
        check(inputs.extra_flags.get(mode, []), mode, f"extra_flags.{mode}")
    for pid, per_mode in inputs.pipeline_extra_flags.items():
        for mode in MODES:
            check(per_mode.get(mode, []), mode, f"pipeline_extra_flags.{pid}.{mode}")
    return errors


def parse_dry_run_inner(output: str) -> list[str] | None:
    lines = strip_ansi(output).splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "Inner command:" and index + 1 < len(lines):
            try:
                return shlex.split(lines[index + 1].strip())
            except ValueError:
                return None
    return None


def injected_flags(stage_argv: list[str], inner: list[str] | None) -> list[str]:
    """Path flags present in the wrapper's inner cli.py command that the runner did not pass.

    docker/iqf fills missing required paths from .iqf/docker-paths.json; the report surfaces
    that so a reader knows the run used a value that did not come from the inputs file.
    """
    if not inner:
        return []
    passed = {token for token in stage_argv if token.startswith("--")}
    wrapper_added = {"--mode", "--adb"}
    return sorted(
        {
            token
            for token in inner
            if token.startswith("--")
            and token not in passed
            and token not in wrapper_added
        }
    )


@dataclass
class _Validation:
    """Accumulates validation findings; each _check_* helper fills one area."""

    inputs: Inputs
    config: dict
    selected: list[Pipeline]
    quiet: bool
    global_errors: list[str]
    global_warnings: list[str]
    checks: dict[str, object]
    shared_errors: dict[str, list[str]]

    def say(self, message: str) -> None:
        if not self.quiet:
            info(message)

    def wants(self, mode: str) -> bool:
        return mode in self.inputs.modes


def _check_docker(v: _Validation) -> None:
    wrapper = v.inputs.repo_root / "docker" / "iqf"
    if not os.access(wrapper, os.X_OK):
        v.global_errors.append(f"{wrapper} is not executable")
    code, output = run_capture(["docker", "info", "--format", "{{.ServerVersion}}"])
    if code != 0:
        tail = output.strip().splitlines()[-1:] or [output]
        v.global_errors.append(f"docker is not usable: {tail[0]}")
        return
    v.checks["docker_server"] = output.strip()
    code, image_id = run_capture(
        ["docker", "image", "inspect", v.inputs.docker_image, "--format", "{{.Id}}"]
    )
    if code != 0:
        v.global_errors.append(
            f"docker image not found locally: {v.inputs.docker_image}"
        )
        return
    v.checks["docker_image_id"] = image_id.strip()
    v.say(f"docker image {v.inputs.docker_image} -> {image_id.strip()[:19]}")


def _check_models_pt(v: _Validation) -> None:
    for model_type in sorted({p.type for p in v.selected}):
        errs: list[str] = []
        path = v.inputs.models_pt.get(model_type)
        if (
            _check_path(path, "file", f"models_pt.{model_type}", errs)
            and path.suffix != ".pt"
        ):
            errs.append(f"models_pt.{model_type}: expected a .pt file")
        v.shared_errors[f"pt:{model_type}"] = errs


def _check_image_dir(
    v: _Validation, path: Path | None, label: str, key: str
) -> list[str]:
    errs: list[str] = []
    if _check_path(path, "dir", label, errs):
        n = count_images(path)
        v.checks[key] = n
        if n == 0:
            errs.append(f"{label} has no images: {path}")
        else:
            v.say(f"{label}: {n} images in {path}")
    return errs


def _check_calibration(v: _Validation) -> None:
    errs: list[str] = []
    if v.wants("qc") and any(p.needs_calibration for p in v.selected):
        errs = _check_image_dir(
            v, v.inputs.calib_dir, "calibration.dir", "calibration_images"
        )
    v.shared_errors["calib"] = errs


def _check_map_inputs(v: _Validation) -> None:
    errs: list[str] = []
    if v.wants("mAP"):
        annotations = v.inputs.map_annotations
        if (
            _check_path(annotations, "file_or_dir", "map.annotations", errs)
            and annotations.is_file()
            and annotations.suffix == ".json"
        ):
            try:
                coco = json.loads(annotations.read_text(encoding="utf-8"))
                v.checks["map_annotation_images"] = len(coco.get("images", []))
                v.checks["map_annotation_boxes"] = len(coco.get("annotations", []))
                v.checks["map_annotation_categories"] = len(coco.get("categories", []))
                v.say(
                    f"map.annotations: {v.checks['map_annotation_images']} images, "
                    f"{v.checks['map_annotation_boxes']} boxes, "
                    f"{v.checks['map_annotation_categories']} categories"
                )
            except (OSError, ValueError) as exc:
                errs.append(f"map.annotations is not valid JSON: {exc}")
        errs += _check_image_dir(v, v.inputs.map_images, "map.images", "map_images")
    v.shared_errors["map"] = errs


def _check_test_inputs(v: _Validation) -> None:
    errs: list[str] = []
    if v.wants("test"):
        inputs = v.inputs
        if bool(inputs.test_image) == bool(inputs.test_images):
            errs.append("test: set exactly one of test.image or test.images")
        elif inputs.test_image:
            _check_path(inputs.test_image, "file", "test.image", errs)
        else:
            errs += _check_image_dir(
                v, inputs.test_images, "test.images", "test_images"
            )
        if _check_path(inputs.test_yaml, "file", "test.yaml", errs):
            try:
                data = (
                    yaml.safe_load(inputs.test_yaml.read_text(encoding="utf-8")) or {}
                )
                names = data.get("names") if isinstance(data, dict) else None
                if not names:
                    errs.append(f"test.yaml has no 'names': {inputs.test_yaml}")
                else:
                    v.checks["test_yaml_classes"] = len(names)
                    v.say(f"test.yaml: {len(names)} classes in {inputs.test_yaml}")
            except (OSError, yaml.YAMLError) as exc:
                errs.append(f"test.yaml is not valid YAML: {exc}")
    v.shared_errors["test"] = errs


def _check_qai_hub(v: _Validation, skip_hub_check: bool) -> None:
    ini_errs: list[str] = []
    login_errs: list[str] = []
    builds = [
        p for p in v.selected if v.wants("qc") and p.id not in v.inputs.prebuilt_models
    ]
    client_ini = Path.home() / ".qai_hub" / "client.ini"
    if builds:
        if client_ini.is_file():
            v.say(f"QAI Hub config present: {client_ini}")
            for message in hub_file_permission_warnings():
                v.global_warnings.append(message)
        else:
            ini_errs.append(
                f"{client_ini} not found (docker/iqf mounts it for every qc run); "
                f"run `hub-login` with ${TOKEN_ENV}"
            )
    needs_hub = [p for p in builds if v.config["qc_backend"][p.runtime] == "qai_hub"]
    if needs_hub and not ini_errs and "docker_image_id" in v.checks:
        if skip_hub_check:
            v.global_warnings.append("QAI Hub login check skipped (--skip-hub-check)")
        else:
            hub_ok, detail = check_qai_hub_login(v.inputs)
            v.checks["qai_hub_login"] = detail
            if hub_ok:
                v.say(f"QAI Hub login works ({detail})")
            else:
                login_errs.append(f"QAI Hub login check failed: {detail}")
    v.shared_errors["hub_ini"] = ini_errs
    v.shared_errors["hub_login"] = login_errs


def _list_adb_devices(v: _Validation) -> tuple[list[dict[str, str]], str]:
    if shutil.which("adb"):
        code, output = run_capture(["adb", "devices", "-l"], timeout=60)
        devices = parse_adb_devices(output) if code == 0 else []
        if devices:
            return devices, "host"
    if "docker_image_id" not in v.checks:
        return [], "none"
    code, output = run_capture(
        [
            "docker",
            "run",
            "--rm",
            "--privileged",
            "-v",
            "/dev/bus/usb:/dev/bus/usb",
            "--entrypoint",
            "bash",
            v.inputs.docker_image,
            "-lc",
            "adb start-server >/dev/null 2>&1; adb devices -l",
        ],
        timeout=120,
    )
    return (parse_adb_devices(output) if code == 0 else []), "container"


def _check_device(v: _Validation) -> None:
    errs: list[str] = []
    v.shared_errors["device"] = errs
    if not (v.wants("mAP") or v.wants("test")):
        return
    devices, source = _list_adb_devices(v)
    v.checks["adb_devices"] = devices
    seen = [(d["serial"], d["state"]) for d in devices] or "none"
    ready = [d for d in devices if d["state"] == "device"]
    chosen: dict[str, str] | None = None
    serial = v.inputs.adb_serial
    if serial:
        chosen = next((d for d in ready if d["serial"] == serial), None)
        if chosen is None:
            errs.append(
                f"adb serial {serial} is not attached in 'device' state (seen: {seen})"
            )
    elif len(ready) == 1:
        chosen = ready[0]
    elif not ready:
        errs.append(f"no adb device in 'device' state (seen: {seen})")
    else:
        errs.append(
            f"{len(ready)} adb devices attached; set device.adb_serial to one of "
            f"{[d['serial'] for d in ready]}"
        )
    if chosen is not None:
        v.checks["adb_serial"] = chosen["serial"]
        v.checks["adb_source"] = source
        if source == "host":
            # The IQ9 target is embedded Linux over adb, not Android, so identify it from the
            # device tree and os-release instead of getprop.
            adb = ["adb", "-s", chosen["serial"], "shell"]
            probes = {
                "device_model": "tr -d '\\0' < /proc/device-tree/model 2>/dev/null",
                "device_os": '. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"',
                "device_python": "python3 --version 2>/dev/null",
            }
            for key, probe in probes.items():
                code, value = run_capture([*adb, probe], timeout=30)
                v.checks[key] = value.strip() if code == 0 and value.strip() else None
        label = v.checks.get("device_model") or chosen.get("model", "?")
        v.say(f"adb device: {chosen['serial']} ({label}) via {source}")
    host_adb_kill_server()
    v.say(
        "host adb server released (adb kill-server) so the container can own the device"
    )


def _stage_errors(v: _Validation, p: Pipeline) -> dict[str, list[str]]:
    shared = v.shared_errors
    pt = shared.get(f"pt:{p.type}", [])
    errors = {
        "qc": [
            *pt,
            *(shared["calib"] if p.needs_calibration else []),
            *shared["hub_ini"],
        ],
        "mAP": [*pt, *shared["map"], *shared["device"]],
        "test": [*shared["test"], *shared["device"]],
    }
    if v.config["qc_backend"][p.runtime] == "qai_hub":
        errors["qc"] += shared["hub_login"]
    prebuilt = v.inputs.prebuilt_models.get(p.id)
    if prebuilt is not None:
        ext = v.config["artifact_ext"][p.runtime]
        errs: list[str] = []
        if _check_path(prebuilt, "file", f"prebuilt_models.{p.id}", errs) and not (
            prebuilt.name.endswith((ext, ext + ".zip"))
        ):
            errs.append(
                f"prebuilt_models.{p.id}: expected a {ext} file for {p.runtime}"
            )
        errors["mAP"] += errs
        errors["test"] += errs
    elif not v.wants("qc"):
        msg = "qc is not selected and no prebuilt converted model was supplied"
        errors["mAP"].append(msg)
        errors["test"].append(msg)
    return {mode: list(dict.fromkeys(errs)) for mode, errs in errors.items()}


def _dry_run_stage(v: _Validation, plan: StagePlan, entry: dict) -> None:
    code, output = run_capture(
        [*plan.argv, "--dry-run"], cwd=v.inputs.repo_root, timeout=120
    )
    inner = parse_dry_run_inner(output)
    entry["inner_command"] = shlex.join(inner) if inner else None
    entry["injected_flags"] = injected_flags(plan.argv, inner)
    if code == 0:
        entry["dry_run"] = "ok"
    else:
        tail = [line for line in output.strip().splitlines() if line.strip()]
        entry["dry_run"] = "failed"
        entry["feasibility"] = NOT_FEASIBLE
        entry["errors"] = [f"wrapper --dry-run exit {code}: {tail[-1] if tail else ''}"]
    if entry["injected_flags"] and not v.quiet:
        warn(
            f"{plan.pipeline.id}:{plan.mode}: docker/iqf added {entry['injected_flags']} "
            "from .iqf/docker-paths.json (saved configure paths); the run will use them"
        )


def _plan_stages(v: _Validation, p: Pipeline, run_dir: Path) -> dict[str, dict]:
    layout = run_layout(run_dir, p, v.config["artifact_ext"][p.runtime])
    for parent in (layout["qc_dir"], layout["map_dir"], layout["test_dir"]):
        parent.mkdir(parents=True, exist_ok=True)
    errors = _stage_errors(v, p)
    stages: dict[str, dict] = {}
    qc_feasible = True
    for plan in build_stage_plans(v.inputs, v.config, p, run_dir):
        entry: dict = {
            "selected": plan.selected,
            "command": shlex.join(plan.argv),
            "errors": errors[plan.mode] if plan.selected else [],
            "dry_run": None,
            "inner_command": None,
            "injected_flags": [],
            "skip_reason": plan.skip_reason,
        }
        if not plan.selected:
            entry["feasibility"] = SKIPPED
        elif entry["errors"]:
            entry["feasibility"] = NOT_FEASIBLE
        elif plan.model_from_qc and not qc_feasible:
            entry["feasibility"] = NOT_FEASIBLE
            entry["errors"] = ["qc for this pipeline is not feasible"]
        else:
            entry["feasibility"] = "OK"
            if plan.model_from_qc:
                entry["dry_run"] = "deferred (model produced by qc)"
            else:
                _dry_run_stage(v, plan, entry)
        if plan.mode == "qc" and plan.selected and entry["feasibility"] != "OK":
            qc_feasible = False
        stages[plan.mode] = entry
    return stages


def validate(
    inputs: Inputs,
    config: dict,
    pipelines: list[Pipeline],
    run_dir: Path,
    skip_hub_check: bool = False,
    quiet: bool = False,
) -> dict:
    """Check everything a person would check before starting, without running a stage."""
    selected = select_pipelines(inputs.pipeline_patterns, pipelines)
    v = _Validation(
        inputs=inputs,
        config=config,
        selected=selected,
        quiet=quiet,
        global_errors=[],
        global_warnings=[],
        checks={},
        shared_errors={},
    )
    yaml_errors, yaml_warnings = cross_check_pipelines(
        pipelines, load_mapper(inputs.repo_root)
    )
    v.global_errors += [f"pipelines.yaml: {e}" for e in yaml_errors]
    v.global_warnings += yaml_warnings
    v.global_errors += _controlled_flag_errors(inputs)
    selected_ids = {p.id for p in selected}
    for section, keys in (
        ("prebuilt_models", inputs.prebuilt_models),
        ("pipeline_extra_flags", inputs.pipeline_extra_flags),
    ):
        for pid in keys:
            if pid not in selected_ids:
                v.global_warnings.append(
                    f"{section}.{pid}: pipeline not selected; ignored"
                )
    v.say(
        f"selected {len(selected)} pipeline(s): "
        + ", ".join(f"{p.num}:{p.id}" for p in selected)
    )
    v.say(f"modes: {', '.join(inputs.modes)}")

    _check_docker(v)
    _check_models_pt(v)
    _check_calibration(v)
    _check_map_inputs(v)
    _check_test_inputs(v)
    _check_qai_hub(v, skip_hub_check)
    _check_device(v)

    stages = {p.id: _plan_stages(v, p, run_dir) for p in selected}
    all_entries = [s for per_mode in stages.values() for s in per_mode.values()]
    infeasible = sum(1 for s in all_entries if s["feasibility"] == NOT_FEASIBLE)
    return {
        "validated_at": now_iso(),
        "inputs": inputs.to_dict(),
        "global_errors": v.global_errors,
        "global_warnings": v.global_warnings,
        "checks": v.checks,
        "pipeline_numbers": {p.id: p.num for p in selected},
        "stages": stages,
        "summary": {
            "pipelines": len(selected),
            "stages_ok": sum(1 for s in all_entries if s["feasibility"] == "OK"),
            "stages_not_feasible": infeasible,
        },
        "ok": not v.global_errors and infeasible == 0,
    }


def print_validation(result: dict) -> None:
    rows = []
    numbers = result.get("pipeline_numbers") or {}
    for pid, per_mode in result["stages"].items():
        for mode in MODES:
            entry = per_mode[mode]
            reason = (
                "; ".join(entry["errors"])
                or entry.get("skip_reason")
                or entry.get("dry_run")
                or ""
            )
            rows.append(
                [str(numbers.get(pid, "-")), pid, mode, entry["feasibility"], reason]
            )
    print()
    print(format_table(["NO", "PIPELINE", "MODE", "FEASIBILITY", "DETAIL"], rows))
    print()
    for message in result["global_warnings"]:
        warn(message)
    for message in result["global_errors"]:
        error(message)
    summary = result["summary"]
    line = (
        f"{summary['pipelines']} pipeline(s), {summary['stages_ok']} stage(s) OK, "
        f"{summary['stages_not_feasible']} NOT_FEASIBLE, {len(result['global_errors'])} global error(s)"
    )
    if result["ok"]:
        ok(f"validation passed: {line}")
    else:
        error(f"validation failed: {line}")


# ----------------------------
# Stage execution
# ----------------------------


def docker_container_ids(image: str) -> set[str]:
    code, output = run_capture(
        ["docker", "ps", "-q", "--filter", f"ancestor={image}"], timeout=30
    )
    return set(output.split()) if code == 0 else set()


def execute_stage(
    argv: list[str],
    cwd: Path,
    log_path: Path,
    timeout_s: int,
    prefix: str,
    image: str,
    header: list[str],
) -> tuple[int | None, str, float]:
    """Run one wrapper command, tee its output, enforce the timeout.

    Returns (exit_code, outcome, duration_s) where outcome is "exited", "timeout" or "interrupted".
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    before = docker_container_ids(image)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        for line in header:
            log.write(f"# {line}\n")
        log.write("#" + "-" * 78 + "\n")
        log.flush()
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )

        def pump() -> None:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                log.write(strip_ansi(raw_line))
                log.flush()
                sys.stdout.write(f"[{prefix}] {raw_line}")
                sys.stdout.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        outcome = "exited"
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            outcome = "timeout"
        except KeyboardInterrupt:
            outcome = "interrupted"
        if outcome != "exited":
            _terminate(proc)
            leaked = docker_container_ids(image) - before
            for container_id in sorted(leaked):
                run_capture(["docker", "kill", container_id], timeout=60)
            note = (
                f"timed out after {timeout_s}s"
                if outcome == "timeout"
                else "interrupted by user"
            )
            log.write(
                f"\n# [regression] {note}; killed wrapper and containers {sorted(leaked) or '[]'}\n"
            )
        reader.join(timeout=10)
    duration = time.monotonic() - start
    return proc.returncode, outcome, duration


def _terminate(proc: subprocess.Popen) -> None:
    for sig, wait_s in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def parse_map_result(path: Path) -> dict[str, object]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    reference = values.get("reference_map50", values.get("fp_map50"))
    converted = values.get("converted_map50", values.get("int_map50"))
    metrics: dict[str, object] = {
        "num_images": values.get("num_images"),
        "reference_map50": float(reference) if reference else None,
        "converted_map50": float(converted) if converted else None,
        "pct_delta_vs_reference": values.get("pct_delta_vs_reference"),
        "trend": values.get("trend"),
    }
    pct = metrics["pct_delta_vs_reference"]
    match = re.match(r"^([+-]?\d+(?:\.\d+)?)%$", str(pct or ""))
    metrics["pct_delta_value"] = float(match.group(1)) if match else None
    return metrics


def parse_test_log(text: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for key in ("avg_total_inference_ms", "avg_model_invoke_ms"):
        matches = re.findall(rf"{key}=([0-9.]+|n/a|N/A)", text)
        if matches:
            value = matches[-1]
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
    return metrics


def error_excerpt(text: str) -> str | None:
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("# ")
    ]
    for line in reversed(lines):
        if "[error]" in line:
            return line.strip()
    traceback_seen = any(line.startswith("Traceback") for line in lines)
    if traceback_seen:
        for line in reversed(lines):
            if re.match(r"^[\w.]+(Error|Exception|Exit|Interrupt)\b", line.strip()):
                return line.strip()
    return lines[-1].strip() if lines else None


def evaluate_stage(
    mode: str,
    exit_code: int | None,
    outcome: str,
    artifact: Path,
    log_text: str,
    config: dict,
) -> dict[str, object]:
    """Decide the stage status from the exit code *and* the artifact it must have produced."""
    result: dict[str, object] = {
        "exit_code": exit_code,
        "metrics": {},
        "warnings": [],
        "notes": [],
    }
    if outcome == "timeout":
        result["status"] = TIMEOUT
    elif outcome == "interrupted":
        result["status"] = ABORTED
    elif exit_code != 0:
        result["status"] = FAIL
        result["notes"].append(f"exit code {exit_code}")
    else:
        result["status"] = PASS

    if mode == "qc":
        if artifact.is_file():
            result["metrics"] = {
                "artifact": str(artifact),
                "size_bytes": artifact.stat().st_size,
            }
            extra = sorted(p.name for p in artifact.parent.iterdir() if p != artifact)
            if extra:
                result["metrics"]["sidecar_files"] = extra
            if artifact.stat().st_size == 0 and result["status"] == PASS:
                result["status"] = FAIL
                result["notes"].append("qc artifact is empty")
        elif result["status"] == PASS:
            result["status"] = FAIL
            result["notes"].append(f"exit 0 but qc artifact missing: {artifact}")
    elif mode == "mAP":
        if artifact.is_file():
            metrics = parse_map_result(artifact)
            result["metrics"] = metrics
            if result["status"] == PASS:
                if (
                    metrics["reference_map50"] is None
                    or metrics["converted_map50"] is None
                ):
                    result["status"] = FAIL
                    result["notes"].append(
                        "mAP result file has no reference/converted mAP50"
                    )
                else:
                    threshold = (config["defaults"].get("mAP") or {}).get(
                        "warn_if_pct_delta_below"
                    )
                    pct = metrics.get("pct_delta_value")
                    if (
                        threshold is not None
                        and pct is not None
                        and pct < float(threshold)
                    ):
                        result["warnings"].append(
                            f"converted mAP50 is {pct:+.2f}% vs reference (threshold {float(threshold):+.2f}%)"
                        )
        elif result["status"] == PASS:
            result["status"] = FAIL
            result["notes"].append(f"exit 0 but mAP result file missing: {artifact}")
    elif mode == "test":
        metrics = parse_test_log(log_text)
        if artifact.is_dir():
            files = [p for p in artifact.iterdir() if p.is_file()]
            images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
            labels = [
                p for p in files if p.suffix == ".txt" and p.name != "classes.txt"
            ]
            metrics.update(
                {
                    "output_dir": str(artifact),
                    "annotated_images": len(images),
                    "label_files": len(labels),
                }
            )
            if result["status"] == PASS and not [
                p for p in files if p.name != "classes.txt"
            ]:
                result["status"] = FAIL
                result["notes"].append(
                    "test output has no artifacts besides classes.txt"
                )
        elif result["status"] == PASS:
            result["status"] = FAIL
            result["notes"].append(f"exit 0 but test output dir missing: {artifact}")
        result["metrics"] = metrics

    for pattern in config["defaults"].get("warning_patterns") or []:
        if pattern in log_text:
            result["warnings"].append(f"log contains '{pattern}'")
    if result["status"] == PASS and result["warnings"]:
        result["status"] = PASS_WITH_WARNINGS
    if result["status"] in {FAIL, TIMEOUT, ABORTED}:
        result["error_excerpt"] = error_excerpt(log_text)
        tail = [line for line in log_text.splitlines() if line.strip()][-20:]
        result["log_tail"] = tail
    return result


# ----------------------------
# Results & reports
# ----------------------------


def results_path(run_dir: Path) -> Path:
    return run_dir / "results.json"


def write_results(run_dir: Path, results: dict) -> None:
    path = results_path(run_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_results(run_dir: Path) -> dict:
    path = results_path(run_dir)
    if not path.is_file():
        raise RegressionError(f"no results.json in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def overall_status(stages: dict[str, dict]) -> str:
    statuses = [stages[m]["status"] for m in MODES if m in stages]
    for status in OVERALL_PRIORITY:
        if status in statuses:
            return status
    return SKIPPED


def key_result(mode: str, stage: dict) -> str:
    metrics = stage.get("metrics") or {}
    status = stage.get("status")
    if status in {SKIPPED, BLOCKED, NOT_FEASIBLE, PENDING}:
        return stage.get("reason") or "-"
    if mode == "qc" and metrics.get("artifact"):
        text = f"{Path(metrics['artifact']).name} ({format_size(metrics.get('size_bytes'))})"
        if metrics.get("sidecar_files"):
            text += f" + {', '.join(metrics['sidecar_files'])}"
        return text
    if mode == "mAP" and metrics.get("converted_map50") is not None:
        ref = metrics.get("reference_map50")
        conv = metrics.get("converted_map50")
        return (
            f"ref {ref:.4f} -> conv {conv:.4f} ({metrics.get('pct_delta_vs_reference')}, "
            f"{metrics.get('num_images')} imgs)"
        )
    if mode == "test" and "annotated_images" in metrics:
        parts = [f"{metrics['annotated_images']} annotated imgs"]
        if isinstance(metrics.get("avg_total_inference_ms"), float):
            parts.append(f"avg total {metrics['avg_total_inference_ms']:.1f} ms")
        if isinstance(metrics.get("avg_model_invoke_ms"), float):
            parts.append(f"invoke {metrics['avg_model_invoke_ms']:.1f} ms")
        return ", ".join(parts)
    return stage.get("error_excerpt") or "; ".join(stage.get("notes") or []) or "-"


def render_report(results: dict) -> str:
    env = results.get("environment") or {}
    validation = results.get("validation_checks") or {}
    inputs = results.get("inputs") or {}
    lines: list[str] = [
        "# iQ-Foundry pipeline regression report",
        "",
        f"- **Run id:** `{results.get('run_id')}`",
        f"- **Run dir:** `{results.get('run_dir')}`",
        f"- **Started:** {results.get('started_at')}  **Finished:** {results.get('finished_at') or '(running)'}"
        f"  **Duration:** {format_duration(results.get('duration_s'))}",
        f"- **Repo:** `{inputs.get('repo_root')}` branch `{env.get('git_branch')}` "
        f"commit `{env.get('git_commit')}` ({env.get('git_dirty_entries')} uncommitted entries)",
        f"- **Docker image:** `{env.get('docker_image')}` (`{(env.get('docker_image_id') or '?')[:19]}`)",
        f"- **Device:** `{validation.get('adb_serial') or '-'}` {validation.get('device_model') or ''}"
        f" / {validation.get('device_os') or '?'}"
        f" (device python: {validation.get('device_python') or '?'})",
        f"- **Host:** {env.get('host')} / python {env.get('host_python')} / runner v{env.get('runner_version')}",
        f"- **Modes selected:** {', '.join(inputs.get('modes') or [])}",
        "",
        "## Summary",
        "",
    ]
    summary_rows = []
    totals: dict[str, int] = {}
    for pipeline in results.get("pipelines", []):
        stages = pipeline["stages"]
        for mode in MODES:
            status = stages[mode]["status"]
            totals[status] = totals.get(status, 0) + 1
        summary_rows.append(
            [
                pipeline.get("num", "-"),
                pipeline["id"],
                *(stages[m]["status"] for m in MODES),
                overall_status(stages),
            ]
        )
    lines.append(
        md_table(["No", "Pipeline", "qc", "mAP", "test", "Overall"], summary_rows)
    )
    lines.append("")
    lines.append(
        "**Stage totals:** "
        + ", ".join(f"{status} {count}" for status, count in sorted(totals.items()))
    )
    lines.append("")

    for pipeline in results.get("pipelines", []):
        lines.append(
            f"## {pipeline.get('num', '-')}. {pipeline['id']} — {pipeline['type']} / {pipeline['runtime']} / {pipeline['precision']}"
            f" — **{overall_status(pipeline['stages'])}**"
        )
        lines.append("")
        rows = []
        for mode in MODES:
            stage = pipeline["stages"][mode]
            exit_code = stage.get("exit_code")
            rows.append(
                [
                    mode,
                    stage["status"],
                    "-" if exit_code is None else str(exit_code),
                    format_duration(stage.get("duration_s")),
                    key_result(mode, stage),
                    f"`{stage['log']}`" if stage.get("log") else "-",
                ]
            )
        lines.append(
            md_table(["Mode", "Status", "Exit", "Duration", "Key result", "Log"], rows)
        )
        notes = []
        for mode in MODES:
            stage = pipeline["stages"][mode]
            for message in stage.get("warnings") or []:
                notes.append(f"- `{mode}` warning: {message}")
            if stage.get("injected_flags"):
                notes.append(
                    f"- `{mode}`: docker/iqf added {', '.join(stage['injected_flags'])} from "
                    "`.iqf/docker-paths.json`"
                )
        if notes:
            lines.append("")
            lines.extend(notes)
        lines.append("")

    failures = [
        (pipeline["id"], mode, pipeline["stages"][mode])
        for pipeline in results.get("pipelines", [])
        for mode in MODES
        if pipeline["stages"][mode]["status"] in {FAIL, TIMEOUT, ABORTED}
    ]
    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.extend(["None.", ""])
    for pid, mode, stage in failures:
        lines.append(
            f"### {pid} : {mode} — {stage['status']} (exit {stage.get('exit_code')})"
        )
        lines.append("")
        lines.append(f"- Command: `{stage.get('command')}`")
        lines.append(f"- Log: `{stage.get('log')}`")
        if stage.get("error_excerpt"):
            lines.append(f"- Error: `{stage['error_excerpt']}`")
        for note in stage.get("notes") or []:
            lines.append(f"- Note: {note}")
        if stage.get("log_tail"):
            lines.append("")
            lines.append("```text")
            lines.extend(stage["log_tail"])
            lines.append("```")
        lines.append("")

    lines.append("## Status legend")
    lines.append("")
    lines.append(
        md_table(
            ["Status", "Meaning"],
            [
                [PASS, "exit code 0 and the expected artifact was produced"],
                [
                    PASS_WITH_WARNINGS,
                    "passed, but a warning pattern appeared in the log or mAP dropped past the threshold",
                ],
                [FAIL, "non-zero exit code, or exit 0 without the expected artifact"],
                [TIMEOUT, "stage exceeded its time limit and was killed"],
                [BLOCKED, "not run because qc for the same pipeline did not pass"],
                [
                    NOT_FEASIBLE,
                    "not run because validation found a missing input or prerequisite",
                ],
                [SKIPPED, "mode not selected, or qc replaced by a prebuilt model"],
                [ABORTED, "run interrupted before or during this stage"],
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


def write_report(run_dir: Path, results: dict) -> Path:
    path = run_dir / "report.md"
    path.write_text(render_report(results), encoding="utf-8")
    return path


# ----------------------------
# Commands
# ----------------------------


def _load_context(args) -> tuple[dict, list[Pipeline], Inputs]:
    config, pipelines = load_pipelines(Path(args.pipelines).resolve())
    inputs = load_inputs(Path(args.inputs).resolve(), config.get("defaults") or {})
    return config, pipelines, inputs


def cmd_list(args) -> int:
    config, pipelines = load_pipelines(Path(args.pipelines).resolve())
    rows = [
        [
            str(p.num),
            p.id,
            p.type,
            p.runtime,
            p.precision,
            "yes" if p.needs_calibration else "no",
            config["qc_backend"][p.runtime],
            config["artifact_ext"][p.runtime],
        ]
        for p in pipelines
    ]
    print(
        format_table(
            [
                "NO",
                "PIPELINE",
                "TYPE",
                "RUNTIME",
                "PRECISION",
                "CALIB",
                "QC BACKEND",
                "ARTIFACT",
            ],
            rows,
        )
    )
    print()
    info(
        f"{len(pipelines)} pipelines; each runs modes {' -> '.join(config['defaults'].get('modes', MODES))}"
    )
    return EXIT_OK


def cmd_init(args) -> int:
    run_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (
            SCRIPT_REPO_ROOT
            / DEFAULT_RUNS_DIR
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    template = SCRIPT_REPO_ROOT / TEMPLATE_FILE
    if not template.is_file():
        raise RegressionError(f"inputs template not found: {template}")
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "inputs.yaml"
    if target.exists():
        raise RegressionError(f"{target} already exists; choose another --out")
    shutil.copyfile(template, target)
    ok(f"created run directory {run_dir}")
    ok(f"fill in {target}")
    return EXIT_OK


QAI_HUB_DIR = Path.home() / ".qai_hub"
QAI_HUB_SECRET_FILES = ("client.ini", "client.ini.bak")


def hub_file_permission_warnings() -> list[str]:
    """Warn when a file holding a QAI Hub token is readable by other users."""
    messages: list[str] = []
    for name in QAI_HUB_SECRET_FILES:
        path = QAI_HUB_DIR / name
        if path.is_file() and path.stat().st_mode & 0o077:
            messages.append(
                f"{path} holds a QAI Hub token and is readable by other users "
                f"(mode {oct(path.stat().st_mode & 0o777)}); consider `chmod 600 {path}`"
            )
    return messages


class TokenMasker:
    """Mask a secret in streamed output, even when it is split across read chunks."""

    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.pending = ""

    def feed(self, text: str) -> str:
        # Mask every complete occurrence first, then hold back the last len(secret) - 1
        # characters: they may be the beginning of a secret that the next chunk completes.
        self.pending = (self.pending + text).replace(self.secret, "***")
        keep = len(self.secret) - 1
        if len(self.pending) <= keep:
            return ""
        ready, self.pending = self.pending[:-keep], self.pending[-keep:]
        return ready

    def flush(self) -> str:
        ready, self.pending = self.pending, ""
        return ready.replace(self.secret, "***")


def files_containing(
    secret: str, root: Path, max_bytes: int = 5_000_000, max_files: int = 20_000
) -> list[Path]:
    """Files under root whose contents include secret (large files are skipped)."""
    needle = secret.encode()
    hits: list[Path] = []
    for index, path in enumerate(root.rglob("*")):
        if index >= max_files:
            break
        try:
            if (
                path.is_file()
                and path.stat().st_size <= max_bytes
                and needle in path.read_bytes()
            ):
                hits.append(path)
        except OSError:
            continue
    return sorted(hits)


def _tighten_hub_permissions() -> None:
    # qai-hub writes client.ini (and client.ini.bak with the previous token) with the default
    # umask, which is often group/world readable. Only the owner needs them.
    try:
        QAI_HUB_DIR.chmod(0o700)
        for name in QAI_HUB_SECRET_FILES:
            path = QAI_HUB_DIR / name
            if path.is_file():
                path.chmod(0o600)
        ok(f"restricted {QAI_HUB_DIR} to the current user (dir 700, files 600)")
    except OSError as exc:
        warn(f"could not restrict permissions on {QAI_HUB_DIR}: {exc}")


def cmd_hub_login(args) -> int:
    _, _, inputs = _load_context(args)
    # The token is only ever read from the environment: never from argv (visible in `ps` and
    # shell history) and never from a file (it would be copied into run directories).
    token = os.environ.pop(TOKEN_ENV, "").strip()
    if not token:
        raise RegressionError(
            f"${TOKEN_ENV} is not set; set it only for this one command"
        )
    if len(token) < 16 or any(ch.isspace() for ch in token):
        raise RegressionError(
            f"${TOKEN_ENV} does not look like a QAI Hub API token "
            "(too short or contains whitespace); nothing was run"
        )
    leaked = files_containing(token, inputs.source.parent)
    if leaked:
        raise RegressionError(
            "the QAI Hub token is written in these run files; remove it and consider "
            f"rotating the token (nothing was run): {', '.join(str(p) for p in leaked)}"
        )
    script = inputs.repo_root / "qaihub_login.sh"
    if not script.is_file():
        raise RegressionError(f"{script} not found")
    client_ini = QAI_HUB_DIR / "client.ini"
    if client_ini.exists():
        warn(
            f"{client_ini} exists and will be overwritten by qaihub_login.sh; qai-hub keeps "
            "the previous token in client.ini.bak"
        )
    argv = [str(script), "--key", token, "--image", inputs.docker_image]
    info(f"running: {script} --key *** --image {inputs.docker_image}")
    before = client_ini.stat().st_mtime if client_ini.exists() else None

    # qaihub_login.sh uses `docker run -it`, so it needs a terminal even when this runner is
    # driven by an agent without one. The documented script takes the key as an argument;
    # that is the product's own login flow and is used unchanged.
    child_env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
    pid, fd = pty.fork()
    if pid == 0:  # child
        try:
            os.chdir(inputs.repo_root)
            os.execve(argv[0], argv, child_env)
        finally:
            os._exit(127)
    masker = TokenMasker(token)
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        sys.stdout.write(masker.feed(chunk.decode("utf-8", errors="replace")))
        sys.stdout.flush()
    sys.stdout.write(masker.flush())
    sys.stdout.flush()
    _, status = os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    # The token must never end up in the run directory (inputs, logs, reports).
    leaked = files_containing(token, inputs.source.parent)
    del token, argv, masker
    if leaked:
        error(
            "the QAI Hub token was found in these run files; delete or edit them and "
            f"consider rotating the token: {', '.join(str(p) for p in leaked)}"
        )
        return EXIT_ERROR
    after = client_ini.stat().st_mtime if client_ini.exists() else None
    if code != 0:
        error(f"qaihub_login.sh exited {code}")
        return EXIT_ERROR
    if after is None or after == before:
        error(f"{client_ini} was not written")
        return EXIT_ERROR
    _tighten_hub_permissions()
    ok(f"QAI Hub login saved to {client_ini}")
    return EXIT_OK


def _resolve_run_dir(inputs: Inputs) -> Path:
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    return inputs.output_dir


def cmd_validate(args) -> int:
    config, pipelines, inputs = _load_context(args)
    run_dir = _resolve_run_dir(inputs)
    result = validate(
        inputs, config, pipelines, run_dir, skip_hub_check=args.skip_hub_check
    )
    (run_dir / "validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print_validation(result)
    info(f"wrote {run_dir / 'validation.json'}")
    return EXIT_OK if result["ok"] else EXIT_VALIDATION


def cmd_plan(args) -> int:
    config, pipelines, inputs = _load_context(args)
    run_dir = _resolve_run_dir(inputs)
    selected = select_pipelines(inputs.pipeline_patterns, pipelines)
    step = 0
    print(f"cd {inputs.repo_root}")
    for p in selected:
        print()
        print(f"# ===== {p.num}. {p.id} =====")
        for plan in build_stage_plans(inputs, config, p, run_dir):
            if not plan.selected:
                print(f"# {plan.mode}: SKIPPED ({plan.skip_reason})")
                continue
            step += 1
            if plan.uses_adb:
                print(f"# step {step}: {plan.mode}  (host first runs: adb kill-server)")
            else:
                print(f"# step {step}: {plan.mode}")
            print(shlex.join(plan.argv))
    print()
    info(f"{step} stage(s) across {len(selected)} pipeline(s); outputs under {run_dir}")
    return EXIT_OK


# ----------------------------
# Run state (lets `status` tell "starting" apart from "died")
# ----------------------------

STATE_FILE = "run_state.json"
RUNNER_OUT = "runner.out"
ACTIVE_PHASES = {"starting", "validating", "running"}


def state_path(run_dir: Path) -> Path:
    return run_dir / STATE_FILE


def write_state(run_dir: Path, phase: str, pid: int | None = None, **extra) -> None:
    path = state_path(run_dir)
    state = load_state(run_dir) or {}
    state.update(extra)
    state["phase"] = phase
    state["pid"] = pid if pid is not None else state.get("pid", os.getpid())
    state["updated_at"] = now_iso()
    state.setdefault("created_at", state["updated_at"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_state(run_dir: Path) -> dict | None:
    try:
        return json.loads(state_path(run_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def runner_alive(pid: int | None) -> bool:
    """True if pid is a live run_pipelines.py process (guards against pid reuse)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():
        try:
            text = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return True
        if "run_pipelines.py" not in text:
            return False
        stat = Path(f"/proc/{pid}/stat")
        try:
            # A zombie ("Z") has exited and only waits to be reaped by its parent.
            if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return False
        except (OSError, IndexError):
            pass
    return True


def _tail(path: Path, lines: int = 15) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line for line in text.splitlines() if line.strip()][-lines:]


def _refuse_reuse(run_dir: Path) -> None:
    """A run directory holds exactly one run; only an attempt that never started a stage
    (for example a runner killed during re-validation) may be replaced."""
    state = load_state(run_dir) or {}
    if (
        state.get("phase") in ACTIVE_PHASES
        and runner_alive(state.get("pid"))
        and (state.get("pid") != os.getpid())
    ):
        raise RegressionError(
            f"another runner (pid {state['pid']}) is already active in {run_dir}; "
            "use `status` to follow it"
        )
    if not results_path(run_dir).exists():
        return
    try:
        previous = load_results(run_dir)
    except (OSError, ValueError, RegressionError):
        previous = {}
    started_any = any(
        stage.get("status") != PENDING
        for record in previous.get("pipelines", [])
        for stage in record.get("stages", {}).values()
    )
    if started_any or previous.get("finished_at"):
        raise RegressionError(
            f"{results_path(run_dir)} already holds a run; every run needs a fresh "
            "output_dir (run `init` again)"
        )
    warn("replacing results of an earlier attempt that never started a stage")


def _new_results(
    inputs: Inputs, run_dir: Path, validation: dict, selected: list[Pipeline]
) -> dict:
    return {
        "runner_version": RUNNER_VERSION,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "started_at": now_iso(),
        "finished_at": None,
        "duration_s": None,
        "environment": collect_environment(inputs),
        "inputs": inputs.to_dict(),
        "validation_checks": validation["checks"],
        "validation_global_warnings": validation["global_warnings"],
        "pipelines": [
            {
                "num": p.num,
                "id": p.id,
                "type": p.type,
                "runtime": p.runtime,
                "precision": p.precision,
                "stages": {m: {"status": PENDING} for m in MODES},
            }
            for p in selected
        ],
    }


def cmd_run(args) -> int:
    if not args.confirmed_by_user:
        raise RegressionError(
            "refusing to run: pass --confirmed-by-user only after the user typed `run`"
        )
    config, pipelines, inputs = _load_context(args)
    run_dir = _resolve_run_dir(inputs)
    _refuse_reuse(run_dir)
    write_state(run_dir, "validating", pid=os.getpid())
    try:
        return _run(args, config, pipelines, inputs, run_dir)
    except BaseException as exc:
        state = load_state(run_dir) or {}
        if state.get("phase") in ACTIVE_PHASES:
            phase = "interrupted" if isinstance(exc, KeyboardInterrupt) else "crashed"
            write_state(run_dir, phase, error=f"{type(exc).__name__}: {exc}")
            if results_path(run_dir).exists():
                write_report(run_dir, load_results(run_dir))
        raise


def _run(args, config, pipelines, inputs: Inputs, run_dir: Path) -> int:
    info("re-validating inputs before execution")
    validation = validate(
        inputs,
        config,
        pipelines,
        run_dir,
        skip_hub_check=args.skip_hub_check,
        quiet=True,
    )
    (run_dir / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    if validation["global_errors"]:
        print_validation(validation)
        error("global validation errors; nothing was executed")
        write_state(run_dir, "validation_failed", errors=validation["global_errors"])
        return EXIT_VALIDATION

    selected = select_pipelines(inputs.pipeline_patterns, pipelines)
    results = _new_results(inputs, run_dir, validation, selected)
    write_results(run_dir, results)
    write_state(run_dir, "running")
    started = time.monotonic()
    interrupted = False
    info(f"run directory: {run_dir}")

    for index, p in enumerate(selected):
        record = results["pipelines"][index]
        plans = build_stage_plans(inputs, config, p, run_dir)
        qc_status: str | None = None
        for plan in plans:
            stage = record["stages"][plan.mode]
            stage["command"] = shlex.join(plan.argv)
            check = validation["stages"][p.id][plan.mode]
            stage["injected_flags"] = check.get("injected_flags") or []

            if interrupted:
                stage.update({"status": ABORTED, "reason": "run interrupted"})
                continue
            if not plan.selected:
                stage.update({"status": SKIPPED, "reason": plan.skip_reason})
                continue
            if check["feasibility"] == NOT_FEASIBLE:
                stage.update(
                    {"status": NOT_FEASIBLE, "reason": "; ".join(check["errors"])}
                )
                if plan.mode == "qc":
                    qc_status = NOT_FEASIBLE
                continue
            if plan.model_from_qc and qc_status not in (PASS, PASS_WITH_WARNINGS):
                stage.update(
                    {
                        "status": BLOCKED,
                        "reason": f"qc {qc_status or 'did not run'}; no converted model",
                    }
                )
                continue

            log_path = run_dir / "logs" / p.id / f"{plan.mode}.log"
            stage.update(
                {
                    "status": RUNNING,
                    "started_at": now_iso(),
                    "log": str(log_path.relative_to(run_dir)),
                }
            )
            write_results(run_dir, results)
            print()
            info(f"==== [{index + 1}/{len(selected)}] {p.id} : {plan.mode} ====")
            info(shlex.join(plan.argv))
            if plan.uses_adb:
                host_adb_kill_server()
            timeout_s = int(inputs.timeouts_s.get(plan.mode, 3600))
            exit_code, outcome, duration = execute_stage(
                plan.argv,
                cwd=inputs.repo_root,
                log_path=log_path,
                timeout_s=timeout_s,
                prefix=f"{p.id}:{plan.mode}",
                image=inputs.docker_image,
                header=[
                    f"pipeline: {p.id}  mode: {plan.mode}",
                    f"started: {stage['started_at']}",
                    f"cwd: {inputs.repo_root}",
                    f"timeout_s: {timeout_s}",
                    f"host pre-step: {'adb kill-server' if plan.uses_adb else 'none'}",
                    f"command: {shlex.join(plan.argv)}",
                ],
            )
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            evaluation = evaluate_stage(
                plan.mode, exit_code, outcome, plan.artifact, log_text, config
            )
            stage.update(evaluation)
            stage["duration_s"] = round(duration, 1)
            stage["finished_at"] = now_iso()
            if plan.mode == "qc":
                qc_status = stage["status"]
            level = ok if stage["status"] in (PASS, PASS_WITH_WARNINGS) else error
            level(
                f"{p.id} : {plan.mode} -> {stage['status']} in {format_duration(duration)}"
            )
            if outcome == "interrupted":
                interrupted = True
            write_results(run_dir, results)
            write_report(run_dir, results)

    results["finished_at"] = now_iso()
    results["duration_s"] = round(time.monotonic() - started, 1)
    write_results(run_dir, results)
    report = write_report(run_dir, results)
    write_state(run_dir, "interrupted" if interrupted else "finished")
    print()
    print(_summary_table(results))
    print()
    ok(f"report: {report}")
    ok(f"results: {results_path(run_dir)}")
    if interrupted:
        return EXIT_INTERRUPTED
    any_failed = any(
        record["stages"][m]["status"] in FAILED_STATUSES
        for record in results["pipelines"]
        for m in MODES
    )
    return EXIT_FAILURES if any_failed else EXIT_OK


def _summary_table(results: dict) -> str:
    rows = [
        [
            str(record.get("num", "-")),
            record["id"],
            *(record["stages"][m]["status"] for m in MODES),
            overall_status(record["stages"]),
        ]
        for record in results["pipelines"]
    ]
    return format_table(["NO", "PIPELINE", "QC", "MAP", "TEST", "OVERALL"], rows)


def cmd_start(args) -> int:
    """Launch `run` fully detached from this shell, then confirm it is alive.

    Agent shell tools often kill every process of a command (including `nohup ... &` jobs)
    as soon as the command returns. The runner is started in its own session so it survives
    that, and this command waits until the runner has proven it is alive.
    """
    if not args.confirmed_by_user:
        raise RegressionError(
            "refusing to start: pass --confirmed-by-user only after the user typed `run`"
        )
    _, _, inputs = _load_context(args)
    run_dir = _resolve_run_dir(inputs)
    _refuse_reuse(run_dir)
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--pipelines",
        str(Path(args.pipelines).resolve()),
        "run",
        "--inputs",
        str(inputs.source),
        "--confirmed-by-user",
    ]
    if args.skip_hub_check:
        argv.append("--skip-hub-check")
    out_path = run_dir / RUNNER_OUT
    # pid 0 = "not known yet"; the child records its own pid as soon as it starts.
    write_state(run_dir, "starting", pid=0, runner_out=str(out_path))
    with out_path.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            argv,
            cwd=inputs.repo_root,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    if (load_state(run_dir) or {}).get("phase") == "starting":
        write_state(run_dir, "starting", pid=proc.pid)
    info(f"started runner pid {proc.pid}; output: {out_path}")
    # The runner rewrites the state to "validating" first thing; wait for that proof of life.
    deadline = time.monotonic() + args.wait_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        state = load_state(run_dir) or {}
        if state.get("phase") != "starting" and state.get("pid") == proc.pid:
            break
        time.sleep(0.5)
    if proc.poll() is not None:
        phase = (load_state(run_dir) or {}).get("phase")
        error(f"runner exited immediately with code {proc.returncode} (phase: {phase})")
        for line in _tail(out_path):
            print(f"  {line}")
        return EXIT_ERROR
    ok(
        f"runner pid {proc.pid} is alive in its own session (phase: "
        f"{(load_state(run_dir) or {}).get('phase')}); follow it with: "
        f"run_pipelines.py status --run-dir {run_dir}"
    )
    return EXIT_OK


def cmd_status(args) -> int:
    """Progress of a run directory.

    Exit codes: 0 = starting, running or finished normally; 1 = no run here, or the runner
    stopped without finishing (killed, crashed or interrupted).
    """
    run_dir = Path(args.run_dir).expanduser().resolve()
    state = load_state(run_dir)
    has_results = results_path(run_dir).exists()
    if state is None and not has_results:
        error(
            f"no run has been started in {run_dir} (no {STATE_FILE}, no results.json)"
        )
        return EXIT_ERROR
    state = state or {}
    phase = state.get("phase", "running")
    pid = state.get("pid")
    alive = runner_alive(pid)
    out_path = run_dir / RUNNER_OUT

    if has_results:
        results = load_results(run_dir)
        print(_summary_table(results))
        print()
    else:
        results = None

    if phase == "finished" or (results and results.get("finished_at")):
        ok(f"run finished at {results['finished_at']}; report: {run_dir / 'report.md'}")
        return EXIT_OK
    if phase == "validation_failed":
        error("run stopped: global validation errors, nothing was executed:")
        for message in state.get("errors", []):
            print(f"  {message}")
        return EXIT_ERROR
    if phase in ACTIVE_PHASES and alive:
        if results is None:
            info(
                f"runner pid {pid} is {phase} (since {state.get('updated_at')}); "
                "re-validation takes up to a minute, then stages start. Check again shortly."
            )
            return EXIT_OK
        running = [
            (record["id"], m, record["stages"][m])
            for record in results["pipelines"]
            for m in MODES
            if record["stages"][m]["status"] == RUNNING
        ]
        pending = sum(
            1
            for record in results["pipelines"]
            for m in MODES
            if record["stages"][m]["status"] == PENDING
        )
        for pipeline_id, mode, stage in running:
            tail = _tail(run_dir / stage["log"], 1)
            info(
                f"running {pipeline_id} : {mode} since {stage.get('started_at')} "
                f"- last log line: {tail[-1] if tail else ''}"
            )
        info(f"{pending} stage(s) pending (runner pid {pid})")
        return EXIT_OK

    # Not finished and the runner is gone: killed, crashed or interrupted.
    reason = (
        state.get("error") or "no error recorded; the process was killed from outside"
    )
    error(
        f"runner pid {pid} is not running and the run did not finish (last phase: {phase}): {reason}"
    )
    if phase in ACTIVE_PHASES:
        error(
            "a runner that vanishes without output was usually killed by the shell that "
            "launched it; start it with `start`, or ask the user to run it in their own terminal"
        )
    tail = _tail(out_path)
    if tail:
        info(f"last lines of {out_path}:")
        for line in tail:
            print(f"  {line}")
    elif out_path.exists():
        info(f"{out_path} is empty: the runner died before printing anything")
    return EXIT_ERROR


def cmd_report(args) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    results = load_results(run_dir)
    path = write_report(run_dir, results)
    ok(f"wrote {path}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipelines.py",
        description="iQ-Foundry pipeline regression runner (see module docstring).",
    )
    parser.add_argument(
        "--pipelines",
        default=str(SCRIPT_REPO_ROOT / DEFAULT_PIPELINES_FILE),
        help="pipeline definitions yaml (default: regression/pipelines.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show all pipeline definitions").set_defaults(
        func=cmd_list
    )

    p_init = sub.add_parser(
        "init", help="create a run dir with an inputs.yaml to fill in"
    )
    p_init.add_argument(
        "--out", help="run directory (default: out/regression/<timestamp>)"
    )
    p_init.set_defaults(func=cmd_init)

    for name, func, help_text in (
        ("hub-login", cmd_hub_login, f"run qaihub_login.sh with ${TOKEN_ENV}"),
        (
            "validate",
            cmd_validate,
            "check inputs and prerequisites, write validation.json",
        ),
        ("plan", cmd_plan, "print the exact commands `run` will execute"),
        ("run", cmd_run, "execute the selected pipelines (foreground)"),
        (
            "start",
            cmd_start,
            "execute the selected pipelines detached, in the background",
        ),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--inputs",
            required=True,
            help="inputs.yaml written from the user's answers",
        )
        if name in ("validate", "run", "start"):
            p.add_argument(
                "--skip-hub-check",
                action="store_true",
                help="do not contact QAI Hub to verify the saved login",
            )
        if name in ("run", "start"):
            p.add_argument(
                "--confirmed-by-user",
                action="store_true",
                help="required: the user explicitly typed `run`",
            )
        if name == "start":
            p.add_argument(
                "--wait-s",
                type=float,
                default=20.0,
                help="seconds to wait for the runner to prove it is alive (default 20)",
            )
        p.set_defaults(func=func)

    for name, func, help_text in (
        ("status", cmd_status, "show progress of a run directory"),
        ("report", cmd_report, "regenerate report.md from results.json"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run-dir", required=True)
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RegressionError as exc:
        error(str(exc))
        return EXIT_ERROR
    except KeyboardInterrupt:
        error("interrupted")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
