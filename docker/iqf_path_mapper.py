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
from __future__ import annotations

from collections.abc import Callable
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_IMAGE = "innodiskorg/iqf:latest"
REPO_MOUNT_TARGET = "/workspace/iQ-Foundry"
CONTAINER_WORKDIR = REPO_MOUNT_TARGET
CONTAINER_INPUT_ROOT = "/inputs"
CONTAINER_OUTPUT_ROOT = "/outputs"
DEFAULT_DOCKERFILE = "docker/Dockerfile"
DOCKER_CONFIG_RELATIVE_PATH = Path(".iqf") / "docker-paths.json"
MODE_CHOICES = ("qc", "mAP", "test")
MODEL_TYPES = ("yolov10", "yolov11", "yolov26")
WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
WSL_UNC_PATH_RE = re.compile(
    r"^\\\\(?P<host>wsl(?:\.localhost|\$))\\(?P<distro>[^\\]+)(?P<suffix>(?:\\.*)?)$",
    re.IGNORECASE,
)


class WrapperError(RuntimeError):
    pass


ConsolePrinter = Callable[[str], None]


@dataclass(frozen=True)
class PathSpec:
    flag: str
    kind: str
    required_input: bool = False


@dataclass(frozen=True)
class MountSpec:
    source: str
    target: str
    read_only: bool

    def docker_args(self) -> list[str]:
        suffix = ":ro" if self.read_only else ""
        return ["-v", f"{self.source}:{self.target}{suffix}"]


@dataclass
class CommandPlan:
    docker_command: list[str]
    inner_command: list[str] | None = None
    warnings: list[str] = field(default_factory=list)
    cwd: str | None = None
    host_pre_command: list[str] | None = None


MODE_PATH_SPECS: dict[str, dict[str, PathSpec]] = {
    "qc": {
        "model": PathSpec("--model", "file", required_input=True),
        "calib_dir": PathSpec("--calib_dir", "dir", required_input=True),
        "output": PathSpec("--output", "output_file"),
    },
    "mAP": {
        "annotations": PathSpec("--annotations", "file_or_dir", required_input=True),
        "fp_model": PathSpec("--fp-model", "file", required_input=True),
        "images": PathSpec("--images", "dir", required_input=True),
        "int_model": PathSpec("--int-model", "file", required_input=True),
        "output_text": PathSpec("--output_text", "output_file"),
        "remote_runner_local": PathSpec("--remote-runner-local", "file"),
    },
    "test": {
        "model": PathSpec("--model", "file", required_input=True),
        "yaml": PathSpec("--yaml", "file", required_input=True),
        "image": PathSpec("--image", "file", required_input=True),
        "images": PathSpec("--images", "dir", required_input=True),
        "output": PathSpec("--output", "output_dir"),
    },
}


def default_repo_root(script_path: str) -> Path:
    return Path(script_path).resolve().parent.parent


def docker_config_path(repo_root: Path) -> Path:
    return repo_root / DOCKER_CONFIG_RELATIVE_PATH


def resolve_image_name(image_override: str | None) -> str:
    return image_override or os.environ.get("IQF_DOCKER_IMAGE") or DEFAULT_IMAGE


def _run_wslpath(path_value: str) -> str:
    wslpath_command = shutil.which("wslpath")
    if not wslpath_command:
        raise WrapperError(
            f"[error] Cannot translate Windows/WSL path without wslpath on PATH: {path_value}"
        )

    completed = subprocess.run(
        [wslpath_command, "-u", path_value],
        capture_output=True,
        text=True,
        check=False,
    )
    translated = completed.stdout.strip()
    if completed.returncode == 0 and translated:
        return translated

    detail = completed.stderr.strip() or translated or "wslpath failed"
    raise WrapperError(
        f"[error] Failed to translate Windows/WSL path: {path_value}\n{detail}"
    )


def _normalize_windows_style_path(path_value: str) -> str | None:
    # Wrapper prompts may receive native Windows paths or WSL UNC paths when the user is driving
    # the wrapper from Windows/WSL. Normalize those into the current distro before validation.
    if WINDOWS_DRIVE_PATH_RE.match(path_value):
        return _run_wslpath(path_value)

    unc_match = WSL_UNC_PATH_RE.match(path_value)
    if unc_match:
        current_distro = os.environ.get("WSL_DISTRO_NAME")
        requested_distro = unc_match.group("distro")
        if current_distro and requested_distro.casefold() != current_distro.casefold():
            raise WrapperError(
                "[error] Unsupported WSL UNC path for a different distro: "
                f"{requested_distro} (current distro: {current_distro})"
            )
        return _run_wslpath(path_value)

    if path_value.startswith("\\\\"):
        raise WrapperError(
            "[error] Unsupported UNC path. Use a Windows drive path like C:\\path\\to\\file "
            "or a current-distro WSL UNC path like \\\\wsl.localhost\\Distro\\path."
        )

    return None


def normalize_host_path(path_value: str) -> Path:
    translated = _normalize_windows_style_path(path_value)
    normalized_input = translated if translated is not None else path_value
    return Path(normalized_input).expanduser().resolve()


def _empty_config() -> dict:
    return {"version": 1, "types": {}}


def load_wrapper_config(config_path: Path) -> dict:
    if not config_path.exists():
        return _empty_config()

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WrapperError(f"[error] Invalid JSON in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise WrapperError(f"[error] Invalid config format in {config_path}")
    if raw.get("version") != 1:
        raise WrapperError(f"[error] Unsupported config version in {config_path}")
    if not isinstance(raw.get("types"), dict):
        raise WrapperError(f"[error] Invalid types section in {config_path}")
    return raw


def load_saved_mode_paths(config_path: Path, model_type: str, mode: str) -> dict[str, str]:
    config = load_wrapper_config(config_path)
    type_entry = config["types"].get(model_type, {})
    mode_entry = type_entry.get(mode, {})
    raw_paths = mode_entry.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise WrapperError(f"[error] Invalid saved path data for {model_type}/{mode}")

    saved: dict[str, str] = {}
    for field_name, payload in raw_paths.items():
        if not isinstance(payload, dict):
            raise WrapperError(
                f"[error] Invalid saved path entry for {model_type}/{mode}/{field_name}"
            )
        host = payload.get("host")
        if not isinstance(host, str) or not host.strip():
            raise WrapperError(
                f"[error] Missing host path for {model_type}/{mode}/{field_name}"
            )
        saved[field_name] = host
    return saved


def save_mode_paths(
    config_path: Path,
    model_type: str,
    mode: str,
    saved_inputs: dict[str, tuple[str, str]],
) -> None:
    config = load_wrapper_config(config_path)
    config.setdefault("types", {})
    config["types"].setdefault(model_type, {})
    config["types"][model_type][mode] = {
        "paths": {
            field_name: {"kind": kind, "host": host}
            for field_name, (kind, host) in saved_inputs.items()
        }
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_existing_path(path_value: str, kind: str) -> str:
    path = normalize_host_path(path_value)
    if not path.exists():
        raise WrapperError(f"[error] Path does not exist: {path}")
    if kind == "file":
        if not path.is_file():
            raise WrapperError(f"[error] Expected a file: {path}")
    elif kind == "dir":
        if not path.is_dir():
            raise WrapperError(f"[error] Expected a directory: {path}")
    elif kind == "file_or_dir":
        if not (path.is_file() or path.is_dir()):
            raise WrapperError(f"[error] Expected an existing file or directory: {path}")
    else:
        raise WrapperError(f"[error] Unsupported validation kind: {kind}")
    return str(path)


def validate_output_path(path_value: str, kind: str) -> str:
    path = normalize_host_path(path_value)
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        raise WrapperError(f"[error] Output parent directory does not exist: {parent}")
    if kind == "output_file":
        if path.exists() and path.is_dir():
            raise WrapperError(f"[error] Output path must be a file path: {path}")
    elif kind == "output_dir":
        if path.exists() and not path.is_dir():
            raise WrapperError(f"[error] Output path must be a directory path: {path}")
    else:
        raise WrapperError(f"[error] Unsupported output kind: {kind}")
    return str(path)


def _persisted_kind(mode: str, field_name: str) -> str:
    return MODE_PATH_SPECS[mode][field_name].kind


def merge_required_input_paths(
    mode: str,
    model_type: str,
    provided_paths: dict[str, str | None],
    saved_paths: dict[str, str],
) -> dict[str, str]:
    merged: dict[str, str] = {}

    if mode == "test":
        # test is the only mode with mutually exclusive required inputs. A direct --image/--images
        # choice replaces the saved choice as a pair so we never merge both into one run.
        direct_image = provided_paths.get("image")
        direct_images = provided_paths.get("images")
        if direct_image or direct_images:
            if bool(direct_image) == bool(direct_images):
                raise WrapperError(
                    "[error] test requires exactly one of --image or --images"
                )
            if direct_image:
                merged["image"] = direct_image
            if direct_images:
                merged["images"] = direct_images
        else:
            saved_image = saved_paths.get("image")
            saved_images = saved_paths.get("images")
            if saved_image and saved_images:
                raise WrapperError(
                    f"[error] Saved config for {model_type}/test contains both image and images"
                )
            if saved_image:
                merged["image"] = saved_image
            if saved_images:
                merged["images"] = saved_images

        for field_name in ("model", "yaml"):
            value = provided_paths.get(field_name) or saved_paths.get(field_name)
            if value:
                merged[field_name] = value
        return merged

    for field_name, spec in MODE_PATH_SPECS[mode].items():
        if not spec.required_input:
            continue
        value = provided_paths.get(field_name) or saved_paths.get(field_name)
        if value:
            merged[field_name] = value
    return merged


def missing_required_paths(mode: str, merged_inputs: dict[str, str]) -> list[str]:
    missing: list[str] = []
    if mode == "test":
        for field_name in ("model", "yaml"):
            if not merged_inputs.get(field_name):
                missing.append(MODE_PATH_SPECS[mode][field_name].flag)
        if not merged_inputs.get("image") and not merged_inputs.get("images"):
            missing.append("exactly one of --image or --images")
        if merged_inputs.get("image") and merged_inputs.get("images"):
            missing.append("exactly one of --image or --images")
        return missing

    for field_name, spec in MODE_PATH_SPECS[mode].items():
        if spec.required_input and not merged_inputs.get(field_name):
            missing.append(spec.flag)
    return missing


def missing_required_path_message(mode: str, model_type: str, missing_flags: list[str]) -> str:
    if len(missing_flags) == 1:
        detail = missing_flags[0]
    else:
        detail = ", ".join(missing_flags)
    return (
        f"[error] Missing required path for {mode}: {detail}\n"
        f"Run: ./docker/iqf configure {mode} --type {model_type}\n"
        f"or pass it directly with the required host-path flags"
    )


def _emit_prompt_message(message: str, printer: ConsolePrinter | None) -> None:
    if printer is None:
        print(message)
        return
    printer(message)


def _normalize_configure_prompt_path(path_value: str) -> str:
    # Windows copy/paste commonly wraps full paths in double quotes. Configure should accept that
    # pasted form without widening path handling for non-configure wrapper code paths.
    normalized = path_value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] == '"':
        return normalized[1:-1]
    return normalized


def prompt_yes_no(
    prompt: str,
    default: bool = True,
    *,
    error_printer: ConsolePrinter | None = None,
) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        _emit_prompt_message("[error] Please answer y/yes or n/no.", error_printer)


def prompt_choice(
    prompt: str,
    choices: list[tuple[str, str]],
    default: str,
    *,
    error_printer: ConsolePrinter | None = None,
) -> str:
    labels = "/".join(key for key, _ in choices)
    descriptions = ", ".join(f"{key}={desc}" for key, desc in choices)
    while True:
        answer = input(f"{prompt} [{labels}] ({descriptions}) [{default}]: ").strip()
        if not answer:
            return default
        for key, _ in choices:
            if answer == key:
                return key
        _emit_prompt_message(
            f"[error] Choose one of: {', '.join(key for key, _ in choices)}",
            error_printer,
        )


def prompt_for_path(
    field_name: str,
    prompt: str,
    kind: str,
    current_value: str | None,
    *,
    info_printer: ConsolePrinter | None = None,
    warning_printer: ConsolePrinter | None = None,
    error_printer: ConsolePrinter | None = None,
) -> str:
    normalized_current_value = (
        _normalize_configure_prompt_path(current_value)
        if current_value is not None
        else None
    )
    if normalized_current_value:
        try:
            validated_current = validate_existing_path(normalized_current_value, kind)
        except WrapperError as exc:
            _emit_prompt_message(
                str(exc).replace("[error]", "[warn]", 1),
                warning_printer,
            )
        else:
            _emit_prompt_message(
                f"[info] Existing saved path for {field_name}: {validated_current}",
                info_printer,
            )
            if prompt_yes_no(
                "Reuse this path",
                default=True,
                error_printer=error_printer,
            ):
                return validated_current

    while True:
        value = _normalize_configure_prompt_path(input(f"{prompt}: "))
        if not value:
            _emit_prompt_message("[error] Path is required.", error_printer)
            continue
        try:
            return validate_existing_path(value, kind)
        except WrapperError as exc:
            _emit_prompt_message(str(exc), error_printer)


def prompt_for_mode_paths(
    config_path: Path,
    model_type: str,
    mode: str,
    *,
    info_printer: ConsolePrinter | None = None,
    warning_printer: ConsolePrinter | None = None,
    error_printer: ConsolePrinter | None = None,
) -> dict[str, tuple[str, str]]:
    saved_paths = load_saved_mode_paths(config_path, model_type, mode)
    saved_inputs: dict[str, tuple[str, str]] = {}

    prompts = {
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
    }

    if mode == "test":
        default_choice = "image" if saved_paths.get("image") else "images"
        choice = prompt_choice(
            "Save a single image or an image directory for test mode",
            [("image", "single file"), ("images", "directory")],
            default_choice,
            error_printer=error_printer,
        )
        test_fields = [
            ("model", "Enter INT model path for test (--model)"),
            ("yaml", "Enter class YAML path for test (--yaml)"),
            (
                choice,
                "Enter single test image path for test (--image)"
                if choice == "image"
                else "Enter image directory for test (--images)",
            ),
        ]
        for field_name, prompt in test_fields:
            kind = _persisted_kind(mode, field_name)
            saved_inputs[field_name] = (
                kind,
                prompt_for_path(
                    field_name,
                    prompt,
                    kind,
                    saved_paths.get(field_name),
                    info_printer=info_printer,
                    warning_printer=warning_printer,
                    error_printer=error_printer,
                ),
            )
        return saved_inputs

    for field_name, prompt in prompts[mode]:
        kind = _persisted_kind(mode, field_name)
        saved_inputs[field_name] = (
            kind,
            prompt_for_path(
                field_name,
                prompt,
                kind,
                saved_paths.get(field_name),
                info_printer=info_printer,
                warning_printer=warning_printer,
                error_printer=error_printer,
            ),
        )
    return saved_inputs


def _translate_path(
    mode: str,
    field_name: str,
    path_value: str,
    kind: str,
) -> tuple[str, MountSpec]:
    # Input files mount their parent directory read-only so the container path keeps the original
    # filename, while input directories and output overrides map to deterministic wrapper targets.
    host_path = normalize_host_path(path_value)
    if kind in {"file", "file_or_dir"} and host_path.is_file():
        container_base = f"{CONTAINER_INPUT_ROOT}/{mode}/{field_name}"
        return (
            f"{container_base}/{host_path.name}",
            MountSpec(str(host_path.parent), container_base, True),
        )
    if kind == "dir" or (kind == "file_or_dir" and host_path.is_dir()):
        container_path = f"{CONTAINER_INPUT_ROOT}/{mode}/{field_name}"
        return (container_path, MountSpec(str(host_path), container_path, True))
    if kind == "output_file":
        container_base = f"{CONTAINER_OUTPUT_ROOT}/{mode}/{field_name}"
        return (
            f"{container_base}/{host_path.name}",
            MountSpec(str(host_path.parent), container_base, False),
        )
    if kind == "output_dir":
        container_base = f"{CONTAINER_OUTPUT_ROOT}/{mode}/{field_name}"
        return (
            f"{container_base}/{host_path.name}",
            MountSpec(str(host_path.parent), container_base, False),
        )
    raise WrapperError(f"[error] Unsupported path mapping for {field_name}: {kind}")


def dedupe_mounts(mounts: list[MountSpec]) -> list[MountSpec]:
    seen: set[tuple[str, str, bool]] = set()
    deduped: list[MountSpec] = []
    for mount in mounts:
        key = (mount.source, mount.target, mount.read_only)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(mount)
    return deduped


def build_inner_cli_command(
    mode: str,
    model_type: str,
    translated_paths: dict[str, str],
    passthrough_args: list[str],
    use_adb: bool,
) -> list[str]:
    command = ["python3", "cli.py", "--type", model_type, "--mode", mode]

    if mode == "qc":
        ordered_fields = ("model", "calib_dir", "output")
    elif mode == "mAP":
        ordered_fields = (
            "annotations",
            "fp_model",
            "images",
            "int_model",
            "output_text",
            "remote_runner_local",
        )
    else:
        ordered_fields = ("model", "yaml", "image", "images", "output")

    for field_name in ordered_fields:
        value = translated_paths.get(field_name)
        if not value:
            continue
        command.extend([MODE_PATH_SPECS[mode][field_name].flag, value])

    if mode == "test" and use_adb:
        command.append("--adb")

    command.extend(passthrough_args)
    return command


def _runtime_check(
    path: Path | None,
    description: str,
    dry_run: bool,
    warnings: list[str],
    expected_kind: str | None = None,
) -> None:
    if path is None:
        return
    if not path.exists():
        _warn_or_error(f"{description} not found: {path}", dry_run, warnings)
        return
    if expected_kind == "file" and not path.is_file():
        _warn_or_error(f"{description} is not a file: {path}", dry_run, warnings)
    if expected_kind == "dir" and not path.is_dir():
        _warn_or_error(f"{description} is not a directory: {path}", dry_run, warnings)


def _warn_or_error(detail: str, dry_run: bool, warnings: list[str]) -> None:
    if dry_run:
        warnings.append(f"[warn] {detail}")
        return
    raise WrapperError(f"[error] {detail}")


def _prepare_android_config_dir(
    dry_run: bool,
    warnings: list[str],
) -> Path:
    android_dir = Path.home() / ".android"
    if android_dir.exists():
        if not android_dir.is_dir():
            _warn_or_error(
                f"Android config directory is not a directory: {android_dir}",
                dry_run,
                warnings,
            )
        return android_dir

    if not dry_run:
        android_dir.mkdir(parents=True, exist_ok=True)
    return android_dir


def _append_adb_runtime_mounts(
    mounts: list[MountSpec],
    dry_run: bool,
    warnings: list[str],
) -> None:
    # Design A relies on live USB visibility from WSL. The wrapper can prepare an empty host-side
    # Android config directory, but the USB passthrough itself still has to exist already.
    android_dir = _prepare_android_config_dir(dry_run, warnings)
    usb_bus = Path("/dev/bus/usb")
    _runtime_check(usb_bus, "USB bus mount path", dry_run, warnings, expected_kind="dir")
    mounts.append(MountSpec(str(usb_bus), "/dev/bus/usb", False))
    mounts.append(MountSpec(str(android_dir), "/root/.android", False))


def _adb_container_preflight_segments() -> list[str]:
    # mAP and test --adb run adb inside the container before invoking cli.py so the container owns
    # the adb server for the duration of the run.
    return [
        "adb kill-server >/dev/null 2>&1 || true",
        "adb start-server",
        "adb devices",
    ]


def _output_display_env_args(
    repo_root: Path,
    output_display_map: dict[str, str],
) -> list[str]:
    return [
        "-e",
        f"IQF_HOST_REPO_ROOT={repo_root}",
        "-e",
        f"IQF_CONTAINER_REPO_ROOT={REPO_MOUNT_TARGET}",
        "-e",
        f"IQF_OUTPUT_DISPLAY_MAP={json.dumps(output_display_map, sort_keys=True)}",
    ]


def plan_shell_command(
    repo_root: Path,
    image_name: str,
    use_qai_hub: bool,
    use_adb: bool,
    dry_run: bool,
) -> CommandPlan:
    warnings: list[str] = []
    mounts = [MountSpec(str(repo_root), REPO_MOUNT_TARGET, False)]

    if use_qai_hub:
        qai_hub_dir = Path.home() / ".qai_hub"
        _runtime_check(qai_hub_dir, "QAI Hub config directory", dry_run, warnings, expected_kind="dir")
        mounts.append(MountSpec(str(qai_hub_dir), "/root/.qai_hub", True))

    if use_adb:
        _append_adb_runtime_mounts(mounts, dry_run, warnings)

    docker_command = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-v",
        f"{repo_root}:{REPO_MOUNT_TARGET}",
        "-w",
        CONTAINER_WORKDIR,
    ]
    if use_adb:
        docker_command.append("--privileged")
    for mount in dedupe_mounts(mounts[1:]):
        docker_command.extend(mount.docker_args())
    docker_command.extend(["--entrypoint", "bash", image_name])
    return CommandPlan(docker_command=docker_command, warnings=warnings)


def plan_build_command(repo_root: Path, image_name: str) -> CommandPlan:
    return CommandPlan(
        docker_command=[
            "docker",
            "build",
            "-f",
            DEFAULT_DOCKERFILE,
            "-t",
            image_name,
            ".",
        ],
        cwd=str(repo_root),
    )


def plan_run_command(
    repo_root: Path,
    image_name: str,
    mode: str,
    model_type: str,
    provided_paths: dict[str, str | None],
    passthrough_args: list[str],
    use_adb: bool,
    dry_run: bool,
) -> tuple[CommandPlan, dict[str, tuple[str, str]]]:
    config_path = docker_config_path(repo_root)
    saved_paths = load_saved_mode_paths(config_path, model_type, mode)
    merged_inputs = merge_required_input_paths(mode, model_type, provided_paths, saved_paths)
    missing_flags = missing_required_paths(mode, merged_inputs)
    if missing_flags:
        raise WrapperError(missing_required_path_message(mode, model_type, missing_flags))

    translated_paths: dict[str, str] = {}
    mounts = [MountSpec(str(repo_root), REPO_MOUNT_TARGET, False)]
    warnings: list[str] = []
    saved_inputs: dict[str, tuple[str, str]] = {}
    output_display_map: dict[str, str] = {}

    for field_name, spec in MODE_PATH_SPECS[mode].items():
        if spec.required_input:
            raw_value = merged_inputs.get(field_name)
        else:
            raw_value = provided_paths.get(field_name)
        if not raw_value:
            continue

        if spec.kind.startswith("output_"):
            normalized = validate_output_path(raw_value, spec.kind)
        else:
            normalized = validate_existing_path(raw_value, spec.kind)
        translated, mount = _translate_path(mode, field_name, normalized, spec.kind)
        translated_paths[field_name] = translated
        mounts.append(mount)
        if spec.required_input:
            saved_inputs[field_name] = (spec.kind, normalized)
        if spec.kind.startswith("output_"):
            output_display_map[translated] = normalized

    needs_adb = mode == "mAP" or (mode == "test" and use_adb)
    if mode == "qc":
        qai_hub_ini = Path.home() / ".qai_hub" / "client.ini"
        _runtime_check(qai_hub_ini, "QAI Hub config", dry_run, warnings, expected_kind="file")
        mounts.append(MountSpec(str(qai_hub_ini.parent), "/root/.qai_hub", True))

    if needs_adb:
        _append_adb_runtime_mounts(mounts, dry_run, warnings)
        host_pre_command = None
    else:
        host_pre_command = None

    inner_command = build_inner_cli_command(
        mode=mode,
        model_type=model_type,
        translated_paths=translated_paths,
        passthrough_args=passthrough_args,
        use_adb=use_adb,
    )
    bash_segments: list[str] = []
    if needs_adb:
        bash_segments.extend(_adb_container_preflight_segments())
    bash_segments.append(shlex.join(inner_command))

    docker_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{repo_root}:{REPO_MOUNT_TARGET}",
        "-w",
        CONTAINER_WORKDIR,
    ]
    if needs_adb:
        docker_command.append("--privileged")
    for mount in dedupe_mounts(mounts[1:]):
        docker_command.extend(mount.docker_args())
    docker_command.extend(_output_display_env_args(repo_root, output_display_map))
    docker_command.extend(
        [
            "--entrypoint",
            "bash",
            image_name,
            "-lc",
            "; ".join(bash_segments),
        ]
    )

    return (
        CommandPlan(
            docker_command=docker_command,
            inner_command=inner_command,
            warnings=warnings,
            host_pre_command=host_pre_command,
        ),
        saved_inputs,
    )


def format_command(command: list[str]) -> str:
    return shlex.join(command)
