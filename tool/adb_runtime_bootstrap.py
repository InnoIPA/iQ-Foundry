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
"""ADB runtime bootstrap for persistent on-device Python environment."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PERSISTENT_ADB_RUNTIME_BASE_DIR = "/etc/innodisk/iq-qnn"
PERSISTENT_ADB_RUNTIME_VENV_DIR = f"{PERSISTENT_ADB_RUNTIME_BASE_DIR}/.venv"
PERSISTENT_ADB_RUNTIME_REQUIREMENTS = (
    f"{PERSISTENT_ADB_RUNTIME_BASE_DIR}/requirements_target.txt"
)
PERSISTENT_ADB_RUNTIME_WHEEL_DIR = f"{PERSISTENT_ADB_RUNTIME_BASE_DIR}/wheels"
ANSI_YELLOW = "\033[33m"
ANSI_RESET = "\033[0m"


def _adb_prefix(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def _run_cmd(cmd: Sequence[str]) -> None:
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(list(cmd), check=True)


def _run_cmd_capture(cmd: Sequence[str]) -> str:
    print(" ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(list(cmd), check=True, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.stdout or ""


def _adb_shell(serial: str | None, command: str) -> None:
    _run_cmd(_adb_prefix(serial) + ["shell", command])


def _adb_shell_capture(serial: str | None, command: str) -> str:
    return _run_cmd_capture(_adb_prefix(serial) + ["shell", command])


def _adb_push(serial: str | None, src: str, dst: str) -> None:
    _run_cmd(_adb_prefix(serial) + ["push", src, dst])


def _print_yellow(message: str) -> None:
    print(f"{ANSI_YELLOW}{message}{ANSI_RESET}", flush=True)


def _validate_optional_wheel(local_ort_qnn_wheel_path: str | None) -> Path | None:
    if not local_ort_qnn_wheel_path:
        return None

    wheel_path = Path(local_ort_qnn_wheel_path).expanduser().resolve()
    if not wheel_path.is_file():
        raise FileNotFoundError(f"ORT_QNN wheel not found: {wheel_path}")
    if wheel_path.suffix.lower() != ".whl":
        raise ValueError(f"ORT_QNN wheel must end with .whl: {wheel_path}")
    return wheel_path


def _parse_wheel_distribution_version(wheel_path: Path) -> tuple[str, str]:
    """Return (distribution, version) parsed from a PEP 427 wheel filename.

    Wheel names are
    `{distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl`, so the first
    two fields are all we need. Parsing the filename keeps the bundled wheel the single
    source of truth for the version the target is required to run.
    """
    parts = wheel_path.stem.split("-")
    if len(parts) < 5 or not parts[0] or not parts[1]:
        raise ValueError(
            "ORT_QNN wheel filename is not a valid PEP 427 wheel name (expected "
            "<distribution>-<version>-<python>-<abi>-<platform>.whl): "
            f"{wheel_path.name}"
        )
    return parts[0].replace("_", "-"), parts[1]


def _probe_remote_ort_origin(
    adb_serial: str | None,
    remote_venv_dir: str,
    remote_python: str,
) -> tuple[str, str | None, str | None]:
    probe_script = """
import json
import pathlib
import sys

venv_dir = pathlib.Path(sys.argv[1]).resolve()
try:
    import onnxruntime
except ModuleNotFoundError:
    print(json.dumps({"status": "missing", "path": None, "version": None}))
    raise SystemExit(0)

origin = getattr(onnxruntime, "__file__", None)
resolved = str(pathlib.Path(origin).resolve()) if origin else None
origin_path = pathlib.Path(resolved) if resolved else None
try:
    in_venv = origin_path is not None and origin_path.is_relative_to(venv_dir)
except AttributeError:
    try:
        in_venv = origin_path is not None and origin_path.relative_to(venv_dir) is not None
    except ValueError:
        in_venv = False

print(
    json.dumps(
        {
            "status": "present_in_venv" if in_venv else "present_outside_venv",
            "path": resolved,
            "version": getattr(onnxruntime, "__version__", None),
        }
    )
)
"""
    ort_probe_cmd = (
        "set -e; "
        f". {shlex.quote(remote_venv_dir + '/bin/activate')}; "
        f"{shlex.quote(remote_python)} -c {shlex.quote(probe_script)} "
        f"{shlex.quote(remote_venv_dir)}"
    )
    probe_out = _adb_shell_capture(adb_serial, ort_probe_cmd)
    probe_lines = [line.strip() for line in probe_out.splitlines() if line.strip()]
    if not probe_lines:
        raise RuntimeError("Remote onnxruntime probe returned no output.")
    try:
        payload = json.loads(probe_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Remote onnxruntime probe returned invalid JSON: {probe_lines[-1]}"
        ) from exc
    status = str(payload.get("status") or "").strip()
    path_value = payload.get("path")
    if status not in {"missing", "present_in_venv", "present_outside_venv"}:
        raise RuntimeError(
            f"Remote onnxruntime probe returned unsupported status: {status!r}"
        )
    if path_value is not None:
        path_value = str(path_value)
    version_value = payload.get("version")
    if version_value is not None:
        version_value = str(version_value).strip() or None
    return status, path_value, version_value


def _uninstall_remote_ort(
    adb_serial: str | None,
    remote_venv_dir: str,
    wheel_distribution: str,
) -> None:
    """Remove any onnxruntime distribution already present in the remote venv.

    Both `onnxruntime` and `onnxruntime-qnn` are always attempted: they ship the same
    `onnxruntime` import package, so leaving either behind would shadow the wheel we are
    about to install, and `uv pip install --force-reinstall` of one does not remove the
    other. Each distribution is uninstalled in its own command with failures ignored, so
    one that simply is not installed cannot abort the rest.
    """
    distributions = ["onnxruntime", "onnxruntime-qnn"]
    if wheel_distribution not in distributions:
        distributions.append(wheel_distribution)
    uninstall_steps = "; ".join(
        f'"$UV_BIN" pip uninstall {shlex.quote(dist)} || true'
        for dist in distributions
    )
    uninstall_cmd = (
        "set -e; "
        'export PATH="$HOME/.local/bin:$PATH"; '
        'export PATH="$(python3 -m site --user-base)/bin:$PATH"; '
        'UV_BIN="$(command -v uv)"; '
        f". {shlex.quote(remote_venv_dir + '/bin/activate')}; "
        f"{uninstall_steps}"
    )
    _adb_shell(adb_serial, uninstall_cmd)


def _install_optional_ort_qnn_wheel(
    adb_serial: str | None,
    remote_venv_dir: str,
    remote_python: str,
    local_ort_qnn_wheel_path: Path | None,
    remote_wheel_dir: str = PERSISTENT_ADB_RUNTIME_WHEEL_DIR,
) -> None:
    if local_ort_qnn_wheel_path is None:
        return

    required_dist, required_version = _parse_wheel_distribution_version(
        local_ort_qnn_wheel_path
    )

    ort_status, ort_path, ort_version = _probe_remote_ort_origin(
        adb_serial=adb_serial,
        remote_venv_dir=remote_venv_dir,
        remote_python=remote_python,
    )
    uninstall_first = False
    if ort_status == "present_in_venv":
        if ort_version == required_version:
            _print_yellow(
                f"[warn] Reusing existing onnxruntime {ort_version} in remote venv "
                "(matches the bundled wheel)"
            )
            return
        # Version mismatch, or a build whose __version__ could not be read. Reinstalling
        # is cheap; running against an unidentified build is not. The old distribution is
        # removed first so it cannot shadow the wheel we are about to install.
        _print_yellow(
            "[warn] Remote onnxruntime version mismatch: found "
            f"{ort_version or '<unknown>'}, bundled wheel provides {required_version}. "
            "Removing the existing installation and reinstalling."
        )
        uninstall_first = True
    elif ort_status == "present_outside_venv":
        _print_yellow(
            "[warn] Found onnxruntime outside the remote venv at: "
            f"{ort_path}. It will be ignored and ORT_QNN will be installed into "
            f"{remote_venv_dir}."
        )
    else:
        _print_yellow(
            "[warn] onnxruntime is not available in the remote venv. "
            f"Installing ORT_QNN into {remote_venv_dir}."
        )

    if uninstall_first:
        _uninstall_remote_ort(
            adb_serial=adb_serial,
            remote_venv_dir=remote_venv_dir,
            wheel_distribution=required_dist,
        )

    remote_wheel_path = f"{remote_wheel_dir}/{local_ort_qnn_wheel_path.name}"
    _adb_shell(adb_serial, f"mkdir -p {shlex.quote(remote_wheel_dir)}")
    _adb_push(adb_serial, str(local_ort_qnn_wheel_path), remote_wheel_path)
    install_wheel_cmd = (
        "set -e; "
        'export PATH="$HOME/.local/bin:$PATH"; '
        'export PATH="$(python3 -m site --user-base)/bin:$PATH"; '
        'UV_BIN="$(command -v uv)"; '
        f". {shlex.quote(remote_venv_dir + '/bin/activate')}; "
        f'"$UV_BIN" pip install --force-reinstall {shlex.quote(remote_wheel_path)}'
    )
    _adb_shell(adb_serial, install_wheel_cmd)
    ort_status, ort_path, ort_version = _probe_remote_ort_origin(
        adb_serial=adb_serial,
        remote_venv_dir=remote_venv_dir,
        remote_python=remote_python,
    )
    if ort_status != "present_in_venv":
        raise RuntimeError(
            "Remote onnxruntime must resolve inside the target venv after "
            f"installation. Expected prefix {remote_venv_dir}, got "
            f"{ort_path or '<missing>'} (status: {ort_status})."
        )
    if ort_version != required_version:
        raise RuntimeError(
            "Remote onnxruntime version does not match the bundled wheel after "
            f"installation. Expected {required_version}, got "
            f"{ort_version or '<unknown>'}."
        )


def ensure_adb_runtime_venv(
    adb_serial: str | None,
    local_requirements_path: str,
    remote_base_dir: str = PERSISTENT_ADB_RUNTIME_BASE_DIR,
    remote_venv_dir: str = PERSISTENT_ADB_RUNTIME_VENV_DIR,
    remote_requirements_path: str = PERSISTENT_ADB_RUNTIME_REQUIREMENTS,
    local_ort_qnn_wheel_path: str | None = None,
) -> str:
    """Ensure persistent remote venv exists and dependencies are installed.

    Returns remote venv Python executable path.
    """
    requirements = Path(local_requirements_path).expanduser().resolve()
    if not requirements.is_file():
        raise FileNotFoundError(f"Target requirements file not found: {requirements}")
    optional_wheel = _validate_optional_wheel(local_ort_qnn_wheel_path)

    _run_cmd(_adb_prefix(adb_serial) + ["get-state"])
    _adb_shell(adb_serial, f"mkdir -p {shlex.quote(remote_base_dir)}")
    _adb_push(adb_serial, str(requirements), remote_requirements_path)
    remote_python = f"{remote_venv_dir}/bin/python"
    venv_state_cmd = (
        f"if [ -x {shlex.quote(remote_python)} ]; then "
        "echo __VENV_EXISTS__; "
        "else echo __VENV_MISSING__; fi"
    )
    venv_state_out = _adb_shell_capture(adb_serial, venv_state_cmd)
    venv_exists = "__VENV_EXISTS__" in venv_state_out
    if venv_exists:
        _print_yellow(
            f"[warn] Reusing remote virtual environment at: {remote_venv_dir}"
        )
    else:
        _print_yellow(
            f"[warn] Creating remote virtual environment at: {remote_venv_dir}"
        )

    resolve_uv_cmd = (
        "set -e; "
        'export PATH="$HOME/.local/bin:$PATH"; '
        "if ! command -v uv >/dev/null 2>&1; then "
        "python3 -m pip install --user uv; "
        "fi; "
        'export PATH="$(python3 -m site --user-base)/bin:$PATH"; '
        'UV_BIN="$(command -v uv || true)"; '
        'if [ -z "$UV_BIN" ]; then '
        "echo '[error] uv is not available after installation. "
        "Ensure target PATH includes user bin or install uv system-wide.' >&2; "
        "exit 1; "
        "fi"
    )
    _adb_shell(adb_serial, resolve_uv_cmd)

    create_venv_cmd = (
        "set -e; "
        'export PATH="$HOME/.local/bin:$PATH"; '
        'export PATH="$(python3 -m site --user-base)/bin:$PATH"; '
        'UV_BIN="$(command -v uv)"; '
        f"if [ ! -x {shlex.quote(remote_python)} ]; then "
        f'"$UV_BIN" venv --system-site-packages {shlex.quote(remote_venv_dir)}; '
        "fi; "
        f"if [ ! -x {shlex.quote(remote_python)} ]; then "
        f"echo '[error] remote venv creation failed: {remote_python} not found.' >&2; "
        "exit 1; "
        "fi"
    )
    _adb_shell(adb_serial, create_venv_cmd)
    if not venv_exists:
        _print_yellow(
            f"[warn] Remote virtual environment created at: {remote_venv_dir}"
        )

    install_deps_cmd = (
        "set -e; "
        'export PATH="$HOME/.local/bin:$PATH"; '
        'export PATH="$(python3 -m site --user-base)/bin:$PATH"; '
        'UV_BIN="$(command -v uv)"; '
        f". {shlex.quote(remote_venv_dir + '/bin/activate')}; "
        f'"$UV_BIN" pip install -r {shlex.quote(remote_requirements_path)}'
    )
    _adb_shell(adb_serial, install_deps_cmd)
    _install_optional_ort_qnn_wheel(
        adb_serial=adb_serial,
        remote_venv_dir=remote_venv_dir,
        remote_python=remote_python,
        local_ort_qnn_wheel_path=optional_wheel,
    )
    return remote_python
