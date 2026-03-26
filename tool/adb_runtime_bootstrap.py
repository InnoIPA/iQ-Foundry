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


def ensure_adb_runtime_venv(
    adb_serial: str | None,
    local_requirements_path: str,
    remote_base_dir: str = PERSISTENT_ADB_RUNTIME_BASE_DIR,
    remote_venv_dir: str = PERSISTENT_ADB_RUNTIME_VENV_DIR,
    remote_requirements_path: str = PERSISTENT_ADB_RUNTIME_REQUIREMENTS,
) -> str:
    """Ensure persistent remote venv exists and dependencies are installed.

    Returns remote venv Python executable path.
    """
    requirements = Path(local_requirements_path).expanduser().resolve()
    if not requirements.is_file():
        raise FileNotFoundError(f"Target requirements file not found: {requirements}")

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
    return remote_python
