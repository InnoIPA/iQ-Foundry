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
"""Run one long iQ-Foundry command (e.g. `./docker/iqf run qc ...`) detached, with a log.

Agent shell tools often kill everything a command started once it returns, and qc can take
10-30 minutes. `start` launches a small supervisor in its own session; the supervisor runs the
command, writes all output to output.log and records the exit code in state.json.

    iqf_job.py start --name qc -- ./docker/iqf run qc --type yolov26 ...
    iqf_job.py status <job_dir>          # STATE: running | finished | died, plus the log tail
    iqf_job.py stop <job_dir>            # only when the user asks

When the command finished, `status` also prints one OUTPUT: line per "[ok] wrote: <path>" line
that cli.py logged, i.e. the exact host path of each result.

Job folders live in <repo>/out/iqf-assistant/jobs/<timestamp>_<name>/ (command.txt,
output.log, state.json). Only one job may run at a time, because the device is shared.
Never put a secret in the command: command.txt and output.log are plain files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# cli.py prints the host path of every result it writes as "[ok] wrote: <path>".
WROTE_RE = re.compile(r"\[ok\] wrote: (.+?)\s*$")


def default_repo_root() -> Path:
    # <repo>/.agents/skills/iqf-assistant/scripts/iqf_job.py
    return Path(__file__).resolve().parents[4]


def jobs_root(repo_root: Path) -> Path:
    return repo_root / "out" / "iqf-assistant" / "jobs"


def read_state(job_dir: Path) -> dict:
    try:
        return json.loads((job_dir / "state.json").read_text())
    except (OSError, ValueError):
        return {}


def write_state(job_dir: Path, state: dict) -> None:
    tmp = job_dir / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(job_dir / "state.json")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A finished child that nobody reaped yet is a zombie: treat it as not alive.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[-1].split()[0] != "Z"
    except OSError:
        return True


def job_state(job_dir: Path) -> str:
    state = read_state(job_dir)
    if not state:
        return "unknown"
    if "exit_code" in state:
        return "finished"
    if pid_alive(state.get("supervisor_pid")):
        return "running"
    return "died"


def tail(path: Path, lines: int) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    # Progress bars rewrite the line with \r; keep only the last version of each line.
    cleaned = [ANSI_RE.sub("", line.split("\r")[-1]) for line in text.splitlines()]
    return "\n".join(cleaned[-lines:])


def running_jobs(repo_root: Path) -> list[Path]:
    root = jobs_root(repo_root)
    if not root.is_dir():
        return []
    return [
        d for d in sorted(root.iterdir()) if d.is_dir() and job_state(d) == "running"
    ]


def written_outputs(log_path: Path) -> list[str]:
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return []
    found = []
    for line in text.splitlines():
        match = WROTE_RE.search(ANSI_RE.sub("", line.split("\r")[-1]))
        if match and match.group(1) not in found:
            found.append(match.group(1))
    return found


def print_result(job_dir: Path, state: dict) -> None:
    print(f"FINISHED: {state.get('finished_at')}")
    print(f"EXIT_CODE: {state.get('exit_code')}")
    for path in written_outputs(job_dir / "output.log"):
        print(f"OUTPUT: {path}")


def cmd_start(args) -> int:
    repo_root = Path(args.repo).resolve() if args.repo else default_repo_root()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print(
            "[error] no command given; usage: iqf_job.py start --name <name> -- <command...>"
        )
        return 1
    busy = running_jobs(repo_root)
    if busy:
        print(
            f"[error] another job is still running: {busy[0]}; wait for it or stop it first"
        )
        return 1

    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.name)[:40] or "job"
    job_dir = jobs_root(repo_root) / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    try:
        job_dir.mkdir(parents=True, exist_ok=False)
    except PermissionError:
        print(
            f"[error] cannot write to {job_dir.parent} (it belongs to another user, usually "
            "because Docker created out/ first). Ask the user to run in their own terminal: "
            f"sudo chown -R $USER {repo_root / 'out'}"
        )
        return 1
    (job_dir / "command.txt").write_text(shlex.join(command) + "\n")
    write_state(
        job_dir,
        {
            "name": name,
            "command": command,
            "cwd": str(repo_root),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    supervisor = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_supervise", str(job_dir)],
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    state = read_state(job_dir)
    state["supervisor_pid"] = supervisor.pid
    write_state(job_dir, state)

    deadline = time.time() + args.wait_s
    while time.time() < deadline:
        if supervisor.poll() is not None:
            break
        time.sleep(0.5)
    current = job_state(job_dir)
    print(f"JOB_DIR: {job_dir}")
    print(f"LOG: {job_dir / 'output.log'}")
    print(f"STATE: {current}")
    if current == "finished":
        print_result(job_dir, read_state(job_dir))
    elif current == "died":
        print("[error] the job stopped right away; see the log above")
        print(tail(job_dir / "output.log", 20))
        return 1
    return 0


def cmd_supervise(args) -> int:
    job_dir = Path(args.job_dir)
    state = read_state(job_dir)
    with (job_dir / "output.log").open("ab", buffering=0) as log:
        try:
            child = subprocess.Popen(
                state["command"],
                cwd=state.get("cwd") or None,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            log.write(f"[error] could not start the command: {exc}\n".encode())
            state.update(
                exit_code=127, finished_at=datetime.now().isoformat(timespec="seconds")
            )
            write_state(job_dir, state)
            return 127
        state["child_pid"] = child.pid
        write_state(job_dir, state)

        def forward(signum, _frame):
            child.send_signal(signum)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        code = child.wait()
    state = read_state(job_dir)
    state.update(
        exit_code=code, finished_at=datetime.now().isoformat(timespec="seconds")
    )
    write_state(job_dir, state)
    return code


def cmd_status(args) -> int:
    job_dir = Path(args.job_dir)
    if not (job_dir / "state.json").is_file():
        print(f"[error] {job_dir} is not a job folder")
        return 1
    state = read_state(job_dir)
    current = job_state(job_dir)
    print(f"JOB_DIR: {job_dir}")
    print(f"COMMAND: {(job_dir / 'command.txt').read_text().strip()}")
    print(f"STARTED: {state.get('started_at')}")
    print(f"STATE: {current}")
    if current == "finished":
        print_result(job_dir, state)
    print(f"--- last {args.lines} log lines ({job_dir / 'output.log'}) ---")
    print(tail(job_dir / "output.log", args.lines))
    return 0


def cmd_stop(args) -> int:
    job_dir = Path(args.job_dir)
    state = read_state(job_dir)
    if job_state(job_dir) != "running":
        print(f"[info] job is not running (state: {job_state(job_dir)})")
        return 0
    pid = state.get("supervisor_pid")
    os.killpg(pid, signal.SIGTERM)
    for _ in range(60):
        if job_state(job_dir) != "running":
            break
        time.sleep(0.5)
    print(f"STATE: {job_state(job_dir)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one long iQ-Foundry command detached, with a log."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("start", help="start a command in the background")
    p.add_argument("--name", required=True, help="short job name, e.g. qc, mAP, test")
    p.add_argument(
        "--repo", help="iQ-Foundry folder (default: the one containing this skill)"
    )
    p.add_argument(
        "--wait-s",
        type=float,
        default=5.0,
        help="seconds to wait before reporting (default 5)",
    )
    p.add_argument(
        "command", nargs=argparse.REMAINDER, help="-- followed by the command"
    )
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("status", help="show the state and log tail of a job")
    p.add_argument("job_dir")
    p.add_argument("--lines", type=int, default=20)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("stop", help="stop a running job")
    p.add_argument("job_dir")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("_supervise")
    p.add_argument("job_dir")
    p.set_defaults(func=cmd_supervise)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
