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
"""Token-safe QAI Hub login through the product's own ./qaihub_login.sh.

The API token is read ONLY from the IQF_QAI_HUB_TOKEN environment variable, set for this one
command. It is never accepted as an argument, never written to a file by this script, never
printed (all child output is masked as ***), and removed from the environment before any child
process starts. After a successful login ~/.qai_hub is restricted to the current user
(directory 700, files 600).

Recommended use (the user types the token hidden in their own terminal):

    read -rsp "QAI Hub API token: " IQF_QAI_HUB_TOKEN && echo && \
      IQF_QAI_HUB_TOKEN="$IQF_QAI_HUB_TOKEN" python3 .agents/skills/iqf-assistant/scripts/hub_login.py; \
      unset IQF_QAI_HUB_TOKEN

The login logic mirrors `hub-login` in tool/test/regression/run_pipelines.py, which cannot be
imported here because it requires PyYAML on the host.

Exit codes: 0 = login saved, 1 = error (the token is never part of the message).
"""

from __future__ import annotations

import argparse
import os
import pty
import sys
from pathlib import Path

TOKEN_ENV = "IQF_QAI_HUB_TOKEN"
QAI_HUB_SECRET_FILES = ("client.ini", "client.ini.bak")


def qai_hub_dir() -> Path:
    return Path.home() / ".qai_hub"


def default_repo_root() -> Path:
    # <repo>/.agents/skills/iqf-assistant/scripts/hub_login.py
    return Path(__file__).resolve().parents[4]


class TokenMasker:
    """Mask a secret in streamed output, even when it is split across read chunks."""

    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.pending = ""

    def feed(self, text: str) -> str:
        # Mask every complete occurrence, then hold back the last len(secret) - 1 characters:
        # they may be the beginning of a secret that the next chunk completes.
        self.pending = (self.pending + text).replace(self.secret, "***")
        keep = len(self.secret) - 1
        if len(self.pending) <= keep:
            return ""
        ready, self.pending = self.pending[:-keep], self.pending[-keep:]
        return ready

    def flush(self) -> str:
        ready, self.pending = self.pending, ""
        return ready.replace(self.secret, "***")


def token_problem(token: str) -> str | None:
    if not token:
        return f"${TOKEN_ENV} is not set; set it only for this one command"
    if len(token) < 16 or any(ch.isspace() for ch in token):
        return (
            f"${TOKEN_ENV} does not look like a QAI Hub API token (too short or contains "
            "spaces; often a copy/paste slip). Nothing was run."
        )
    return None


def tighten_permissions(hub_dir: Path) -> None:
    # qai-hub writes client.ini (and client.ini.bak with the previous token) with the default
    # umask, which is often readable by other users. Only the owner needs them.
    hub_dir.chmod(0o700)
    for name in QAI_HUB_SECRET_FILES:
        path = hub_dir / name
        if path.is_file():
            path.chmod(0o600)


def run_masked(argv: list[str], cwd: Path, env: dict[str, str], token: str) -> int:
    """Run argv in a pseudo-terminal (qaihub_login.sh uses `docker run -it`), masking token."""
    pid, fd = pty.fork()
    if pid == 0:  # child
        try:
            os.chdir(cwd)
            os.execve(argv[0], argv, env)
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
    return os.waitstatus_to_exitcode(status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Token-safe QAI Hub login. The token is read only from $"
        + TOKEN_ENV
        + "."
    )
    parser.add_argument(
        "--repo", help="iQ-Foundry folder (default: the one containing this skill)"
    )
    parser.add_argument(
        "--image", help="Docker image for the login (default: same as ./docker/iqf)"
    )
    args = parser.parse_args(argv)

    token = os.environ.pop(TOKEN_ENV, "").strip()
    problem = token_problem(token)
    if problem:
        print(f"[error] {problem}")
        return 1

    repo_root = Path(args.repo).resolve() if args.repo else default_repo_root()
    script = repo_root / "qaihub_login.sh"
    if not script.is_file():
        print(f"[error] {script} not found")
        return 1

    hub_dir = qai_hub_dir()
    client_ini = hub_dir / "client.ini"
    if client_ini.exists():
        print(
            f"[warn] {client_ini} exists and will be replaced; qai-hub keeps the previous "
            "login in client.ini.bak"
        )
    login_argv = [str(script), "--key", token]
    shown = f"{script} --key ***"
    if args.image:
        login_argv += ["--image", args.image]
        shown += f" --image {args.image}"
    print(f"[info] running: {shown}")
    sys.stdout.flush()

    before = client_ini.stat().st_mtime if client_ini.exists() else None
    child_env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
    code = run_masked(login_argv, repo_root, child_env, token)
    del token, login_argv

    after = client_ini.stat().st_mtime if client_ini.exists() else None
    if code != 0:
        print(f"[error] qaihub_login.sh exited {code}")
        return 1
    if after is None or after == before:
        print(f"[error] {client_ini} was not written")
        return 1
    try:
        tighten_permissions(hub_dir)
        print(f"[ok] restricted {hub_dir} to the current user (dir 700, files 600)")
    except OSError as exc:
        print(f"[warn] could not restrict permissions on {hub_dir}: {exc}")
    print(f"[ok] QAI Hub login saved to {client_ini}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
