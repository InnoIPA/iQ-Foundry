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
"""Read-only iQ-Foundry host setup check for Ubuntu 22.04 and WSL (Windows host).

Checks every prerequisite from Ubuntu_host.md / Windows_host.md and prints one line per check:

    <STATUS>  <check id>  <detail>

STATUS is OK, WARN or MISSING. The fix for every check id is described in
references/setup_fixes.md. Nothing is installed or configured. The only side effect is
`adb kill-server` on the host after listing devices, which the mode docs require anyway
(a host adb server hides the device from the container); pass --no-kill-adb to skip it.

The QAI Hub credential file (~/.qai_hub/client.ini) is never opened: only its existence and
permission bits are checked. --check-hub-login asks QAI Hub for the target device from inside
the iqf image with the file mounted read-only, and prints only the device count.

Exit codes: 0 = nothing MISSING, 2 = something MISSING, 1 = usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_IMAGE = "innodiskorg/iqf:latest"
QAI_HUB_DIR = Path.home() / ".qai_hub"
QAI_HUB_SECRET_FILES = ("client.ini", "client.ini.bak")
HUB_DEVICE_NAME = "Dragonwing IQ-9075 EVK"
MIN_RAM_GIB = 15.0  # 16 GB machines report slightly less than 16 GiB
MIN_FREE_DISK_GIB = 20.0

OK, WARN, MISSING = "OK", "WARN", "MISSING"


@dataclass
class Check:
    id: str
    status: str
    detail: str


def default_repo_root() -> Path:
    # <repo>/.agents/skills/iqf-assistant/scripts/check_setup.py
    return Path(__file__).resolve().parents[4]


def resolve_image(repo_root: Path, override: str | None) -> str:
    """Same precedence as ./docker/iqf: --image, then $IQF_DOCKER_IMAGE, then the default."""
    if override:
        return override
    mapper_dir = repo_root / "docker"
    if (mapper_dir / "iqf_path_mapper.py").is_file():
        sys.path.insert(0, str(mapper_dir))
        try:
            from iqf_path_mapper import resolve_image_name  # type: ignore

            return resolve_image_name(None)
        except Exception:
            pass
        finally:
            sys.path.pop(0)
    return os.environ.get("IQF_DOCKER_IMAGE") or DEFAULT_IMAGE


def run(argv: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return 127, f"{argv[0]}: command not found"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


# ----------------------------
# Parsers (pure, unit-tested)
# ----------------------------


def parse_os_release(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"')
    return out


def parse_meminfo_gib(text: str) -> float | None:
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1)) / (1024 * 1024)


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


def wsl_conf_has_systemd(text: str) -> bool:
    section = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
        elif section == "boot" and "=" in line:
            key, value = (part.strip().lower() for part in line.split("=", 1))
            if key == "systemd" and value == "true":
                return True
    return False


def is_wsl(proc_version: str, env: dict[str, str]) -> bool:
    return bool(env.get("WSL_DISTRO_NAME")) or "microsoft" in proc_version.lower()


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


# ----------------------------
# Checks
# ----------------------------


def check_os(wsl: bool) -> Check:
    info = parse_os_release(_read("/etc/os-release"))
    name = info.get("PRETTY_NAME") or info.get("NAME") or platform.platform()
    arch = platform.machine()
    where = "WSL (Windows host)" if wsl else "native Linux"
    detail = f"{name}, {arch}, {where}"
    if info.get("ID") != "ubuntu":
        return Check("os", MISSING, f"{detail}; iQ-Foundry is tested on Ubuntu 22.04")
    if arch not in ("x86_64", "amd64"):
        return Check("os", MISSING, f"{detail}; an x86_64 host is required")
    if info.get("VERSION_ID") != "22.04":
        return Check("os", WARN, f"{detail}; the tested version is Ubuntu 22.04")
    return Check("os", OK, detail)


def check_ram() -> Check:
    gib = parse_meminfo_gib(_read("/proc/meminfo"))
    if gib is None:
        return Check("ram", WARN, "could not read /proc/meminfo")
    detail = f"{gib:.1f} GiB"
    if gib < MIN_RAM_GIB:
        return Check("ram", WARN, f"{detail}; at least 16 GB is recommended")
    return Check("ram", OK, detail)


def check_disk(repo_root: Path) -> Check:
    try:
        free = shutil.disk_usage(repo_root).free / (1024**3)
    except OSError as exc:
        return Check("disk", WARN, f"could not check free space: {exc}")
    detail = f"{free:.0f} GiB free at {repo_root}"
    if free < MIN_FREE_DISK_GIB:
        return Check("disk", WARN, f"{detail}; models, images and results need space")
    return Check("disk", OK, detail)


def check_repo(repo_root: Path) -> Check:
    wrapper = repo_root / "docker" / "iqf"
    if not wrapper.is_file():
        return Check(
            "repo", MISSING, f"{wrapper} not found; is this the iQ-Foundry folder?"
        )
    if not os.access(wrapper, os.X_OK):
        return Check("repo", MISSING, f"{wrapper} is not executable")
    return Check("repo", OK, str(repo_root))


def check_docker() -> list[Check]:
    if not shutil.which("docker"):
        return [Check("docker_cli", MISSING, "docker command not found")]
    checks = [Check("docker_cli", OK, shutil.which("docker") or "docker")]

    user = os.environ.get("USER") or ""
    _, groups_now = run(["id", "-nG"])
    in_session = "docker" in groups_now.split()
    _, group_line = run(["getent", "group", "docker"])
    members = (
        group_line.strip().rsplit(":", 1)[-1].split(",") if group_line.strip() else []
    )
    if in_session:
        checks.append(
            Check("docker_group", OK, f"{user or 'user'} is in the docker group")
        )
    elif user and user in members:
        checks.append(
            Check(
                "docker_group",
                MISSING,
                "you were added to the docker group, but this terminal session started "
                "before that; log out and back in (or close and reopen WSL)",
            )
        )
    elif not group_line.strip():
        checks.append(Check("docker_group", MISSING, "the docker group does not exist"))
    else:
        checks.append(
            Check(
                "docker_group", MISSING, f"{user or 'user'} is not in the docker group"
            )
        )

    code, out = run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=30)
    if code == 0 and out.strip():
        checks.append(
            Check("docker_daemon", OK, f"Docker Engine {out.strip().splitlines()[0]}")
        )
    else:
        reason = out.strip().splitlines()[-1] if out.strip() else f"exit {code}"
        if "permission denied" in out.lower():
            reason = "permission denied talking to Docker (docker group not active yet)"
        elif (
            "cannot connect" in out.lower()
            or "is the docker daemon running" in out.lower()
        ):
            reason = "the Docker service is not running"
        checks.append(Check("docker_daemon", MISSING, reason))
    return checks


def check_image(image: str, docker_ok: bool) -> Check:
    if not docker_ok:
        return Check("image", MISSING, f"{image}: cannot check until Docker works")
    code, out = run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], timeout=30
    )
    if code == 0:
        return Check("image", OK, f"{image} ({out.strip()[:19]})")
    return Check("image", MISSING, f"{image} is not pulled yet")


def hub_permission_warnings() -> list[str]:
    messages = []
    for name in QAI_HUB_SECRET_FILES:
        path = QAI_HUB_DIR / name
        try:
            mode = path.stat().st_mode
        except OSError:
            continue
        if stat.S_ISREG(mode) and mode & 0o077:
            messages.append(
                f"{path} is readable by other users (mode {oct(mode & 0o777)})"
            )
    return messages


def check_qai_hub_file() -> Check:
    # Existence and permission bits only. The file holds the API token: never open it.
    ini = QAI_HUB_DIR / "client.ini"
    if not ini.is_file():
        return Check(
            "qai_hub", MISSING, f"{ini} not found (QAI Hub login not done yet)"
        )
    warnings = hub_permission_warnings()
    if warnings:
        return Check("qai_hub", WARN, "login file present, but " + "; ".join(warnings))
    return Check("qai_hub", OK, f"login file present ({ini})")


def check_qai_hub_login(image: str) -> Check:
    snippet = (
        "import qai_hub as hub; "
        f"d = hub.get_devices(name='{HUB_DEVICE_NAME}'); "
        "print('hub_devices_found=%d' % len(d))"
    )
    code, out = run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{QAI_HUB_DIR}:/root/.qai_hub:ro",
            "--entrypoint",
            "python3",
            image,
            "-c",
            snippet,
        ],
        timeout=180,
    )
    match = re.search(r"hub_devices_found=(\d+)", out)
    if code == 0 and match:
        return Check(
            "qai_hub_login",
            OK,
            f"QAI Hub accepted the login ({match.group(1)} target device(s) visible)",
        )
    lines = [line for line in out.strip().splitlines() if line.strip()]
    last = lines[-1] if lines else f"exit {code}"
    return Check("qai_hub_login", MISSING, f"QAI Hub login check failed: {last[:200]}")


def check_usb() -> Check:
    if Path("/dev/bus/usb").is_dir():
        return Check("usb", OK, "/dev/bus/usb is available")
    return Check(
        "usb",
        MISSING,
        "/dev/bus/usb not found (USB is not passed through to this system)",
    )


def list_devices(
    image: str, image_ok: bool, kill_host_adb: bool
) -> tuple[list[dict[str, str]], str]:
    devices: list[dict[str, str]] = []
    source = "none"
    if shutil.which("adb"):
        code, out = run(["adb", "devices", "-l"], timeout=60)
        devices = parse_adb_devices(out) if code == 0 else []
        source = "host adb"
        if kill_host_adb:
            # A host adb server holding the USB interface hides the device from the container.
            run(["adb", "kill-server"], timeout=30)
    if not devices and image_ok and Path("/dev/bus/usb").is_dir():
        code, out = run(
            [
                "docker",
                "run",
                "--rm",
                "--privileged",
                "-v",
                "/dev/bus/usb:/dev/bus/usb",
                "--entrypoint",
                "bash",
                image,
                "-lc",
                "adb start-server >/dev/null 2>&1; adb devices -l; adb kill-server >/dev/null 2>&1",
            ],
            timeout=120,
        )
        devices = parse_adb_devices(out) if code == 0 else []
        source = "container adb"
    return devices, source


def check_device(devices: list[dict[str, str]], source: str) -> Check:
    if source == "none":
        return Check(
            "device",
            MISSING,
            "could not look for the device (Docker image or USB not available)",
        )
    ready = [d for d in devices if d.get("state") == "device"]
    others = [d for d in devices if d.get("state") != "device"]
    if len(ready) == 1:
        d = ready[0]
        label = d.get("model") or d.get("product") or "device"
        return Check(
            "device", OK, f"1 device attached: {d['serial']} ({label}, via {source})"
        )
    if len(ready) > 1:
        serials = ", ".join(d["serial"] for d in ready)
        return Check(
            "device",
            WARN,
            f"{len(ready)} devices attached ({serials}); you will be asked which one to use",
        )
    if others:
        states = ", ".join(f"{d['serial']}={d['state']}" for d in others)
        return Check("device", MISSING, f"device seen but not usable: {states}")
    return Check("device", MISSING, f"no EXMP-Q911 device found (via {source})")


def check_wsl_systemd() -> Check:
    if wsl_conf_has_systemd(_read("/etc/wsl.conf")):
        return Check("wsl_systemd", OK, "systemd is enabled in /etc/wsl.conf")
    return Check(
        "wsl_systemd",
        WARN,
        "systemd is not enabled in /etc/wsl.conf (setup-windows-wsl.ps1 enables it)",
    )


# ----------------------------
# Main
# ----------------------------


def readiness(checks: list[Check]) -> dict[str, bool]:
    status = {c.id: c.status for c in checks}

    def ok(*ids: str) -> bool:
        # Checks that were not run (e.g. --skip-device) do not count against a mode.
        return all(status[i] != MISSING for i in ids if i in status)

    base = ("os", "repo", "docker_cli", "docker_group", "docker_daemon", "image")
    qc_ready = ok(*base, "qai_hub", "qai_hub_login")
    device_ready = ok(*base, "usb", "device")
    return {"qc": qc_ready, "mAP": device_ready, "test": device_ready}


def collect(args) -> list[Check]:
    repo_root = Path(args.repo).resolve() if args.repo else default_repo_root()
    image = resolve_image(repo_root, args.image)
    wsl = is_wsl(_read("/proc/version"), dict(os.environ))

    checks = [check_os(wsl), check_ram(), check_disk(repo_root), check_repo(repo_root)]
    if wsl:
        checks.append(check_wsl_systemd())
    docker_checks = check_docker()
    checks += docker_checks
    docker_ok = any(c.id == "docker_daemon" and c.status == OK for c in docker_checks)
    image_check = check_image(image, docker_ok)
    checks.append(image_check)
    image_ok = image_check.status == OK

    hub_file = check_qai_hub_file()
    checks.append(hub_file)
    if args.check_hub_login:
        if hub_file.status == MISSING:
            checks.append(Check("qai_hub_login", MISSING, "skipped: no login file yet"))
        elif not image_ok:
            checks.append(
                Check(
                    "qai_hub_login",
                    MISSING,
                    "skipped: the Docker image is not available",
                )
            )
        else:
            checks.append(check_qai_hub_login(image))

    usb = check_usb()
    checks.append(usb)
    if not args.skip_device:
        devices, source = list_devices(
            image, image_ok, kill_host_adb=not args.no_kill_adb
        )
        checks.append(check_device(devices, source))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo", help="iQ-Foundry folder (default: the one containing this skill)"
    )
    parser.add_argument(
        "--image", help="Docker image to check (default: same as ./docker/iqf)"
    )
    parser.add_argument(
        "--check-hub-login",
        action="store_true",
        help="also ask QAI Hub whether the saved login works",
    )
    parser.add_argument(
        "--skip-device",
        action="store_true",
        help="do not look for the EXMP-Q911 device",
    )
    parser.add_argument(
        "--no-kill-adb",
        action="store_true",
        help="do not run `adb kill-server` on the host",
    )
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON"
    )
    args = parser.parse_args(argv)

    if sys.platform.startswith("win"):
        print(
            "[error] this checker runs inside Ubuntu/WSL; on Windows use check_setup_windows.ps1"
        )
        return 1
    if sys.platform == "darwin":
        print(
            "[error] macOS hosts are not supported by iQ-Foundry (Ubuntu 22.04 or Windows 11 + WSL)"
        )
        return 1

    checks = collect(args)
    ready = readiness(checks)
    missing = [c.id for c in checks if c.status == MISSING]

    if args.json:
        print(
            json.dumps(
                {
                    "checks": [asdict(c) for c in checks],
                    "ready_for": ready,
                    "missing": missing,
                },
                indent=2,
            )
        )
    else:
        width = max(len(c.id) for c in checks)
        for c in checks:
            print(f"{c.status:<8} {c.id:<{width}}  {c.detail}")
        print(
            "ready_for: "
            + " ".join(
                f"{mode}={'yes' if value else 'no'}" for mode, value in ready.items()
            )
        )
        if missing:
            print(f"RESULT: NOT_READY (missing: {', '.join(missing)})")
        else:
            print("RESULT: READY")
    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
