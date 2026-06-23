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
# limitations under the License.set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "[error] python3 or python not found on PATH. Required to resolve the default Docker image." >&2
    exit 1
fi
DOCKER_BIN="${IQF_DOCKER_BIN:-docker}"
DEFAULT_IMAGE="$(
"${PYTHON_BIN}" - "$SCRIPT_DIR/docker/iqf_path_mapper.py" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
match = re.search(r'^DEFAULT_IMAGE\s*=\s*"([^"]+)"', text, re.MULTILINE)
if not match:
    raise SystemExit(f"[error] Could not resolve DEFAULT_IMAGE from {path}")
print(match.group(1))
PY
)"
IMAGE_NAME="${IQF_DOCKER_IMAGE:-$DEFAULT_IMAGE}"
API_KEY=""

print_usage() {
    cat <<'EOF'
Usage:
  ./qaihub_login.sh --key <YOUR_QAI_HUB_API_KEY> [--image <docker-image>]

Examples:
  ./qaihub_login.sh --key <YOUR_QAI_HUB_API_KEY>
  ./qaihub_login.sh --key <YOUR_QAI_HUB_API_KEY> --image innodiskorg/iqf:latest

Notes:
  - Writes the QAI Hub config to ~/.qai_hub on the WSL host.
  - Uses Docker to run qai-hub configure inside the image.
  - Defaults to the same Docker image configured in docker/iqf_path_mapper.py.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --key)
            if [[ $# -lt 2 ]]; then
                echo "[error] --key requires a value" >&2
                exit 1
            fi
            API_KEY="$2"
            shift 2
            ;;
        --image)
            if [[ $# -lt 2 ]]; then
                echo "[error] --image requires a value" >&2
                exit 1
            fi
            IMAGE_NAME="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "[error] Unknown argument: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${API_KEY}" ]]; then
    echo "[error] Missing required argument: --key" >&2
    print_usage >&2
    exit 1
fi

if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
    echo "[error] docker not found on PATH. Install Docker Engine in WSL first." >&2
    exit 1
fi

mkdir -p "${HOME}/.qai_hub"

"${DOCKER_BIN}" run --rm -it \
    -v "${HOME}/.qai_hub:/root/.qai_hub" \
    "${IMAGE_NAME}" \
    qai-hub configure --api_token "${API_KEY}"

echo "[ok] QAI Hub config saved to ${HOME}/.qai_hub/client.ini"
