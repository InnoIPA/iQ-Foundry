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
"""Raw model outputs on disk: one <image>_<role>.npy file per output role.

The target writes these files after running the model and the host reads them back, so both
sides must agree on the names. The roles of a task are listed in specs.TaskSpec.outputs.
"""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


def raw_output_path(directory, stem: str, role: str) -> Path:
    return Path(directory) / f"{stem}_{role}.npy"


INVOKE_MS_FILE = "avg_model_invoke_ms.txt"


def save_avg_invoke_ms(directory, invoke_seconds: list) -> None:
    """Write the mean model invoke time of a batch, in ms, next to the raw outputs."""
    if invoke_seconds:
        avg_ms = sum(invoke_seconds) / len(invoke_seconds) * 1000.0
        (Path(directory) / INVOKE_MS_FILE).write_text(f"{avg_ms:.3f}\n")


def load_avg_invoke_ms(directory):
    """The value written by save_avg_invoke_ms, or None if there is none."""
    path = Path(directory) / INVOKE_MS_FILE
    return float(path.read_text()) if path.exists() else None


def save_raw_outputs(directory, stem: str, outputs: dict) -> None:
    """Write {role: array} as <stem>_<role>.npy files."""
    for role, array in outputs.items():
        np.save(raw_output_path(directory, stem, role), array)


def load_raw_outputs(directory, stem: str, roles) -> dict:
    """Read the files written by save_raw_outputs back as {role: array}."""
    paths = {role: raw_output_path(directory, stem, role) for role in roles}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    return {role: np.load(path) for role, path in paths.items()}
