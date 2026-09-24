# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
import shutil
import tempfile
from json import load
from logging import getLogger
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple, Union

from qti.aisw.core.model_level_api.target.target import Target
from qti.aisw.core.model_level_api.utils.qnn_sdk import qnn_sdk_root
from qti.aisw.tools.core.modules.api.backend.backend import Backend
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.tools.core.utilities.devices.utils.subprocess_helper import execute


logger = getLogger(__name__)
default_profiling_log_name = "qnn-profiling-data_0.log"
log_counter = 0


class ProfilingData(NamedTuple):
    """Profiling data class"""

    profiling_log: Path
    backend_profiling_artifacts: Optional[List[Path]]


def profiling_log_to_dict(profiling_log: Union[PathLike, str]) -> Dict[str, Any]:
    """A helper function to generate a dictionary from a profiling log.

    Args:
        profiling_log (Union[PathLike, str]): The profiling log to generate a dictionary from

    Returns:
       Dict[str, Any]: A dictionary containing the profiling events reported from the application
                       and backend
    """
    profiling_log = Path(profiling_log)
    if not profiling_log.is_file():
        raise FileNotFoundError("Provided profiling log could not be found")

    sdk_root = qnn_sdk_root()
    target = Target.create_host_target()

    # Change target name for WoS from default which is ARM64x
    if target.target_platform_type == DevicePlatformType.WOS:
        target_name = "aarch64-windows-msvc"
    else:
        target_name = target.target_name

    if target.target_platform_type in [DevicePlatformType.X86_64_LINUX, DevicePlatformType.LINUX_EMBEDDED]:
        qnn_profile_viewer_file_name = "qnn-profile-viewer"
        json_reader = "libQnnJsonProfilingReader.so"
    elif target.target_platform_type in [DevicePlatformType.X86_64_WINDOWS_MSVC, DevicePlatformType.WOS]:
        qnn_profile_viewer_file_name = "qnn-profile-viewer.exe"
        json_reader = "QnnJsonProfilingReader.dll"
    else:
        raise ValueError(f"Unsupported target for profiling log to dictionary: {target.target_platform_type}")

    qnn_profile_viewer = Path(sdk_root, "bin", target_name, qnn_profile_viewer_file_name)
    if not qnn_profile_viewer.is_file():
        raise FileNotFoundError(f"Could not locate {qnn_profile_viewer} in QNN SDK")

    json_reader = Path(sdk_root, "lib", target_name, json_reader)
    if not json_reader.is_file():
        raise FileNotFoundError(f"Could not locate {json_reader} in QNN SDK")

    with TemporaryDirectory() as temp_dir:
        output_json = Path(temp_dir, "profiling_output.json")

        profile_viewer_args = (
            f"--input_log {profiling_log} --output {output_json} "
            f" --standardized_json_output"
            f" --reader {json_reader}"
        )

        logger.debug(f"Running command: {qnn_profile_viewer} {profile_viewer_args}")
        completed_process = execute(str(qnn_profile_viewer), profile_viewer_args.split())
        if completed_process.returncode != 0:
            raise RuntimeError(
                f"qnn-profile-viewer execution failed, stdout: "
                f"{completed_process.stdout}, stderr: {completed_process.stderr}"
            )

        with output_json.open() as f:
            output_dict = load(f)

    return output_dict


def generate_optrace_profiling_output(
    schematic_bin: Union[PathLike, str],
    profiling_log: Union[PathLike, str],
    output_dir: Union[PathLike, str] = "./output/",
    *,
    qhas_output_type: Literal["html", "json"] = "json",
) -> Tuple[Path, Path]:
    """A helper function to generate optrace artifacts (a JSON viewable via chrometrace and an HTML
    summary report) from a profiling log and a schematic bin.

    Args:
        schematic_bin (Union[PathLike, str]): The schematic bin file created during context binary
                                              generation
        profiling_log (Union[PathLike, str]): The profiling log file created during execution
        output_dir (Union[PathLike, str]): The output directory where outputs will be stored
        qhas_output_type (Literal["html", "json"]): The type of summary output to generate.
                                        `           Either "html" or "json". Defaults to "json".

    Returns:
        Tuple[Path, Path]: A tuple containing (path to output JSON, path to HTML summary report)
    """
    schematic_bin = Path(schematic_bin)
    if not schematic_bin.is_file():
        raise FileNotFoundError("Provided schematic bin could not be found")

    profiling_log = Path(profiling_log)
    if not profiling_log.is_file():
        raise FileNotFoundError("Provided profiling log could not be found")

    sdk_root = qnn_sdk_root()
    target = Target.create_host_target()
    if target.target_platform_type == DevicePlatformType.WOS:
        target_name = "aarch64-windows-msvc"
    else:
        target_name = target.target_name

    if target.target_platform_type in [DevicePlatformType.X86_64_LINUX, DevicePlatformType.LINUX_EMBEDDED]:
        qnn_profile_viewer_file_name = "qnn-profile-viewer"
        Optrace_reader = "libQnnHtpOptraceProfilingReader.so"
    elif target.target_platform_type in [DevicePlatformType.X86_64_WINDOWS_MSVC, DevicePlatformType.WOS]:
        qnn_profile_viewer_file_name = "qnn-profile-viewer.exe"
        Optrace_reader = "QnnHtpOptraceProfilingReader.dll"
    else:
        raise ValueError(
            f"Unsupported target for generating optrace profiling output: {target.target_platform_type}"
        )

    qnn_profile_viewer = Path(sdk_root, "bin", target_name, qnn_profile_viewer_file_name)
    if not qnn_profile_viewer.is_file():
        raise FileNotFoundError(f"Could not locate {qnn_profile_viewer} in QNN SDK")

    htp_optrace_reader = Path(sdk_root, "lib", target_name, Optrace_reader)
    if not htp_optrace_reader.is_file():
        raise FileNotFoundError(f"Could not locate {htp_optrace_reader} in QNN SDK")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / "chrometrace.json"

    # Output qhas as json
    if qhas_output_type == "json":
        config_dict = {"features": {"qhas_schema": True, "qhas_json": True}}

        # create temp file for config json
        tmp_config_json_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=output_dir)
        with open(tmp_config_json_path.name, "w") as f:
            json.dump(config_dict, f)

        output_qhas = output_dir / "chrometrace_qnn_htp_analysis_summary.json"
        config_arg = f" --config {tmp_config_json_path.name}"

    else:
        # Legacy path to dump html summary report
        output_qhas = output_dir / "chrometrace_qnn_htp_analysis_summary.html"
        config_arg = ""

    profile_viewer_args = (
        f"--input_log {profiling_log} --output {output_json} "
        f" --standardized_json_output"
        f" --schematic {schematic_bin} --reader {htp_optrace_reader}"
        f" {config_arg}"
    )
    logger.debug(f"Running command: {qnn_profile_viewer} {profile_viewer_args}")
    completed_process = execute(str(qnn_profile_viewer), profile_viewer_args.split())
    if completed_process.returncode != 0:
        raise RuntimeError(
            f"qnn-profile-viewer execution failed, stdout: "
            f"{completed_process.stdout}, stderr: {completed_process.stderr}"
        )

    if not output_json.is_file():
        raise RuntimeError("Could not locate optrace json after running qnn-profile-viewer")

    if not output_qhas.is_file():
        raise RuntimeError("Could not locate optrace summary after running qnn-profile-viewer")

    return output_json, output_qhas


def move_backend_profiling_artifacts(
    backend: Backend, output_dir: Union[PathLike, str]
) -> Optional[List[Path]]:
    """A helper function to query backend profiling artifacts stored in a temp directory and move
    them to a user-specified output directory

    Args:
        backend (Backend): The Backend instance which may have accumulated profiling artifacts
        output_dir (Union[PathLike, str]): The output directory where the artifacts will be stored

    Returns:
        Optional[List[Path]]: A list of profiling artifacts which have been moved into the output
        directory, or None if the Backend instance did not have any accumulated profiling artifacts
    """
    moved_profiling_artifacts = None
    backend_profiling_artifacts = backend.get_profiling_artifacts()
    if backend_profiling_artifacts:
        moved_profiling_artifacts = []
        for artifact in backend_profiling_artifacts:
            shutil.copy(artifact, output_dir)
            moved_profiling_artifacts.append(Path(output_dir, artifact.name))
        backend.clear_profiling_artifacts()

    return moved_profiling_artifacts


def get_backend_profiling_data(
    backend: Backend, output_dir: Union[PathLike, str], temp_dir: Optional[Union[PathLike, str]] = None
) -> ProfilingData:
    """A helper function to locate and return a profiling log generated during inference or context
    binary generation. Any backend profiling artifacts will be moved and returned as well.

    Args:
        backend (Backend): The Backend instance that will be queried for backend profiling artifacts
        output_dir (Union[PathLike, str]): The user-provided output directory where the profiling
        log and any backend profiling artifacts should be placed in
        temp_dir (Optional[Union[PathLike, str]]): The temporary directory where the profiling log
        was created in. Should be None if the profiling log is already present in output_dir
    Returns:
         ProfilingData: An object containing the profiling log and any backend profiling artifacts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiling_log_dir = Path(temp_dir) if temp_dir else output_dir
    profiling_log = profiling_log_dir / default_profiling_log_name
    if not profiling_log.is_file():
        raise RuntimeError(f"Could not locate profiling log at {profiling_log}")

    # if the log is in a temp directory instead of the desired output directory, move it
    if temp_dir:
        global log_counter

        # use the log counter to avoid overwriting existing profiling logs in the output directory
        moved_profiling_log = (output_dir / f"qnn-profiling-data_{log_counter}.log").resolve()

        # increment the log counter
        log_counter += 1

        # if there is a symlink to a profiling log in the output directory already, the
        # symlink will be followed and the linked file will be overwritten instead of the
        # symlink. To avoid this, remove any existing symlinks before moving the profile log
        # to the output directory.
        if moved_profiling_log.is_symlink():
            moved_profiling_log.unlink()

        shutil.copy(profiling_log, moved_profiling_log)
        profiling_log = moved_profiling_log

    backend_profiling_artifacts = move_backend_profiling_artifacts(backend, output_dir)
    return ProfilingData(profiling_log=profiling_log, backend_profiling_artifacts=backend_profiling_artifacts)
