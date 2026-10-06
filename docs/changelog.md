# Changelogs

## v1.0.0

### features
- Added QAIRT (Qualcomm AI Runtime) support for `qc`, `test`, and `mAP`, covering QAIRT INT8, W8A16, and FP16 flows for `yolov10`, `yolov11`, and `yolov26`.
- Added offline conversion for QAIRT: `qc` builds a pre-compiled HTP context binary locally through ONNX export, `qairt-converter`, `qairt-quantizer`, and `qnn-context-binary-generator`, with no Qualcomm AI Hub account and no device-side compilation.
- Added `tool/qairt_inference.py` for QAIRT execution, reusing the shared geometry, decode, NMS, and drawing helpers so all three runtimes produce identical detections from identical tensors.
- Added the vendored QAIRT `2.47.0.260601` host toolchain under `vendor/qairt/`, including the five OS libraries the container image does not provide, so no image rebuild is required.
- Added QAIRT `.bin` output naming for `qc`, and QAIRT rows to the runtime and precision matrix in both `./docker/iqf` and backend `cli.py`.
- Added QLI2.0 target support by replacing TensorFlow with `ai-edge-litert==2.2.0` on both host and target (`requirements/host.txt`, `requirements/target.txt`), and switching `tool/inference_tflite.py`, `tool/remote_tflite_raw_runner.py`, and `tool/test_map.py` to the LiteRT interpreter. QLI2.0 ships Python 3.14, for which TensorFlow has no wheel.
- Moved the target `numpy` pin to `2.5.3` for Python 3.14 (no cp314 wheel exists for `1.26.4`); the host stays on `1.26.4` because `torch==2.4.1` requires `numpy<2`.
- Replaced the bundled ORT-QNN wheel (`1.23.0` cp312) with the `1.25.1` cp314 aarch64 build for the QLI2.0 target.
- Added a version check to the on-device `onnxruntime` bootstrap in `tool/adb_runtime_bootstrap.py`: a mismatch against the bundled wheel now uninstalls the existing `onnxruntime`/`onnxruntime-qnn`, reinstalls the wheel, and verifies the installed version.
- Added the `iqf-assistant` agent skill (`.agents/skills/iqf-assistant/`, symlinked for Claude Code) so users can run `qc`, `mAP`, and `test` by chatting with a coding agent. It includes a read-only host setup checker (`check_setup.py`, `check_setup_windows.ps1`), a token-safe QAI Hub login (`hub_login.py`), a detached job runner (`iqf_job.py`), a mAP data inspector and reformatter (`prepare_map_data.py`), troubleshooting references, and offline tests.
- Added an end-to-end pipeline regression suite: `regression/pipelines.yaml` (21 pipelines across model type, runtime, and precision), the deterministic runner `tool/test/regression/run_pipelines.py` (`qc` -> `mAP` -> `test`, detached runs, QAI Hub token masking), the `iqf-pipeline-regression` agent skill, and offline tests in `tests/test_regression_runner.py`.

### refactor
- Moved the calibration loader, ONNX finalize step, and QAIRT conversion helpers, previously copied verbatim into each model file, into a single `yolo_models/common.py` shared by `yolov10`, `yolov11`, and `yolov26`.
- Removed dead code from `cli.py`, `tool/inference_tflite.py`, `tool/test_map.py`, `tool/onnx_inference.py`, `tool/qairt_inference.py`, and the regression runner.

### docs
- Updated `README.md` with a QAIRT runtime logo cell, an `FP16` column in the runtime-versus-precision support matrix, and a v1.0.0 feature callout.
- Reworked `docs/qc_mode.md`, `docs/test_mode.md`, and `docs/mAP_mode.md` to document the QAIRT matrix, QAIRT calibration behavior, context-binary handling, and runtime-specific execution notes.
- Scoped the Qualcomm AI Hub references to `litert` and `onnx`, since `qairt` converts offline and needs no API token.
- Added an ADB troubleshooting note to `docs/test_mode.md` and `docs/mAP_mode.md` covering the host-side `adb` server holding the USB interface.
- Updated `Ubuntu_host.md` and `Windows_host.md` with the QAIRT runtime and precision combinations.
- Documented the vendored QAIRT SDK in `requirements/host.txt` and recorded that the QAIRT target path installs nothing on device.
- Added `docs/iqf_assistant.md` with usage instructions for the `iqf-assistant` skill, and linked it from `Ubuntu_host.md`, `Windows_host.md`, `docs/qc_mode.md`, `docs/test_mode.md`, and `docs/mAP_mode.md`.
- Updated the workflow, QC overview, and runtime/precision images, and added the QAIRT logo.
- Updated the ORT-QNN wheel filename in `docs/test_mode.md`.
- Corrected the `--remote-workdir` default in `docs/mAP_mode.md` to `/data/local/tmp/yolo_map_eval`.

### fixes
- Fixed box/class output selection for models with 4 or 64 classes: outputs are now narrowed by the box channels expected for `--type` and ties are broken by export order, which resolves the "Expected exactly one box output" and "Could not identify raw ONNX box/class outputs" errors in LiteRT and ONNX `test` and `mAP`.
- Fixed ONNX `qc` overwriting `model.data` when several models share an output directory: each artifact now gets a self-contained `<stem>.data` sidecar with the graph reference rewritten, the downloaded `.onnx.zip` is removed once unpacked, and ADB runs push the correct sidecar via `collect_model_sidecars`.
- Fixed the container's `adb` missing the board when a host-side `adb` server held the USB interface: the wrapper now stops the host server before ADB runs (`--dry-run` prints the step).
- Fixed host `onnxruntime` to `1.23.2` (from `1.22.0`) so ONNX IR v11 models, such as W8A16, load.
- Cleared lint debt introduced by the QAIRT runtime in `tool/onnx_inference.py`, `tool/qairt_inference.py`, and `tool/test_map.py`.

## v0.0.3

### features
- Added ONNX Runtime support for `qc`, `test`, and `mAP`, including ONNX FP32 and ONNX W8A16 flows for `yolov10`, `yolov11`, and `yolov26`.
- Added LiteRT FP32 support across `qc`, `test`, and `mAP` while preserving the existing LiteRT INT8 workflow.
- Added explicit `--runtime` and `--precision` requirements to both `./docker/iqf` and backend `cli.py`, with the supported matrix `litert/int8`, `litert/fp32`, `onnx/fp32`, and `onnx/w8a16`.
- Added runtime/precision-scoped saved-path reuse in `.iqf/docker-paths.json` so configure flow state is isolated by model type, mode, runtime, and precision.
- Added runtime/precision-aware default output naming for `qc`, `test`, and `mAP`.
- Added public `mAP` model flags `--reference-model` and `--converted-model`.
- Added `tool/onnx_inference.py` for ONNX Runtime execution and renamed the shared LiteRT inference runner to `tool/inference_tflite.py`.

### docs
- Updated `README.md` with refreshed model and deployment support tables, a runtime-versus-precision support matrix, runtime logos, and a v0.0.3 feature callout.
- Refined `README.md` images with minor layout and presentation updates.
- Updated `Ubuntu_host.md` and `Windows_host.md` to use the explicit runtime/precision CLI, the new `mAP` model flag names, and the current default output names.
- Reworked `docs/qc_mode.md`, `docs/test_mode.md`, and `docs/mAP_mode.md` to document the full v0.0.3 runtime matrix, FP32 calibration behavior, ONNX bundle handling, and runtime-specific execution notes.
- Added direct on-device ONNX Runtime installation guidance in `docs/test_mode.md`, including the `wheels/`-based `onnxruntime_qnn` install command.
- Refreshed backend help output in `cli.py` so QC, test, and `mAP` help reflect LiteRT FP32, ONNX FP32, ONNX W8A16, and runtime-specific ADB behavior.

### fixes
- Fixed `mAP` category mapping to use name-based `model class index -> COCO category id` resolution, supporting reordered or non-contiguous COCO ids and subset-category datasets.
- Fixed repo-local runtime references after renaming `tool/inference.py` to `tool/inference_tflite.py`.
- Fixed `setup-windows-wsl.ps1` to detect the WSL "requires update" state, attempt `wsl.exe --update`, retry with `wsl.exe --update --web-download`, and then continue distro discovery.
- Fixed Windows USB auto-detection in `setup-windows-wsl.ps1` so it now accepts `exmp-q911` in addition to `Qualcomm`.
- Fixed the Windows host repository setup flow to use ZIP download and extraction instead of `git clone`, avoiding a separate Git installation step.
- Fixed shebangs in bash and powershell scripts througout the repo

## v0.0.2

### feat
- Added wrapper commands for `build`, `shell`, `configure`, and `run`.
- Added wrapper-side saved host-path management in `.iqf/docker-paths.json`.
- Added wrapper flags for `--dry-run`, `--save`, `--image`, and `--repo-root`.
- Added `shell --qai-hub` and `shell --adb` runtime helper flows.
- Added Windows host workflow support through WSL Ubuntu.
- Added Windows path translation support for native Windows paths and current-distro WSL UNC paths.
- Added `setup-windows-wsl.ps1` for WSL setup, first-launch handling, and USB passthrough preparation.
- Added `docker_install.sh` for Docker Engine installation on Ubuntu and WSL Ubuntu.
- Added `qaihub_login.sh` for Docker-based Qualcomm AI Hub login with host-side config persistence.

### refactor
- Migrated the primary host-side workflow from direct `cli.py` usage to `./docker/iqf`.
- Moved interactive configure flow from `cli.py` to the Docker wrapper.
- Updated `cli.py` help and validation messaging to guide Docker-wrapper usage while keeping direct backend execution available for prepared environments.

### docs
- Reworked the main `README.md` into a wrapper-first gateway for the repository.
- Added dedicated host guides for `Ubuntu_host.md` and `Windows_host.md`.
- Added `docker/Docker.md` as the direct image build guide and fallback build workflow reference.
- Added `docs/other_model_flow.md` as the flow guide for converting unsupported vision models through Qualcomm AI Hub Workbench.
- Updated the `qc`, `mAP`, and `test` mode documents to use `./docker/iqf` as the primary host workflow.
- Preserved and clarified direct on-device `test` guidance as the remaining primary direct-`cli.py` workflow.
- Added Windows WSL setup guidance, Docker Engine installation guidance, Docker Hub image pull guidance, and Qualcomm AI Hub login guidance.
- Added route to the iQ-Studio YOLO26 tutorial.
- Added redirect pages for legacy wrapper-related docs in `docker/README.md` and `Windows.md`.
- Updated docs to use the published default image `innodiskorg/iqf:latest`.
- Clarified that the legacy `config.json` configure flow has been replaced by `.iqf/docker-paths.json` in the wrapper flow.
- Updated the overview image with Bring Your Own Model messaging and clarified the supported input model type.

## v0.0.1

### feat
- Added `qc` mode to quantize and compile supported YOLO `.pt` models to INT8 `.tflite` artifacts through QAI Hub.
- Added `mAP` mode for pairwise FP-vs-INT mAP@0.5 evaluation.
- Added `test` mode for compiled-model inference on EXMP-Q911 (Qualcomm QCS9075).
- Added both direct EXMP-Q911 (Qualcomm QCS9075)-native inference and ADB-orchestrated inference flows.
- Added model-family support for `yolov10`, `yolov11`, and `yolov26`.
- Added per-model QC defaults and override flags for quant scheme and output head selection.
- Added persistent ADB runtime bootstrap and reuse for remote execution on EXMP-Q911 (Qualcomm QCS9075).
- Added custom annotation directory support for `mAP`, including YOLO `.txt`, VOC `.xml`, and COCO `.json`.
- Added configure flow commands to save required mode paths in the legacy `config.json` flow for simpler repeated runs.
- Added direct run flow support for passing paths and advanced flags directly through the CLI.
- Added improved CLI help output with quick-start guidance for configure flow and direct run usage.
- Improved inference compatibility and evaluation behavior in `test` and `mAP` flows.
- Improved YAML class-name validation and dynamic class handling.

### docs
- Added mode-specific documentation for `qc`, `mAP`, and `test`.
- Added getting-started and usage guidance in the repository documentation.
- Added advanced configuration and detailed usage examples for all modes.
- Added documentation for custom annotation directory support in `mAP`.
- Added USB-C connection and setup guidance for target-device usage.
- Updated `mAP` documentation for custom dataset handling and evaluation behavior.
- Updated the README with configure flow commands and direct run commands guidance.
- Added Bring Your Own Model messaging and pretrained model guidance with an Ultralytics reference.
- Added a changelog section link in the README.
