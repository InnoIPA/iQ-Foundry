# Changelogs

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
- Updated `Ubuntu_host.md` and `Windows_host.md` to use the explicit runtime/precision CLI, the new `mAP` model flag names, and the current default output names.
- Reworked `docs/qc_mode.md`, `docs/test_mode.md`, and `docs/mAP_mode.md` to document the full v0.0.3 runtime matrix, FP32 calibration behavior, ONNX bundle handling, and runtime-specific execution notes.
- Added direct on-device ONNX Runtime installation guidance in `docs/test_mode.md`, including the `wheels/`-based `onnxruntime_qnn` install command.
- Refreshed backend help output in `cli.py` so QC, test, and `mAP` help reflect LiteRT FP32, ONNX FP32, ONNX W8A16, and runtime-specific ADB behavior.

### fixes
- Fixed `mAP` category mapping to use name-based `model class index -> COCO category id` resolution, supporting reordered or non-contiguous COCO ids and subset-category datasets.
- Fixed repo-local runtime references after renaming `tool/inference.py` to `tool/inference_tflite.py`.
- Fixed `setup-windows-wsl.ps1` to detect the WSL "requires update" state, attempt `wsl.exe --update`, retry with `wsl.exe --update --web-download`, and then continue distro discovery.
- Fixed Windows USB auto-detection in `setup-windows-wsl.ps1` so it now accepts `exmp-q911` in addition to `Qualcomm`.
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
