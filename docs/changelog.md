# Changelogs

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
