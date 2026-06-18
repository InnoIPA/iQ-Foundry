# Windows Host Guide

<p align="center">
  <img src="./docs/Images/windows_logo.png" alt="Windows logo" width="96">
</p>

> [!IMPORTANT]
> This is the canonical setup guide for a Windows 11 host.

## Prerequisites

| Category | Requirement |
| --- | --- |
| Host platform | x86 Windows 11 |
| Memory | Recommended minimum 16 GB RAM |
| Target for `mAP` and host-side `test` | EXMP-Q911 (Qualcomm QCS9075) |
| ADB connection | USB-C cable between the host and EXMP-Q911 |
| QAI Hub access | [Qualcomm AI Hub](https://aihub.qualcomm.com/) API token |

## Getting Started

### STEP 1: Connect the Device

Connect the host to EXMP-Q911 with USB-C before using `mAP` or host-side `test`.

<p align="center">
  <img src="./docs/Images/usb-c-target-connection.png" alt="Host to target USB-C connection" width="720">
</p>

### STEP 2: Open PowerShell as Administrator

Open Windows PowerShell with Administrator rights:

1. Click the `Start` button.
2. Search for `PowerShell`.
3. Right-click `Windows PowerShell`.
4. Select `Run as administrator`.

### STEP 3: Clone the Repository

```powershell
cd "$env:USERPROFILE"
git clone https://github.com/InnoIPA/iQ-Foundry.git
cd iQ-Foundry
```

### STEP 4: Run the Windows WSL Setup Helper

This step typically takes 5-10 minutes.

Run the Windows-side setup helper from PowerShell as Administrator:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-windows-wsl.ps1
```

This is an interactive setup. During the WSL setup flow, when Windows asks whether to continue,
answer `Y` or `Yes`. If `Ubuntu-22.04` is being initialized for the first time, create the Ubuntu
account when prompted and set the username and password for that WSL environment.

This script prepares WSL and USB passthrough. It does not install Docker Engine inside WSL and it
does not store QAI Hub credentials.

### STEP 5: Install Docker Engine Inside WSL

Continue in the WSL Ubuntu terminal. Keeping the repo on the Windows filesystem is supported.

```bash
bash ./docker_install.sh
```

This installs Docker Engine inside WSL and prepares the `docker` group workflow used by
`./docker/iqf`.

### STEP 6: Pull the Tested Image from Docker Hub

This step typically takes 3-5 minutes.

```bash
docker pull innodiskorg/iqf:latest
```

The default published image used is `innodiskorg/iqf:latest`.

If you need to build the image directly from source, use [docker/Docker.md](./docker/Docker.md).
Pulling from Docker Hub remains the recommended workflow, and direct image build is provided as a
fallback option.

### STEP 7: Authenticate with Qualcomm AI Hub

Log in to the [Qualcomm AI Hub Workbench](https://aihub.qualcomm.com/).

Navigate to `Account -> Settings -> API Token` to find your unique API token.

Authenticate through the helper script from WSL:

```bash
./qaihub_login.sh --key <YOUR_QAI_HUB_API_KEY>
```

This stores the host-side QAI Hub configuration in `~/.qai_hub/client.ini` inside WSL.

After these steps, the Windows host is ready for `qc`, `mAP`, and host-side `test`.

## Quick Start

`iQ-Foundry` supports two ways to run each mode:

- [Configure Flow](#configure-flow): simple, repeatable wrapper commands for each mode. Save the required host paths in `.iqf/docker-paths.json`, then run the mode with shorter commands. For a guided walkthrough of the repeated-flow experience, see the [iQ-Studio YOLO26 tutorial](https://github.com/InnoIPA/iQ-Studio/blob/main/tutorials/model-deploy/cv/yolo26/README.md).
- [One-Shot Commands](#one-shot-commands): full wrapper usage with more explicit control. Pass the required host paths and any extra flags directly in the command.

Use `Configure Flow` when you want an easier repeated workflow. Use `One-Shot Commands` when you
need more control over paths and flags for a specific run.

## Configure Flow Commands

Configure flow lets you save the required host paths in `.iqf/docker-paths.json` and run each
mode later with shorter commands.

If you want a simpler guided setup for configure flow, start with the
[iQ-Studio](https://github.com/InnoIPA/iQ-Studio/tree/main) tutorial:
[YOLO26 Configure Flow Tutorial](https://github.com/InnoIPA/iQ-Studio/blob/main/tutorials/model-deploy/cv/yolo26/README.md).

The examples below use `yolov26`. You can also use `--type yolov10` or `--type yolov11`.

### QC

Use `qc` to quantize and compile a supported FP `.pt` model into a deployment-ready `.tflite`
model through QAI Hub.

Configure the required paths: FP model path and calibration image directory.

```bash
./docker/iqf configure qc --type yolov26
```

Run the mode:

```bash
./docker/iqf run qc --type yolov26
```

Output location: `out/model/yolov26/yolov26_<quant>_<timestamp>.tflite`

### mAP

Use `mAP` to compare source versus converted model quality at `mAP@0.5`. The source model runs on
the host, and the converted model runs on EXMP-Q911 (Qualcomm QCS9075) through ADB.

Configure the required paths: annotations path, image directory, FP model path, and converted
model path.

```bash
./docker/iqf configure mAP --type yolov26
```

Run the mode:

```bash
./docker/iqf run mAP --type yolov26
```

For a smaller validation run, you can limit the number of images:

```bash
./docker/iqf run mAP --type yolov26 --max-images 5
```

Output location: `out/mAP_results/yolov26/yolov26_mAP_result_<timestamp>.txt`

### Test

Use `test` to run converted model inference on EXMP-Q911 (Qualcomm QCS9075).

Configure the required paths: prepared model path, YAML file path, and test image or image
directory path.

```bash
./docker/iqf configure test --type yolov26
```

Run the mode:

```bash
./docker/iqf run test --type yolov26 --adb
```

Output location: `out/test/yolov26/yolov26_inference_<timestamp>/`

> NOTE:
> `test` in configure flow commands uses ADB-based host execution. Direct on-device inference is
> documented in [docs/test_mode.md](./docs/test_mode.md).

> 💡 TIP:
> To review the currently saved mode paths, open `.iqf/docker-paths.json`.

## One-Shot Commands

The examples below use `yolov26`. The same workflow also supports `yolov10` and `yolov11` by
changing `--type` and supplying the matching model files directly on the command line.

### QC Mode

Use `qc` to quantize and compile a supported FP `.pt` model into a deployment-ready `.tflite`
model through QAI Hub.

```bash
./docker/iqf run qc \
  --type yolov26 \
  --model /path/to/yolov26n.pt \
  --calib_dir /path/to/calibration_images
```

By default, this writes the compiled model to
`out/model/yolov26/yolov26_<quant>_<timestamp>.tflite`.

For advanced `qc` options, see [docs/qc_mode.md](./docs/qc_mode.md).

### mAP Mode

Use `mAP` to compare source versus converted model quality at `mAP@0.5`. The source model runs on
the host, and the converted model runs on EXMP-Q911 (Qualcomm QCS9075) through ADB.

```bash
./docker/iqf run mAP \
  --type yolov26 \
  --annotations /path/to/instances_val2017.json \
  --images /path/to/val2017 \
  --fp-model /path/to/yolov26n.pt \
  --int-model /path/to/yolov26_int8.tflite
```

`--annotations` can be either a COCO `.json` file or a custom annotation directory containing
separate YOLO `.txt` labels or VOC `.xml` labels for each image in the images directory.

By default, this writes the report to
`out/mAP_results/yolov26/yolov26_mAP_result_<timestamp>.txt`.

For `yolov10` and `yolov26`, if the converted model was generated with `--qc-head one2one`, run
`mAP` with `--fp-head one2one` so the FP branch matches the converted model.

For advanced `mAP` options, see [docs/mAP_mode.md](./docs/mAP_mode.md).

### Test Mode

Use `test` to run converted model inference on EXMP-Q911 (Qualcomm QCS9075). The example below
runs from the host through ADB.

```bash
./docker/iqf run test \
  --type yolov26 \
  --model /path/to/yolov26_int8.tflite \
  --yaml /path/to/coco.yaml \
  --images /path/to/test_images \
  --adb
```

By default, this writes annotated images, detection `.txt` files, and `classes.txt` to
`out/test/yolov26/yolov26_inference_<timestamp>/`.

Use `--output` to override the default test output directory.

For advanced `test` options, see [docs/test_mode.md](./docs/test_mode.md).

## Advanced Mode Details

Use the mode documents for advanced options and mode-specific notes:

- [./docs/qc_mode.md](./docs/qc_mode.md)
- [./docs/mAP_mode.md](./docs/mAP_mode.md)
- [./docs/test_mode.md](./docs/test_mode.md)

## Notes

- Run Docker commands from the WSL terminal, not from PowerShell.
- Docker Desktop is not required for this workflow.
