# Test Mode

`test` mode runs a compiled TFLite model on EXMP-Q911 (Qualcomm QCS9075) and writes visual and text outputs for quick inspection. Use this mode after model compilation and quality validation to confirm that detections look correct on real input images and that on-device behavior matches expectations.

![Test mode overview](Images/test-mode-overview.png)

## Purpose

`test` mode is the final quick-check stage in this repository workflow.

It is used to:

- run a compiled `.tflite` model on sample images
- verify that the selected `--type` matches the compiled model
- inspect annotated outputs and detection text files
- confirm practical on-device behavior after `qc` and, if needed, after `mAP`


## Basic Command

Use the command below as the starting point for `yolov26` from the host through ADB:

```bash
python3 cli.py \
  --mode test \
  --type yolov26 \
  --model /path/to/yolov26_compiled.tflite \
  --yaml /path/to/coco.yaml \
  --images /path/to/test_images \
  --adb
```

For saved-path usage, see [Configure Flow Commands](../README.md#configure-flow-commands) in the README.

## How Test Mode Works

The current `test` pipeline works as follows:

1. Validate the compiled model path, class-name YAML file, and image input.
2. Resolve the effective postprocess settings from the selected model type and any CLI overrides.
3. Run inference on EXMP-Q911 (Qualcomm QCS9075), either through ADB from the host or directly on the device.
4. Postprocess the raw outputs and write annotated images, detection `.txt` files, and `classes.txt`.

## ADB Mode vs Direct On-Device Mode

`test` mode supports two execution paths.

### ADB Mode

Use ADB mode when you are running the command from the x86 host and want the tool to push files, execute inference on EXMP-Q911 (Qualcomm QCS9075), and pull the results back automatically.

Sample command:

```bash
python3 cli.py \
  --mode test \
  --type yolov26 \
  --model /path/to/yolov26_compiled.tflite \
  --yaml /path/to/coco.yaml \
  --images /path/to/test_images \
  --adb
```

In ADB mode, the tool:

- prepares a temporary run directory on the target
- pushes the model, YAML file, inference script, and input images to EXMP-Q911 (Qualcomm QCS9075)
- runs inference remotely
- pulls the generated outputs back to the host output directory

### Direct On-Device Mode

Use direct on-device mode when you are logged into EXMP-Q911 (Qualcomm QCS9075) and want to run the command locally on the target without `--adb`.

This mode requires [Step 4 of the setup process in README.md](../README.md#step-4-target-setup-required-only-for-on-device-inference-without-adb), because the target environment must already be prepared on EXMP-Q911 (Qualcomm QCS9075).

Sample command:

```bash
python3 cli.py \
  --mode test \
  --type yolov26 \
  --model /path/to/yolov26_compiled.tflite \
  --yaml /path/to/coco.yaml \
  --images /path/to/test_images
```

In direct on-device mode:

- the command must run on EXMP-Q911 (Qualcomm QCS9075) itself
- `--adb` is not used
- the repository and target dependencies must already be installed on the device

## Required Inputs

`test` requires the following:

- `--mode test`
- `--type` with one of `yolov10`, `yolov11`, or `yolov26`
- `--model` pointing to the compiled `.tflite` model
- `--yaml` pointing to the class-name YAML file
- exactly one of:
  - `--images`
  - `--image`

## Output

By default, `test` writes results to:

```text
out/test/<type>/<type>_inference_<timestamp>/
```

The output directory contains:

- annotated images
- detection `.txt` files
- `classes.txt`

Use `--output` to override the default location.

## Default Settings by Model

| `--type` | Default flow | Default `--conf` | Default `--nms` | Default `--topk` | Default `--max-det` |
| --- | --- | --- | --- | --- | --- |
| `yolov10` | `o2m` | `0.25` | `0.6` | `300` | `100` |
| `yolov11` | `default` | `0.25` | `0.6` | `300` | `100` |
| `yolov26` | `o2m` | `0.25` | `0.6` | `300` | `100` |

## Flags, Defaults, and Options

| Flag | Purpose | Options | Default |
| --- | --- | --- | --- |
| `--model` | Path to the compiled `.tflite` model. | filesystem path | Required |
| `--yaml` | Path to the class-name YAML file. | filesystem path | Required |
| `--images` | Directory of input images. | filesystem path | Required unless `--image` is used |
| `--image` | Single input image. | filesystem path | Required unless `--images` is used |
| `--adb` | Run inference on EXMP-Q911 (Qualcomm QCS9075) through ADB from the host. | enabled or omitted | off |
| `--output` | Override the output directory. | filesystem path | `out/test/<type>/<type>_inference_<timestamp>/` |
| `--conf` | Confidence threshold used in postprocess. | float | model default |
| `--nms` | NMS IoU threshold used in postprocess. | float | model default |
| `--topk` | Number of candidates kept before NMS. | integer | model default |
| `--max-det` | Maximum detections kept per image. | integer | model default |
| `--postprocess-flow` | Override the postprocess flow. | `auto`, `default`, `o2o`, `o2m` | `auto` |
| `--o2o-nms` | Enable class-wise NMS when using `o2o`. | enabled or omitted | off |
| `--disable-int8-prefilter` | Disable the INT8 class prefilter in postprocess. | enabled or omitted | off |
| `--adb-serial` | Select a specific ADB target device. | ADB serial string | first available ADB target |
| `--remote-workdir` | Remote working directory used in ADB mode. | filesystem path on target | `/data/local/tmp/yolo_map_eval` |
| `--qnn-lib` | QNN delegate library path on EXMP-Q911 (Qualcomm QCS9075). | filesystem path on target | `/usr/lib/libQnnTFLiteDelegate.so` |
| `--backend` | QNN backend type. | string | `htp` |
| `--no-qnn` | Disable the QNN delegate. | enabled or omitted | off |

## Notes

- `yolov11` supports only the default postprocess flow. If `o2o` or `o2m` is requested, the current implementation ignores it and uses `default`.
- `--o2o-nms` is meaningful only when the active flow is `o2o`.
- When running without `--adb`, the command must be executed on EXMP-Q911 (Qualcomm QCS9075) directly.
- It is recommended to start with the default settings and only change postprocess flags when debugging unexpected detection behavior.
