# QC Mode

`qc` mode prepares a supported YOLO `.pt` model for deployment by converting it into a compiled
`.tflite` artifact through QAI Hub. This is the model-preparation step used before model quality
validation and device inference on EXMP-Q911 (Qualcomm QCS9075).

![QC mode overview](Images/qc-mode-overview.png)

> [!IMPORTANT]
> Recommended host flow: run `./docker/iqf run qc ...`. If you repeat this workflow, save the
> required host paths first with `./docker/iqf configure qc --type <type>`. Saved paths live in
> `.iqf/docker-paths.json`.

> [!TIP]
> For detailed mode help, run `./docker/iqf run qc --help`.

## Basic Command

Use the command below as the starting point for `yolov26`:

```bash
./docker/iqf run qc \
  --type yolov26 \
  --model /path/to/yolov26n.pt \
  --calib_dir /path/to/calibration_images
```

For host setup, start from [README.md](../README.md) and choose either
[Ubuntu_host.md](../Ubuntu_host.md) or [Windows_host.md](../Windows_host.md).

## Required Inputs

`qc` requires the following inputs:

- the wrapper subcommand `./docker/iqf run qc`
- `--type` with one of `yolov10`, `yolov11`, or `yolov26`
- `--model` pointing to a supported FP `.pt` model
- `--calib_dir` pointing to a calibration image directory

When you use the wrapper, pass the path flags directly or save them first through
`./docker/iqf configure qc --type <type>`.

## Output

By default, `qc` writes the generated model to:

```text
out/model/<type>/<type>_<quant>_<timestamp>.tflite
```

Use `--output` to override the default location.

## How QC Mode Works

`qc` mode runs the following high-level flow:

1. Validate that `--model` and `--calib_dir` were provided.
2. Resolve the effective output head, quantization scheme, and output path for the selected model type.
3. Compile and quantize the model through QAI Hub.
4. Download the generated `.tflite` artifact to the resolved output path.

## Flags, Defaults, and Options

| Flag | Purpose | Options | Default |
| --- | --- | --- | --- |
| `--type` | Select the model family. | `yolov10`, `yolov11`, `yolov26` | Required |
| `--model` | Path to the FP `.pt` model. | filesystem path | Required |
| `--calib_dir` | Path to the calibration image directory. | filesystem path | Required |
| `--output` | Override the output model path. | filesystem path | `out/model/<type>/<type>_int8_<timestamp>.tflite` |
| `--max_calib` | Maximum calibration images used for quantization. | integer | `200` |
| `--qc-head` | Override the export head for supported models. | `one2many`, `one2one` | `one2many` for `yolov10` and `yolov26`; ignored for `yolov11`, which uses `default` |
| `--qc-quant-scheme` | Override the quantization scheme. | `mse`, `minmax` | `mse` for `yolov10`, `minmax` for `yolov11`, `mse` for `yolov26` |

## Backend-Only Note

The wrapper ultimately invokes `cli.py` inside the prepared runtime with explicit paths. Use
`cli.py` directly only when those paths are already valid in the active environment, such as an
interactive container shell.

## Notes

It is recommended to use the default settings first. If the compiled model shows anomalies, try
switching the quantization scheme. The current implementation uses fully INT8 quantization.
