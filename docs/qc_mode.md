# QC Mode

`qc` mode prepares a supported YOLO `.pt` model for deployment by converting it into a compiled `.tflite` artifact through QAI Hub. This is the model-preparation step used before model quality validation and device inference on EXMP-Q911 (Qualcomm QCS9075).

![QC mode overview](Images/qc_mode_1.png)

## Basic Command

Use the command below as the starting point for `yolov26`:

```bash
python3 cli.py \
  --mode qc \
  --type yolov26 \
  --model /path/to/yolov26n.pt \
  --calib_dir /path/to/calibration_images
```

## Required Inputs

`qc` requires the following inputs:

- `--mode qc`
- `--type` with one of `yolov10`, `yolov11`, or `yolov26`
- `--model` pointing to a supported FP `.pt` model
- `--calib_dir` pointing to a calibration image directory

## Output

By default, `qc` writes the generated model to:

```text
out/model/<type>/<type>_<quant>_<timestamp>.tflite
```

Use `--output` to override the default location.

## How QC Mode Works

The current `qc` pipeline works as follows:

1. Validate that `--model` and `--calib_dir` were provided.
2. Resolve the effective output head, quantization scheme, and output path for the selected model type.
3. Load the selected YOLO model through Ultralytics.
4. Wrap the model so export exposes only the raw output tensors needed by this pipeline.
5. Trace the wrapped model with a fixed NHWC input shape of `1 x 640 x 640 x 3`.
6. Submit the traced model to QAI Hub for ONNX compilation.
7. Build calibration samples from the images in `--calib_dir`.
8. Apply the configured quantization settings to the compiled model.
9. Compile the resulting model to TFLite.
10. Download the final `.tflite` artifact to the resolved output path.

In the current implementation, step 8 uses fully INT8 quantization with INT8 weights and INT8 activations, and step 9 keeps quantized I/O.

## Default Settings by Model

| `--type` | Default compilation format | Default quant scheme | Default output head |
| --- | --- | --- | --- |
| `yolov10` | `w8a8 INT8` | `mse` | `one2many` |
| `yolov11` | `w8a8 INT8` | `minmax` | `default` |
| `yolov26` | `w8a8 INT8` | `mse` | `one2many` |

## Available Configurations

| Setting | Available choices | Default by model | CLI flag | Notes |
| --- | --- | --- | --- | --- |
| Quantization | `w8a8` | `yolov10=w8a8`, `yolov11=w8a8`, `yolov26=w8a8` | `not configurable` | The current implementation uses fully INT8 quantization. |
| Quant scheme | `mse`, `minmax` | `yolov10=mse`, `yolov11=minmax`, `yolov26=mse` | `--qc-quant-scheme` |  |
| Output head | `one2many`, `one2one` | `yolov10=one2many`, `yolov11=default`, `yolov26=one2many` | `--qc-head` | `yolov10` and `yolov26` support `one2many` and `one2one`. `yolov11` uses only the default head and ignores `--qc-head`. |

## Note

It is recommended to use the default settings first. If the compiled model shows anomalies, try switching the quantization scheme.
