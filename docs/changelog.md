# Changelogs

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
- Improved inference compatibility and evaluation behavior in `test` and `mAP` flows.
- Improved YAML class-name validation and dynamic class handling.

### docs
- Added mode-specific documentation for `qc`, `mAP`, and `test`.
- Added getting-started and usage guidance in the repository documentation.
- Added advanced configuration and detailed usage examples for all modes.
- Added documentation for custom annotation directory support in `mAP`.
- Added USB-C connection and setup guidance for target-device usage.
- Updated `mAP` documentation for custom dataset handling and evaluation behavior.
