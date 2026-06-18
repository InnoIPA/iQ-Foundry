# iQ-Foundry

<br />
<div align="center"><img width="30%" height="30%" src="./docs/Images/project-logo.png"></div>
<br />

<h1 align="center"><em><strong>Simplify the Workflow, Accelerate Deployment.</strong></em></h1>

<h3 align="center">This tool simplifies model quantization and validation for edge AI, reducing friction from preparation to real-device deployment—making workflows repeatable and scalable.</h3>
<h3 align="center"><strong>🚀 Bring Your Own Model</strong></h3>
<p align="center"><i>Supports custom-trained and pretrained models.</i></p>

![Repository overview](./docs/Images/overall-workflow.png)

`iQ-Foundry` helps prepare computer vision models for innodisk Qualcomm solution. The current workflow supports compiling compatible computer vision `.pt` models into `.tflite` artifacts, validating FP-versus-quantized quality with mAP@0.5, and running on-device inference on [EXMP-Q911 (Qualcomm QCS9075)](https://www.innodisk.com/en/products/computing/qualcomm-solution/exec-q911).

`iQ-Foundry` supports a Bring Your Own Model workflow. You can use your own compatible `yolov10`, `yolov11`, or `yolov26`  models with the pipeline. If you need pretrained YOLO weights, you can download official pretrained models from [Ultralytics](https://docs.ultralytics.com/).

> **Note:** `iQ-Foundry` is focused on computer vision model conversion, optimization, and deployment. For more comprehensive deployment information related to hardware and applications, see [iQ-Studio](https://github.com/InnoIPA/iQ-Studio/tree/main).

## Workflow At A Glance

1. Choose your host workflow: Ubuntu 22.04 or Windows 11.
2. Run `qc` to generate the quantized and compiled model.
3. Run `mAP` to compare source and converted model quality on the same dataset.
4. Run `test` for on-device inference, either from the host through ADB or directly on the target.

<p align="center">
  <img src="./docs/Images/modes.gif" alt="iQ-Foundry modes overview">
</p>

## Quick Start

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <div align="center">
          <img src="./docs/Images/ubuntu_logo.png" alt="Ubuntu logo" width="96"><br>
          <strong>Ubuntu Host</strong><br>
          Native Ubuntu 22.04 with Docker Engine<br><br>
          <a href="./Ubuntu_host.md">Open Ubuntu Host Guide</a>
        </div>
      </td>
      <td align="center" width="50%">
        <div align="center">
          <img src="./docs/Images/windows_logo.png" alt="Windows logo" width="96"><br>
          <strong>Windows Host</strong><br>
          Windows 11 + WSL Ubuntu with Docker Engine inside WSL<br><br>
          <a href="./Windows_host.md">Open Windows Host Guide</a>
        </div>
      </td>
    </tr>
  </table>
</div>

## Platform Capabilities Overview

### Workflow Modes

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Mode</th>
        <th>Stage</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>qc</code></td>
        <td>Quantize and Compile</td>
        <td>Convert a supported model into a deployment-ready artifact for validation and device execution.</td>
      </tr>
      <tr>
        <td><code>mAP</code></td>
        <td>Evaluate Converted Model Quality</td>
        <td>Measure detection quality so you can compare the original model against the converted result.</td>
      </tr>
      <tr>
        <td><code>test</code></td>
        <td>On-Device Inference</td>
        <td>Run the prepared model on target device and save outputs for quick functional and visual verification.</td>
      </tr>
    </tbody>
  </table>
</div>

### Model and Deployment Support

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Category</th>
        <th>Current Support</th>
        <th>Upcoming</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Input Format</td>
        <td><code>.pt</code></td>
        <td>-</td>
      </tr>
      <tr>
        <td>Model Families</td>
        <td><code>yolov10</code>, <code>yolov11</code>, <code>yolov26</code></td>
        <td>-</td>
      </tr>
      <tr>
        <td>Quantization</td>
        <td><code>INT8 (W8A8)</code></td>
        <td><code>W8A16</code>, <code>FP16</code>, <code>FP32</code></td>
      </tr>
      <tr>
        <td>Target Device</td>
        <td><a href="https://www.innodisk.com/en/products/computing/qualcomm-solution/exec-q911">EXMP-Q911 (Qualcomm QCS9075)</a></td>
        <td>-</td>
      </tr>
      <tr>
        <td>Runtime</td>
        <td><code>TensorFlow Lite</code></td>
        <td><code>ONNX Runtime</code></td>
      </tr>
      <tr>
        <td>Backend</td>
        <td><code>NPU (Qualcomm HTP)</code>, <code>CPU</code></td>
        <td>-</td>
      </tr>
    </tbody>
  </table>
</div>

## Changelog

Please refer to the [Changelog](./docs/changelog.md) for all updates.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
