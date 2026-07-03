# iQ-Foundry

<br />
<div align="center"><img width="30%" height="30%" src="./docs/Images/project-logo.png"></div>
<br />

<h1 align="center"><em><strong>Simplify the Workflow, Accelerate Deployment.</strong></em></h1>

<h3 align="center">This tool simplifies model quantization and validation for edge AI, reducing friction from preparation to real-device deployment—making workflows repeatable and scalable.</h3>
<h3 align="center"><strong>🚀 Bring Your Own Model</strong></h3>
<p align="center"><i>Supports custom-trained and pretrained models.</i></p>

![Repository overview](./docs/Images/overall-workflow.png)

<p align="center">
  <img src="docs/Images/litert-logo.png" alt="LiteRT" style="height: 42px; width: auto; vertical-align: middle;">
  <img src="docs/Images/ort-logo.png" alt="ONNX Runtime" style="height: 38px; width: auto; vertical-align: middle;">
  <br>
  <strong>New in v0.0.3:</strong> FP32 and mixed-precision deployment support!!
</p>

`iQ-Foundry` helps prepare computer vision models for innodisk Qualcomm solution. The current workflow supports compiling compatible computer vision `.pt` models into `.tflite` and `.onnx` artifacts, validating reference-versus-converted quality with mAP@0.5, and running on-device inference on [EXMP-Q911 (Qualcomm QCS9075)](https://www.innodisk.com/en/products/computing/qualcomm-solution/exec-q911).

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
          <img src="./docs/Images/ubuntu_logo.png" alt="Ubuntu logo" height="96"><br>
          <strong>Ubuntu Host</strong><br>
          Native Ubuntu 22.04 with Docker Engine<br><br>
          <a href="./Ubuntu_host.md">Open Ubuntu Host Guide</a>
        </div>
      </td>
      <td align="center" width="50%">
        <div align="center">
          <img src="./docs/Images/windows_logo.png" alt="Windows logo" height="96"><br>
          <strong>Windows Host</strong><br>
          Windows 11 with Docker Engine inside WSL<br><br>
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
        <th>Support</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Input Format</td>
        <td><code>.pt</code></td>
      </tr>
      <tr>
        <td>Model Families</td>
        <td><code>yolov10</code>, <code>yolov11</code>, <code>yolov26</code></td>
      </tr>
      <tr>
        <td>Quantization</td>
        <td><code>FP32 (float)</code>, <code>INT8 (W8A8)</code>, <code>W8A16 (INT mixed precision)</code></td>
      </tr>
      <tr>
        <td>Target Device</td>
        <td><a href="https://www.innodisk.com/en/products/computing/qualcomm-solution/exec-q911">EXMP-Q911 (Qualcomm QCS9075)</a></td>
      </tr>
      <tr>
        <td>Runtime</td>
        <td>
          <table>
            <tr>
              <td align="center">
                <img src="docs/Images/litert-logo.png" alt="LiteRT" height="24"><br>
                <code>LiteRT (TensorFlow Lite)</code>
              </td>
              <td align="center">
                <img src="docs/Images/ort-logo.png" alt="ONNX Runtime" height="24"><br>
                <code>ONNX Runtime</code>
              </td>
            </tr>
          </table>
        </td>
      </tr>
      <tr>
        <td>Backend</td>
        <td><code>NPU (Qualcomm HTP)</code>, <code>CPU</code></td>
      </tr>
    </tbody>
  </table>
</div>

<p><strong>Runtime and Precision Support Matrix</strong></p>

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Runtime</th>
        <th><code>FP32</code></th>
        <th><code>INT8</code></th>
        <th><code>W8A16</code></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>LiteRT</code></td>
        <td align="center">✓</td>
        <td align="center">✓</td>
        <td align="center">✗</td>
      </tr>
      <tr>
        <td><code>ONNX Runtime</code></td>
        <td align="center">✓</td>
        <td align="center">✗</td>
        <td align="center">✓</td>
      </tr>
    </tbody>
  </table>
</div>

## Explore Other Documentation

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Document</th>
        <th>Purpose</th>
        <th>Use It When</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="./docs/other_model_flow.md"><code>docs/other_model_flow.md</code></a></td>
        <td>High-level flow guide for unsupported or custom models outside the current native iQ-Foundry path.</td>
        <td>You want to explore bring-your-own-model onboarding beyond the built-in workflows.</td>
      </tr>
      <tr>
        <td><a href="./docker/Docker.md"><code>docker/Docker.md</code></a></td>
        <td>Fallback guide for building the iQF Docker image locally.</td>
        <td>You need to build the container image instead of pulling it from Docker Hub.</td>
      </tr>
    </tbody>
  </table>
</div>

## Changelog

Please refer to the [Changelog](./docs/changelog.md) for all updates.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
