# What common errors mean, in plain words

Use this file to explain a failure to the user and to suggest what they can change. **Never
change iQ-Foundry's code to make an error go away.** Find the error line in the job log with
`iqf_job.py status <JOB_DIR> --lines 60`, then look for a matching message below.

## Board and USB (mAP, test)

| Message contains | Plain explanation | What the user can do |
|---|---|---|
| `no devices/emulators found`, `device not found`, `device offline` | The board is not visible to iQ-Foundry. | Check the cable and power, then run the setup check again (see `device` in setup_fixes.md). Close other programs that use adb. On WSL, attach the USB again with `usbipd attach --wsl --busid <BUSID>`. |
| `more than one device/emulator` | Several boards are connected. | Choose one. Pass its serial with `--adb-serial <serial>`. |
| `/dev/bus/usb` missing | USB is not passed through to this system. | See `usb` in setup_fixes.md. |
| `Falling back to CPUExecutionProvider`, `QNNExecutionProvider is not available` (warning, not an error) | The ONNX model ran on the board's CPU instead of its AI accelerator. The results are valid, but speed is not representative. | Mention it. The device setup that iQ-Foundry ran is in the log. |

## QAI Hub (qc)

| Message contains | Plain explanation | What the user can do |
|---|---|---|
| `client.ini` not found, `QAI Hub config` | Not logged in to Qualcomm AI Hub. qc needs this for every runtime. | Log in (SKILL.md, "QAI Hub login", option (b)). |
| `QAI Hub login check failed`, `401`, `Unauthorized`, `invalid token` | The saved login was rejected. | Create a new token on the QAI Hub website and log in again. |
| `Job failed`, `UserError`, a `qai_hub` traceback | The conversion job failed on Qualcomm's servers. | The log contains a job link. The user can open it on the QAI Hub website to see why. Trying another precision is a reasonable next step. |
| Connection, proxy or timeout errors | No internet connection to QAI Hub. | Check the internet connection or company proxy. The qairt runtimes convert without internet. |

## Inputs (all modes)

| Message contains | Plain explanation | What the user can do |
|---|---|---|
| `[error] ... does not exist`, `not found`, `Expected a file` / `Expected a directory` | A path is wrong, or a file was given where a folder is needed (or the other way round). | Check the path together. Remember that `--images` for test is a folder and `--image` is one picture. |
| `Unsupported combination`, `Unsupported runtime`, `Unsupported precision` | That combination does not exist. | Pick one of the 7 options in SKILL.md Step 2. |
| `Output directory already exists` | The test output folder already exists. | Leave `--output` out (the default name has a time stamp), or pick a new folder. |
| `Permission denied` under `out/` | `out/` belongs to root, because Docker created it. | The user runs `sudo chown -R $USER <iQ-Foundry folder>/out` in their own terminal. |
| `No images found` | The folder has no `.jpg/.jpeg/.png/.bmp` pictures directly inside it. | Point to the folder that holds the pictures themselves, not a parent folder. Convert other picture formats first. |

## Accuracy check (mAP)

| Message contains | Plain explanation | What the user can do |
|---|---|---|
| `Annotation category '...' does not exist in reference model classes` | The labels use a class name the model does not know. | Run `prepare_map_data.py inspect`. It suggests a mapping, or leaving the class out. |
| `XML label '...' is not present in FP model classes` | The same problem, for VOC labels. | The same fix. |
| `YOLO class id N ... is out of range` | A label uses a class number the model does not have. | Make sure the labels belong to this model. If there is a `data.yaml` or `classes.txt`, pass it with `--dataset-names`. |
| `Expected 5 values` | A YOLO label line is not `class cx cy w h`: it may be a polygon, or have a confidence column. | `prepare_map_data.py convert` fixes this. |
| `FileNotFoundError` for a picture | The COCO file lists a picture that is not in the pictures folder. | `prepare_map_data.py convert` matches pictures by file name and leaves out the missing ones. |
| `Converted model class count mismatch` | The converted model and the `.pt` model have a different number of classes. | They are not the same model. Use the `.pt` that the converted model was made from. |
| `Invalid bounding box after clipping` | A label box has no area, for example because it is completely outside the picture. | `prepare_map_data.py convert` drops such boxes and reports how many. |
| `No images selected for evaluation` | No pictures matched the labels, or `--max-images` is 0. | Check the folders with `prepare_map_data.py inspect`. |

**Reading a mAP result.** A converted model scoring a few percent below the original is normal
for int8 or w8a16. A much bigger drop (more than about 10%) is worth mentioning. Ideas to
improve it:
- more calibration pictures, or more typical ones;
- `fp32` / `fp16` instead of `int8`;
- for yolov10/yolov26, keep `--qc-head` and `--fp-head` the same.

## Try it on pictures (test)

| Message contains | Plain explanation | What the user can do |
|---|---|---|
| `class count` / `names` mismatch with the YAML | The class-names file does not fit the model. | Make one from the model: `prepare_map_data.py class-yaml --model <model.pt>`. |
| `names` missing in YAML | The file has no `names:` list. | The same fix. |
| The output has only `classes.txt` and no pictures | The model ran but saved nothing, usually because there were no readable pictures. | Check the picture folder and formats. |
| `--no-qnn is not supported by the QAIRT runtime` | That option does not apply to qairt models. | Leave it out. |

## Helper scripts

| Message | Meaning | What to do |
|---|---|---|
| `iqf_job.py`: `another job is still running` | Only one mode can run at a time. | Wait for it, or ask the user whether to stop it. |
| `iqf_job.py status`: `STATE: died` | The background job was stopped from outside, for example because the computer slept or the agent's shell was closed. | Offer the user the same command to run in their own terminal. |
| `prepare_map_data.py`: `VERDICT: UNSUPPORTED` | The label format is not one the tool reads. | Ask the user to export COCO, YOLO, VOC, LabelMe or CVAT from their labelling tool. |
| `prepare_map_data.py`: `VERDICT: FAILED_SELF_CHECK` | The converted copy did not pass mAP mode's own loader. The tool has a bug, or the data is unusual. | Show the `ERROR:` line. Do not run mAP with that copy. |
| `hub_login.py`: `does not look like a QAI Hub API token` | The token is too short or contains spaces, often a copy/paste slip. Nothing was run. | Copy the token again from the QAI Hub website. |
