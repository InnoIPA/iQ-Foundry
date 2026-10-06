# Troubleshooting: what common errors mean

Use this only to **explain** errors to the user. The regression skill never fixes them. A fix is
a separate task that the user must request explicitly.

| Where you see it | Message (substring) | Likely meaning | What the user can do |
|---|---|---|---|
| validate / mAP / test log | `no devices/emulators found`, `device not found` | The target isn't attached, or a host-side adb server still holds the USB interface, so the container can't see the device. | Check the cable and `adb devices`. The runner already runs `adb kill-server` on the host before each adb stage. If another tool (IDE, a second terminal) keeps restarting adb, close it. |
| validate | `more than one ... adb devices attached` | Several targets are attached. | Set `device.adb_serial`. |
| validate / qc | `~/.qai_hub/client.ini not found` | QAI Hub was never configured on this host. `docker/iqf` mounts it for **every** qc run, qairt included. | Give a token, and the skill runs `hub-login`. |
| validate | `QAI Hub login check failed` | The saved token was rejected, or there is no network access to QAI Hub. | Log in again with a valid token, and check network or proxy. |
| qc log | `Traceback` from `qai_hub` (e.g. `Job failed`, `UserError`) | The QAI Hub compile/quantize job failed on the Hub side. | Open the job URL printed in the log on the QAI Hub website. |
| qc log (qairt) | `qairt-converter` / `qairt-quantizer` / `qnn-context-binary-generator` errors | The offline QAIRT toolchain in `vendor/qairt/` failed for this model/precision. | Report it with the log; this is a product bug or an SDK limitation. |
| qc | `exit 0 but qc artifact missing` | cli.py reported success but did not write the requested `--output`. | Product bug; report it with the log. |
| mAP log | `Converted model class count mismatch` | The converted model's class count differs from the reference `.pt`. | Use a matching `.pt` / converted model pair. |
| mAP | `mAP result file has no reference/converted mAP50` | The result text was written but in an unexpected format. | Product change or bug; attach `artifacts/<id>/mAP/mAP_result.txt`. |
| mAP/test | `PASS_WITH_WARNINGS`: log contains `Falling back to CPUExecutionProvider` / `QNNExecutionProvider is not available` | ONNX Runtime on the device silently ran on the CPU instead of QNN/HTP. The numbers are valid, but performance is not representative. | Check the device ORT/QNN install (`tool/adb_runtime_bootstrap.py` output in the log). |
| mAP | `PASS_WITH_WARNINGS`: `converted mAP50 is -X% vs reference` | Accuracy dropped more than `defaults.mAP.warn_if_pct_delta_below` in `regression/pipelines.yaml`. | Expected for some int8 paths; compare with previous runs. |
| test | `test output has no artifacts besides classes.txt` | Inference ran but produced no annotated images or labels. | Product bug; attach the log. |
| test log | `Output directory already exists` | The test `--output` directory existed before the stage started. | Should not happen with a fresh run directory; start a new run (`init`). |
| any stage | `TIMEOUT` | The stage exceeded `timeouts_s` and was killed along with its container. | Raise `timeouts_s.<mode>` in `inputs.yaml` for the next run if the stage is slow but healthy. |
| run | `results.json already exists` | The run directory was reused. | Create a new run directory with `init`. |
| hub-login | `does not look like a QAI Hub API token` | The token is too short or has whitespace (often a copy/paste error). Nothing was run. | Copy the token again from the QAI Hub website. |
| hub-login | `the QAI Hub token is written in these run files` | The token was found inside the run directory, e.g. pasted into `inputs.yaml`. | Remove it from the named files and rotate the token on QAI Hub. |
| validate | `holds a QAI Hub token and is readable by other users` | `~/.qai_hub/client.ini` or `.bak` has group/world read permission. | `chmod 600` the named file (and `chmod 700 ~/.qai_hub`). |
| status | `runner pid N is not running and the run did not finish`, empty `runner.out` | The process that launched the runner was cleaned up by the agent's shell tool, which killed its background job (typical for `nohup ... &`). | Use `start`, or run `run` in your own terminal (SKILL.md Step 7, option C). If no stage had started, the same run directory can be reused. |
| status | `no results.json` / `validating ... Check again shortly` right after starting | `run` re-validates (Docker, QAI Hub, adb, dry-runs) before the first stage; this takes up to a minute. | Wait and check again. |
| run | `refusing to run: pass --confirmed-by-user` | The confirmation flag is missing. | Only add it after the user typed `run`. |
