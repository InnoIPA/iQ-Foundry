---
name: iqf-pipeline-regression
description: Run the iQ-Foundry end-to-end pipeline regression (qc -> mAP -> test for yolov10/yolov11/yolov26 on litert, onnx and qairt) against an adb-attached IQ9 target, then produce a per-pipeline, per-mode status report. Use when the user asks to regression-test, smoke-test or verify iQ-Foundry pipelines, check that "all pipelines work", or run qc/mAP/test across models, runtimes and precisions.
---

# iQ-Foundry pipeline regression

This skill checks that iQ-Foundry pipelines work cleanly **exactly as a person following the
docs would run them**. A *pipeline* is one model type + runtime + precision, for example
`yolov26-litert-fp32`. Every pipeline runs three *modes* in this fixed order:

1. `qc`: converts the `.pt` model into a runtime model on the host (`.tflite`, `.onnx` or `.bin`).
2. `mAP`: compares the reference `.pt` with the converted model. The converted model runs on the device over adb.
3. `test`: runs inference with the converted model on the device over adb.

All the work is done by one script: `tool/test/regression/run_pipelines.py`. You collect the inputs,
run that script's subcommands in order, and show the user what they print. The script drives the
documented `./docker/iqf run ...` wrapper, so nothing is simulated.

## Hard rules. Read them before doing anything.

1. **Do not modify anything in the repository.** Do not edit code, configs, docs, `.iqf/`,
   `regression/pipelines.yaml` or the runner. The only files you write are the
   `inputs.yaml` inside the run directory made by `init`.
2. **Do not fix, retry or work around failures.** If a stage fails, the runner records it and moves on.
   Do not re-run it with different flags, do not patch code and do not switch images. The report is the
   result, and a FAIL is a valid result.
3. **Never guess or hardcode a path.** Every path comes from the user's answers. If an answer is
   missing or unclear, ask again.
4. **The QAI Hub API token is a secret.** Follow every rule in
   [QAI Hub token security](#qai-hub-token-security) below. In short: never show, repeat, log,
   store or forward it; it may appear only as the `IQF_QAI_HUB_TOKEN` environment variable of the
   single `hub-login` command. Whenever you refer to it, write `***`.
5. **Do not execute pipelines until the user types exactly `run`.** Anything else, such as
   "yes", "ok", "go" or "sure", is **not** confirmation. Ask again.
6. **Run the commands exactly as written here**, from the repository root. Replace only the
   `<placeholders>`.
7. If a command's output does not match what this file says to expect, **stop**. Show the output to
   the user and ask how to continue. Do not improvise.
8. Only one run at a time. The device is shared, so never start two `run` commands in parallel.

## Status words used everywhere

| Status | Meaning |
|---|---|
| `PASS` | Exit code 0 and the expected artifact was produced. |
| `PASS_WITH_WARNINGS` | It passed, but the log contains a known warning (e.g. ONNX CPU fallback) or mAP dropped past the threshold. |
| `FAIL` | Non-zero exit code, or exit 0 without the expected artifact. |
| `TIMEOUT` | The stage took longer than its time limit and was killed. |
| `BLOCKED` | Not run because `qc` for the same pipeline did not pass. |
| `NOT_FEASIBLE` | Not run because validation found a missing input or prerequisite. |
| `SKIPPED` | The mode was not selected, or `qc` was replaced by a prebuilt model. |
| `ABORTED` | The run was interrupted. |

## QAI Hub token security

The QAI Hub API token gives full access to the user's QAI Hub account (jobs, models, quota).
Treat it like a password. These rules apply in every step, including summaries and later turns.

**Preferred: the token never passes through you.** In Step 2 (question 8), always offer the user
the option to log in **themselves** in their own terminal (Step 4, option A). Recommend it. Only
handle the token yourself if the user explicitly chooses to give it to you.

If the user does give you the token:

1. **Use it exactly once**, in the single `hub-login` command of Step 4 (option B), as the
   `IQF_QAI_HUB_TOKEN=...` prefix of that command. That is the only place it may appear.
2. **Never write it anywhere else**:
   - not in `inputs.yaml`, not in any other file, not in a comment;
   - not as a command-line argument of any other command (arguments are visible in `ps`);
   - not with `export`, so it does not stay in the shell's environment;
   - not in a background job, a script file, a heredoc or a scheduled task;
   - not in a commit message, PR, issue, report, summary or progress update.
3. **Never repeat it back**, not even partly (no "starts with w9u…", no last 4 characters).
   When you confirm the inputs, write `QAI Hub: token received (***)`.
4. **Never send it anywhere else.** No web requests, other tools, sub-agents or other services.
   Only `hub-login` sends it, through the product's own `qaihub_login.sh`, to QAI Hub.
5. **Never read the stored login.** Do not `cat`, `grep`, open, copy or attach
   `~/.qai_hub/client.ini` or `~/.qai_hub/client.ini.bak`; both contain the token. To check the
   login, use only `validate`, which reports `QAI Hub login works (...)` without printing it.
6. **Forget it after Step 4.** Do not reuse it in later turns or later runs. If a new login is
   needed, ask the user again, and recommend option A again.
7. **If the token appears in any output or file**, for example unmasked in a log or pasted by the user
   into a file, stop. Do not repeat it. Tell the user which file or output contains it, and
   recommend that they (a) remove it and (b) rotate the token on the QAI Hub website. Do not
   delete or edit their files yourself unless they ask.
8. **If the user pastes the token in chat**, remind them once that chat transcripts may be
   stored, and that they can rotate the token after testing. Then continue with option B.

What the runner already does to back this up (you still must follow the rules above):
- It reads the token **only** from `IQF_QAI_HUB_TOKEN`, then removes it from its environment
  before starting any child process.
- It masks the token as `***` in all output, even when the token is split across output chunks.
- It refuses to run if the token appears in any file of the run directory, and checks again after the login.
- It rejects token-like keys in `inputs.yaml`.
- It restricts `~/.qai_hub` to the current user (directory 700, files 600) after login.
  `validate` warns if these files are readable by other users.

---

## Step 1: Go to the repository and list the pipelines

Ask the user: **"What is the absolute path of the iQ-Foundry repository to test?"**
(If you are already inside the repository and it contains `docker/iqf`, you may propose that path.
The user must still confirm it.)

Run:

```bash
cd <repo_root>
python3 tool/test/regression/run_pipelines.py list
```

Expected: a table of pipelines (21 at the time of writing) with columns `NO PIPELINE TYPE RUNTIME
PRECISION CALIB QC BACKEND ARTIFACT`, ending with `[info] <N> pipelines; ...`. Show this table to
the user **with the `NO` column first**. `NO` is the pipeline's short numeric ID, and the user picks
pipelines by it in Step 2. The numbers are fixed in `regression/pipelines.yaml` and never change
between runs.

If the command fails, stop and show the error (see rule 7).

## Step 2: Ask the user for the inputs

Ask all of the questions below in **one** message, as a numbered list, using this wording. The user may
answer them in one reply or several. Keep asking until every **required** item has an answer.
For optional items, tell the user the default. Never fill in an answer yourself.

1. **Pipelines** (required): "Which pipelines should I run? Give the pipeline numbers from the
   `NO` column (e.g. `3, 7, 15` or a range `1-7`), or `all`. Pipeline ids and glob patterns such as
   `yolov26-*` also work."
2. **Modes** (optional, default `qc, mAP, test`): "Which modes should run for each pipeline?"
3. **Docker image** (required): "Which Docker image tag should be used (see `docker images`)?"
4. **Input models** (required when `qc` or `mAP` is selected): "Give the absolute path of the `.pt`
   file for each model type you selected: yolov10 / yolov11 / yolov26." Ask only for the types present in
   the selected pipelines.
5. **Calibration images** (required if any selected pipeline has `CALIB = yes` and `qc` is
   selected): "Absolute path of the calibration image directory?" Optional follow-up: "Max
   calibration images (default: product default 200)?"
6. **mAP data** (required when `mAP` is selected): "Absolute path of the COCO annotations file
   (e.g. `instances_val2017.json`) and of the mAP image directory?" Optional follow-up: "Max images for
   mAP (default: product default 300)?"
7. **Test data** (required when `test` is selected): "Absolute path of the test image **or** test
   image directory, and of the class-names yaml (e.g. `coco.yaml`)?"
8. **QAI Hub login** (required when `qc` is selected): "How should QAI Hub be logged in?
   **(a)** reuse the existing login in `~/.qai_hub/client.ini`,
   **(b)** you log in yourself in your own terminal with a command I give you (recommended if you
   need a new login, because the token never passes through me), or
   **(c)** give me the token and I run the login once."
9. **ADB device** (optional, default: the single attached device): "Which adb serial should be
   used? Leave empty if only one device is attached."
10. **Extra flags** (optional, default none): "Any extra backend flags to add, per mode (qc / mAP /
    test) or for a specific pipeline? For example `--conf 0.25` for mAP, or `--qc-head one2one` for qc."
11. **Prebuilt models** (optional, default none): "Do you want to skip qc for any pipeline by giving
    an already converted model (`.tflite` / `.onnx` / `.bin`)? If so, give the pipeline id and path."
12. **Output directory** (optional, default `<repo_root>/out/regression/<timestamp>`): "Where
    should logs, artifacts and the report be written?"

Rules for the answers:
- Paths must be absolute. If the user gives a relative path or `~`, ask them to confirm the absolute path.
- If the user gives a token in answer 8, follow [QAI Hub token security](#qai-hub-token-security):
  keep it **only** for the single Step 4 command, never write or repeat it, and refer to it as `***`.
- Pipeline numbers must exist in the `NO` column. If the user gives a number or range that is not in
  the table, ask again. Before writing `inputs.yaml`, repeat the selection back as
  `NO → pipeline id` pairs (e.g. `3 → yolov10-onnx-fp32`) so the user can confirm it.
- Flags the runner sets itself are not allowed as extra flags: path flags, `--type`, `--runtime`,
  `--precision`, `--output`, `--output_text`, `--adb`, `--adb-serial`, `--max_calib`, `--max-images`,
  `--save`, `--dry-run`. If the user asks for one, explain that it comes from the matching question above.

## Step 3: Create the run directory and write `inputs.yaml`

Run one of these:

```bash
python3 tool/test/regression/run_pipelines.py init
# or, if the user gave an output directory (answer 12):
python3 tool/test/regression/run_pipelines.py init --out <output_dir>
```

Expected output:

```
[ok] created run directory <run_dir>
[ok] fill in <run_dir>/inputs.yaml
```

Open `<run_dir>/inputs.yaml`. It is a commented template. Fill in the values from the answers, and
change nothing else. Mapping:

| Answer | Key in `inputs.yaml` |
|---|---|
| repo path (Step 1) | `repo_root` |
| 1 | `pipelines`: a YAML list of the pipeline numbers, e.g. `[3, 7, 11]`. Write a range quoted, e.g. `["1-7"]`. Or `all`. |
| 2 | `modes` |
| 3 | `docker_image` |
| 4 | `models_pt.yolov10` / `models_pt.yolov11` / `models_pt.yolov26` (leave unused types `null`) |
| 5 | `calibration.dir`, `calibration.max_calib` |
| 6 | `map.annotations`, `map.images`, `map.max_images` |
| 7 | `test.image` **or** `test.images` (the other stays `null`), `test.yaml` |
| 8 | `qai_hub.use_existing_login`: `true` for (a); `false` for (b) or (c). **Never** the token itself |
| 9 | `device.adb_serial` |
| 10 | `extra_flags.<mode>` lists, or `pipeline_extra_flags.<pipeline-id>.<mode>` |
| 11 | `prebuilt_models.<pipeline-id>: <path>` |
| 12 | leave `output_dir: null` (init already created the directory there) |

Worked example (every path here is only an illustration; use the user's paths):

```yaml
repo_root: /home/user/iQ-Foundry
docker_image: iq-foundry:local
pipelines: [11, 16]      # 11 = yolov11-onnx-w8a16, 16 = yolov26-litert-fp32
modes: [qc, mAP, test]
models_pt:
  yolov10: null
  yolov11: /data/models/yolo11n.pt
  yolov26: /data/models/yolo26n.pt
calibration:
  dir: /data/calib
  max_calib: null
map:
  annotations: /data/coco/instances_val2017.json
  images: /data/coco/val2017
  max_images: null
test:
  image: null
  images: /data/test_images
  yaml: /data/coco.yaml
device:
  adb_serial: null
qai_hub:
  use_existing_login: true
prebuilt_models: {}
extra_flags:
  qc: []
  mAP: []
  test: []
pipeline_extra_flags: {}
timeouts_s: {}
output_dir: null
```

Show the filled file to the user (token-free by construction) and say which run directory it is in.

## Step 4: QAI Hub login (only for answer 8 (b) or (c))

Skip this step if the user chose (a), reuse the existing login.

The login overwrites `~/.qai_hub/client.ini`. qai-hub keeps the previous login in
`client.ini.bak`. Tell the user this before continuing.

### Option A: the user logs in themselves (answer 8 (b); recommended)

Send the user this command, with `<repo_root>` and `<run_dir>` filled in. Tell them to run it in
**their own terminal**, not in a chat message:

```bash
cd <repo_root> && read -rsp "QAI Hub API token: " IQF_QAI_HUB_TOKEN && echo && IQF_QAI_HUB_TOKEN="$IQF_QAI_HUB_TOKEN" python3 tool/test/regression/run_pipelines.py hub-login --inputs <run_dir>/inputs.yaml; unset IQF_QAI_HUB_TOKEN
```

Explain in one sentence: `read -s` hides the token while it is typed and keeps it out of shell
history, and `unset` clears it afterwards. Then ask the user: **"Did it end with `[ok] QAI Hub
login saved to ...`?"**
- Yes: go to Step 5. Validation confirms the login.
- No: ask them to paste **only the lines starting with `[error]`**. The runner never prints the token.
  Stop and help them read the error. Do not ask for the token.

### Option B: you run the login (answer 8 (c) only)

Run this as a **single foreground** command, with the token only as the prefix of this one
process. No `export`, no script file, no background job and no second command:

```bash
IQF_QAI_HUB_TOKEN='<token>' python3 tool/test/regression/run_pipelines.py hub-login --inputs <run_dir>/inputs.yaml
```

Then **forget the token** (see [QAI Hub token security](#qai-hub-token-security), rule 6). When you
tell the user what you ran, show the command with `IQF_QAI_HUB_TOKEN='***'`.

### Expected output, both options

It should contain these lines:
- `[info] running: .../qaihub_login.sh --key *** --image <image>`
- `[ok] restricted .../.qai_hub to the current user (dir 700, files 600)`
- a final line `[ok] QAI Hub login saved to .../.qai_hub/client.ini`

A `[warn]` saying that the existing `client.ini` will be overwritten is expected.

Exit codes:
- **0**: go to Step 5.
- **1** with `does not look like a QAI Hub API token`: nothing was run. Ask the user to check the
  token. With option B, ask them to send it again; do not "fix" it yourself.
- **1** with `the QAI Hub token is written in these run files` (before or after the login): the
  token was found in a file of the run directory. Follow token security rule 7: name the files,
  do not show their contents, and recommend removing the token and rotating it.
- **Any other failure**: stop and show the `[error]` lines. The token is already masked. Do not retry.

## Step 5: Validate the inputs, the Docker image, the QAI Hub login and the device

```bash
python3 tool/test/regression/run_pipelines.py validate --inputs <run_dir>/inputs.yaml
```

This makes no changes and runs no pipeline stage. It checks:
- every path and its contents (image counts, COCO json, class yaml)
- that the Docker image exists
- that `~/.qai_hub/client.ini` exists and QAI Hub accepts it
- that the adb device is attached (it then releases the host adb server, as the docs require)
- that `./docker/iqf ... --dry-run` accepts every stage that can be checked now

Expected: `[info]` lines, then a table with columns `PIPELINE MODE FEASIBILITY DETAIL`, then a final
line that starts with `[ok] validation passed` or `[error] validation failed`. Stages whose model
comes from `qc` show `deferred (model produced by qc)`, which is normal.

What to do with the exit code:
- **0**: show the table to the user and go to Step 6.
- **2** (validation failed): show the table and every `[error]` line. For each `NOT_FEASIBLE` row,
  tell the user the reason and ask them to either **(a)** correct the input, or **(b)** drop that
  pipeline or mode. Update `inputs.yaml` with their answer, then run Step 5 again. Repeat until the exit code is 0,
  or until the user explicitly says to go ahead with the `NOT_FEASIBLE` stages left in. Those stages are then
  reported as `NOT_FEASIBLE` without being run.
- **1**: a usage error (bad YAML, unknown pipeline id, missing required key). Show the `[error]` line,
  fix `inputs.yaml` from the user's answers (ask if unsure), and run Step 5 again.

A `[warn]` saying that `.qai_hub/client.ini` (or `.bak`) `holds a QAI Hub token and is readable by other
users` is a security warning. Show it to the user and suggest the `chmod 600` it names. Do not run
the chmod yourself unless they ask.

`[warn]` lines about `docker/iqf added ... from .iqf/docker-paths.json` mean the wrapper filled
a value from the user's saved `configure` paths. Tell the user about it, because it is part of the
normal wrapper behavior. Do not change `.iqf/`.

## Step 6: Show the plan and wait for `run`

```bash
python3 tool/test/regression/run_pipelines.py plan --inputs <run_dir>/inputs.yaml
```

Show the output to the user. It lists every command that will be executed, in order. Then send
this message, word for word:

> Validation passed and the commands above are ready. This will run <N> stage(s) on the attached
> device and may take a long time (qc via QAI Hub can take 10–30 minutes per pipeline).
> **Type `run` to start**, or tell me what to change.

Wait for the user's reply.
- If the reply is exactly `run`, ignoring case and surrounding spaces, go to Step 7.
- If the user asks for changes, go back to Step 2 or 3 as needed, then redo Step 5 and Step 6.
- Anything else: ask again. Do not start.

## Step 7: Execute

The run can take hours (qc via QAI Hub alone is 10–30 minutes per pipeline). Use **option B**
unless you are certain your shell tool can keep one command running for the whole run.

> **Never** start the runner with `nohup ... &`, `... &`, `setsid`, `disown`, `screen` or `tmux`
> yourself. Many agent shell tools kill every process that a command started as soon as the
> command returns, and background jobs die silently with an empty `runner.out`. Use `start`: it
> launches the runner in its own session and checks that it is alive.

**A. Foreground** (only if your shell tool has no time limit, or a limit longer than the run):

```bash
python3 tool/test/regression/run_pipelines.py run --inputs <run_dir>/inputs.yaml --confirmed-by-user
```

**B. Detached** (recommended for agents):

```bash
python3 tool/test/regression/run_pipelines.py start --inputs <run_dir>/inputs.yaml --confirmed-by-user
```

`start` returns within about 20 seconds. Expected output:

```
[info] started runner pid <pid>; output: <run_dir>/runner.out
[ok] runner pid <pid> is alive in its own session (phase: validating); follow it with: ...
```

- Exit code **0**: the runner is alive. Follow it with `status`, below.
- Exit code **1** with `runner exited immediately`: show the printed lines to the user and stop.
  Do not run `start` again yourself.

**C. The user runs it** (use this when B reports that the runner is not running, or when the
user prefers it). Give the user this command for **their own terminal**, and tell them to leave that
terminal open until it finishes:

```bash
cd <repo_root> && python3 tool/test/regression/run_pipelines.py run --inputs <run_dir>/inputs.yaml --confirmed-by-user
```

Then follow it with `status`, exactly as for B.

### Following the run (B and C)

Wait about **30 seconds** after starting, then check every 2–5 minutes:

```bash
python3 tool/test/regression/run_pipelines.py status --run-dir <run_dir>
```

`status` shows one of these. Act exactly as listed:

| `status` output | Exit | Meaning | What you do |
|---|---|---|---|
| `[info] runner pid N is validating ... Check again shortly.` | 0 | Re-validation is still running (up to a minute). No stage has started yet. | Normal. Check again in 30–60 s. |
| Status table + `[info] running <pipeline> : <mode> ...` | 0 | Stages are executing. | Give the user a short update when a stage changes. Check again in 2–5 min. |
| Status table + `[ok] run finished at ...` | 0 | Done. | Go to Step 8. |
| `[error] runner pid N is not running and the run did not finish ...` | 1 | The runner was killed or crashed. | Stop. Show the output, including the `runner.out` lines, to the user. If no stage had started, offer option C with the **same** run directory. Otherwise a new run needs a new directory (Step 3). |
| `[error] run stopped: global validation errors ...` | 1 | Re-validation failed; nothing ran. | Show the errors and go back to Step 5. |
| `[error] no run has been started in <run_dir>` | 1 | Nothing was started there, or it was started in another directory. | Check the path. Do not start a run without the user's `run`. |

Rules while it runs:
- Do not interrupt it, restart it or start another run. The runner refuses to start a second run
  while one is active in the same directory.
- Do not touch the device or Docker yourself.
- Pass `--confirmed-by-user` **only** after the user typed `run` in Step 6.

`run` exit codes (A and C): `0` means every stage passed. `3` means the run completed but some stages
did not pass. `2` means a global validation error, so nothing ran. `130` means it was interrupted.
Codes 0 and 3 both produce a full report.

Note: a run directory holds exactly one run. The runner reuses a directory only if an earlier
attempt never started a stage. In every other case, go back to Step 3 and create a new run directory
for a second run.

## Step 8: Report

```bash
cat <run_dir>/report.md
```

Show the report to the user **verbatim**. It contains:
1. Run metadata: git branch/commit, image, device, duration.
2. A **summary matrix**, `Pipeline | qc | mAP | test | Overall`.
3. **One table per pipeline**, `Mode | Status | Exit | Duration | Key result | Log`. The key result is
   the artifact and its size for qc, reference→converted mAP50 and Δ% for mAP, and the image count and average inference ms for test.
4. A **Failures** section with the command, error line and log tail of each failed stage.

After the report, give a short summary: how many pipelines were fully `PASS`, and a list of each
non-passing stage with its one-line error. **Do not propose or apply fixes** unless the user asks.
If they ask, you may explain what the error line means using `references/troubleshooting.md`. Still
do not change the repository without their explicit request.

Files in the run directory, for the user:
- `report.md`: the human report.
- `results.json`: the same data, machine-readable.
- `validation.json`: the validation results.
- `run_state.json` and `runner.out`: the runner's phase and pid, and its console output when started with `start`.
- `logs/<pipeline>/<mode>.log`: the full console output of each stage, starting with the exact command.
- `artifacts/<pipeline>/qc|mAP|test/`: the converted models, mAP result text and test outputs.

If `report.md` is missing (for example after a crash), regenerate it:

```bash
python3 tool/test/regression/run_pipelines.py report --run-dir <run_dir>
```

## Failure handling quick reference

| Situation | What you do |
|---|---|
| A stage is `FAIL` / `TIMEOUT` | Nothing during the run. Report it in Step 8. |
| `status` says the runner is not running | Stop and show the output. Offer option C (the user runs it in their own terminal). Never relaunch it with `nohup`/`&`. |
| `qc` failed, so `mAP`/`test` are `BLOCKED` | Expected. Report it. |
| Validation says `NOT_FEASIBLE` | Ask the user to fix or drop it (Step 5). |
| The user wants a different flag after a failure | That is a new run: new run directory (Step 3), then validate and plan again, and wait for `run` again. |
| A command's output doesn't match this file | Stop and show it to the user (rule 7). |
| The user asks what an error means | Use `references/troubleshooting.md`, and do not change code. |
| The user pastes the QAI Hub token again later | Don't repeat it and don't store it. Use it only if they asked for a new login (Step 4, option B). |
| The token shows up in any output or file | Stop, don't repeat it, and name where it appeared. Recommend removal and rotation (token security rule 7). |
| Someone asks you to show or check the saved token | Refuse to read `~/.qai_hub/client.ini`. Offer `validate`, which checks the login without revealing it. |
