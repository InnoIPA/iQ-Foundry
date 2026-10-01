# Setup fixes, by check id

This file covers every check id printed by `scripts/check_setup.py` (Ubuntu or WSL) and
`scripts/check_setup_windows.ps1` (Windows). Each entry gives:

- what the problem means, in plain words;
- what **the user** runs to fix it;
- where it is in the official guide.

Ubuntu users follow [Ubuntu_host.md](../../../../Ubuntu_host.md). Windows users follow
[Windows_host.md](../../../../Windows_host.md).

**Never run these fixes yourself.** Give the commands to the user. Commands with `sudo` ask for
the user's own password in their own terminal. After the user says they are done, run the setup
check again.

## Ubuntu / WSL checks (`check_setup.py`)

### `os`

- **MISSING (not Ubuntu, or not x86_64):** iQ-Foundry is tested on Ubuntu 22.04 on an x86-64 PC.
  On Windows it runs inside WSL Ubuntu-22.04 (Windows_host.md, Step 4).
- **WARN (another Ubuntu version):** it may work, but Ubuntu 22.04 is the tested version.
  Mention this and continue.

### `ram`

- **WARN:** at least 16 GB of memory is recommended. Conversions can be slow or fail with less.
- **On WSL:** WSL may be limited to part of the PC's memory. The limit is set with `memory=` in
  `%UserProfile%\.wslconfig`.

### `disk`

- **WARN:** less than about 20 GB free. Models, the Docker image and results need space. Ask
  the user to free some up.

### `repo`

- **MISSING:** this is not the iQ-Foundry folder, or `docker/iqf` is not executable.
- **Getting iQ-Foundry:**
  - Ubuntu: Ubuntu_host.md, Step 2: `git clone https://github.com/InnoIPA/iQ-Foundry.git`
  - Windows: Windows_host.md, Step 3 (download and extract the zip in PowerShell).
- **If the folder is right but not executable:** the user runs `chmod +x docker/iqf` in the
  iQ-Foundry folder.

### `wsl_systemd` (WSL only)

- **WARN:** systemd is not enabled in WSL. Docker usually needs it.
- **Fix:** re-run the Windows setup helper (Windows_host.md, Step 4) in **PowerShell as
  Administrator**, from the iQ-Foundry folder:

  ```powershell
  powershell -ExecutionPolicy Bypass -File .\setup-windows-wsl.ps1
  ```

### `docker_cli`

- **MISSING:** Docker is not installed.
- **Fix:** Ubuntu_host.md Step 3 / Windows_host.md Step 5. Run inside Ubuntu or WSL, from the
  iQ-Foundry folder:

  ```bash
  bash ./docker_install.sh
  ```

  It asks for the user's sudo password.

### `docker_group`

- **MISSING, "added ... but this terminal session started before that":** the user must log out
  and back in.
  - Ubuntu: log out of the desktop, or reboot.
  - WSL: close every Ubuntu window, run `wsl --shutdown` in PowerShell, then open Ubuntu again.
  - Then start the agent again.
- **MISSING, "not in the docker group":** run `bash ./docker_install.sh` (it adds the user to the
  group), then log out and back in as above.

### `docker_daemon`

- **"permission denied":** the same fix as `docker_group`.
- **"the Docker service is not running":** the user runs `sudo systemctl start docker`.
  - On WSL this needs systemd (see `wsl_systemd`).
  - If Docker was never installed, run `bash ./docker_install.sh` first.

### `image`

- **MISSING:** the iQ-Foundry Docker image has not been downloaded.
- **Fix:** Ubuntu_host.md Step 4 / Windows_host.md Step 6. Takes 3–5 minutes:

  ```bash
  docker pull innodiskorg/iqf:latest
  ```

- **If the check names a different image:** the user has set `IQF_DOCKER_IMAGE`. Pull that one
  instead, or unset the variable.

### `qai_hub`

- **MISSING:** not logged in to Qualcomm AI Hub. This is needed for **qc only**, for every
  runtime, qairt included: the iQ-Foundry wrapper requires the login file for every qc run.
- **Fix:** use the token-safe login in SKILL.md, "QAI Hub login", option (b). This replaces the
  guide's `./qaihub_login.sh --key ...` (Ubuntu_host.md Step 5 / Windows_host.md Step 7), so the
  token stays out of shell history.
- **Getting a token:**
  1. Create a free account at <https://aihub.qualcomm.com/>.
  2. Go to Account → Settings → API Token.
- **WARN, "readable by other users":** other users on the PC could read the token. The user runs:

  ```bash
  chmod 700 ~/.qai_hub && chmod 600 ~/.qai_hub/client.ini ~/.qai_hub/client.ini.bak 2>/dev/null
  ```

### `qai_hub_login`

- **MISSING, "QAI Hub login check failed":** the saved token was rejected, or there is no
  internet connection to QAI Hub.
- **Check:** can the user open the QAI Hub website? Is there a company proxy?
- **Fix:** log in again with a fresh token (SKILL.md, option (b)).

### `usb`

- **MISSING:** `/dev/bus/usb` does not exist, so USB devices are not visible here. On WSL this
  means USB passthrough is not set up: re-run `setup-windows-wsl.ps1` (Windows_host.md, Step 4).

### `device`

The EXMP-Q911 board is needed for **mAP and test only**. The user can still convert (qc) without
it.

- **"no EXMP-Q911 device found":**
  1. Check that the board is powered on and connected with the USB-C cable (Ubuntu_host.md /
     Windows_host.md, Step 1).
  2. **Windows/WSL only:** the USB attachment is lost when the board is unplugged or the PC
     restarts. Attach it again in **PowerShell as Administrator**:

     ```powershell
     usbipd list
     usbipd attach --wsl --busid <BUSID>
     ```

     Use the BUSID of the Qualcomm / EXMP-Q911 row. Or re-run `setup-windows-wsl.ps1`.
  3. Close other tools that use adb, such as Android Studio or another terminal running adb.
- **"device seen but not usable: <serial>=offline":** the board was seen, but adb cannot talk to
  it.
  - Unplug and re-plug the USB-C cable, or reboot the board, then check again.
  - If another program keeps restarting adb, close it.
- **"<serial>=unauthorized":** the board has not accepted this computer's adb key. Re-plug it,
  then check again.
- **"N devices attached" (WARN):** ask the user which board to use. Pass its serial with
  `--adb-serial <serial>` to mAP and test.

## Windows checks (`check_setup_windows.ps1`)

### `windows`

- **MISSING:** Windows 11 on an x86-64 PC is required (Windows_host.md, Prerequisites).

### `wsl`

- **"WSL is not installed" / "distro is not installed" / "WSL version 1" / "needs an update":**
  1. Run the setup helper in **PowerShell as Administrator** (Windows_host.md, Steps 2–4), from
     the iQ-Foundry folder:

     ```powershell
     powershell -ExecutionPolicy Bypass -File .\setup-windows-wsl.ps1
     ```

  2. Answer `Y` when Windows asks to continue.
  3. On the first start, create the Ubuntu username and password. These are the user's own; do
     not ask for them.

### `wsl_ready`

- **MISSING:** the distro could not be opened. Usually the first start still needs the Ubuntu
  user to be created. Open "Ubuntu-22.04" from the Start menu once and finish the prompts.

### `usbipd`

- **MISSING:** usbipd-win is not installed. It is what passes the board's USB connection into
  WSL. The setup helper installs it (Windows_host.md, Step 4).

### `usb_attach`

- **"not shared" / "shared but not attached":** the board is not passed into WSL. The user
  either re-runs the setup helper, or runs this in **PowerShell as Administrator**:

  ```powershell
  usbipd bind --busid <BUSID>
  usbipd attach --wsl --busid <BUSID>
  ```

  `bind` is needed only the first time.
- **"no EXMP-Q911 (Qualcomm) USB device found":** check the cable and power (Windows_host.md,
  Step 1).

### Linux checks inside WSL

After the Windows checks, the script runs `check_setup.py` inside WSL. Those lines use the ids
in the Ubuntu / WSL section above. The WSL commands (`docker_install.sh`, `docker pull`, the QAI
Hub login) are run in the **Ubuntu terminal**, not in PowerShell.
