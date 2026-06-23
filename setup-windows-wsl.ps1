#!/usr/bin/env python3
# Copyright 2026 Innodisk Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu-22.04",
    [string]$BusId,
    [switch]$SkipUsb,
    [switch]$ForceUsbAttach
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$script:LastWslListResult = $null
$script:UsbStatusLabel = "Innodisk EXMP-Q911(Qualcomm IQ9075)"
$script:WslKeepAlive = $null

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "== $Title ==" -ForegroundColor Cyan
}

function Write-Info {
    param([string]$Message)
    Write-Host "[info] $Message"
}

function Write-WarnLine {
    param([string]$Message)
    Write-Host "[warn] $Message" -ForegroundColor Yellow
}

function Write-ErrorLine {
    param([string]$Message)
    Write-Host "[error] $Message" -ForegroundColor Red
}

function Read-YesNo {
    param(
        [string]$Prompt,
        [bool]$DefaultYes = $true
    )

    # Keep the current recursive re-prompt behavior so the existing prompt flow stays intact.
    $suffix = if ($DefaultYes) { "[Y/n]" } else { "[y/N]" }
    $response = Read-Host "$Prompt $suffix"
    if ([string]::IsNullOrWhiteSpace($response)) {
        return $DefaultYes
    }

    switch -Regex ($response.Trim()) {
        '^(?i)y(?:es)?$' { return $true }
        '^(?i)n(?:o)?$' { return $false }
        default {
            Write-WarnLine "Please answer y or n."
            return Read-YesNo -Prompt $Prompt -DefaultYes:$DefaultYes
        }
    }
}

function Read-RequiredValue {
    param(
        [string]$Prompt,
        [string]$MissingMessage
    )

    $value = Read-Host $Prompt
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw $MissingMessage
    }

    return $value.Trim()
}

function Format-CommandText {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    $parts = @($FilePath)
    foreach ($argument in $ArgumentList) {
        if ($argument -match '\s') {
            $escaped = $argument.Replace('"', '\"')
            $parts += '"' + $escaped + '"'
        }
        else {
            $parts += $argument
        }
    }
    return ($parts -join " ")
}

function Invoke-ExternalCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$ArgumentList = @(),
        [switch]$AllowFailure,
        [switch]$SuppressCommand,
        [switch]$SuppressOutput
    )

    if (-not $SuppressCommand) {
        Write-Host ("[cmd] " + (Format-CommandText -FilePath $FilePath -ArgumentList $ArgumentList))
    }

    $previousErrorActionPreference = $ErrorActionPreference
    $nativeErrorPreference = Get-Variable -Name "PSNativeCommandUseErrorActionPreference" -ErrorAction SilentlyContinue

    try {
        $script:ErrorActionPreference = "Continue"
        if ($null -ne $nativeErrorPreference) {
            $previousNativePreference = [bool]$nativeErrorPreference.Value
            $script:PSNativeCommandUseErrorActionPreference = $false
        }

        $output = & $FilePath @ArgumentList 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $script:ErrorActionPreference = $previousErrorActionPreference
        if ($null -ne $nativeErrorPreference) {
            $script:PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
    }

    if ($null -eq $exitCode) {
        $exitCode = 0
    }

    $lines = @()
    foreach ($line in @($output)) {
        if ($null -eq $line) {
            continue
        }
        $lines += ($line.ToString() -replace "`0", "")
    }

    if (-not $SuppressOutput -and $lines.Count -gt 0) {
        foreach ($line in $lines) {
            Write-Host $line
        }
    }

    if (($exitCode -ne 0) -and (-not $AllowFailure)) {
        throw "Command failed with exit code ${exitCode}: $(Format-CommandText -FilePath $FilePath -ArgumentList $ArgumentList)"
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Lines    = $lines
        Text     = ($lines -join [Environment]::NewLine).TrimEnd()
    }
}

function Test-IsAdmin {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-UsbipdCommand {
    $command = Get-Command -Name "usbipd.exe" -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        $command = Get-Command -Name "usbipd" -ErrorAction SilentlyContinue
    }
    if ($null -ne $command) {
        return $command.Source
    }

    $defaultPath = "C:\Program Files\usbipd-win\usbipd.exe"
    if (Test-Path $defaultPath) {
        return $defaultPath
    }

    return $null
}

function Get-WslDistros {
    # Read the live distro table first so later steps can decide whether WSL itself is missing,
    # whether the requested distro needs to be installed, or whether it just needs conversion.
    $result = Invoke-ExternalCommand -FilePath "wsl.exe" -ArgumentList @("-l", "-v") -AllowFailure -SuppressOutput
    $script:LastWslListResult = $result
    $distros = @()

    $installableMissingPatterns = @(
        'no installed distributions',
        'optional component',
        'virtual machine platform',
        'wsl\.exe --install',
        'windows subsystem for linux has not been enabled',
        'wsl is not installed'
    )
    $installableMissing = $false
    foreach ($pattern in $installableMissingPatterns) {
        if ($result.Text -match $pattern) {
            $installableMissing = $true
            break
        }
    }

    if (($result.ExitCode -ne 0) -and (-not $installableMissing)) {
        throw "Command failed with exit code $($result.ExitCode): wsl.exe -l -v"
    }

    foreach ($rawLine in $result.Lines) {
        $line = $rawLine.TrimEnd()
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line -match '^\s*NAME\s+STATE\s+VERSION\s*$') {
            continue
        }

        $isDefault = $false
        $normalized = $line.Trim()
        if ($normalized.StartsWith("*")) {
            $isDefault = $true
            $normalized = $normalized.Substring(1).Trim()
        }

        $parts = $normalized -split '\s{2,}', 3
        if ($parts.Count -lt 3) {
            continue
        }

        $distros += [pscustomobject]@{
            Name      = $parts[0].Trim()
            State     = $parts[1].Trim()
            Version   = $parts[2].Trim()
            IsDefault = $isDefault
        }
    }

    return @($distros)
}

function Test-WslLaunchReady {
    param(
        [string]$TargetDistro,
        [int]$TimeoutSeconds = 8
    )

    # A distro can be installed but still blocked on its first interactive user-creation flow.
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = "wsl.exe"
    $processInfo.Arguments = "-d $TargetDistro -- echo __IQF_READY__"
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo

    $null = $process.Start()
    $completed = $process.WaitForExit($TimeoutSeconds * 1000)

    if (-not $completed) {
        try {
            $process.Kill()
        }
        catch {
        }

        return [pscustomobject]@{
            Ready  = $false
            Reason = "Launch the distro once, create the Linux user/password, then rerun this script."
        }
    }

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $text = (($stdout + [Environment]::NewLine + $stderr) -replace "`0", "").Trim()

    if (($process.ExitCode -eq 0) -and ($text -match "__IQF_READY__")) {
        return [pscustomobject]@{
            Ready  = $true
            Reason = $null
        }
    }

    if ($text -match '(?i)OOBE|default Unix user account|create the Linux user|create a UNIX user') {
        return [pscustomobject]@{
            Ready  = $false
            Reason = "Launch the distro once, create the Linux user/password, then rerun this script."
        }
    }

    return [pscustomobject]@{
        Ready  = $false
        Reason = "WSL distro '$TargetDistro' is installed but not ready for scripted commands yet. Launch it once manually, complete first-run setup, then rerun this script."
    }
}

function Invoke-WslFirstLaunch {
    param([string]$TargetDistro)

    Write-Section "WSL First Launch"
    Write-Info "WSL will prompt for the Linux user and password now."
    Write-Info "After setup finishes, this script will continue automatically without leaving you in the WSL shell."
    Write-WarnLine "If Ubuntu still leaves you at a shell prompt, type 'exit' once and this script will resume."
    Write-Host ("[cmd] wsl.exe -d " + $TargetDistro + " -- bash -lc `"exit 0`"")

    & wsl.exe -d $TargetDistro -- bash -lc "exit 0"
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) {
        $exitCode = 0
    }

    $launchCheck = Test-WslLaunchReady -TargetDistro $TargetDistro
    if ($launchCheck.Ready) {
        Write-Info "WSL first-launch setup completed and distro '$TargetDistro' is ready."
        return
    }

    if ($exitCode -ne 0) {
        throw "WSL first-launch setup for distro '$TargetDistro' did not complete cleanly. Run 'wsl -d $TargetDistro' manually, finish setup, then rerun this script."
    }

    throw $launchCheck.Reason
}

function Ensure-WslDistro {
    param([string]$TargetDistro)

    Write-Section "WSL Distro"
    # This covers initial WSL installation, distro installation, WSL 2 conversion, and first-run
    # readiness before the script proceeds to any USB-specific setup.
    $distros = @(Get-WslDistros)
    if ($distros.Count -eq 0) {
        Write-WarnLine "No WSL distros were parsed from 'wsl -l -v'."
    }

    $wslListText = ""
    if ($null -ne $script:LastWslListResult) {
        $wslListText = $script:LastWslListResult.Text
    }
    $missingWslBase = $wslListText -match '(?i)optional component|virtual machine platform|wsl\.exe --install|windows subsystem for linux has not been enabled|wsl is not installed'

    $target = $distros | Where-Object { $_.Name -ieq $TargetDistro } | Select-Object -First 1
    if ($null -eq $target) {
        if (-not (Test-IsAdmin)) {
            if ($missingWslBase) {
                throw "WSL is not fully installed. Re-run PowerShell as Administrator so this script can install WSL and distro '$TargetDistro'."
            }
            throw "The distro '$TargetDistro' is not installed. Re-run PowerShell as Administrator to install it."
        }

        if ($missingWslBase) {
            if (-not (Read-YesNo -Prompt "WSL is not fully installed. Install WSL and distro '$TargetDistro' now?" -DefaultYes $true)) {
                throw "WSL and distro '$TargetDistro' are required. Install them, then rerun this script."
            }
            Write-Info "Installing WSL and distro '$TargetDistro'."
        }
        else {
            if (-not (Read-YesNo -Prompt "WSL distro '$TargetDistro' is missing. Install it now?" -DefaultYes $true)) {
                throw "WSL distro '$TargetDistro' is required. Install it, then rerun this script."
            }
            Write-Info "Installing missing WSL distro '$TargetDistro'."
        }

        Write-Info "This step can take several minutes and may stay quiet while Windows downloads and registers the distro."
        Invoke-ExternalCommand -FilePath "wsl.exe" -ArgumentList @("--install", "-d", $TargetDistro, "--no-launch")
        Invoke-WslFirstLaunch -TargetDistro $TargetDistro

        $distros = @(Get-WslDistros)
        $target = $distros | Where-Object { $_.Name -ieq $TargetDistro } | Select-Object -First 1
        if ($null -eq $target) {
            throw "WSL distro '$TargetDistro' was installed, but it could not be found afterward. Run 'wsl -l -v' and then rerun this script."
        }
    }

    Write-Info "Found distro '$($target.Name)' in state '$($target.State)' with WSL version '$($target.Version)'."
    Invoke-ExternalCommand -FilePath "wsl.exe" -ArgumentList @("--set-default-version", "2")

    if ($target.Version -ne "2") {
        Write-Info "Converting distro '$TargetDistro' to WSL 2."
        Invoke-ExternalCommand -FilePath "wsl.exe" -ArgumentList @("--set-version", $TargetDistro, "2")
        $distros = @(Get-WslDistros)
        $target = $distros | Where-Object { $_.Name -ieq $TargetDistro } | Select-Object -First 1
    }

    $launchCheck = Test-WslLaunchReady -TargetDistro $TargetDistro
    if (-not $launchCheck.Ready) {
        Write-WarnLine $launchCheck.Reason
        if (-not (Read-YesNo -Prompt "Open '$TargetDistro' now to finish first-launch setup?" -DefaultYes $true)) {
            throw $launchCheck.Reason
        }

        Invoke-WslFirstLaunch -TargetDistro $TargetDistro
        $launchCheck = Test-WslLaunchReady -TargetDistro $TargetDistro
        if (-not $launchCheck.Ready) {
            throw $launchCheck.Reason
        }
    }

    return $target
}

function Ensure-Systemd {
    param([string]$TargetDistro)

    Write-Section "WSL systemd"
    # The script edits /etc/wsl.conf in place, then shuts WSL down so the next launch picks up the change.
    $checkCommand = "if grep -Eq '^[[:space:]]*systemd=true[[:space:]]*$' /etc/wsl.conf 2>/dev/null; then echo __IQF_SYSTEMD_ENABLED__; fi"
    $checkResult = Invoke-ExternalCommand -FilePath "wsl.exe" -ArgumentList @("-d", $TargetDistro, "--", "bash", "-lc", $checkCommand) -SuppressOutput
    if ($checkResult.Text -match "__IQF_SYSTEMD_ENABLED__") {
        Write-Info "systemd is already enabled in distro '$TargetDistro'."
        return
    }

    Write-WarnLine "Enabling systemd may require your WSL sudo password."

    $enableCommand = @'
sudo python3 - <<'PY'
from pathlib import Path

path = Path("/etc/wsl.conf")
text = path.read_text(encoding="utf-8") if path.exists() else ""
lines = text.splitlines()
out = []
boot_found = False
in_boot = False
systemd_written = False

for line in lines:
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        if in_boot and not systemd_written:
            out.append("systemd=true")
            systemd_written = True
        in_boot = stripped.lower() == "[boot]"
        if in_boot:
            boot_found = True
        out.append(line)
        continue
    if in_boot and stripped.lower().startswith("systemd="):
        if not systemd_written:
            out.append("systemd=true")
            systemd_written = True
        continue
    out.append(line)

if in_boot and not systemd_written:
    out.append("systemd=true")
    systemd_written = True

if not boot_found:
    if out and out[-1] != "":
        out.append("")
    out.append("[boot]")
    out.append("systemd=true")

path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
PY
'@
    Invoke-ExternalCommand -FilePath "wsl.exe" -ArgumentList @("-d", $TargetDistro, "--", "bash", "-lc", $enableCommand)
    Write-Info "Shutting down WSL so the systemd change can take effect."
    Invoke-ExternalCommand -FilePath "wsl.exe" -ArgumentList @("--shutdown")
}

function Ensure-Usbipd {
    param()

    Write-Section "usbipd-win"
    $usbipdPath = Get-UsbipdCommand
    if ($null -ne $usbipdPath) {
        Write-Info "Found usbipd at '$usbipdPath'."
        return $usbipdPath
    }

    if (-not (Read-YesNo -Prompt "usbipd-win is missing. Install it now?" -DefaultYes $true)) {
        throw "usbipd-win is required for USB passthrough. Install it, or rerun this script with -SkipUsb."
    }

    if (-not (Test-IsAdmin)) {
        throw "Installing usbipd-win requires Administrator PowerShell. Re-run this script from an elevated prompt."
    }

    Write-Info "Installing usbipd-win with winget."
    Invoke-ExternalCommand -FilePath "winget.exe" -ArgumentList @(
        "install",
        "--interactive",
        "--id", "dorssel.usbipd-win",
        "--exact",
        "--source", "winget",
        "--accept-source-agreements",
        "--accept-package-agreements"
    )
    $usbipdPath = Get-UsbipdCommand
    if ($null -eq $usbipdPath) {
        throw "usbipd-win was installed, but usbipd is still not on PATH. Reopen PowerShell and rerun this script."
    }

    Write-Info "Found usbipd at '$usbipdPath' after installation."
    return $usbipdPath
}

function Get-UsbipdList {
    param([string]$UsbipdPath)

    $result = Invoke-ExternalCommand -FilePath $UsbipdPath -ArgumentList @("list") -SuppressOutput
    $devices = @()

    foreach ($rawLine in $result.Lines) {
        $line = ($rawLine -replace "`0", "").TrimEnd()
        if ($line -notmatch '^\s*\d+-\d+\s+') {
            continue
        }

        $parts = $line.Trim() -split '\s{2,}', 4
        if ($parts.Count -lt 4) {
            continue
        }

        $devices += [pscustomobject]@{
            BusId  = $parts[0].Trim()
            VidPid = $parts[1].Trim()
            Device = $parts[2].Trim()
            State  = $parts[3].Trim()
            Raw    = $line.Trim()
        }
    }

    return [pscustomobject]@{
        Text    = $result.Text
        Devices = @($devices)
    }
}

function Confirm-QcOnlyContinuation {
    param([string]$Reason)

    # This bypass only exists for qc-only setup when there is no usable target device. The normal
    # USB attach and verification path for mAP/test stays unchanged once a BUSID is resolved.
    Write-WarnLine $Reason
    Write-WarnLine "Only qc mode can work without a connected target device."

    if (Read-YesNo -Prompt "Continue to the WSL shell anyway?" -DefaultYes $false) {
        Write-Info "Continuing without USB target setup. You can use qc mode after WSL opens."
        return $true
    }

    throw "USB target setup was cancelled. Connect the target device and rerun this script for mAP or test."
}

function Resolve-UsbBusId {
    param(
        [pscustomobject]$UsbList,
        [string]$RequestedBusId
    )

    Write-Section "USB Device Detection"

    if ($UsbList.Devices.Count -eq 0) {
        if ($UsbList.Text) {
            Write-Host $UsbList.Text
        }
        if ($RequestedBusId) {
            throw "BUSID '$RequestedBusId' was not found in the current usbipd device list."
        }
        if (Confirm-QcOnlyContinuation -Reason "usbipd did not report any attachable USB devices, so no target device is available for mAP or test.") {
            return $null
        }
    }

    if ($RequestedBusId) {
        $requested = $UsbList.Devices | Where-Object { $_.BusId -ieq $RequestedBusId } | Select-Object -First 1
        if ($null -eq $requested) {
            Write-Host $UsbList.Text
            throw "BUSID '$RequestedBusId' was not found in the current usbipd device list."
        }
        Write-Info "Using explicit BUSID '$RequestedBusId'."
        return $requested.BusId
    }

    $matches = @($UsbList.Devices | Where-Object {
        $_.Raw -match 'Qualcomm' -or $_.Device -match 'Qualcomm'
    })

    if ($matches.Count -eq 1) {
        Write-Info "Auto-detected Qualcomm USB device '$($matches[0].BusId)'."
        return $matches[0].BusId
    }

    if ($matches.Count -gt 1) {
        Write-WarnLine "Multiple Qualcomm USB devices matched. Choose the correct BUSID."
        foreach ($match in $matches) {
            Write-Host $match.Raw
        }

        $selection = Read-RequiredValue -Prompt "Enter the BUSID to use" -MissingMessage "A BUSID is required when multiple Qualcomm USB devices are present. Re-run with -BusId <BUSID>."

        $chosen = $UsbList.Devices | Where-Object { $_.BusId -ieq $selection } | Select-Object -First 1
        if ($null -eq $chosen) {
            throw "BUSID '$selection' was not found in the current usbipd device list."
        }
        return $chosen.BusId
    }

    Write-Host $UsbList.Text
    $manualBusId = Read-Host "Could not auto-detect a Qualcomm USB device. Enter a BUSID manually, or press Enter for qc-only options"
    if (-not [string]::IsNullOrWhiteSpace($manualBusId)) {
        $chosen = $UsbList.Devices | Where-Object { $_.BusId -ieq $manualBusId.Trim() } | Select-Object -First 1
        if ($null -eq $chosen) {
            throw "BUSID '$manualBusId' was not found in the current usbipd device list."
        }
        Write-Info "Using manually entered BUSID '$($chosen.BusId)'."
        return $chosen.BusId
    }

    if (Confirm-QcOnlyContinuation -Reason "No Qualcomm USB target device was selected, so mAP and test cannot run from this host right now.") {
        return $null
    }
}

function Get-UsbipdDeviceState {
    param(
        [pscustomobject]$UsbList,
        [string]$ResolvedBusId
    )

    $device = $UsbList.Devices | Where-Object { $_.BusId -ieq $ResolvedBusId } | Select-Object -First 1
    if ($null -eq $device) {
        return $null
    }

    if ($device.State -match '^(?i)Not shared\b') {
        return "Not shared"
    }
    if ($device.State -match '^(?i)Shared\b') {
        return "Shared"
    }
    if ($device.State -match '^(?i)Attached\b') {
        return "Attached"
    }
    return $null
}

function Invoke-WslVerification {
    param([string]$TargetDistro)

    Write-Section "WSL Verification"
    $verificationCommand = "command -v lsusb >/dev/null && lsusb || true; command -v adb >/dev/null && adb devices || true"
    Invoke-ExternalCommand -FilePath "wsl.exe" -ArgumentList @("-d", $TargetDistro, "--", "bash", "-lc", $verificationCommand)
}

function Assert-SetupSuccessful {
    param(
        [string]$TargetDistro,
        [switch]$UsbChecked,
        [string]$UsbipdPath,
        [string]$ResolvedBusId
    )

    # Re-read final WSL and usbipd state here instead of trusting earlier commands so the success
    # banner reflects the post-setup reality the user is about to rely on.
    $distros = @(Get-WslDistros)
    $target = $distros | Where-Object { $_.Name -ieq $TargetDistro } | Select-Object -First 1
    if ($null -eq $target) {
        throw "WSL and Ubuntu setup could not be verified because distro '$TargetDistro' was not found after setup."
    }

    if ($target.Version -ne "2") {
        throw "WSL and Ubuntu setup could not be verified because distro '$TargetDistro' is not running as WSL 2."
    }

    $launchCheck = Test-WslLaunchReady -TargetDistro $TargetDistro
    if (-not $launchCheck.Ready) {
        throw "WSL and Ubuntu setup could not be verified because distro '$TargetDistro' is not ready yet."
    }

    $usbStatusLine = $null
    if ($UsbChecked) {
        if ([string]::IsNullOrWhiteSpace($ResolvedBusId)) {
            throw "USB setup could not be verified because no Qualcomm device BUSID was resolved."
        }
        if ([string]::IsNullOrWhiteSpace($UsbipdPath)) {
            throw "USB setup could not be verified because usbipd-win was not available for a final check."
        }

        $usbList = Get-UsbipdList -UsbipdPath $UsbipdPath
        $deviceState = Get-UsbipdDeviceState -UsbList $usbList -ResolvedBusId $ResolvedBusId
        if ($deviceState -ne "Attached") {
            throw "USB setup could not be verified because device '$ResolvedBusId' is not attached to WSL."
        }

        $usbStatusLine = $script:UsbStatusLabel + " detected and attached to WSL"
    }

    return [pscustomobject]@{
        WslStatusLine = "WSL and $TargetDistro installed and ready"
        UsbStatusLine = $usbStatusLine
    }
}

function Start-WslKeepAlive {
    param([string]$TargetDistro)

    $distros = @(Get-WslDistros)
    $target = $distros | Where-Object { $_.Name -ieq $TargetDistro } | Select-Object -First 1
    if (($null -ne $target) -and ($target.State -ieq "Running")) {
        return $null
    }

    # usbipd attach is more reliable when the target distro is already running, so this background
    # process keeps WSL alive just for the attach/verify window and gets cleaned up afterward.
    Write-Info "Starting a temporary WSL session so usbipd can attach the device."
    $process = Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $TargetDistro, "--", "bash", "-lc", "while true; do sleep 300; done") -WindowStyle Hidden -PassThru

    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        $distros = @(Get-WslDistros)
        $target = $distros | Where-Object { $_.Name -ieq $TargetDistro } | Select-Object -First 1
        if (($null -ne $target) -and ($target.State -ieq "Running")) {
            return $process
        }
    }

    if ($process -and (-not $process.HasExited)) {
        Stop-Process -Id $process.Id -Force
    }

    throw "WSL distro '$TargetDistro' did not stay running long enough for usbipd attach."
}

function Stop-WslKeepAlive {
    param($Process)

    if (($null -ne $Process) -and (-not $Process.HasExited)) {
        Stop-Process -Id $Process.Id -Force
    }
}

function Attach-UsbToWsl {
    param(
        [string]$UsbipdPath,
        [string]$ResolvedBusId,
        [switch]$Reattach
    )

    Write-Section "USB Attach"
    # Reattach stays opt-in because some sessions only need state verification while others need a
    # full detach/attach cycle after cable or device-state changes.
    $usbList = Get-UsbipdList -UsbipdPath $UsbipdPath
    $deviceState = Get-UsbipdDeviceState -UsbList $usbList -ResolvedBusId $ResolvedBusId
    if ($null -eq $deviceState) {
        if ($usbList.Text) {
            Write-Host $usbList.Text
        }
        throw "Could not determine the current state for USB device '$ResolvedBusId'."
    }

    switch ($deviceState) {
        "Not shared" {
            if (-not (Test-IsAdmin)) {
                throw "usbipd bind requires Administrator PowerShell. Re-run this script as Administrator."
            }
            Write-Info "Binding and attaching USB device '$ResolvedBusId' to WSL."
            Invoke-ExternalCommand -FilePath $UsbipdPath -ArgumentList @("bind", "--busid", $ResolvedBusId)
            Invoke-ExternalCommand -FilePath $UsbipdPath -ArgumentList @("attach", "--wsl", "--busid", $ResolvedBusId)
        }
        "Shared" {
            Write-Info "Attaching shared USB device '$ResolvedBusId' to WSL."
            Invoke-ExternalCommand -FilePath $UsbipdPath -ArgumentList @("attach", "--wsl", "--busid", $ResolvedBusId)
        }
        "Attached" {
            if ($Reattach) {
                Write-Info "Reattaching USB device '$ResolvedBusId' to WSL."
                Invoke-ExternalCommand -FilePath $UsbipdPath -ArgumentList @("detach", "--busid", $ResolvedBusId)
                Invoke-ExternalCommand -FilePath $UsbipdPath -ArgumentList @("attach", "--wsl", "--busid", $ResolvedBusId)
            }
            else {
                Write-Info "USB device '$ResolvedBusId' is already attached to WSL."
            }
        }
    }

    Write-Section "usbipd Status"
    Invoke-ExternalCommand -FilePath $UsbipdPath -ArgumentList @("list")
}

function Print-SetupStatus {
    param(
        [string]$WslStatusLine,
        [string]$UsbStatusLine
    )

    Write-Section "Status"
    Write-Host "iQ-Foundry Windows Setup Successful"
    Write-Host ("- " + $WslStatusLine)

    if (-not [string]::IsNullOrWhiteSpace($UsbStatusLine)) {
        Write-Host ("- " + $UsbStatusLine)
    }
}

function Launch-WslShell {
    param([string]$TargetDistro)

    Write-Section "WSL Shell"
    Write-Info "Opening '$TargetDistro' now."
    Write-Host ("[cmd] wsl.exe -d " + $TargetDistro)
    & wsl.exe -d $TargetDistro
}

try {
    Write-Section "iQ-Foundry Windows WSL Prep"
    $null = Ensure-WslDistro -TargetDistro $Distro
    Ensure-Systemd -TargetDistro $Distro

    $resolvedBusId = $null
    $usbipdPath = $null
    $usbChecked = $false
    if ($SkipUsb) {
        Write-Section "USB"
        Write-Info "Skipping usbipd and USB verification because -SkipUsb was provided."
    }
    else {
        $usbipdPath = Ensure-Usbipd
        $usbList = Get-UsbipdList -UsbipdPath $usbipdPath
        $resolvedBusId = Resolve-UsbBusId -UsbList $usbList -RequestedBusId $BusId
        if (-not [string]::IsNullOrWhiteSpace($resolvedBusId)) {
            $usbChecked = $true
            # Keep WSL running while usbipd attaches so the distro is immediately available as a target.
            $script:WslKeepAlive = Start-WslKeepAlive -TargetDistro $Distro
            Attach-UsbToWsl -UsbipdPath $usbipdPath -ResolvedBusId $resolvedBusId -Reattach:$ForceUsbAttach
            Invoke-WslVerification -TargetDistro $Distro
        }
    }

    # Final verification runs before the shell opens so users land in a distro that matches the
    # success state printed by the script.
    $setupStatus = Assert-SetupSuccessful -TargetDistro $Distro -UsbChecked:$usbChecked -UsbipdPath $usbipdPath -ResolvedBusId $resolvedBusId
    Print-SetupStatus -WslStatusLine $setupStatus.WslStatusLine -UsbStatusLine $setupStatus.UsbStatusLine
    Launch-WslShell -TargetDistro $Distro
    Stop-WslKeepAlive -Process $script:WslKeepAlive
    exit 0
}
catch {
    Stop-WslKeepAlive -Process $script:WslKeepAlive
    Write-ErrorLine $_.Exception.Message
    exit 1
}
