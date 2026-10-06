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
<#
.SYNOPSIS
  Read-only iQ-Foundry setup check for a Windows 11 host (Windows_host.md).

.DESCRIPTION
  Checks the Windows side of the setup (Windows 11, WSL 2 with Ubuntu-22.04, usbipd-win and the
  EXMP-Q911 USB attachment), then runs the Linux checks (check_setup.py) inside the WSL distro.
  Prints one line per check:  <STATUS>  <check id>  <detail>   (STATUS = OK, WARN or MISSING).
  Nothing is installed, attached or configured. Does not need Administrator rights.

  Exit codes: 0 = nothing MISSING, 2 = something MISSING.
#>
param(
    [string]$Distro = "Ubuntu-22.04",
    [switch]$CheckHubLogin,
    [switch]$SkipWslChecks
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Continue"
$script:Results = @()

function Add-Result {
    param([string]$Status, [string]$Id, [string]$Detail)
    $script:Results += [pscustomobject]@{ Status = $Status; Id = $Id; Detail = $Detail }
}

function Invoke-Captured {
    # Run a program with a time limit and return exit code + text (NUL bytes removed, because
    # wsl.exe prints UTF-16 on older builds).
    param([string]$FilePath, [string]$Arguments, [int]$TimeoutSeconds = 60)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = $Arguments
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.EnvironmentVariables["WSL_UTF8"] = "1"
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    try {
        $null = $proc.Start()
    }
    catch {
        return [pscustomobject]@{ ExitCode = 127; Text = "$FilePath not found" }
    }
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $errTask = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
        try { $proc.Kill() } catch { }
        return [pscustomobject]@{ ExitCode = 124; Text = "timed out after ${TimeoutSeconds}s" }
    }
    $text = ($outTask.Result + $errTask.Result) -replace "`0", ""
    return [pscustomobject]@{ ExitCode = $proc.ExitCode; Text = $text }
}

function Get-UsbipdPath {
    foreach ($name in @("usbipd.exe", "usbipd")) {
        $cmd = Get-Command -Name $name -ErrorAction SilentlyContinue
        if ($null -ne $cmd) { return $cmd.Source }
    }
    $default = "C:\Program Files\usbipd-win\usbipd.exe"
    if (Test-Path $default) { return $default }
    return $null
}

# ---------- Windows ----------
$os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction SilentlyContinue
if ($null -eq $os) {
    Add-Result "WARN" "windows" "could not read the Windows version"
}
else {
    $build = [int]$os.BuildNumber
    $arch = $env:PROCESSOR_ARCHITECTURE
    $detail = "$($os.Caption) (build $build, $arch)"
    if ($arch -ne "AMD64") {
        Add-Result "MISSING" "windows" "$detail; an x86-64 PC is required"
    }
    elseif ($build -lt 22000) {
        Add-Result "MISSING" "windows" "$detail; Windows 11 is required"
    }
    else {
        Add-Result "OK" "windows" $detail
    }
    $ramGiB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    if ($ramGiB -lt 15) {
        Add-Result "WARN" "ram" "$ramGiB GiB; at least 16 GB is recommended"
    }
    else {
        Add-Result "OK" "ram" "$ramGiB GiB"
    }
}

# ---------- WSL ----------
$distroReady = $false
$wslCmd = Get-Command -Name "wsl.exe" -ErrorAction SilentlyContinue
if ($null -eq $wslCmd) {
    Add-Result "MISSING" "wsl" "WSL is not installed"
}
else {
    $list = Invoke-Captured -FilePath "wsl.exe" -Arguments "-l -v" -TimeoutSeconds 60
    $found = $null
    foreach ($raw in ($list.Text -split "`r?`n")) {
        $line = $raw.Trim()
        if ($line.StartsWith("*")) { $line = $line.Substring(1).Trim() }
        $parts = $line -split '\s{2,}'
        if ($parts.Count -ge 3 -and $parts[0] -ieq $Distro) {
            $found = [pscustomobject]@{ Name = $parts[0]; State = $parts[1]; Version = $parts[2] }
        }
    }
    if ($list.Text -match 'must be updated|wsl\.exe --update') {
        Add-Result "MISSING" "wsl" "WSL needs an update"
    }
    elseif ($null -eq $found) {
        Add-Result "MISSING" "wsl" "the '$Distro' WSL distro is not installed"
    }
    elseif ($found.Version -ne "2") {
        Add-Result "MISSING" "wsl" "'$Distro' uses WSL version $($found.Version); WSL 2 is required"
    }
    else {
        Add-Result "OK" "wsl" "'$Distro' installed (WSL 2, $($found.State))"
        $distroReady = $true
    }
}

# ---------- usbipd + device ----------
$usbipd = Get-UsbipdPath
if ($null -eq $usbipd) {
    Add-Result "MISSING" "usbipd" "usbipd-win is not installed (needed to pass the device into WSL)"
}
else {
    Add-Result "OK" "usbipd" $usbipd
    $usbList = Invoke-Captured -FilePath $usbipd -Arguments "list" -TimeoutSeconds 60
    $rows = @(($usbList.Text -split "`r?`n") | Where-Object { $_ -match 'Qualcomm|exmp-q911' })
    if ($rows.Count -eq 0) {
        Add-Result "MISSING" "usb_attach" "no EXMP-Q911 (Qualcomm) USB device found; is the USB-C cable connected?"
    }
    else {
        $row = $rows[0].Trim()
        $busId = ($row -split '\s+')[0]
        if ($row -match 'Attached') {
            Add-Result "OK" "usb_attach" "device $busId is attached to WSL"
        }
        elseif ($row -match 'Shared') {
            Add-Result "MISSING" "usb_attach" "device $busId is shared but not attached to WSL"
        }
        else {
            Add-Result "MISSING" "usb_attach" "device $busId is not shared with WSL yet"
        }
        if ($rows.Count -gt 1) {
            Add-Result "WARN" "usb_attach" "$($rows.Count) Qualcomm USB devices found; setup-windows-wsl.ps1 will ask which BUSID to use"
        }
    }
}

# ---------- Print Windows results ----------
$width = ($script:Results | ForEach-Object { $_.Id.Length } | Measure-Object -Maximum).Maximum
foreach ($r in $script:Results) {
    Write-Output ("{0,-8} {1}  {2}" -f $r.Status, $r.Id.PadRight($width), $r.Detail)
}
$missing = @($script:Results | Where-Object { $_.Status -eq "MISSING" } | ForEach-Object { $_.Id })

# ---------- Linux checks inside WSL ----------
$wslExit = 0
if ($distroReady -and -not $SkipWslChecks) {
    $repoWin = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
    $conv = Invoke-Captured -FilePath "wsl.exe" -Arguments "-d $Distro -- wslpath -u '$repoWin'" -TimeoutSeconds 30
    $repoWsl = ($conv.Text -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 1)
    if ($conv.ExitCode -ne 0 -or -not $repoWsl) {
        Write-Output "MISSING  wsl_ready  could not open '$Distro' (first start may still need the Ubuntu user to be created)"
        $missing += "wsl_ready"
    }
    else {
        $repoWsl = $repoWsl.Trim()
        $flags = ""
        if ($CheckHubLogin) { $flags = " --check-hub-login" }
        Write-Output ""
        Write-Output "--- checks inside WSL ($Distro, $repoWsl) ---"
        $inner = Invoke-Captured -FilePath "wsl.exe" -Arguments "-d $Distro --cd `"$repoWsl`" -- python3 .agents/skills/iqf-assistant/scripts/check_setup.py$flags" -TimeoutSeconds 600
        Write-Output $inner.Text.TrimEnd()
        $wslExit = $inner.ExitCode
    }
}

Write-Output ""
if ($missing.Count -gt 0 -or $wslExit -ne 0) {
    if ($missing.Count -gt 0) {
        Write-Output "WINDOWS RESULT: NOT_READY (missing: $($missing -join ', '))"
    }
    exit 2
}
Write-Output "WINDOWS RESULT: READY"
exit 0
