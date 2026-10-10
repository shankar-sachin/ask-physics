# Run by the Ask Physics Windows installer (askphysics.iss); not meant to be run by hand.
#
#   askphysics-setup.ps1 -Action install   -Ref v0.4.0 -Log C:\...\AskPhysics-setup.log
#   askphysics-setup.ps1 -Action uninstall            -Log C:\...\AskPhysics-setup.log
#
# install: runs install.ps1 (next to this file) pinned to -Ref, in a plain-output child
#   PowerShell whose output goes to -Log, and exits with install.ps1's exit code. If
#   ASKPHYSICS_SOURCE is set in the environment, install.ps1 installs from that instead.
# uninstall: runs `uv tool uninstall askphysics`. Always exits 0: a missing uv or a tool that
#   is already gone must not stop the program from being removed. Models in the user's cache
#   are not touched.

param(
    [Parameter(Mandatory = $true)][ValidateSet("install", "uninstall")][string]$Action,
    [string]$Ref = "",
    [string]$Log = ""
)

$ErrorActionPreference = "Stop"
if (-not $Log) { $Log = Join-Path ([System.IO.Path]::GetTempPath()) "AskPhysics-setup.log" }
$ErrLog = "$Log.err"

# Runs a program with its output (stdout, then stderr) appended to $Log; returns its exit code.
function Invoke-Logged([string]$Exe, [string[]]$Arguments) {
    $quoted = foreach ($a in $Arguments) { if ($a -match '[\s"]') { '"' + ($a -replace '"', '\"') + '"' } else { $a } }
    $process = Start-Process -FilePath $Exe -ArgumentList ($quoted -join " ") -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$Log.out" -RedirectStandardError $ErrLog
    $null = $process.Handle  # keep the handle, so the exit code is available afterwards
    $process.WaitForExit()
    foreach ($f in "$Log.out", $ErrLog) {
        if (Test-Path $f) {
            Get-Content -ErrorAction SilentlyContinue $f | Add-Content -Path $Log
            Remove-Item -Force -ErrorAction SilentlyContinue $f
        }
    }
    return $process.ExitCode
}

function Find-Uv {
    $found = Get-Command uv -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe")
    )
    foreach ($dir in $env:XDG_BIN_HOME, $env:UV_INSTALL_DIR) {
        if ($dir) { $candidates += (Join-Path $dir "uv.exe") }
    }
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

Set-Content -Path $Log -Value ("Ask Physics setup: {0} {1}" -f $Action, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

if ($Action -eq "install") {
    $script = Join-Path $PSScriptRoot "install.ps1"
    if (-not (Test-Path $script)) {
        Add-Content -Path $Log -Value "error: install.ps1 is missing from $PSScriptRoot"
        exit 2
    }
    if ($Ref) { $env:ASKPHYSICS_REF = $Ref }
    $env:NO_COLOR = "1"  # plain start/done lines, which read well in a log
    $shell = (Get-Process -Id $PID).Path
    $code = Invoke-Logged $shell @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script)
    Add-Content -Path $Log -Value "install.ps1 exited with code $code"
    exit $code
}

# uninstall
try {
    $uv = Find-Uv
    if (-not $uv) {
        Add-Content -Path $Log -Value "uv was not found; nothing to uninstall with it"
    } else {
        $code = Invoke-Logged $uv @("tool", "uninstall", "askphysics")
        Add-Content -Path $Log -Value "uv tool uninstall askphysics exited with code $code (ignored)"
    }
} catch {
    Add-Content -Path $Log -Value "uninstall step failed and was ignored: $($_.Exception.Message)"
}
exit 0
