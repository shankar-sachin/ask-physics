# Ask Physics installer for Windows (PowerShell 5.1 or 7+).
#
#   irm https://askphysics.vercel.app/installers/install.ps1 | iex
#
# Installs the `askphysics` command in its own isolated environment with uv
# (https://docs.astral.sh/uv/), installing uv first if it's missing. Nothing
# touches your system Python.
#
# Each step is a spinner that turns into a tick with the time it took; the full output of a
# step is kept in a log file whose path is shown if it fails. Without a terminal (CI, a pipe)
# or with NO_COLOR set, the same steps print as plain start/done lines, as install.sh does.
#
# Environment variables:
#   ASKPHYSICS_REF           git tag, branch, or commit to install (default: latest release)
#   ASKPHYSICS_SOURCE        install from this local path or URL instead of GitHub
#   ASKPHYSICS_SKIP_MODELS   set to 1 to skip downloading the Fermi models
#
# Uninstall:  uv tool uninstall askphysics

$ErrorActionPreference = "Stop"
$Repo = "shankar-sachin/ask-physics"
$PythonVersion = "3.12"

# ---- The look (the same theme, glyphs and columns as install.sh and scripts/lib.sh) ----------

$script:Pretty = $false
try {
    $script:Pretty = (-not [Console]::IsOutputRedirected) -and (-not $env:NO_COLOR) -and ($env:TERM -ne "dumb")
} catch {
    $script:Pretty = $false
}
$script:Width = 80
try { $script:Width = $Host.UI.RawUI.WindowSize.Width } catch { $script:Width = 80 }
if ($script:Width -gt 100) { $script:Width = 100 }
if ($script:Width -lt 48) { $script:Width = 48 }
if ($script:Pretty) {
    try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
}

# Characters by code, so the script reads the same in any encoding.
$Tick = [string][char]0x2713
$Cross = [string][char]0x2717
$Bullet = [string][char]0x25C9
$Dot = [string][char]0x00B7
$Rule = [string][char]0x2500
$Frames = 0x280B, 0x2819, 0x2839, 0x2838, 0x283C, 0x2834, 0x2826, 0x2827, 0x2807, 0x280F | ForEach-Object { [string][char]$_ }

# The theme colours, as the nearest console colours.
$Brand = "Cyan"; $Accent = "Magenta"; $Muted = "DarkGray"; $Ok = "Green"
$Bad = "Red"; $Value = "White"

$script:Step = 0
$script:Total = 0
$script:Started = Get-Date
$script:LogDir = Join-Path ([System.IO.Path]::GetTempPath()) ("askphysics-logs\install-" + (Get-Date -Format "yyyyMMdd-HHmmss"))

function Write-Part([string]$Text, [string]$Color) {
    if ($script:Pretty) { Write-Host $Text -ForegroundColor $Color -NoNewline } else { Write-Host $Text -NoNewline }
}

# 45s, 2m 05s, 1h 03m 09s: the same wording as install.sh.
function Format-Elapsed([double]$Seconds) {
    $s = [int][math]::Round($Seconds)
    if ($s -ge 3600) { return ("{0}h {1:00}m {2:00}s" -f [math]::Floor($s / 3600), [math]::Floor(($s % 3600) / 60), ($s % 60)) }
    if ($s -ge 60) { return ("{0}m {1:00}s" -f [math]::Floor($s / 60), ($s % 60)) }
    return "${s}s"
}

# One aligned line, without the line ending: mark, counter, title, detail, and the time at the right.
function Write-Row([string]$Kind, [string]$Mark, [string]$Counter, [string]$Title, [string]$Detail, [string]$Time) {
    $markColor = $Accent; $titleColor = $Brand; $detailColor = $Muted
    if ($Kind -eq "ok") { $markColor = $Ok; $titleColor = $Value }
    if ($Kind -eq "bad") { $markColor = $Bad; $titleColor = $Bad; $detailColor = $Bad }
    Write-Part "  " $Muted
    Write-Part $Mark $markColor
    if ($Counter) { Write-Part (" " + $Counter) $Muted }
    Write-Part (" " + $Title) $titleColor
    if ($Detail) { Write-Part ("  " + $Detail) $detailColor }
    if ($Time) {
        try {
            $left = $script:Width - $Time.Length
            if ($left -gt [Console]::CursorLeft) { [Console]::CursorLeft = $left }
        } catch { }
        Write-Part $Time $Muted
    }
}

function Write-Header([string]$Title, [string]$About, [int]$Steps) {
    $count = "$Steps steps"
    if ($Steps -eq 1) { $count = "1 step" }
    if (-not $script:Pretty) {
        Write-Host "Ask Physics: $Title ($count)"
        Write-Host "  $About"
        return
    }
    Write-Host ""
    Write-Part "  " $Muted; Write-Part ($Bullet + " ") $Accent; Write-Part "Ask Physics" $Brand
    Write-Part ("  " + $Dot + "  ") $Muted; Write-Host $Title -ForegroundColor $Value
    Write-Host ("    " + $About + "  " + $Dot + "  " + $count) -ForegroundColor $Muted
    Write-Host ""
}

function Write-Fail([string]$Message) {
    if ($script:Pretty) {
        Write-Host "  " -NoNewline
        Write-Host $Cross -ForegroundColor $Bad -NoNewline
        Write-Host (" " + $Message)
    } else {
        Write-Host "error: $Message"
    }
    exit 1
}

function Write-Info([string]$Message) {
    if ($script:Pretty) {
        Write-Host "  " -NoNewline
        Write-Host ([string][char]0x203A) -ForegroundColor $Accent -NoNewline
        Write-Host (" " + $Message)
    } else {
        Write-Host $Message
    }
}

# Start-Process joins its arguments with spaces and quotes nothing, so quote the ones that need it.
function ConvertTo-ArgumentString([string[]]$Arguments) {
    $quoted = foreach ($a in $Arguments) {
        if ($a -match '[\s"]') { '"' + ($a -replace '"', '\"') + '"' } else { $a }
    }
    return ($quoted -join " ")
}

# Invoke-Step TITLE EXE ARGUMENTS...: one numbered step; the installer stops if it fails.
function Invoke-Step([string]$Title, [string]$Exe, [string[]]$Arguments) {
    $script:Step++
    $counter = "$($script:Step)/$($script:Total)"
    $started = Get-Date
    if (-not $script:Pretty) {
        Write-Host "start: [$counter] $Title"
        & $Exe @Arguments
        $code = $LASTEXITCODE
        $elapsed = Format-Elapsed ((Get-Date) - $started).TotalSeconds
        if ($code -eq 0) {
            Write-Host "done: [$counter] $Title ($elapsed)"
        } else {
            Write-Host "failed: [$counter] $Title (exit $code, $elapsed)"
            Write-Host "  command: $Exe $($Arguments -join ' ')"
            exit 1
        }
        return
    }
    New-Item -ItemType Directory -Force -Path $script:LogDir | Out-Null
    $slug = ($Title.ToLower() -replace "[^a-z0-9]+", "-").Trim("-")
    $log = Join-Path $script:LogDir ("{0:00}-{1}.log" -f $script:Step, $slug)
    $errLog = "$log.err"
    $process = Start-Process -FilePath $Exe -ArgumentList (ConvertTo-ArgumentString $Arguments) -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError $errLog
    $null = $process.Handle  # keep the handle, so the exit code is available afterwards
    $frame = 0
    while (-not $process.HasExited) {
        $elapsed = Format-Elapsed ((Get-Date) - $started).TotalSeconds
        Write-Host "`r" -NoNewline
        Write-Row "run" $Frames[$frame % $Frames.Count] $counter $Title "" $elapsed
        $frame++
        Start-Sleep -Milliseconds 80
    }
    $process.WaitForExit()
    $code = $process.ExitCode
    $elapsed = Format-Elapsed ((Get-Date) - $started).TotalSeconds
    Write-Host ("`r" + (" " * ($script:Width - 1)) + "`r") -NoNewline
    if ($code -eq 0) {
        Write-Row "ok" $Tick $counter $Title "" $elapsed
        Write-Host ""
        return
    }
    Write-Row "bad" $Cross $counter $Title "exit $code" $elapsed
    Write-Host ""
    Write-Host ("      `$ " + $Exe + " " + ($Arguments -join " ")) -ForegroundColor $Muted
    Write-Host ""
    $lines = @(Get-Content -ErrorAction SilentlyContinue $log) + @(Get-Content -ErrorAction SilentlyContinue $errLog)
    $lines | Where-Object { $_ -and $_.Trim() } | Select-Object -Last 12 | ForEach-Object { Write-Host ("      " + $_) }
    Write-Host ""
    Write-Part "      full log" $Muted; Write-Part ("  " + $log) $Accent; Write-Host ""
    exit 1
}

# Like Invoke-Step, for a command that draws its own display (the model download's bars).
function Invoke-Live([string]$Title, [string]$Exe, [string[]]$Arguments) {
    $script:Step++
    $counter = "$($script:Step)/$($script:Total)"
    $started = Get-Date
    if ($script:Pretty) {
        Write-Part "  " $Muted; Write-Part ($Bullet + " ") $Accent; Write-Host $Title -ForegroundColor $Brand
    } else {
        Write-Host "start: [$counter] $Title"
    }
    # Start-Process keeps the command on the real console, so its own bars still draw.
    $process = Start-Process -FilePath $Exe -ArgumentList (ConvertTo-ArgumentString $Arguments) -NoNewWindow -Wait -PassThru
    $code = $process.ExitCode
    $script:LiveCode = $code
    $elapsed = Format-Elapsed ((Get-Date) - $started).TotalSeconds
    if ($script:Pretty) {
        if ($code -eq 0) { Write-Row "ok" $Tick $counter $Title "" $elapsed } else { Write-Row "bad" $Cross $counter $Title "exit $code" $elapsed }
        Write-Host ""
    } elseif ($code -eq 0) {
        Write-Host "done: [$counter] $Title ($elapsed)"
    } else {
        Write-Host "failed: [$counter] $Title (exit $code, $elapsed)"
    }
}

function Write-Result([string]$Title, [string]$Detail) {
    $script:Step++
    $counter = "$($script:Step)/$($script:Total)"
    if ($script:Pretty) {
        Write-Row "ok" $Tick $counter $Title $Detail ""
        Write-Host ""
    } else {
        Write-Host "done: [$counter] $Title - $Detail"
    }
}

function Write-Summary([string]$Title) {
    $elapsed = Format-Elapsed ((Get-Date) - $script:Started).TotalSeconds
    if (-not $script:Pretty) {
        Write-Host "ok: $Title ($($script:Step) of $($script:Total) steps, $elapsed)"
        return
    }
    Write-Part "  " $Muted; Write-Part ($Rule * ($script:Width - 2)) $Muted; Write-Host ""
    Write-Part "  " $Muted; Write-Part ($Tick + " " + $Title) $Ok
    Write-Host ("   all $($script:Total) steps  " + $Dot + "  " + $elapsed) -ForegroundColor $Muted
    Write-Host ""
}

# The panel a setup ends on; each row is a command and what it does.
function Write-Ready([string]$Title, [string[]]$Commands, [string[]]$About) {
    if (-not $script:Pretty) {
        Write-Host "ready: $Title"
        for ($i = 0; $i -lt $Commands.Count; $i++) { Write-Host ("  " + $Commands[$i] + "  - " + $About[$i]) }
        return
    }
    $inner = $script:Width - 4
    $widest = ($Commands | Measure-Object -Property Length -Maximum).Maximum
    Write-Part "  " $Muted
    Write-Part ([string][char]0x256D + $Rule + " " + $Tick + " " + $Title + " ") $Ok
    Write-Part ($Rule * ($inner - $Title.Length - 5)) $Ok
    Write-Part ([string][char]0x256E) $Ok; Write-Host ""
    $edge = [string][char]0x2502
    Write-Part "  " $Muted; Write-Part $edge $Ok; Write-Part (" " * $inner) $Muted; Write-Part $edge $Ok; Write-Host ""
    for ($i = 0; $i -lt $Commands.Count; $i++) {
        $pad = $widest - $Commands[$i].Length + 3
        $fill = $inner - 6 - $widest - $About[$i].Length
        Write-Part "  " $Muted; Write-Part $edge $Ok; Write-Part "   " $Muted
        Write-Part $Commands[$i] $Brand; Write-Part (" " * $pad) $Muted
        Write-Part $About[$i] $Muted; Write-Part (" " * $fill) $Muted
        Write-Part $edge $Ok; Write-Host ""
    }
    Write-Part "  " $Muted; Write-Part $edge $Ok; Write-Part (" " * $inner) $Muted; Write-Part $edge $Ok; Write-Host ""
    Write-Part "  " $Muted; Write-Part ([string][char]0x2570 + ($Rule * $inner) + [string][char]0x256F) $Ok; Write-Host ""
    Write-Host ""
}

function Has($Name) { [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

# ---- The install ---------------------------------------------------------------------------

# What to install, and how many steps that makes.
$Ref = $null
$Spec = $null
if ($env:ASKPHYSICS_SOURCE) {
    $Spec = $env:ASKPHYSICS_SOURCE
    $What = "from $Spec"
} else {
    $Ref = $env:ASKPHYSICS_REF
    if (-not $Ref) {
        try {
            $Ref = (Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest" -TimeoutSec 10).tag_name
        } catch {
            $Ref = $null
        }
        if (-not $Ref) { $Ref = "main" }
    }
    $Spec = "askphysics @ git+https://github.com/$Repo@$Ref"
    $What = $Ref
}
$script:Total = 2
if (-not (Has "uv")) { $script:Total++ }
if ($env:ASKPHYSICS_SKIP_MODELS -ne "1") { $script:Total++ }

$Arch = $env:PROCESSOR_ARCHITECTURE
Write-Header "Install" "Windows ($Arch) $Dot Ask Physics $What, in its own environment" $script:Total

# 1. uv
if (-not (Has "uv")) {
    $shell = (Get-Process -Id $PID).Path
    Invoke-Step "Install uv" $shell @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "irm https://astral.sh/uv/install.ps1 | iex")
    # Make uv visible to the rest of this session.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Has "uv")) { Write-Fail "uv was installed but is not on PATH; open a new terminal and rerun" }
}

# 2. Install (torch from PyPI is already CPU-only on Windows).
$Title = "Install Ask Physics"
if ($Ref) { $Title = "Install Ask Physics $Ref" }
Invoke-Step $Title "uv" @("tool", "install", "--force", "--python", $PythonVersion, $Spec)

# 3. PATH
$BinDir = (uv tool dir --bin).Trim()
$NewPath = $false
if (-not (($env:Path -split ";") -contains $BinDir)) {
    uv tool update-shell | Out-Null
    $env:Path = "$BinDir;$env:Path"
    $NewPath = $true
}

# The installed command starts, and says which version it is.
$Exe = Join-Path $BinDir "askphysics.exe"
$Version = & $Exe version | Select-Object -First 1
if ($LASTEXITCODE -ne 0) { Write-Fail "installed, but askphysics did not start" }
Write-Result "Check that askphysics starts" "$Version"

# 4. Models (tellus and solem, checked against the manifest the package pins)
if ($env:ASKPHYSICS_SKIP_MODELS -eq "1") {
    Write-Info "Skipping the Fermi models (ASKPHYSICS_SKIP_MODELS=1)"
} else {
    Invoke-Live "Download the Fermi models" $Exe @("model", "pull", "--if-published")
    if ($script:LiveCode -ne 0) {
        Write-Info "Couldn't download the models just now. Run later: askphysics model pull"
    }
}

if ($NewPath) {
    Write-Info "$BinDir is now on your PATH. Open a new terminal to pick up the change."
}
Write-Summary "Ask Physics is installed"
Write-Ready "Ask Physics is ready" `
    @('askphysics ask "A 20 m drop: how fast does it land?"', "askphysics --help", "uv tool uninstall askphysics") `
    @("ask a question", "see every command", "remove it again")
