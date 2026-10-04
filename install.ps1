# Ask Physics installer for Windows (PowerShell 5.1 or 7+).
#
#   irm https://askphysics.vercel.app/installers/install.ps1 | iex
#
# Installs the `askphysics` command in its own isolated environment with uv
# (https://docs.astral.sh/uv/), installing uv first if it's missing. Nothing
# touches your system Python.
#
# Environment variables:
#   ASKPHYSICS_REF     git tag, branch, or commit to install (default: latest release)
#   ASKPHYSICS_SOURCE  install from this local path or URL instead of GitHub
#
# Uninstall:  uv tool uninstall askphysics

$ErrorActionPreference = "Stop"
$Repo = "shankar-sachin/ask-physics"
$PythonVersion = "3.12"

function Info($Message) { Write-Host "==> " -ForegroundColor Cyan -NoNewline; Write-Host $Message }
function Fail($Message) { Write-Host "error: " -ForegroundColor Red -NoNewline; Write-Host $Message; exit 1 }
function Has($Name) { [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

Write-Host ""
Write-Host "  (o) Ask Physics installer" -ForegroundColor Cyan
Write-Host "      grounded answers, real math" -ForegroundColor DarkGray
Write-Host ""

# 1. uv
if (-not (Has "uv")) {
    Info "Installing uv (Python package manager)"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # Make uv visible to the rest of this session.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Has "uv")) { Fail "uv was installed but is not on PATH; open a new terminal and rerun" }
}
Info "Using $(uv --version)"

# 2. What to install
if ($env:ASKPHYSICS_SOURCE) {
    $Spec = $env:ASKPHYSICS_SOURCE
    Info "Installing from $Spec"
} else {
    $Ref = $env:ASKPHYSICS_REF
    if (-not $Ref) {
        try {
            $Ref = (Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest").tag_name
        } catch {
            $Ref = $null
        }
        if (-not $Ref) { $Ref = "main" }
    }
    $Spec = "askphysics @ git+https://github.com/$Repo@$Ref"
    Info "Installing Ask Physics $Ref"
}

# 3. Install (torch from PyPI is already CPU-only on Windows).
uv tool install --force --python $PythonVersion $Spec
if ($LASTEXITCODE -ne 0) { Fail "uv tool install failed" }

# 4. PATH
$BinDir = (uv tool dir --bin).Trim()
if (-not (($env:Path -split ";") -contains $BinDir)) {
    Info "Adding $BinDir to your PATH (uv tool update-shell)"
    uv tool update-shell | Out-Null
    $env:Path = "$BinDir;$env:Path"
    Write-Host "    Open a new terminal to pick up the PATH change." -ForegroundColor DarkGray
}

Write-Host ""
& (Join-Path $BinDir "askphysics.exe") version
if ($LASTEXITCODE -ne 0) { Fail "installed, but askphysics did not start" }
Write-Host ""
Write-Host "  Try it:"
Write-Host '    askphysics ask "How fast does a ball dropped from 20 m hit the ground?"'
Write-Host ""
