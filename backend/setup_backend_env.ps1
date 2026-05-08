$ErrorActionPreference = "Stop"

param(
    [switch]$UseMinimalRequirements,
    [switch]$ForceRecreate
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv310"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$hfHome = Join-Path $repoRoot ".hf_cache"
$transformersCache = Join-Path $hfHome "transformers"
$localAppData = Join-Path $repoRoot ".localappdata"
$minimalRequirements = Join-Path $PSScriptRoot "requirements.txt"
$lockedRequirements = Join-Path $PSScriptRoot "requirements-lock.txt"
$requirementsFile = if ($UseMinimalRequirements) {
    $minimalRequirements
} else {
    $lockedRequirements
}

function New-BackendVenv {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & $pyLauncher.Source -3.10 -m venv $venvDir
        return
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python launcher not found. Install Python 3.10 and rerun this script."
    }

    $pythonVersion = & $pythonCommand.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($pythonVersion -ne "3.10") {
        throw "Expected Python 3.10, found Python $pythonVersion. Install Python 3.10 or use the 'py' launcher."
    }

    & $pythonCommand.Source -m venv $venvDir
}

if ($ForceRecreate -and (Test-Path $venvDir)) {
    Remove-Item -LiteralPath $venvDir -Recurse -Force
}

if (-not (Test-Path $pythonExe)) {
    New-BackendVenv
}

if (-not (Test-Path $pythonExe)) {
    throw "Python virtualenv could not be created at $pythonExe"
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r $requirementsFile

New-Item -ItemType Directory -Force $hfHome | Out-Null
New-Item -ItemType Directory -Force $transformersCache | Out-Null
New-Item -ItemType Directory -Force $localAppData | Out-Null

Write-Host "Backend environment is ready."
Write-Host "Virtualenv: $venvDir"
Write-Host "Requirements file: $requirementsFile"
Write-Host "Next steps:"
Write-Host "  1. .\\backend\\run_ingest.ps1"
Write-Host "  2. .\\backend\\run_api.ps1"

