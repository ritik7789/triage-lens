$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv310\Scripts\python.exe"
$hfHome = Join-Path $repoRoot ".hf_cache"
$transformersCache = Join-Path $hfHome "transformers"
$localAppData = Join-Path $repoRoot ".localappdata"

if (-not (Test-Path $pythonExe)) {
    throw "Python virtualenv not found at $pythonExe. Run .\backend\setup_backend_env.ps1 first."
}

New-Item -ItemType Directory -Force $hfHome | Out-Null
New-Item -ItemType Directory -Force $transformersCache | Out-Null
New-Item -ItemType Directory -Force $localAppData | Out-Null

$env:LOCALAPPDATA = $localAppData
$env:HF_HOME = $hfHome
$env:TRANSFORMERS_CACHE = $transformersCache
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:GIT_HTTP_PROXY = ""
$env:GIT_HTTPS_PROXY = ""

& $pythonExe (Join-Path $PSScriptRoot "main.py")
