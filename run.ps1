$ErrorActionPreference = "Stop"

# Always run from the repo root
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is required. Install with 'pip install uv' before running."
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual env with uv..."
    & uv venv --python=3.13
}

& uv pip install -r requirements.txt --python ".\.venv\Scripts\python.exe"

$venvActivate = Join-Path ".venv" "Scripts/Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
} else {
    Write-Host "Virtual env activate script not found; rerun setup."
    exit 1
}

python -m cu.agent @args
