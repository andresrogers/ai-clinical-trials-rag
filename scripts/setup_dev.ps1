<#
PowerShell helper to bootstrap development environment on Windows.
Run from repo root in an elevated PowerShell if required by execution policy.
#>

$ErrorActionPreference = 'Stop'

# Create venv
python -m venv .venv

Write-Host "Created virtual environment .venv"

# Activate venv for this script (affects this session)
$activate = Join-Path -Path $PWD -ChildPath '.venv\Scripts\Activate.ps1'
if (Test-Path $activate) {
    Write-Host "Activating virtual environment..."
    & $activate
} else {
    Write-Error "Activation script not found: $activate"
}

# Upgrade pip and install build tools
python -m pip install --upgrade pip
pip install build wheel

# Install project and all optional dependencies in editable mode
pip install -e ".[all]"

# Register ipykernel
python -m ipykernel install --user --name=ai-clinical-trials-rag-env --display-name "Python (ai-clinical-trials-rag)"

# Install pre-commit and hooks
pip install pre-commit
pre-commit install
try {
    pre-commit run --all-files
} catch {
    Write-Warning "pre-commit reported issues. Inspect output above."
}

# Save installed packages snapshot
if (!(Test-Path -Path "reports")) { New-Item -ItemType Directory -Path "reports" | Out-Null }
pip freeze > reports\installed_packages.txt

Write-Host "Development environment bootstrapped. Activate the venv in new shells with: .\\.venv\\Scripts\\Activate.ps1"
