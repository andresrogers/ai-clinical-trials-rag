#!/usr/bin/env bash
set -euo pipefail

# Unix / macOS bootstrap script
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install build wheel

# Install editable with all extras
pip install -e ".[all]"

# Register kernel
python -m ipykernel install --user --name=ai-clinical-trials-rag-env --display-name "Python (ai-clinical-trials-rag)"

# Pre-commit
pip install pre-commit || true
pre-commit install || true
pre-commit run --all-files || true

# Ensure reports dir exists and export packages
mkdir -p reports
pip freeze > reports/installed_packages.txt

echo "Bootstrap complete. Activate the venv in new shells with: source .venv/bin/activate"
