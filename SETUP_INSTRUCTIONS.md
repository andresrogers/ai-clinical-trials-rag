# Development Environment Setup

This document shows a modern, repeatable Python development setup for the `ai-clinical-trials-rag` project.

## Quick Setup (5 minutes)

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

(For Bash / macOS / Linux):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Upgrade `pip` and install build tools:

```bash
python -m pip install --upgrade pip
pip install build wheel
```

3. Install the project in editable mode (all extras):

```bash
pip install -e ".[all]"
```

- To install only development tools: `pip install -e "[dev]"`
- To install only runtime deps: `pip install -e .`

4. Register the Jupyter kernel (optional, useful for notebooks):

```bash
python -m ipykernel install --user --name=ai-clinical-trials-rag-env --display-name "Python (ai-clinical-trials-rag)"
```

Verify kernel registration:

```bash
jupyter kernelspec list
```

5. (Optional) Install and enable pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

6. Export installed packages (snapshot):

```bash
pip freeze > reports/installed_packages.txt
```

7. Run tests to validate the environment:

```bash
pytest tests/ -v
```

## Modern Packaging Notes

- This project uses `pyproject.toml` (PEP 517/518).
- Use editable installs with `pip install -e .` or `pip install -e "[all]"` during development.
- Build distributions with `python -m build` when needed.

## Development Workflow (common commands)

- Install dev deps: `pip install -e "[dev]"`
- Run tests: `pytest tests/`
- Format code: `black src/ tests/`
- Lint code: `ruff check src/ tests/`
- Type check: `mypy src/`
- Build distribution: `python -m build`

## Troubleshooting

- If imports fail after changes: editable installs take effect immediately; restart the interpreter/kernel if needed.
- If `ipykernel` registration fails, ensure the venv is activated and `ipykernel` is installed in that environment.
- On PowerShell, you may need to set execution policy to allow scripts: run PowerShell as Admin and use `Set-ExecutionPolicy RemoteSigned`.

## Automation scripts

Two helper scripts are included:

- `scripts/setup_dev.ps1` — PowerShell automation for Windows
- `scripts/setup_dev.sh` — Unix shell automation for macOS/Linux

Run these scripts from the repository root.

## Notes

Keep secrets out of the repository. Use `.env` (based on `.env.example`) or CI secrets for API keys.
