"""Centralized configuration for the project.

Keep lightweight defaults here; sensitive keys belong in `.env` and are not committed.
"""

from typing import Final

PROJECT_NAME: Final[str] = "biotech-trial-forecasting"
DEFAULT_EMBEDDING_NAME: Final[str] = "text-embedding-3-small"
