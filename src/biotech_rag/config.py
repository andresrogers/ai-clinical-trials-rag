"""Centralized configuration for the biotech RAG pipeline.

This module defines project paths and application settings loaded from
environment variables via `pydantic-settings`.
"""

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

try:
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - pydantic v1 compatibility fallback
    ConfigDict = None

load_dotenv()

# Project paths
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
TRIAL_PDFS_DIR: Final[Path] = RAW_DATA_DIR / "trial_pdfs"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
CHUNKS_DIR: Final[Path] = INTERIM_DATA_DIR / "chunks"
METADATA_DIR: Final[Path] = INTERIM_DATA_DIR / "metadata"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
EMBEDDINGS_DIR: Final[Path] = PROCESSED_DATA_DIR / "embeddings"
VECTORSTORE_DIR: Final[Path] = PROCESSED_DATA_DIR / "vectorstore"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
PROMPTS_DIR: Final[Path] = MODELS_DIR / "prompts"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"

# Create directories if they don't exist
for dir_path in [
    TRIAL_PDFS_DIR,
    CHUNKS_DIR,
    METADATA_DIR,
    EMBEDDINGS_DIR,
    VECTORSTORE_DIR,
    PROMPTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    CACHE_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)


# API Configuration
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Allow extra env vars to be present without failing validation (helps local dev)
    if ConfigDict is not None:
        model_config = ConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")
    else:
        model_config = {"extra": "ignore", "env_file": ".env", "env_file_encoding": "utf-8"}

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_org_id: str | None = os.getenv("OPENAI_ORG_ID")

    # Vector Database
    chroma_persist_dir: Path = VECTORSTORE_DIR / "chroma"
    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "ai-clinical-trials-rag")

    # Google Drive
    google_drive_folder_id: str | None = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    google_drive_credentials_path: Path | None = None

    # Model Configuration
    # Embeddings (centralized defaults)
    embedding_model: str = "qwen/qwen3-embedding-8b:floor"
    embedding_dim: int = 4096
    embedder_backend: str = os.getenv("EMBEDDER_BACKEND", "openrouter")
    embedder_model: str = os.getenv("EMBEDDER_MODEL", "qwen/qwen3-embedding-8b:floor")
    # LLM (centralized defaults)
    llm_model: str = os.getenv("LLM_MODEL", "deepseek/deepseek-r1-distill-llama-70b:floor")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    # Prefer using OpenRouter-based remote verifiers for NLI/QA (avoid local HF downloads)
    use_openrouter_verifier: bool = bool(os.getenv("USE_OPENROUTER_VERIFIER", "True") in ("True", "true", "1"))
    # Backwards-compat aliases
    max_tokens: int = llm_max_tokens
    temperature: float = llm_temperature

    # Retrieval Configuration (aligned for scientific chunking)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "750"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Caching
    cache_dir: Path = CACHE_DIR
    enable_cache: bool = True

    # Cost Optimization & Limits
    max_claims_per_answer: int = int(os.getenv("MAX_CLAIMS_PER_ANSWER", "5"))
    max_ncts_per_run: int = int(os.getenv("MAX_NCTS_PER_RUN", "20"))
    llm_batch_size: int = int(os.getenv("LLM_BATCH_SIZE", "5"))

    # model_config set above configures env loading and extra handling


settings = Settings()

# Model parameters
RANDOM_SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
CV_FOLDS: Final[int] = 5
