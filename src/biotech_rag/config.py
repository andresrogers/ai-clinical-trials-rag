"""Centralized configuration for the biotech RAG pipeline.

This module defines project paths and application settings loaded from
environment variables via `pydantic-settings`.
"""

from pathlib import Path
from typing import Final
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

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
	TRIAL_PDFS_DIR, CHUNKS_DIR, METADATA_DIR, EMBEDDINGS_DIR,
	VECTORSTORE_DIR, PROMPTS_DIR, REPORTS_DIR, FIGURES_DIR, CACHE_DIR
]:
	dir_path.mkdir(parents=True, exist_ok=True)


# API Configuration
class Settings(BaseSettings):
	"""Application settings loaded from environment variables."""
	# OpenAI
	openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
	openai_org_id: str | None = os.getenv("OPENAI_ORG_ID")

	# Vector Database
	chroma_persist_dir: Path = VECTORSTORE_DIR / "chroma"
	pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY")
	pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
	pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "biotech-trials")

	# Google Drive
	google_drive_folder_id: str | None = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
	google_drive_credentials_path: Path | None = None

	# Model Configuration
	embedding_model: str = "text-embedding-3-small"
	embedding_dim: int = 1536
	llm_model: str = "gpt-4o-mini"
	max_tokens: int = 4000
	temperature: float = 0.0

	# Retrieval Configuration
	chunk_size: int = 512
	chunk_overlap: int = 128
	top_k_retrieval: int = 20
	top_k_rerank: int = 5
	similarity_threshold: float = 0.7

	# Logging
	log_level: str = os.getenv("LOG_LEVEL", "INFO")

	# Caching
	cache_dir: Path = CACHE_DIR
	enable_cache: bool = True

	# MLflow (optional)
	mlflow_tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI")
	mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "biotech-trial-rag")

	class Config:
		env_file = ".env"
		env_file_encoding = "utf-8"


settings = Settings()

# Model parameters
RANDOM_SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
CV_FOLDS: Final[int] = 5
