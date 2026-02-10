"""
Data loading functions for AI Drug Discovery Clinical Trials dataset.

This module provides functions to load various dataset files with proper
error handling, validation, and logging.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from ..config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_master_dataset(file_path: Path | None = None) -> pd.DataFrame:
    """
    Load the master AI trials dataset with all data sources.
    
    Parameters
    ----------
    file_path : Path, optional
        Custom file path. If None, uses default location.
    
    Returns
    -------
    pd.DataFrame
        Master dataset with all trials and features.
    
    Raises
    ------
    FileNotFoundError
        If dataset file is not found.
    ValueError
        If dataset is empty or malformed.
    """
    if file_path is None:
        file_path = PROCESSED_DATA_DIR / "master_ai_trials_dataset.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {file_path}. Run create_master_dataset.py first."
        )

    try:
        df = pd.read_csv(file_path, low_memory=False)

        if df.empty:
            raise ValueError("Dataset is empty")

        logger.info(f"✓ Loaded master dataset: {len(df)} trials, {df.shape} columns")
        return df

    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file is empty: {file_path}")
    except Exception as e:
        logger.error(f"Error loading master dataset: {e}")
        raise


def load_rag_ready_subset(file_path: Path | None = None) -> pd.DataFrame:
    """
    Load RAG-ready trials subset (trials with PDF or XML data).
    """
    if file_path is None:
        file_path = PROCESSED_DATA_DIR / "rag_ready_trials.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"RAG-ready subset not found at {file_path}. Run create_master_dataset.py first."
        )

    try:
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"✓ Loaded RAG-ready subset: {len(df)} trials")
        return df

    except Exception as e:
        logger.error(f"Error loading RAG subset: {e}")
        raise


def load_dataset_summary(file_path: Path | None = None) -> dict:
    """
    Load dataset summary statistics and metadata.
    """
    if file_path is None:
        file_path = PROCESSED_DATA_DIR / "dataset_summary.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset summary not found at {file_path}")

    try:
        with open(file_path, encoding='utf-8') as f:
            summary = json.load(f)

        logger.info("✓ Loaded dataset summary")
        return summary

    except Exception as e:
        logger.error(f"Error loading dataset summary: {e}")
        raise


def load_pmc_publications(file_path: Path | None = None) -> pd.DataFrame:
    """
    Load PMC publications metadata.
    """
    if file_path is None:
        file_path = INTERIM_DATA_DIR / "pmc_publications.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"PMC publications not found at {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"✓ Loaded PMC publications: {len(df)} records")
        return df

    except Exception as e:
        logger.error(f"Error loading PMC publications: {e}")
        raise


def validate_data_paths() -> dict[str, bool]:
    """
    Validate that all expected data files exist.
    """
    expected_files = {
        "Master dataset": PROCESSED_DATA_DIR / "master_ai_trials_dataset.csv",
        "RAG-ready subset": PROCESSED_DATA_DIR / "rag_ready_trials.csv",
        "Dataset summary": PROCESSED_DATA_DIR / "dataset_summary.json",
        "PMC publications": INTERIM_DATA_DIR / "pmc_publications.csv",
        "Ground truth template": PROCESSED_DATA_DIR / "ground_truth_template.csv",
        "PDF directory": RAW_DATA_DIR / "pdf_studies",
        "XML directory": RAW_DATA_DIR / "pubmed_semantic_data" / "xml_files",
        "FDA labels directory": RAW_DATA_DIR / "fda_labels"
    }

    results = {}
    for name, path in expected_files.items():
        results[name] = path.exists()

    return results


def load_ground_truth_template(file_path: Path | None = None) -> pd.DataFrame:
    """
    Load ground truth Q&A annotation template.
    """
    if file_path is None:
        file_path = PROCESSED_DATA_DIR / "ground_truth_template.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth template not found at {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"✓ Loaded ground truth template: {len(df)} question slots")
        return df

    except Exception as e:
        logger.error(f"Error loading ground truth template: {e}")
        raise
