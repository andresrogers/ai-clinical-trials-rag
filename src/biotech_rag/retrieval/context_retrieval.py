"""Helpers for context retrieval from Chroma with consistent embeddings.

This module provides a small probe to discover the embedding dimension used by a
Chroma collection. When the collection does not expose embedding vectors (or the
probe fails), we fall back to the configured default from `settings.embedding_dim`.
"""

from __future__ import annotations

import logging
from typing import Any

from biotech_rag.config import settings

logger = logging.getLogger(__name__)


def get_collection_embedding_dim(collection: Any) -> int | None:
    """Return embedding dimension stored in a Chroma collection if available.

    The function attempts a lightweight probe using ``collection.get(..., include=['embeddings'])``.
    If the collection does not expose embeddings or the probe fails, this will
    return ``settings.embedding_dim`` as a conservative default so downstream
    code can still validate dimensions.

    Args:
        collection: Chroma collection object (client-side).

    Returns:
        The embedding vector dimensionality (int) or ``settings.embedding_dim``
        when the collection does not provide an explicit value.
    """
    try:
        sample = collection.get(limit=1, include=["embeddings"])
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        logger.debug("Chroma probe for embeddings failed: %s", exc)
        return settings.embedding_dim

    emb = sample.get("embeddings")

    # Avoid truth-value checks on array-like objects (NumPy arrays raise)
    if emb is None:
        logger.info(
            "No embeddings returned from Chroma probe; falling back to settings.embedding_dim=%d",
            settings.embedding_dim,
        )
        return settings.embedding_dim

    try:
        emb_len = len(emb)
    except Exception:
        emb_len = None

    if emb_len == 0:
        logger.info(
            "Empty embeddings returned from Chroma probe; falling back to settings.embedding_dim=%d",
            settings.embedding_dim,
        )
        return settings.embedding_dim

    # Determine a usable first vector robustly (handle lists, nested lists, numpy arrays)
    first_vec = None
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is not None and isinstance(emb, np.ndarray):
        # If emb is 2D (n_vectors x dim) take the first row, else use it directly
        if getattr(emb, "ndim", 1) >= 2 and emb.shape[0] > 0:
            first_vec = emb[0]
        else:
            first_vec = emb
    elif isinstance(emb, list):
        for item in emb:
            if item is None:
                continue
            try:
                if np is not None and isinstance(item, np.ndarray):
                    if item.size > 0:
                        first_vec = item
                        break
                elif hasattr(item, "__len__") and len(item) > 0:
                    first_vec = item
                    break
            except Exception:
                first_vec = item
                break
    else:
        first_vec = emb

    if first_vec is None:
        logger.warning(
            "Unable to locate a non-empty embedding vector in probe; using settings.embedding_dim=%d",
            settings.embedding_dim,
        )
        return settings.embedding_dim

    try:
        # Prefer numpy shapes when available
        if np is not None and isinstance(first_vec, np.ndarray):
            if getattr(first_vec, "ndim", 1) >= 1:
                return int(first_vec.shape[0])
        return int(len(first_vec))
    except Exception:  # pragma: no cover - defensive
        logger.warning(
            "Failed to determine embedding length from probe result; using settings.embedding_dim=%d",
            settings.embedding_dim,
        )
        return settings.embedding_dim


def retrieve_chunks(
    collection: Any,
    embedder: Any,
    question: str,
    nct_id: str | None = None,
    top_k: int = 5,
    collection_dim: int | None = None,
) -> dict[str, Any]:
    """Retrieve top-k raw chunks for a question, optionally filtered by NCT ID."""
    where_filter = {"nct_id": nct_id} if isinstance(nct_id, str) and nct_id.strip() else None

    query_embedding = embedder.embed([question])[0]
    if collection_dim and len(query_embedding) != collection_dim:
        raise ValueError(
            f"Embedding dimension mismatch: collection={collection_dim}, query={len(query_embedding)}. "
            "Use the same embedding model for indexing and retrieval."
        )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    if where_filter and (not results.get("documents") or not results["documents"][0]):
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    return results


def format_retrieved_chunks(chunks: list[str]) -> str:
    """Format retrieved chunks with stable chunk labels."""
    return "\n".join([f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(chunks)])
