"""Embedding generation wrappers (placeholders)."""


def get_embeddings(texts):
    """Return placeholder embeddings (list of lists)."""
    return [[0.0] * 3 for _ in texts]
