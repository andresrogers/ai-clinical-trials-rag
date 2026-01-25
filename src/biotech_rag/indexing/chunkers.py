"""Chunking strategies (semantic, fixed, hybrid) - placeholders."""


def chunk_text(text: str, size: int = 1000):
    """Yield chunks of approximately `size` characters."""
    for i in range(0, len(text), size):
        yield text[i : i + size]
