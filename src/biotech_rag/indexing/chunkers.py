"""Chunking utilities and a section-aware chunker."""

import logging

try:
    from llama_index.core.node_parser import SentenceSplitter

    _have_llama_index = True
except Exception:
    SentenceSplitter = None  # type: ignore
    _have_llama_index = False


class SectionAwareChunker:
    """Simple section-aware chunker.

    Splits section text into chunks of approx `min_words` words with `overlap_words` overlap.
    """

    def __init__(self, min_tokens: int = 512, overlap_tokens: int = 50):
        # Approximate words per token (heuristic)
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.words_per_token = 0.9

    def chunk_text(self, text: str, section_title: str = None, metadata: dict = None) -> list[dict]:
        """Return list of chunks: {'text': ..., 'metadata': {...}}"""
        if not text:
            return []

        words = text.split()
        min_words = max(64, int(self.min_tokens * self.words_per_token))
        overlap = max(8, int(self.overlap_tokens * self.words_per_token))

        chunks = []
        i = 0
        chunk_id = 0
        while i < len(words):
            j = min(i + min_words, len(words))
            chunk_words = words[i:j]
            chunk_text = " ".join(chunk_words).strip()
            meta = dict(metadata or {})
            if section_title:
                meta.setdefault("section_title", section_title)
            meta.setdefault("chunk_id", chunk_id)
            chunks.append({"text": chunk_text, "metadata": meta})
            chunk_id += 1
            if j >= len(words):
                break
            i = j - overlap

        return chunks


class HybridScientificChunker:
    """Semantic-aware chunker with a SectionAware fallback.

    Uses LlamaIndex SentenceSplitter when available, else falls back to
    SectionAwareChunker with smaller chunks and higher overlap.
    """

    def __init__(
        self,
        chunk_size: int = 750,
        chunk_overlap: int = 150,
        paragraph_separator: str = "\n\n",
        secondary_chunking_regex: str = r"\n(?=[A-Z][a-z]+:|\d+\.\d+|\*\*)",
        fallback_min_tokens: int = 600,
        fallback_overlap_tokens: int = 150,
    ):
        self.semantic_splitter = None
        if _have_llama_index and SentenceSplitter is not None:
            self.semantic_splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                paragraph_separator=paragraph_separator,
                secondary_chunking_regex=secondary_chunking_regex,
            )
            logging.info("Using LlamaIndex SentenceSplitter for semantic chunking.")
        else:
            logging.warning("LlamaIndex not installed; falling back to SectionAwareChunker.")

        self.fallback_chunker = SectionAwareChunker(
            min_tokens=fallback_min_tokens,
            overlap_tokens=fallback_overlap_tokens,
        )

    def chunk_text(
        self,
        text: str,
        section_title: str | None = None,
        metadata: dict | None = None,
    ) -> list[dict]:
        if self.semantic_splitter is not None:
            chunks = self.semantic_splitter.split_text(text)
            base_meta = dict(metadata or {})
            if section_title and "section_title" not in base_meta:
                base_meta["section_title"] = section_title
            return [
                {
                    "text": chunk,
                    "metadata": {**base_meta, "chunk_id": i},
                }
                for i, chunk in enumerate(chunks)
            ]
        return self.fallback_chunker.chunk_text(
            text, section_title=section_title, metadata=metadata
        )


"""Chunking strategies (semantic, fixed, hybrid) - placeholders."""


def chunk_text(text: str, size: int = 1000):
    """Yield chunks of approximately `size` characters."""
    for i in range(0, len(text), size):
        yield text[i : i + size]
