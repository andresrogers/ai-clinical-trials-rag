def test_chunk_text():
    from src.biotech_rag.indexing.chunkers import chunk_text

    text = "a" * 2500
    chunks = list(chunk_text(text, size=1000))
    assert len(chunks) == 3
