def test_get_embeddings():
    from src.biotech_rag.indexing.embedders import get_embeddings

    embs = get_embeddings(["one", "two"])
    assert isinstance(embs, list)
    assert len(embs) == 2
