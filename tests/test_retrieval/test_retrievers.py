def test_retrieve_placeholder():
    from src.biotech_rag.retrieval.retrievers import retrieve

    assert retrieve("anything") == []
