def test_metrics_placeholders():
    from src.biotech_rag.evaluation.retrieval_metrics import precision_at_k

    assert precision_at_k([], 5) == 0.0
