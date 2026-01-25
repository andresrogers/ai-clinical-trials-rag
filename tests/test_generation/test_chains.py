def test_build_chain_placeholder():
    from src.biotech_rag.generation.chains import build_chain

    assert build_chain() is None
