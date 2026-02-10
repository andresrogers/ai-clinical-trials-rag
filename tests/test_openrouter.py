"""Pytest integration test for OpenRouter connectivity.

This test is an integration test and will be skipped unless `OPENROUTER_API_KEY`
is set in the environment. It performs a single embedding request and verifies
the response contains an embedding vector.
"""

import os

import pytest

from biotech_rag.indexing.openrouter_client import test_connection


@pytest.mark.skipif(os.getenv("OPENROUTER_API_KEY") is None, reason="OPENROUTER_API_KEY not set")
def test_openrouter_basic_embedding():
    res = test_connection()
    assert res.get("success") is True, f"OpenRouter test failed: {res.get('error')}"
    assert isinstance(res.get("dim"), int) and res.get("dim") > 0
