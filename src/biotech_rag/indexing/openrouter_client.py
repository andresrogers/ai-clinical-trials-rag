"""Lightweight OpenRouter embedding client helpers.

This module provides small helpers to call the OpenRouter embeddings endpoint
directly using `requests`. It mirrors the "Basic Request" examples from the
OpenRouter docs and wraps network errors into helpful RuntimeErrors.

Usage:
  from biotech_rag.indexing.openrouter_client import test_connection
  res = test_connection(model='qwen/qwen3-embedding-8b:floor')
  print(res)

Configuration:
  - Set `OPENROUTER_API_KEY` in your environment or pass `api_key` to functions.
  - Optionally set `OPENROUTER_BASE_URL` (defaults to https://openrouter.ai/api/v1).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)


def _requests_session_with_retries(
    total_retries: int = 3, backoff: float = 0.3
) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=total_retries, backoff_factor=backoff, status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _default_base_url() -> str:
    # Documentation examples use https://openrouter.ai/api/v1/embeddings
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def embeddings_request(
    texts: list[str],
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Send embedding request to OpenRouter and return parsed JSON.

    Args:
        texts: list of input strings (batch supported) or a single-item list.
        model: model id to request (e.g. 'qwen/qwen3-embedding-8b:floor').
        api_key: optional override of env `OPENROUTER_API_KEY`.
        base_url: optional override of env `OPENROUTER_BASE_URL`.
        provider: provider routing dict (see OpenRouter docs).
        timeout: HTTP request timeout in seconds.

    Returns:
        Parsed JSON response as dict.

    Raises:
        RuntimeError on missing API key or on network/HTTP errors.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; export it or pass `api_key`")

    base = (base_url or _default_base_url()).rstrip("/")
    url = f"{base}/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {"model": model, "input": texts}
    if provider is not None:
        payload["provider"] = provider

    session = _requests_session_with_retries()

    try:
        LOGGER.debug(
            "OpenRouter embeddings request to %s model=%s batch=%d", url, model, len(texts)
        )
        resp = session.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except RequestException as e:
        # Provide an actionable error message (DNS, proxy, timeout, TLS)
        msg = (
            f"OpenRouter request failed: {e}.\n"
            "Check OPENROUTER_API_KEY, OPENROUTER_BASE_URL, network/DNS, and any proxy/firewall settings."
        )
        raise RuntimeError(msg) from e

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"OpenRouter returned non-JSON response: {e}") from e

    return data


def chat_completion_request(
    messages: list[dict[str, str]],
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Send chat completion request to OpenRouter.

    Args:
        messages: List of chat messages ({"role": "user", "content": "..."}).
        model: Model ID (e.g. 'deepseek/deepseek-r1-distill-llama-70b:floor').
        api_key: API key.
        base_url: Base URL.
        temperature: Sampling temperature.
        max_tokens: Max result tokens.
        timeout: Request timeout.

    Returns:
        Parsed JSON response.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY for chat is not set")

    base = (base_url or _default_base_url()).rstrip("/")
    url = f"{base}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/andresrogers/ai-clinical-trials-rag",
        "X-Title": "Biotech Clinical RAG",
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    session = _requests_session_with_retries()

    try:
        resp = session.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except RequestException as e:
        raise RuntimeError(f"OpenRouter chat request failed: {e}")


def test_connection(
    model: str = "qwen/qwen3-embedding-8b:floor", sample_text: str = "hello world"
) -> dict[str, Any]:
    """Test connectivity and basic embedding result.

    Returns a dict with keys: success (bool), dim (Optional[int]), error (Optional[str]).
    """
    try:
        data = embeddings_request([sample_text], model=model)
    except Exception as e:
        return {"success": False, "dim": None, "error": str(e)}

    # Attempt to read embedding from common response shapes
    try:
        if "data" in data and isinstance(data["data"], list) and data["data"]:
            emb = data["data"][0].get("embedding") if isinstance(data["data"][0], dict) else None
        elif "embeddings" in data:
            emb = data["embeddings"][0]
        else:
            # search nested
            emb = None
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict) and "embedding" in v[0]:
                    emb = v[0]["embedding"]
                    break

        if emb is None:
            return {"success": False, "dim": None, "error": "No embedding found in response"}

        return {"success": True, "dim": len(emb), "error": None}
    except Exception as e:
        return {"success": False, "dim": None, "error": f"Failed to parse embedding: {e}"}
