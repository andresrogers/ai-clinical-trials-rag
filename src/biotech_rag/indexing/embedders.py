"""Embedding wrapper supporting multiple backends.

Supported backends: 'openai', 'openrouter', 'nomic', 'local'

Configuration via environment variables:
  - OPENAI_API_KEY (when using OpenAI SDK)
  - OPENROUTER_API_KEY and OPENROUTER_BASE_URL
  - NOMIC_API_KEY
  - EMBEDDER_BACKEND (one of openai|openrouter|nomic|local)

Fallback: sentence-transformers (`all-MiniLM-L6-v2`) when `local` or when remote backends unavailable.
"""

import logging
import os
import time

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

try:
    from openai import OpenAI

    _have_openai = True
except Exception:
    _have_openai = False

try:
    from sentence_transformers import SentenceTransformer

    _have_st = True
except Exception:
    _have_st = False

try:
    import torch

    _have_torch = True
except Exception:
    _have_torch = False


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


class Embedder:
    """Embedder supporting multiple backends.

    Args:
        backend: 'openai' | 'openrouter' | 'nomic' | 'local' (or set EMBEDDER_BACKEND env var)
        model: model name to request from the backend
        batch_size: number of texts per API request
    """

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        batch_size: int = 64,
    ):
        # Centralized defaults from settings (with env overrides)
        try:
            from biotech_rag.config import settings

            default_backend = getattr(settings, "embedder_backend", None) or os.getenv(
                "EMBEDDER_BACKEND"
            )
            default_model = getattr(settings, "embedder_model", None) or os.getenv("EMBEDDER_MODEL")
        except Exception:
            default_backend = os.getenv("EMBEDDER_BACKEND")
            default_model = os.getenv("EMBEDDER_MODEL")

        self.backend = (backend or default_backend or "openrouter").lower()
        self.model = model or default_model or "qwen/qwen3-embedding-8b:floor"
        self.batch_size = batch_size
        self.session = _requests_session_with_retries()

        # OpenAI client if available
        if self.backend == "openai":
            if _have_openai:
                self.client = OpenAI()
            else:
                raise RuntimeError("OpenAI backend selected but openai package is not installed")

        # Determine device for sentence-transformers if available
        self.device = "cpu"
        if _have_torch and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            self.device = "cuda"

        # Local fallback model
        if self.backend == "local" or not any([_have_openai, _have_st]):
            if _have_st:
                try:
                    self._st_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                except TypeError:
                    # Older SentenceTransformer versions may not accept device at init
                    self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                raise RuntimeError("Local backend requires sentence-transformers")

        # API keys
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        # Align default with OpenRouter docs: https://openrouter.ai/api/v1
        self.openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.nomic_key = os.getenv("NOMIC_API_KEY")
        # ensure attribute exists for lazy local fallback
        self._st_model = getattr(self, "_st_model", None)

        # If OpenRouter is selected but key is missing, fall back to local if available
        if self.backend == "openrouter" and not self.openrouter_key:
            if _have_st:
                logging.warning("OPENROUTER_API_KEY not set; falling back to local embeddings.")
                self.backend = "local"
                self._ensure_local_model()
            else:
                raise RuntimeError("OPENROUTER_API_KEY is not set")

    def _ensure_local_model(self):
        """Ensure a local SentenceTransformer model is loaded for fallback."""
        if getattr(self, "_st_model", None) is not None:
            return
        if not _have_st:
            raise RuntimeError("sentence-transformers is not installed for local fallback")
        try:
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        except TypeError:
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _batch(self, items: list[str]):
        for i in range(0, len(items), self.batch_size):
            yield items[i : i + self.batch_size]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts.

        Batches requests according to `batch_size`. Implements retries via requests session.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []

        if self.backend == "openai":
            try:
                for batch in self._batch(texts):
                    resp = self.client.embeddings.create(model=self.model, input=batch)
                    embeddings.extend([r.embedding for r in resp.data])
                return embeddings
            except Exception:
                # fallback to local if available
                if _have_st:
                    self._ensure_local_model()
                    try:
                        return self._st_model.encode(
                            texts, show_progress_bar=False, device=self.device
                        ).tolist()
                    except TypeError:
                        return self._st_model.encode(texts, show_progress_bar=False).tolist()
                raise

        if self.backend == "openrouter":
            if not self.openrouter_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            url = self.openrouter_base.rstrip("/") + "/embeddings"
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
            }
            try:
                for batch in self._batch(texts):
                    payload = {"model": self.model, "input": batch}
                    r = self.session.post(url, json=payload, headers=headers, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    # support OpenAI-style 'data' array
                    if "data" in data and isinstance(data["data"], list):
                        for item in data["data"]:
                            embeddings.append(item.get("embedding"))
                    # support other vendors
                    elif "embeddings" in data:
                        embeddings.extend(data["embeddings"])
                    else:
                        # try nested results
                        for v in data.values():
                            if (
                                isinstance(v, list)
                                and v
                                and isinstance(v[0], dict)
                                and "embedding" in v[0]
                            ):
                                embeddings.extend([it["embedding"] for it in v])
                    time.sleep(0.01)
                return embeddings
            except RequestException as e:
                # Network/DNS/proxy issues -> fallback to local model if available
                if _have_st:
                    self._ensure_local_model()
                    try:
                        return self._st_model.encode(
                            texts, show_progress_bar=False, device=self.device
                        ).tolist()
                    except TypeError:
                        return self._st_model.encode(texts, show_progress_bar=False).tolist()
                raise RuntimeError(
                    f"OpenRouter request failed and no local fallback available: {e}"
                )

        if self.backend == "nomic":
            if not self.nomic_key:
                raise RuntimeError("NOMIC_API_KEY is not set")
            url = "https://api.nomic.ai/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {self.nomic_key}",
                "Content-Type": "application/json",
            }
            try:
                for batch in self._batch(texts):
                    payload = {"model": self.model, "input": batch}
                    r = self.session.post(url, json=payload, headers=headers, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    # nomic response may include 'results' or 'data'
                    if "results" in data:
                        embeddings.extend([it.get("embedding") for it in data["results"]])
                    elif "data" in data:
                        embeddings.extend([it.get("embedding") for it in data["data"]])
                    else:
                        # attempt generic extraction
                        for v in data.values():
                            if (
                                isinstance(v, list)
                                and v
                                and isinstance(v[0], dict)
                                and "embedding" in v[0]
                            ):
                                embeddings.extend([it["embedding"] for it in v])
                    time.sleep(0.01)
                return embeddings
            except RequestException as e:
                if _have_st:
                    self._ensure_local_model()
                    try:
                        return self._st_model.encode(
                            texts, show_progress_bar=False, device=self.device
                        ).tolist()
                    except TypeError:
                        return self._st_model.encode(texts, show_progress_bar=False).tolist()
                raise RuntimeError(f"Nomic request failed and no local fallback available: {e}")

        # local (sentence-transformers)
        if self.backend == "local":
            try:
                return self._st_model.encode(
                    texts, show_progress_bar=False, device=self.device
                ).tolist()
            except TypeError:
                return self._st_model.encode(texts, show_progress_bar=False).tolist()

        # If backend not recognized, try fallbacks
        if _have_st:
            try:
                tmp = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                return tmp.encode(texts, show_progress_bar=False).tolist()
            except TypeError:
                return (
                    SentenceTransformer("all-MiniLM-L6-v2")
                    .encode(texts, show_progress_bar=False)
                    .tolist()
                )

        raise RuntimeError("No embedding backend available or backend not recognized")
