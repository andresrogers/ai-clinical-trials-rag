"""Persistent LLM response cache using diskcache."""

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache

# Use the data/cache directory defined in config if available, else local
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_cache = diskcache.Cache(str(CACHE_DIR / "llm_responses"))


def _make_key(prompt: Any) -> str:
    """Create a hash key for any JSON-serializable prompt content."""
    if not isinstance(prompt, str):
        prompt = json.dumps(prompt, sort_keys=True)
    return hashlib.md5(prompt.encode()).hexdigest()


def get_cached_response(prompt: Any, model: str) -> str | None:
    """Retrieve response from disk cache."""
    key = f"{model}:{_make_key(prompt)}"
    return _cache.get(key)


def set_cached_response(prompt: Any, model: str, response: str):
    """Store response in disk cache."""
    key = f"{model}:{_make_key(prompt)}"
    _cache.set(key, response)


def clear_cache():
    """Clear all cached responses."""
    _cache.clear()
