"""Simple LLM response cache placeholder."""

_cache = {}


def get(key: str):
    return _cache.get(key)


def set(key: str, value):
    _cache[key] = value
