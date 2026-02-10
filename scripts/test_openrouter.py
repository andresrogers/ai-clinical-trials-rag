"""Small CLI to test OpenRouter connectivity and embedding shape.

Usage:
  python scripts/test_openrouter.py --model qwen/qwen3-embedding-8b:floor

The script prints a short result and exits with code 0 on success.
"""
from __future__ import annotations

import argparse
import json
import sys

from biotech_rag.indexing.openrouter_client import test_connection


def main() -> int:
    p = argparse.ArgumentParser(description="Test OpenRouter embeddings connectivity")
    p.add_argument("--model", default="qwen/qwen3-embedding-8b:floor", help="OpenRouter model id")
    p.add_argument("--text", default="OpenRouter connectivity test", help="Sample text to embed")
    args = p.parse_args()

    res = test_connection(model=args.model, sample_text=args.text)
    print(json.dumps(res, indent=2))
    return 0 if res.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
