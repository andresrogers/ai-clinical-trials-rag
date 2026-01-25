"""Simple CLI entry points referenced by pyproject scripts.

These are thin wrappers to call library functions; implement real logic later.
"""
from __future__ import annotations

import argparse
from typing import List
import uvicorn


def index_documents(args: List[str] | None = None) -> int:
    """Placeholder CLI command to index documents."""
    print("Indexing documents (placeholder)...")
    return 0


def query_rag(args: List[str] | None = None) -> int:
    """Placeholder CLI command to run a query against the RAG pipeline."""
    print("Querying RAG (placeholder)...")
    return 0


def run_evaluation(args: List[str] | None = None) -> int:
    """Placeholder CLI command to run evaluations."""
    print("Running evaluation (placeholder)...")
    return 0


def serve_api(args: List[str] | None = None) -> int:
    """Start the FastAPI app via Uvicorn."""
    uvicorn.run("biotech_rag.api.app:app", host="0.0.0.0", port=8000, reload=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biotech-rag")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("index", help="Index documents")
    sub.add_parser("query", help="Query the RAG system")
    sub.add_parser("eval", help="Run evaluation suite")
    sub.add_parser("serve", help="Serve the API locally")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    if ns.cmd == "index":
        return index_documents()
    if ns.cmd == "query":
        return query_rag()
    if ns.cmd == "eval":
        return run_evaluation()
    if ns.cmd == "serve":
        return serve_api()
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
