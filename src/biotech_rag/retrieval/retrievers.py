"""Dense and hybrid retrievers for clinical trial documents.

This module provides a compatibility layer so the project works with
multiple LangChain packaging layouts. If `langchain.retrievers.EnsembleRetriever`
is not available in the runtime, a lightweight `SimpleEnsembleRetriever`
fallback will be used instead.
"""

from typing import Any

from langchain_community.retrievers import BM25Retriever

try:
    # Newer/standard import (may not be present in all langchain distributions)
    from langchain.retrievers import EnsembleRetriever  # type: ignore
except Exception:
    EnsembleRetriever = None  # type: ignore
import logging

# Avoid importing langchain_openai here to prevent hard dependency
# on langchain_openai/langchain_core internals that may differ
# between environments. The retriever does not need OpenAIEmbeddings.
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def create_hybrid_retriever(
    vectorstore: Chroma,
    all_texts: list[str],
    all_metadatas: list[dict],
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    k: int = 5,
) -> Any:
    """Create a hybrid retriever combining Vector (Dense) and BM25 (Sparse) search.

    Args:
        vectorstore: The initialized Chroma vectorstore.
        all_texts: List of all document texts for BM25 indexing.
        all_metadatas: List of all document metadatas for BM25.
        vector_weight: Weight for the vector retriever in the ensemble.
        bm25_weight: Weight for the BM25 retriever in the ensemble.
        k: Number of documents to retrieve.

    Returns:
        EnsembleRetriever: A LangChain ensemble retriever.
    """
    # 1. Setup Vector Retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})

    # 2. Setup BM25 Retriever
    bm25_docs = [Document(page_content=t, metadata=m) for t, m in zip(all_texts, all_metadatas)]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k * 2

    if EnsembleRetriever is not None:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[vector_weight, bm25_weight],
        )
        return ensemble_retriever

    # Fallback: simple ensemble implementation to avoid hard dependency
    logger.warning(
        "langchain.retrievers.EnsembleRetriever not available; using SimpleEnsembleRetriever fallback"
    )

    class SimpleEnsembleRetriever:
        """Lightweight ensemble retriever fallback.

        This collects results from each retriever, assigns a simple rank-based
        score weighted by the provided weights, deduplicates by content/id,
        and returns the top-k documents.
        """

        def __init__(self, retrievers: list[Any], weights: list[float], k: int = 5):
            self.retrievers = retrievers
            self.weights = weights
            self.k = k

        def _invoke_retriever(self, retriever: Any, query: str) -> list[Document]:
            # Try common retriever methods, prefer `invoke` (langchain-core 0.1.46+)
            for method in (
                "invoke",
                "get_relevant_documents",
                "retrieve",
                "similarity_search",
                "get_relevant_texts",
            ):
                fn = getattr(retriever, method, None)
                if not fn:
                    continue
                try:
                    if method == "similarity_search":
                        return fn(query, k=self.k)
                    if method == "invoke":
                        try:
                            return fn(query)
                        except TypeError:
                            # Some `invoke` implementations expect a mapping-like input
                            return fn({"query": query})
                    return fn(query)
                except Exception:
                    continue
            return []

        def get_relevant_documents(self, query: str) -> list[Document]:
            all_docs: list[Document] = []
            for retriever, weight in zip(self.retrievers, self.weights):
                docs = self._invoke_retriever(retriever, query) or []
                for rank, doc in enumerate(docs):
                    score = float(weight) * float(max(0, (len(docs) - rank)))
                    meta = dict(doc.metadata or {})
                    meta.setdefault("_ensemble_score", 0.0)
                    meta["_ensemble_score"] = max(meta.get("_ensemble_score", 0.0), score)
                    all_docs.append(Document(page_content=doc.page_content, metadata=meta))

            # Deduplicate by id or content snippet, keep highest score
            dedup = {}
            for doc in all_docs:
                key = doc.metadata.get("id") or doc.page_content[:256]
                if key not in dedup or doc.metadata.get("_ensemble_score", 0.0) > dedup[
                    key
                ].metadata.get("_ensemble_score", 0.0):
                    dedup[key] = doc

            results = sorted(
                dedup.values(), key=lambda d: d.metadata.get("_ensemble_score", 0.0), reverse=True
            )
            return results[:k]

        # Provide an `invoke` alias for compatibility with older wrappers
        def invoke(self, query: str) -> list[Document]:
            return self.get_relevant_documents(query)

    return SimpleEnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[vector_weight, bm25_weight], k=k
    )


def retrieve_with_rerank(
    retriever: Any,
    query: str,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
    candidate_k: int | None = None,
) -> list[Document]:
    """Retrieve documents and then rerank them using a Cross-Encoder.

    Args:
        retriever: The base (hybrid) retriever.
        query: User query.
        reranker_model: The SentenceTransformer Cross-Encoder model name.
        top_n: Number of final docs to return.
        candidate_k: Number of initial candidates to retrieve before reranking.
            If None, defaults to top_n.

    Returns:
        List[Document]: Reranked documents.
    """
    import json
    import re

    # CrossEncoder availability is probed lazily — we do NOT import sentence_transformers
    # here because the mere import triggers HuggingFace Hub/Xet Storage network traffic.
    # The actual import is deferred to where it is first used (see below).
    has_crossencoder: bool | None = None  # None = not yet probed

    # Get initial candidates
    # Support both `invoke(query)` and `get_relevant_documents(query)`
    requested_k = int(candidate_k) if candidate_k is not None else int(top_n)
    requested_k = max(requested_k, int(top_n))

    def _apply_candidate_k(target: Any, k: int) -> list[tuple[Any, str, Any]]:
        changes: list[tuple[Any, str, Any]] = []
        if hasattr(target, "k"):
            try:
                old = target.k
                target.k = k
                changes.append((target, "k", old))
            except Exception:
                pass
        if hasattr(target, "search_kwargs"):
            try:
                old = dict(target.search_kwargs or {})
                updated = dict(old)
                updated["k"] = k
                target.search_kwargs = updated
                changes.append((target, "search_kwargs", old))
            except Exception:
                pass
        if hasattr(target, "retrievers"):
            try:
                for child in target.retrievers or []:
                    changes.extend(_apply_candidate_k(child, k))
            except Exception:
                pass
        return changes

    def _restore_candidate_k(changes: list[tuple[Any, str, Any]]) -> None:
        for obj, attr, old in changes:
            try:
                setattr(obj, attr, old)
            except Exception:
                continue

    changes = _apply_candidate_k(retriever, requested_k) if requested_k else []
    try:
        if hasattr(retriever, "invoke"):
            try:
                docs = retriever.invoke(query)
            except TypeError:
                docs = retriever.invoke({"query": query, "k": requested_k})
        else:
            try:
                docs = retriever.get_relevant_documents(query)
            except Exception:
                # Last-resort: try calling the retriever as a callable
                docs = retriever(query) if callable(retriever) else []
    finally:
        _restore_candidate_k(changes)
    if not docs:
        return []

    # If CrossEncoder available, use it (preferred — fast when torch is present).
    # Probe lazily here so we never import sentence_transformers at the top of the function
    # (the bare import triggers HuggingFace Hub / Xet Storage network traffic).
    if has_crossencoder is None:
        try:
            from sentence_transformers import CrossEncoder as _CE  # type: ignore  # noqa: PLC0415
            has_crossencoder = True
        except Exception as _ce_err:
            logger.info(
                f"CrossEncoder unavailable ({_ce_err}); falling back to LLM-based reranker."
            )
            has_crossencoder = False

    if has_crossencoder:
        from sentence_transformers import CrossEncoder  # type: ignore  # noqa: PLC0415
        model = CrossEncoder(reranker_model)
        pairs = [[query, d.page_content] for d in docs]
        scores = model.predict(pairs)
        for i, doc in enumerate(docs):
            try:
                doc.metadata["rerank_score"] = float(scores[i])
            except Exception:
                doc.metadata["rerank_score"] = 0.0
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
        return sorted_docs[:top_n]

    # Fallback: use remote LLM (OpenRouter) to score pairs without torch
    try:
        from biotech_rag.generation.llm_clients import query_llm

        # Limit candidate count to keep prompt size reasonable
        candidates = docs[: min(len(docs), 20)]
        items = []
        for i, d in enumerate(candidates):
            snippet = re.sub(r"\s+", " ", str(d.page_content))
            snippet = snippet.strip()
            # Truncate long snippets to avoid huge prompts
            if len(snippet) > 800:
                snippet = snippet[:800] + "..."
            items.append({"idx": i, "snippet": snippet})

        # Compose a strict-scoring prompt asking for a JSON array of integers 0-100
        prompt_lines = [
            "You are a clinical retrieval evaluator.\n",
            "Given a user question and a list of candidate document snippets,\n",
            "assign an integer relevance score from 0 (irrelevant) to 100 (highly relevant) to each snippet.\n",
            "Return ONLY a JSON array of integers in the SAME ORDER as the snippets, for example: [0, 85, 42].\n",
            "Do not add any extra text.\n\n",
            f"QUESTION: {query}\n\n",
            "CANDIDATES:\n",
        ]
        for it in items:
            prompt_lines.append(f"[{it['idx']}] {it['snippet']}\n")

        prompt = "".join(prompt_lines)
        response = query_llm(prompt)

        # Parse JSON array from LLM response (robust) and handle degenerate outputs
        scores = []
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                scores = [float(x) for x in parsed]
        except Exception:
            # Attempt to extract the first JSON array-like substring
            m = re.search(r"\[.*\]", response, flags=re.S)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, list):
                        scores = [float(x) for x in parsed]
                except Exception:
                    scores = []

        # If we didn't get scores, bail out early
        if not scores:
            logger.warning(
                "LLM reranker returned no parseable scores; returning original candidates."
            )
            return docs[:top_n]

        # If scores are 0..1, scale to 0..100
        if all(0.0 <= s <= 1.0 for s in scores):
            scores = [s * 100.0 for s in scores]

        # If the LLM produced an all-zero / no-variance result, derive fallback scores
        try:
            min_s, max_s = float(min(scores)), float(max(scores))
        except Exception:
            min_s, max_s = 0.0, 0.0

        low_variance = (max_s - min_s) < 1e-6
        all_zero = all(float(s) == 0.0 for s in scores)
        if low_variance or all_zero:
            logger.warning(
                "LLM reranker produced low-variance/all-zero scores; using metadata-based fallback."
            )
            # Try to use existing metadata scores (ensemble, stored score, or inverted distance)
            meta_vals = []
            for d in candidates:
                meta = d.metadata or {}
                val = 0.0
                try:
                    val = float(meta.get("_ensemble_score", meta.get("score", 0.0) or 0.0))
                except Exception:
                    val = 0.0
                # Try distance-like keys (smaller distance -> higher relevance)
                if not val:
                    for key in ("distance", "dist", "distances", "distance_score"):
                        if key in meta:
                            try:
                                dv = meta.get(key)
                                # If distances stored as list, take first element
                                if isinstance(dv, (list, tuple)) and dv:
                                    dv = float(dv[0])
                                dv = float(dv)
                                if dv > 0:
                                    val = 1.0 / (1.0 + dv)
                                    break
                            except Exception:
                                continue
                meta_vals.append(float(val))

            # If we have any non-zero meta values, normalize them to 0-100, else fallback to original rank order
            if any(v > 0 for v in meta_vals):
                max_mv = max(meta_vals) if max(meta_vals) > 0 else 1.0
                scores = [(v / max_mv) * 100.0 for v in meta_vals]
            else:
                # fallback: prefer earlier candidates (preserve original retriever ranking)
                n = len(candidates)
                scores = [((n - i) / n) * 100.0 for i in range(n)]

        # Assign scores back to candidates (truncate/extend as needed)
        for i, d in enumerate(candidates):
            try:
                d.metadata["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0
            except Exception:
                d.metadata["rerank_score"] = 0.0

        sorted_docs = sorted(
            candidates, key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True
        )
        return sorted_docs[:top_n]
    except Exception as e:
        logger.warning(f"LLM-based rerank failed: {e}; returning original candidates (no rerank).")
        return docs[:top_n]
