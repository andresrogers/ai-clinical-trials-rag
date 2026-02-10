"""Retrieval evaluation utilities.

Provides a function to evaluate three retrieval strategies (vector-only,
hybrid, hybrid+rerank) against a ground-truth JSON produced by the
`ground_truth` notebook.

Metrics produced: Precision@K, Recall@K, MRR (mean reciprocal rank).

The main entrypoint is `evaluate_retrieval(...)` which is intended to be
called from the notebook environment where `vstore`, `hybrid_retriever`,
and `retrieve_with_rerank` are already available.

Example:
    from biotech_rag.evaluation.retrieval_eval import evaluate_retrieval
    summary = evaluate_retrieval(vstore, hybrid_retriever, retrieve_with_rerank,
                                 final_gt_path, top_k=5, sample_size=50)

"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _words_set(text: str) -> set:
    return set(re.findall(r"\w+", (text or "").lower()))


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _normalize_id(val: Any) -> str:
    return re.sub(r"\s+", "", str(val or "").strip()).lower()


def _coerce_source_chunks(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # try JSON list
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
        return [s]
    return [str(val)]


def _is_relevant(doc_text: str, source_chunks: list[str], overlap_thresh: float = 0.20) -> bool:
    if not source_chunks:
        return False
    doc_words = _words_set(doc_text)
    for chunk in source_chunks:
        if not chunk:
            continue
        chunk_words = _words_set(chunk)
        if not chunk_words:
            # fallback: substring match
            if len(chunk) > 40 and _normalize(chunk) in _normalize(doc_text):
                return True
            continue
        overlap = len(doc_words & chunk_words) / max(1, len(chunk_words))
        if overlap >= overlap_thresh:
            return True
        if len(chunk) > 40 and _normalize(chunk) in _normalize(doc_text):
            return True
    return False


def evaluate_retrieval(
    vstore: Any,
    hybrid_retriever: Any,
    rerank_fn: Any,
    final_gt_path: Any,
    top_k: int = 5,
    sample_size: int | None = None,
    sample_strategy: str = "head",
    seed: int = 42,
    filter_by_nct_id: bool = True,
    rerank_candidate_k: int | None = None,
    save_dir: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate retrieval strategies against a final ground truth JSON.

    Args:
        vstore: LangChain/Chroma vectorstore wrapper (provides `similarity_search`).
        hybrid_retriever: hybrid retriever (supports `invoke()` or `get_relevant_documents()`).
        rerank_fn: callable that implements reranking: `rerank_fn(retriever, query, top_n)`.
        final_gt_path: path to `ground_truth_final.json`.
        top_k: how many documents to evaluate per query.
        sample_size: if set, evaluate only this many queries.
        sample_strategy: "head" for first-N, or "random" for random sampling.
        seed: RNG seed used when sample_strategy="random".
        filter_by_nct_id: if True, only count docs as relevant when their nct_id matches.
        rerank_candidate_k: number of candidates to retrieve before reranking.
            If None, defaults to 2 * top_k.
        save_dir: optional Path to write summary/details (defaults to ../data/processed).
        verbose: print per-query breakdown when True.

    Returns:
        dict: {"summary": {...}, "per_query": [...]}
    """
    final_gt_path = Path(final_gt_path)
    if save_dir is None:
        save_dir = final_gt_path.parent if final_gt_path.exists() else Path("../data/processed")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(final_gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    if sample_size is not None and sample_size < len(gt):
        if sample_strategy.lower() == "random":
            rng = random.Random(seed)
            gt = rng.sample(gt, sample_size)
        else:
            gt = gt[:sample_size]

    methods = ["vector", "hybrid", "hybrid_rerank"]

    per_query: list[dict[str, Any]] = []

    for rec in gt:
        q = rec.get("question", "") or ""
        try:
            row_id = int(rec.get("row_id"))
        except Exception:
            row_id = rec.get("row_id")
        source_chunks = _coerce_source_chunks(rec.get("source_chunks", []))
        expected_nct_id = rec.get("nct_id")

        # run vector-only
        try:
            v_docs = vstore.similarity_search(q, k=top_k)
        except Exception:
            try:
                # some wrappers use different signature
                v_docs = vstore.similarity_search(query=q, k=top_k)  # type: ignore
            except Exception as e:
                logger.warning(f"Vector search failed for row {row_id}: {e}")
                v_docs = []

        # run hybrid
        try:
            if hasattr(hybrid_retriever, "invoke"):
                try:
                    h_docs = hybrid_retriever.invoke(q)
                except TypeError:
                    h_docs = hybrid_retriever.invoke({"query": q})
            else:
                h_docs = hybrid_retriever.get_relevant_documents(q)
        except Exception:
            try:
                h_docs = hybrid_retriever(q) if callable(hybrid_retriever) else []
            except Exception:
                h_docs = []

        # run hybrid + rerank
        try:
            candidate_k = (
                rerank_candidate_k if rerank_candidate_k is not None else max(top_k * 2, top_k)
            )
            try:
                r_docs = rerank_fn(hybrid_retriever, q, top_n=top_k, candidate_k=candidate_k)
            except TypeError:
                r_docs = rerank_fn(hybrid_retriever, q, top_n=top_k)
        except Exception as e:
            logger.warning(f"Rerank failed for row {row_id}: {e}")
            r_docs = []

        def score_list(docs: list[Any], expected_nct: Any) -> dict[str, float]:
            matched_chunk_idxs: set[int] = set()
            hits = 0
            first_rank = 0
            expected_norm = _normalize_id(expected_nct) if expected_nct is not None else ""
            for idx, d in enumerate(docs[:top_k], start=1):
                content = getattr(d, "page_content", None) or str(d)
                meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
                doc_nct = meta.get("nct_id") if isinstance(meta, dict) else None
                if filter_by_nct_id and expected_norm and doc_nct:
                    if _normalize_id(doc_nct) != expected_norm:
                        continue
                doc_matched = False
                # determine which ground-truth chunks this doc matches (avoid double-counting)
                for ci, chunk in enumerate(source_chunks):
                    if not chunk:
                        continue
                    if _is_relevant(content, [chunk]):
                        matched_chunk_idxs.add(ci)
                        doc_matched = True
                if doc_matched:
                    hits += 1
                    if first_rank == 0:
                        first_rank = idx
            precision = hits / top_k if top_k else 0.0
            recall = (len(matched_chunk_idxs) / len(source_chunks)) if source_chunks else 0.0
            mrr = 1.0 / first_rank if first_rank else 0.0
            return {
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
                "hits": hits,
                "matched_chunks": len(matched_chunk_idxs),
            }

        v_score = score_list(v_docs, expected_nct_id)
        h_score = score_list(h_docs, expected_nct_id)
        r_score = score_list(r_docs, expected_nct_id)

        per_query.append(
            {
                "row_id": row_id,
                "nct_id": rec.get("nct_id"),
                "question": q,
                "source_count": len(source_chunks),
                "vector": v_score,
                "hybrid": h_score,
                "hybrid_rerank": r_score,
            }
        )

        if verbose:
            print(
                f"Row {row_id} | vector hits={v_score['hits']} hybrid hits={h_score['hits']} rerank hits={r_score['hits']}"
            )

    # aggregate
    def agg(lst: list[dict[str, Any]]) -> dict[str, float]:
        n = len(lst)
        if n == 0:
            return {"precision": 0.0, "recall": 0.0, "mrr": 0.0}
        return {
            "precision": sum(x["precision"] for x in lst) / n,
            "recall": sum(x["recall"] for x in lst) / n,
            "mrr": sum(x["mrr"] for x in lst) / n,
        }

    agg_vector = agg([p["vector"] for p in per_query])
    agg_hybrid = agg([p["hybrid"] for p in per_query])
    agg_rerank = agg([p["hybrid_rerank"] for p in per_query])

    summary = {
        "n_queries": len(per_query),
        "top_k": top_k,
        "sample_strategy": sample_strategy,
        "seed": seed if sample_strategy.lower() == "random" else None,
        "filter_by_nct_id": filter_by_nct_id,
        "rerank_candidate_k": (
            rerank_candidate_k if rerank_candidate_k is not None else max(top_k * 2, top_k)
        ),
        "vector": agg_vector,
        "hybrid": agg_hybrid,
        "hybrid_rerank": agg_rerank,
    }

    # persist results
    summary_path = save_dir / "retrieval_evaluation_summary.json"
    details_path = save_dir / "retrieval_evaluation_details.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(per_query, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote evaluation summary to {summary_path} and details to {details_path}")
    return {
        "summary": summary,
        "details_path": str(details_path),
        "summary_path": str(summary_path),
    }
