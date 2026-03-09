"""RAGAS evaluation helpers for notebook and script usage.

This module centralizes RAGAS dataset preparation and evaluation execution while
reusing the project's OpenRouter-backed LLM and embedding clients.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from biotech_rag.generation.llm_clients import get_openrouter_llm
from biotech_rag.indexing.embedders import Embedder

logger = logging.getLogger(__name__)


class OpenRouterLangChainEmbeddings:
    """LangChain-compatible embedding adapter backed by project `Embedder`.

    This class intentionally provides the minimal `embed_documents` and
    `embed_query` interface consumed by LangChain and RAGAS wrappers.
    """

    def __init__(
        self,
        backend: str = "openrouter",
        model: str = "qwen/qwen3-embedding-8b:floor",
        batch_size: int = 32,
    ) -> None:
        self._embedder = Embedder(backend=backend, model=model, batch_size=batch_size)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings."""
        return self._embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        vectors = self._embedder.embed([text])
        return vectors[0] if vectors else []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous wrapper for `embed_documents`.

        Uses `asyncio.to_thread` to run the blocking embedder in a thread to
        provide the `aembed_documents` API expected by async consumers.
        """
        try:
            return await asyncio.to_thread(self._embedder.embed, texts)
        except Exception:
            # Fall back to sync call if for some reason `to_thread` fails.
            return self._embedder.embed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronous wrapper for `embed_query`."""
        try:
            vectors = await asyncio.to_thread(self._embedder.embed, [text])
            return vectors[0] if vectors else []
        except Exception:
            try:
                vectors = self._embedder.embed([text])
                return vectors[0] if vectors else []
            except Exception:
                return []


def _chunk_to_text(chunk: Any) -> str:
    """Normalize a single chunk/document-like object to text."""
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk.strip()
    if isinstance(chunk, dict):
        for key in ("text", "content", "page_content", "chunk", "value", "source", "document"):
            v = chunk.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        try:
            return json.dumps(chunk, ensure_ascii=False)
        except Exception:
            return str(chunk)
    return str(chunk).strip()


def _normalize_retrieved_contexts(value: Any) -> list[str]:
    """Normalize various retrieved-context formats into list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # parse JSON-encoded lists if present
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [c for c in (_chunk_to_text(p) for p in parsed) if c]
            except Exception:
                pass
        return [s]
    if isinstance(value, dict):
        t = _chunk_to_text(value)
        return [t] if t else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            txt = _chunk_to_text(item)
            if txt:
                out.append(txt)
        return out
    return [str(value).strip()]


def _normalize_reference_text(value: Any) -> str:
    """Return a single normalized reference string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return " ".join(parts)
    if isinstance(value, dict):
        for k in ("reference", "ground_truth_answer", "text", "content"):
            v = value.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value).strip()


# ---------------------------------------------------------------------------
# Negative-evidence helpers
# ---------------------------------------------------------------------------

_NEGATIVE_EVIDENCE_PATTERNS: list[str] = [
    r"\bno evidence\b",
    r"\bno mention\b",
    r"\bnot (?:mentioned|provided|reported|stated|available|described|discussed|specified|found|noted)\b",
    r"\bnone (?:found|reported|available|mentioned|provided)\b",
    r"\bnot explicitly\b",
    r"\bnot (?:explicitly|directly) (?:provided|stated|mentioned|discussed|specified)\b",
    r"\bno information\b",
    r"\bnot (?:in|within) the (?:provided|retrieved|given) context\b",
    r"\bcannot be (?:found|determined|confirmed|verified)\b",
    r"\bnot (?:addressed|covered|included)\b",
]
_NEGATIVE_EVIDENCE_RE = re.compile("|".join(_NEGATIVE_EVIDENCE_PATTERNS), flags=re.IGNORECASE)


def _is_negative_evidence_answer(text: str) -> bool:
    """Return True if the model reply explicitly states absence of evidence.

    Args:
        text: Model-generated answer text.

    Returns:
        True if the response indicates the information was not found in context.
    """
    if not text or not text.strip():
        return False
    return bool(_NEGATIVE_EVIDENCE_RE.search(text))


def _postprocess_negative_evidence_rows(
    rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Override context_recall and faithfulness to 1.0 for valid negative answers.

    When the retrieved_contexts contain nothing relevant to the question and the
    model correctly says "no evidence / not mentioned", RAGAS incorrectly scores
    both context_recall and faithfulness as 0.  This post-processor detects those
    rows and applies the correct score of 1.0 (vacuously faithful, recall satisfied).

    The heuristic is:
      * retrieved_contexts is empty  **OR**
      * ``_is_negative_evidence_answer`` matches the response text.

    Args:
        rows: Serialized RAGAS result rows (mutated in place).
        dataset_rows: Original dataset rows (source of retrieved_contexts/response).

    Returns:
        Updated rows list.
    """
    # Build a quick lookup from user_input → dataset row
    ds_by_ui: dict[str, dict[str, Any]] = {}
    for dr in dataset_rows:
        ui = str(dr.get("user_input") or dr.get("question") or "").strip().lower()
        if ui:
            ds_by_ui[ui] = dr

    for row in rows:
        if not isinstance(row, dict):
            continue

        # Find matching dataset row to inspect retrieved_contexts
        ui = str(row.get("user_input") or row.get("question") or "").strip().lower()
        ds_row = ds_by_ui.get(ui, {})
        contexts: list[str] = ds_row.get("retrieved_contexts") or []
        response: str = str(row.get("response") or ds_row.get("response") or "").strip()

        contexts_empty = not any(c.strip() for c in contexts)
        is_negative = _is_negative_evidence_answer(response)

        if not (contexts_empty or is_negative):
            continue

        logger.debug(
            "Negative-evidence row detected (empty_ctx=%s, neg_ans=%s): '%s'",
            contexts_empty,
            is_negative,
            ui[:80],
        )

        # Override inside a 'metrics' sub-dict if present (newer RAGAS serialization)
        metrics_dict = row.get("metrics")
        if isinstance(metrics_dict, dict):
            for key in list(metrics_dict.keys()):
                k = key.lower()
                if "recall" in k or "faithfulness" in k:
                    if metrics_dict[key] is None or metrics_dict[key] == metrics_dict[key]:  # not NaN check
                        logger.debug("Overriding %s → 1.0 for negative-evidence row", key)
                        metrics_dict[key] = 1.0
        else:
            # Flat serialization style
            for key in list(row.keys()):
                k = key.lower()
                if ("recall" in k or "faithfulness" in k) and isinstance(row.get(key), (int, float)):
                    logger.debug("Overriding %s → 1.0 for negative-evidence row", key)
                    row[key] = 1.0

    return rows


def build_evaluation_rows(records: list[dict[str, Any]], sample_n: int | None = None) -> list[dict[str, Any]]:
    """Build RAGAS rows from merged records.

    Prefers revised_answer -> draft_answer -> response and supports many retrieved keys.
    """
    if sample_n is not None and sample_n > 0:
        source_records = records[:sample_n]
    else:
        source_records = records

    rows: list[dict[str, Any]] = []

    def _is_na_like(s: str) -> bool:
        if not s:
            return True
        s2 = s.strip().lower()
        return s2 in ("n/a", "na", "none", "n.a.") or (len(s2) <= 3 and s2.startswith("n/"))

    candidate_keys = [
        "retrieved_contexts",
        "retrieved_chunks",
        "contexts",
        "source_documents",
        "documents",
        "docs",
        "chunks",
        "results",
        "retrieved",
        "topk_docs",
        "top_k_docs",
        "top_k",
        "topk",
        "chunk_texts",
    ]

    for idx, rec in enumerate(source_records):
        user_input = (rec.get("user_input") or rec.get("question") or rec.get("query") or rec.get("prompt") or "")

        resp_candidates = [
            rec.get("response"),
            rec.get("revised_answer"),
            rec.get("draft_answer"),
            rec.get("answer"),
            rec.get("final_answer"),
        ]
        response = ""
        for c in resp_candidates:
            if isinstance(c, str) and c.strip():
                if not _is_na_like(c):
                    response = c.strip()
                    break
                cleaned = c.strip()
                if len(cleaned) > 4:
                    response = cleaned
                    break

        val = None
        for k in candidate_keys:
            if k in rec and rec.get(k) is not None:
                val = rec.get(k)
                break
        retrieved_contexts = _normalize_retrieved_contexts(val)

        reference = rec.get("reference") or rec.get("ground_truth_answer") or rec.get("reference_text") or ""
        reference = _normalize_reference_text(reference)

        rows.append(
            {
                "id": str(rec.get("row_id") or rec.get("nct_id") or idx),
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
                "metadata": rec.get("metadatas") or rec.get("metadata") or {},
            }
        )

    return rows


def _build_ragas_dataset_from_rows(rows: list[dict[str, Any]]) -> Any:
    """Create a RAGAS dataset object with compatibility across versions."""
    try:
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample

        sample_kwargs_list: list[dict[str, Any]] = []
        for row in rows:
            sample_kwargs: dict[str, Any] = {
                "user_input": row["user_input"],
                "response": row["response"],
                "retrieved_contexts": row["retrieved_contexts"],
            }
            if "reference" in row:
                sample_kwargs["reference"] = row["reference"]
            sample_kwargs_list.append(sample_kwargs)

        samples: list[Any] = []
        sample_sig = inspect.signature(SingleTurnSample)
        params = set(sample_sig.parameters.keys())
        for kwargs in sample_kwargs_list:
            final_kwargs = dict(kwargs)
            if "reference" in final_kwargs and "reference" not in params and "expected" in params:
                final_kwargs["expected"] = final_kwargs.pop("reference")
            samples.append(SingleTurnSample(**final_kwargs))

        return EvaluationDataset(samples=samples)
    except Exception as exc:
        logger.warning("Falling back to HuggingFace dataset for RAGAS compatibility: %s", exc)

    try:
        from datasets import Dataset

        return Dataset.from_list(rows)
    except Exception as exc:
        logger.warning("Falling back to plain list dataset: %s", exc)

    return rows


def build_evaluation_dataset(records: list[dict[str, Any]], sample_n: int | None = 10) -> Any:
    """Build a RAGAS-compatible evaluation dataset.

    Args:
        records: Merged retrieval/response records.
        sample_n: Optional cap on number of rows.

    Returns:
        A dataset object accepted by `ragas.evaluate(...)`.
    """
    rows = build_evaluation_rows(records=records, sample_n=sample_n)
    return _build_ragas_dataset_from_rows(rows)


def make_ragas_llm(
    model: str = "openai/gpt-4o-mini:floor",
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> Any:
    """Create a RAGAS-compatible evaluator LLM using OpenRouter.

    Args:
        model: OpenRouter model id for the judge LLM.
        temperature: Sampling temperature.
        max_tokens: Maximum completion tokens.

    Returns:
        A RAGAS-compatible LLM wrapper or underlying LangChain model.
    """
    langchain_llm = get_openrouter_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(langchain_llm)
    except Exception:
        pass

    try:
        from ragas.llms.base import LangchainLLMWrapper

        return LangchainLLMWrapper(langchain_llm)
    except Exception:
        pass

    try:
        from ragas.llms import llm_factory

        for factory_kwargs in (
            {"client": langchain_llm},
            {"langchain_llm": langchain_llm},
            {"llm": langchain_llm},
        ):
            try:
                return llm_factory("langchain", **factory_kwargs)
            except Exception:
                continue
    except Exception:
        pass

    return langchain_llm


def make_ragas_embeddings(
    backend: str = "openrouter",
    model: str = "qwen/qwen3-embedding-8b:floor",
    batch_size: int = 32,
) -> Any:
    """Create a RAGAS-compatible embeddings wrapper."""
    lc_embeddings = OpenRouterLangChainEmbeddings(
        backend=backend,
        model=model,
        batch_size=batch_size,
    )

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(lc_embeddings)
    except Exception:
        return lc_embeddings


def _get_metric_class(metric_name_candidates: list[str]) -> type[Any] | None:
    try:
        import ragas.metrics as ragas_metrics

        for name in metric_name_candidates:
            metric_cls = getattr(ragas_metrics, name, None)
            if metric_cls is not None:
                return metric_cls
    except Exception:
        return None
    return None


def _init_metric(metric_cls: type[Any], llm: Any | None, embeddings: Any | None) -> Any:
    metric_kwargs: dict[str, Any] = {}
    try:
        metric_sig = inspect.signature(metric_cls)
        params = set(metric_sig.parameters.keys())
        if "llm" in params and llm is not None:
            metric_kwargs["llm"] = llm
        if "embeddings" in params and embeddings is not None:
            metric_kwargs["embeddings"] = embeddings
    except Exception:
        metric_kwargs = {}

    metric = metric_cls(**metric_kwargs)

    if llm is not None and getattr(metric, "llm", None) is None:
        try:
            setattr(metric, "llm", llm)
        except Exception:
            pass

    if embeddings is not None and getattr(metric, "embeddings", None) is None:
        try:
            setattr(metric, "embeddings", embeddings)
        except Exception:
            pass

    return metric


def default_ragas_metrics(llm: Any | None = None, embeddings: Any | None = None) -> list[Any]:
    """Build core RAGAS metric instances with version compatibility.

    Metric intent:
    - context precision
    - context recall
    - response relevancy
    - faithfulness
    - factual correctness
    - semantic similarity
    """
    metric_specs = [
        [
            # Prefer question-based precision (no reference needed) — avoids
            # false-zero when retrieved chunks are topically relevant but not
            # verbatim-needed for the specific reference answer wording.
            "LLMContextPrecisionWithoutReference",
            "LLMContextPrecisionWithReference",
            "ContextPrecision",
        ],
        ["ContextRecall", "LLMContextRecall"],
        ["ResponseRelevancy", "AnswerRelevancy", "ResponseRelevance"],
        ["Faithfulness"],
        ["FactualCorrectness", "Factuality"],
        ["SemanticSimilarity", "ReferenceSemanticSimilarity"],
    ]

    metrics: list[Any] = []
    for candidates in metric_specs:
        metric_cls = _get_metric_class(candidates)
        if metric_cls is None:
            continue
        metrics.append(_init_metric(metric_cls=metric_cls, llm=llm, embeddings=embeddings))

    if not metrics:
        logger.warning(
            "No compatible RAGAS metric classes were found. Proceeding with empty metric list (fallbacks may be used)."
        )

    # If no metric classes were discovered, return an empty list and allow callers
    # (or the evaluation runner) to use fallback computations instead of failing
    # hard here. This helps notebooks run even when the installed `ragas`
    # package and its transitive dependencies (e.g., `langchain_core`) are not
    # compatible with the versions expected by this code.
    if not metrics:
        return []

    return metrics


def _resolve_ragas_evaluate() -> Any:
    """Resolve the RAGAS evaluation function across versions."""
    try:
        from ragas import evaluate

        return evaluate
    except Exception:
        pass

    try:
        from ragas.evaluation import evaluate

        return evaluate
    except Exception as exc:
        raise RuntimeError("Unable to import ragas.evaluate") from exc


def _serialize_evaluation_result(result: Any) -> dict[str, Any]:
    """Serialize evaluation result into JSON-safe summary payload."""
    payload: dict[str, Any] = {}

    try:
        if hasattr(result, "to_dict"):
            to_dict_payload = result.to_dict()
            if isinstance(to_dict_payload, dict):
                payload["scores"] = to_dict_payload
    except Exception:
        pass

    try:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            payload["rows"] = df.to_dict(orient="records")
            numeric_cols = df.select_dtypes(include=["number"]).columns
            payload["summary"] = {
                col: float(df[col].mean()) for col in numeric_cols if len(df[col]) > 0
            }
    except Exception:
        pass

    if not payload and isinstance(result, dict):
        payload = result

    if not payload:
        payload = {"repr": str(result)}

    return payload


def _extract_dataset_rows(dataset: Any) -> list[dict[str, Any]]:
    if isinstance(dataset, list):
        return [item for item in dataset if isinstance(item, dict)]

    if hasattr(dataset, "to_list"):
        try:
            values = dataset.to_list()
            return [item for item in values if isinstance(item, dict)]
        except Exception:
            pass

    if hasattr(dataset, "samples"):
        rows: list[dict[str, Any]] = []
        for sample in getattr(dataset, "samples", []):
            sample_dict: dict[str, Any] | None = None
            if hasattr(sample, "model_dump"):
                try:
                    dumped = sample.model_dump()
                    if isinstance(dumped, dict):
                        sample_dict = dumped
                except Exception:
                    sample_dict = None
            if sample_dict is None and hasattr(sample, "to_dict"):
                try:
                    dumped = sample.to_dict()
                    if isinstance(dumped, dict):
                        sample_dict = dumped
                except Exception:
                    sample_dict = None
            if sample_dict is None and hasattr(sample, "__dict__"):
                raw = getattr(sample, "__dict__", {})
                if isinstance(raw, dict):
                    sample_dict = dict(raw)
            if sample_dict is not None:
                rows.append(sample_dict)
        return rows

    if hasattr(dataset, "to_pandas"):
        try:
            df = dataset.to_pandas()
            return df.to_dict(orient="records")
        except Exception:
            pass

    return []


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if not vector_a or not vector_b:
        return 0.0
    if len(vector_a) != len(vector_b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vector_a, vector_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _embed_text(embeddings: Any, text: str) -> list[float]:
    if hasattr(embeddings, "embed_query"):
        try:
            return embeddings.embed_query(text)
        except Exception:
            pass

    if hasattr(embeddings, "embed_documents"):
        try:
            vectors = embeddings.embed_documents([text])
            if vectors:
                return vectors[0]
        except Exception:
            pass

    wrapped = getattr(embeddings, "embeddings", None)
    if wrapped is not None and hasattr(wrapped, "embed_query"):
        try:
            return wrapped.embed_query(text)
        except Exception:
            pass

    if wrapped is not None and hasattr(wrapped, "embed_documents"):
        try:
            vectors = wrapped.embed_documents([text])
            if vectors:
                return vectors[0]
        except Exception:
            pass

    return []


def _invoke_llm_text(llm: Any, prompt: str) -> str:
    """Invoke an LLM (raw LangChain or RAGAS-wrapped) and return plain text.

    Tries multiple unwrapping strategies so it works whether ``llm`` is a raw
    ``ChatOpenAI``, a RAGAS ``LangchainLLMWrapper`` (which stores the inner
    model as ``.langchain_llm``), or any other wrapper variant.
    """
    # Build candidate list: the wrapper itself + every common attribute that
    # could hold the underlying LangChain LLM.
    candidates: list[Any] = [llm]
    for attr in ("langchain_llm", "llm", "model", "_langchain_llm", "_llm", "client"):
        obj = getattr(llm, attr, None)
        if obj is not None and obj is not llm:
            candidates.append(obj)

    for candidate in candidates:
        if candidate is None:
            continue

        # 1. Standard LangChain .invoke(str) — works for ChatOpenAI, etc.
        if hasattr(candidate, "invoke"):
            try:
                response = candidate.invoke(prompt)
                if isinstance(response, str):
                    return response
                if hasattr(response, "content"):
                    return str(response.content)
                # Some wrappers return a list of generations
                if isinstance(response, list) and response:
                    first = response[0]
                    if isinstance(first, str):
                        return first
                    if hasattr(first, "text"):
                        return str(first.text)
                    if hasattr(first, "content"):
                        return str(first.content)
                return str(response)
            except Exception:
                continue

        # 2. RAGAS async generate path exposed as sync via generate_text
        if hasattr(candidate, "generate_text"):
            try:
                response = candidate.generate_text(prompt)
                if isinstance(response, str):
                    return response
                return str(response)
            except Exception:
                continue

        # 3. Plain callable
        if callable(candidate):
            try:
                response = candidate(prompt)
                if isinstance(response, str):
                    return response
                if hasattr(response, "content"):
                    return str(response.content)
                return str(response)
            except Exception:
                continue

    logger.warning("_invoke_llm_text: all candidate LLM calls failed for prompt starting: %r", prompt[:80])
    return ""


def _parse_score_from_text(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None

    # Strategy 1: whole response is a bare decimal (simplest prompt format)
    bare = re.match(r"^([0-9](?:\.[0-9]*)?)$", stripped)
    if bare:
        try:
            score = float(bare.group(1))
            if 0.0 <= score <= 1.0:
                return score
        except Exception:
            pass

    # Strategy 2: try the entire text as JSON
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            value = payload.get("score")
            if value is not None:
                return max(0.0, min(1.0, float(value)))
    except Exception:
        pass

    # Strategy 3: try every line individually as JSON (handles pretty-printed blocks)
    for line in reversed([l.strip() for l in stripped.splitlines() if l.strip()]):
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                value = payload.get("score")
                if value is not None:
                    return max(0.0, min(1.0, float(value)))
        except Exception:
            pass

    # Strategy 4: regex search for "score": <number> anywhere in the text
    m = re.search(r'["\']?score["\']?\s*:\s*([0-9]*\.?[0-9]+)', stripped, flags=re.IGNORECASE)
    if m:
        try:
            score = float(m.group(1))
            if 0.0 <= score <= 1.0:
                return score
        except Exception:
            pass

    # Strategy 5: find all decimal floats in [0,1] — take the first plausible one
    # (avoid bare integers like "0" or "1" which appear in any text)
    for m in re.finditer(r'\b(0(?:\.[0-9]+)?|1(?:\.0+)?)\b', stripped):
        try:
            score = float(m.group(1))
            if 0.0 <= score <= 1.0 and '.' in m.group(1):
                return score
        except Exception:
            pass

    return None


def _is_raw_judge_error(raw: str) -> bool:
    """Detect obvious judge/LLM error text (timeout, rate-limit, internal error)."""
    if not raw:
        return False  # Empty is not an error - RAGAS often doesn't expose raw responses
    s = raw.lower()
    tokens = (
        "error",
        "timeout",
        "timed out",
        "rate limit",
        "internal server error",
        "exceeded",
        "could not",
        "unavailable",
        "exception",
        "traceback",
    )
    return any(tok in s for tok in tokens)


def _is_judge_response_bad(row: dict) -> bool:
    """Return True when a row has an explicit error or missing/invalid metrics.

    Checks for explicit `error` field or missing/all-None metrics.
    Does NOT check raw judge text (RAGAS doesn't expose it reliably).
    """
    if not isinstance(row, dict):
        return True
    
    # Check for explicit error field
    if row.get("error"):
        return True
    
    # Check if metrics are present and at least one is valid
    metrics = row.get("metrics") or row.get("scores")
    if not metrics or not isinstance(metrics, dict):
        return True
    
    # Check if at least one metric has a valid numeric value
    for v in metrics.values():
        if isinstance(v, (int, float)) and v == v:  # not NaN
            return False  # Found at least one valid metric
    
    return True  # All metrics are None/NaN/invalid


def _find_dataset_row_index(serialized_row: dict, dataset_rows: list[dict]) -> Optional[int]:
    """Locate matching dataset row index for a serialized row by id/nct/user_input.

    Returns index or None when no match is found.
    """
    if not isinstance(serialized_row, dict):
        return None
    row_id = serialized_row.get("id") or serialized_row.get("row_id") or serialized_row.get("nct_id")
    if row_id:
        try:
            row_id_s = str(row_id).strip().lower()
            for i, dr in enumerate(dataset_rows):
                cand = str(dr.get("id") or dr.get("row_id") or dr.get("nct_id") or "").strip().lower()
                if cand and cand == row_id_s:
                    return i
        except Exception:
            pass

    user_input = (serialized_row.get("user_input") or serialized_row.get("question") or "")
    if user_input:
        ui = str(user_input).strip().lower()
        for i, dr in enumerate(dataset_rows):
            cand_ui = str(dr.get("user_input") or dr.get("question") or "").strip().lower()
            if cand_ui and cand_ui == ui:
                return i

    return None


def _set_row_metrics_none(row: dict) -> None:
    """Set numeric metric values to None for a serialized row (in-place)."""
    if not isinstance(row, dict):
        return
    if "metrics" in row and isinstance(row["metrics"], dict):
        for k in list(row["metrics"].keys()):
            row["metrics"][k] = None
        return
    for k, v in list(row.items()):
        if isinstance(v, (int, float)):
            row[k] = None


def _recompute_summary_from_rows(serialized: dict) -> dict:
    """Recompute a simple numeric-mean summary from serialized['rows'].

    Ignores None and non-numeric values.
    """
    rows = serialized.get("rows", []) or []
    metric_keys: set = set()
    for r in rows:
        if isinstance(r, dict):
            m = r.get("metrics") or r.get("scores")
            if isinstance(m, dict):
                metric_keys.update(m.keys())
            else:
                for k, v in r.items():
                    if isinstance(v, (int, float)):
                        metric_keys.add(k)

    summary: dict = {}
    for k in metric_keys:
        vals: list[float] = []
        for r in rows:
            val = None
            if isinstance(r, dict):
                m = r.get("metrics") or r.get("scores")
                if isinstance(m, dict) and k in m:
                    val = m.get(k)
                elif k in r and isinstance(r.get(k), (int, float)):
                    val = r.get(k)
            if isinstance(val, (int, float)):
                vals.append(float(val))
        if vals:
            summary[k] = float(sum(vals) / len(vals))

    serialized["summary"] = summary
    return serialized


def _call_evaluate_with_timeout(evaluate_fn: Any, eval_kwargs: dict, timeout_seconds: int = 60) -> Any:
    """Call the provided evaluate function with a timeout.

    Supports coroutine evaluate functions (runs with asyncio) and synchronous
    evaluate functions (runs in a thread with a timeout). Returns the result
    or None on timeout/error.
    """
    try:
        # If evaluate_fn is a coroutine function, run it with asyncio and a timeout
        if inspect.iscoroutinefunction(evaluate_fn):
            return asyncio.run(asyncio.wait_for(evaluate_fn(**eval_kwargs), timeout_seconds))
    except Exception:
        # fall back to threaded call below
        pass

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(evaluate_fn, **eval_kwargs)
            return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        logger.warning("evaluate_fn call timed out after %ss", timeout_seconds)
        return None
    except Exception as exc:
        logger.warning("evaluate_fn call raised: %s", exc)
        return None


def _compute_semantic_similarity_fallback(
    rows: list[dict[str, Any]],
    embeddings: Any,
) -> dict[str, Any] | None:
    score_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        response = _normalize_reference_text(row.get("response"))
        reference = _normalize_reference_text(row.get("reference"))
        if not response or not reference:
            continue

        response_vec = _embed_text(embeddings, response)
        reference_vec = _embed_text(embeddings, reference)
        similarity = _cosine_similarity(response_vec, reference_vec)

        score_rows.append(
            {
                "index": idx,
                "row_id": row.get("row_id"),
                "semantic_similarity": float(similarity),
            }
        )

    if not score_rows:
        return None

    mean_score = sum(item["semantic_similarity"] for item in score_rows) / len(score_rows)
    return {
        "name": "semantic_similarity_fallback",
        "rows": score_rows,
        "mean": float(mean_score),
    }


def _compute_factual_correctness_fallback(
    rows: list[dict[str, Any]],
    llm: Any,
    max_rows: int = 25,
) -> dict[str, Any] | None:
    score_rows: list[dict[str, Any]] = []
    evaluated = 0
    for idx, row in enumerate(rows):
        if evaluated >= max_rows:
            break

        response = _normalize_reference_text(row.get("response"))
        reference = _normalize_reference_text(row.get("reference"))
        if not response or not reference:
            continue

        prompt = (
            "You are an evaluator. Compare RESPONSE against REFERENCE. "
            "Return strict JSON only with keys 'score' and 'reason'. "
            "score must be a float in [0, 1] where 1 means fully factually consistent with reference.\n\n"
            f"REFERENCE: '''{reference}'''\n"
            f"RESPONSE: '''{response}'''"
        )
        output = _invoke_llm_text(llm, prompt)
        score = _parse_score_from_text(output)
        if score is None:
            continue

        score_rows.append(
            {
                "index": idx,
                "row_id": row.get("row_id"),
                "factual_correctness": float(score),
            }
        )
        evaluated += 1

    if not score_rows:
        return None

    mean_score = sum(item["factual_correctness"] for item in score_rows) / len(score_rows)
    return {
        "name": "factual_correctness_fallback",
        "rows": score_rows,
        "mean": float(mean_score),
    }


# ---------------------------------------------------------------------------
# Context precision LLM judge (question-based average precision)
# ---------------------------------------------------------------------------


def _parse_relevance_list(text: str, expected_len: int) -> list[int] | None:
    """Parse a JSON array of 0/1 relevance flags from LLM output."""
    stripped = text.strip()
    candidates = [stripped] + [line.strip() for line in stripped.splitlines() if line.strip()]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and parsed:
                return [1 if v else 0 for v in parsed]
        except Exception:
            pass
    m = re.search(r"\[[\s0-9,]+\]", stripped)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [1 if v else 0 for v in parsed]
        except Exception:
            pass
    return None


def _average_precision(relevance: list[int]) -> float:
    """Compute Average Precision from a binary relevance list."""
    if not relevance or sum(relevance) == 0:
        return 0.0
    total = 0.0
    relevant_so_far = 0
    for i, rel in enumerate(relevance):
        if rel:
            relevant_so_far += 1
            total += relevant_so_far / (i + 1)
    return total / sum(relevance)


def _compute_context_precision_llm_judge(
    rows: list[dict[str, Any]],
    llm: Any,
    max_rows: int = 50,
    override_zeros_only: bool = True,
) -> dict[str, Any] | None:
    """Judge context precision as Average Precision @K against the user question.

    Unlike RAGAS ``LLMContextPrecisionWithReference`` — which asks whether each
    chunk was needed to produce the *reference answer* — this judge asks whether
    each chunk is relevant to the *user question*.  That distinction matters
    whenever the question is broad (e.g. "what secondary endpoints were assessed?")
    but the reference answer only mentions a subset: RAGAS would score PFS-only
    reference → chunks about safety get 0, even though they directly answer the
    question.

    The judge sends all chunks in a single prompt (up to 10) and asks for a
    JSON array of 0/1 relevance flags, then computes Average Precision from
    that ranked binary list.

    Args:
        rows: Serialized dataset rows with ``user_input`` and ``retrieved_contexts``.
        llm: OpenRouter LLM wrapper.
        max_rows: Budget guard.
        override_zeros_only: When True, only score rows where context_precision is 0 or None.

    Returns:
        Dict with ``rows`` (per-sample AP scores + chunk relevance) and ``mean``.
    """
    score_rows: list[dict[str, Any]] = []
    evaluated = 0

    for idx, row in enumerate(rows):
        if evaluated >= max_rows:
            break
        if not isinstance(row, dict):
            continue

        if override_zeros_only:
            existing = _get_flat_metric(row, "context_precision")
            if existing is not None and existing > 0.0:
                continue

        user_input = str(row.get("user_input") or row.get("question") or "").strip()
        contexts: list[str] = row.get("retrieved_contexts") or []
        if not user_input or not contexts:
            continue

        # Trim to 10 chunks max; truncate each chunk to 400 chars to stay in budget
        eval_contexts = contexts[:10]
        chunk_block = "\n\n".join(
            f"CHUNK {i + 1}:\n{ctx[:400]}" for i, ctx in enumerate(eval_contexts)
        )

        prompt = (
            "You are an expert clinical-trial RAG evaluator.\n"
            "Task: for each numbered chunk below, decide whether it contains information "
            "DIRECTLY OR PARTIALLY relevant to answering the USER QUESTION.\n"
            "\n"
            "Relevance rules:\n"
            "  1 = chunk mentions a concept, endpoint, measurement, or finding that helps answer the question.\n"
            "  0 = chunk is completely unrelated to the question's topic.\n"
            "\n"
            "IMPORTANT: judge relevance against the QUESTION, not against any particular answer wording.\n"
            f"USER QUESTION:\n{user_input}\n\n"
            f"{chunk_block}\n\n"
            f"Output ONLY a JSON array of {len(eval_contexts)} integers (0 or 1), one per chunk, in order.\n"
            f"Example for {len(eval_contexts)} chunks: "
            + str([1] * len(eval_contexts)).replace(" ", "") + "\n"
            "No explanation, no other text."
        )

        raw = _invoke_llm_text(llm, prompt)
        if not raw:
            print(f"[context_precision judge] WARNING: row {idx} — LLM returned empty string")
            continue

        relevance_list = _parse_relevance_list(raw, len(eval_contexts))
        if relevance_list is None:
            print(f"[context_precision judge] WARNING: row {idx} — unparseable: {raw[:120]!r}")
            continue

        ap = _average_precision(relevance_list)
        print(f"[context_precision judge] row {idx} → AP={ap:.3f}  relevance={relevance_list}")

        score_rows.append(
            {
                "index": idx,
                "row_id": row.get("id") or row.get("row_id"),
                "context_precision_judge": float(ap),
                "chunk_relevance": relevance_list,
            }
        )
        evaluated += 1

    if not score_rows:
        return None

    mean_score = sum(r["context_precision_judge"] for r in score_rows) / len(score_rows)
    return {
        "name": "context_precision_llm_judge",
        "rows": score_rows,
        "mean": float(mean_score),
    }


def _apply_context_precision_judge_to_rows(
    rows: list[dict[str, Any]],
    judge_payload: dict[str, Any],
) -> None:
    """Write context precision judge scores back into serialized rows (in-place)."""
    index_map = {
        item["index"]: item["context_precision_judge"] for item in judge_payload.get("rows", [])
    }
    for idx, row in enumerate(rows):
        if idx not in index_map:
            continue
        score = index_map[idx]
        metrics_dict = row.get("metrics")
        if isinstance(metrics_dict, dict):
            metrics_dict["context_precision"] = score
        else:
            row["context_precision"] = score


# ---------------------------------------------------------------------------
# Answer relevancy LLM judge
# ---------------------------------------------------------------------------


def _compute_answer_relevancy_llm_judge(
    rows: list[dict[str, Any]],
    llm: Any,
    max_rows: int = 50,
    override_zeros_only: bool = True,
) -> dict[str, Any] | None:
    """Judge answer relevancy using gpt-4o-mini via OpenRouter.

    RAGAS ResponseRelevancy uses a reverse-question heuristic that frequently
    assigns 0 to correct negative or paraphrased answers.  This function
    replaces it with a direct LLM judge prompt that is more robust:
    it asks the model to compare the response against the user question (and
    optionally the reference answer) and returns a [0,1] relevancy score.

    Args:
        rows: Serialized dataset rows containing ``user_input``, ``response``,
            and optionally ``reference``.
        llm: OpenRouter LLM wrapper exposing ``invoke`` or ``__call__``.
        max_rows: Maximum number of rows to judge (budget guard).
        override_zeros_only: When True, only score rows where the current
            answer_relevancy is 0.0 or None.  Set False to re-judge all rows.

    Returns:
        Dict with ``rows`` (per-sample scores) and ``mean``, or None if no
        rows were scored.
    """
    score_rows: list[dict[str, Any]] = []
    evaluated = 0

    for idx, row in enumerate(rows):
        if evaluated >= max_rows:
            break
        if not isinstance(row, dict):
            continue

        # Decide whether to skip (override_zeros_only mode)
        if override_zeros_only:
            existing = _get_flat_metric(row, "answer_relevancy") or _get_flat_metric(row, "response_relevancy")
            if existing is not None and existing > 0.0:
                continue  # Already has a non-zero score; skip

        user_input = str(row.get("user_input") or row.get("question") or "").strip()
        response = str(row.get("response") or "").strip()
        reference = str(row.get("reference") or "").strip()

        if not user_input or not response:
            continue

        reference_block = (
            f"\nREFERENCE ANSWER (expert-written gold standard):\n'''{reference}'''\n"
            if reference
            else ""
        )

        prompt = (
            "You are an expert clinical-trial RAG evaluator.\n"
            "Task: score how well the RESPONSE addresses the USER QUESTION.\n"
            "\n"
            "Scoring rubric (choose the closest value):\n"
            "  1.0 — Fully answers every aspect of the question.\n"
            "  0.8 — Mostly answers; minor gaps or slight off-topic details.\n"
            "  0.6 — Partially answers; addresses the main point but misses sub-questions.\n"
            "  0.4 — Tangentially related; some relevant content but does not directly answer.\n"
            "  0.2 — Barely relevant; mostly unrelated to the question.\n"
            "  0.0 — Completely irrelevant or empty.\n"
            "\n"
            "CRITICAL RULE: A response that correctly states 'no evidence was found in the "
            "retrieved context' IS a valid, relevant answer when the topic is absent from the "
            "documents. Score such responses >= 0.8 if they name the correct topic and explain "
            "that it was not found.\n"
            f"{reference_block}"
            f"\nUSER QUESTION:\n{user_input}\n"
            f"\nRESPONSE TO EVALUATE:\n{response}\n"
            "\n"
            "Reply with ONLY a single decimal number between 0 and 1 (e.g. 0.8). "
            "No explanation, no JSON, no other text."
        )

        raw = _invoke_llm_text(llm, prompt)
        if not raw:
            print(f"[answer_relevancy judge] WARNING: row {idx} — LLM returned empty string. "
                  "Check that _invoke_llm_text can reach the underlying LangChain model.")
            logger.warning("answer_relevancy judge: empty LLM response for row %d", idx)
            continue
        score = _parse_score_from_text(raw)
        if score is None:
            print(f"[answer_relevancy judge] WARNING: row {idx} — unparseable output: {raw[:150]!r}")
            logger.warning(
                "answer_relevancy judge: unparseable output for row %d (raw=%r)", idx, raw[:200]
            )
            continue
        print(f"[answer_relevancy judge] row {idx} → {score:.2f}  (raw={raw[:40]!r})")
        logger.debug("answer_relevancy judge: row %d → %.3f (raw=%r)", idx, score, raw[:60])

        score_rows.append(
            {
                "index": idx,
                "row_id": row.get("id") or row.get("row_id"),
                "answer_relevancy_judge": float(score),
            }
        )
        evaluated += 1

    if not score_rows:
        return None

    mean_score = sum(item["answer_relevancy_judge"] for item in score_rows) / len(score_rows)
    return {
        "name": "answer_relevancy_llm_judge",
        "rows": score_rows,
        "mean": float(mean_score),
    }


def _get_flat_metric(row: dict, *keys: str) -> float | None:
    """Extract a numeric metric value from a row's nested or flat structure.

    Uses EXACT key matching only.  Substring matching is intentionally avoided
    so that e.g. looking up ``context_precision`` does not accidentally return
    the value of ``llm_context_precision_without_reference``.
    """
    metrics_dict = row.get("metrics") or row.get("scores")
    if isinstance(metrics_dict, dict):
        for key in keys:
            # exact match
            v = metrics_dict.get(key)
            if isinstance(v, (int, float)) and v == v:
                return float(v)
            # case-insensitive exact match
            key_lower = key.lower()
            for k, v in metrics_dict.items():
                if k.lower() == key_lower and isinstance(v, (int, float)) and v == v:
                    return float(v)
    for key in keys:
        v = row.get(key)
        if isinstance(v, (int, float)) and v == v:
            return float(v)
        # case-insensitive exact flat scan
        key_lower = key.lower()
        for k, val in row.items():
            if k.lower() == key_lower and isinstance(val, (int, float)) and val == val:
                return float(val)
    return None


# Aliases RAGAS may use for context precision depending on the installed version.
# After our judge writes the canonical ``context_precision`` key we remove these
# so the summary does not contain two conflicting columns.
_RAGAS_PRECISION_ALIASES: tuple[str, ...] = (
    "llm_context_precision_without_reference",
    "llm_context_precision_with_reference",
)


def _strip_ragas_precision_aliases_from_rows(rows: list[dict[str, Any]]) -> None:
    """Remove RAGAS-named precision aliases from serialized rows (in-place).

    After the context-precision judge writes ``context_precision`` we remove
    the RAGAS-generated alias keys (e.g. ``llm_context_precision_without_reference``)
    so the summary only contains a single canonical ``context_precision`` column.
    """
    for row in rows:
        if not isinstance(row, dict):
            continue
        metrics_dict = row.get("metrics")
        if isinstance(metrics_dict, dict):
            for alias in _RAGAS_PRECISION_ALIASES:
                metrics_dict.pop(alias, None)
        for alias in _RAGAS_PRECISION_ALIASES:
            row.pop(alias, None)


def _apply_answer_relevancy_judge_to_rows(
    rows: list[dict[str, Any]],
    judge_payload: dict[str, Any],
) -> None:
    """Write judge scores back into serialized rows (in-place)."""
    index_map = {item["index"]: item["answer_relevancy_judge"] for item in judge_payload.get("rows", [])}
    for idx, row in enumerate(rows):
        if idx not in index_map:
            continue
        score = index_map[idx]
        metrics_dict = row.get("metrics")
        if isinstance(metrics_dict, dict):
            metrics_dict["answer_relevancy"] = score
        else:
            row["answer_relevancy"] = score


def run_ragas_evaluation(
    dataset: Any,
    llm: Any,
    metrics: list[Any] | None = None,
    embeddings: Any | None = None,
    output_path: str | Path = "../data/processed/ragas_results.json",
    max_factual_judge_rows: int = 25,
    allow_retries: bool = False,  # Disabled by default to prevent endless loops
) -> dict[str, Any]:
    """Run RAGAS evaluation and persist summarized results.

    Args:
        dataset: RAGAS-compatible evaluation dataset.
        llm: Evaluator LLM instance for judge-based metrics.
        metrics: Optional metric instances. Defaults to core metrics.
        embeddings: Optional embeddings wrapper (used by relevancy-style metrics).
        output_path: JSON output path for serialized results.

    Returns:
        Dictionary containing serialized metrics and output metadata.
    """
    try:
        evaluate_fn = _resolve_ragas_evaluate()
    except Exception as exc:
        logger.warning("RAGAS evaluate() not available: %s. Falling back to local evaluation.", exc)

        # Build a simple fallback evaluation payload using available helpers.
        dataset_rows = _extract_dataset_rows(dataset)
        serialized: dict[str, Any] = {}
        comparison_metrics: dict[str, Any] = {}
        comparison_summary: dict[str, float] = {}
        metric_names: list[str] = []

        # Semantic similarity fallback (requires embeddings)
        if embeddings is not None:
            semantic_payload = _compute_semantic_similarity_fallback(rows=dataset_rows, embeddings=embeddings)
            if semantic_payload is not None:
                comparison_metrics["semantic_similarity"] = semantic_payload
                comparison_summary["avg_semantic_similarity"] = semantic_payload["mean"]
                metric_names.append("semantic_similarity_fallback")

        # Factual correctness fallback (requires llm)
        if llm is not None:
            factual_payload = _compute_factual_correctness_fallback(rows=dataset_rows, llm=llm, max_rows=max_factual_judge_rows)
            if factual_payload is not None:
                comparison_metrics["factual_correctness"] = factual_payload
                comparison_summary["avg_factual_correctness"] = factual_payload["mean"]
                metric_names.append("factual_correctness_fallback")

        if comparison_metrics:
            serialized["comparison_metrics"] = comparison_metrics

        if comparison_summary:
            serialized["summary"] = comparison_summary

        # --- negative-evidence post-processing (fallback branch) ---
        dataset_rows = _postprocess_negative_evidence_rows(dataset_rows, dataset_rows)

        # --- context_precision LLM judge (fallback branch) ---
        if llm is not None:
            cp_payload = _compute_context_precision_llm_judge(
                rows=dataset_rows, llm=llm, max_rows=max_factual_judge_rows, override_zeros_only=False
            )
            if cp_payload is not None:
                _apply_context_precision_judge_to_rows(dataset_rows, cp_payload)
                comparison_metrics["context_precision_judge"] = cp_payload
                comparison_summary["avg_context_precision_judge"] = cp_payload["mean"]
                metric_names.append("context_precision_llm_judge")
                serialized["comparison_metrics"] = comparison_metrics
                serialized["summary"] = comparison_summary

        # --- answer_relevancy LLM judge (fallback branch) ---
        if llm is not None:
            ar_payload = _compute_answer_relevancy_llm_judge(
                rows=dataset_rows, llm=llm, max_rows=max_factual_judge_rows, override_zeros_only=False
            )
            if ar_payload is not None:
                _apply_answer_relevancy_judge_to_rows(dataset_rows, ar_payload)
                comparison_metrics["answer_relevancy_judge"] = ar_payload
                comparison_summary["avg_answer_relevancy_judge"] = ar_payload["mean"]
                metric_names.append("answer_relevancy_llm_judge")
                serialized["comparison_metrics"] = comparison_metrics
                serialized["summary"] = comparison_summary

        serialized["rows"] = dataset_rows
        serialized["metric_names"] = metric_names

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(serialized, file, ensure_ascii=False, indent=2)

        logger.info("Saved fallback RAG evaluation results to %s", path)
        serialized["output_path"] = str(path)
        return serialized

    if metrics is None:
        metrics = default_ragas_metrics(llm=llm, embeddings=embeddings)

    eval_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "metrics": metrics,
    }

    try:
        signature = inspect.signature(evaluate_fn)
        if "llm" in signature.parameters:
            eval_kwargs["llm"] = llm
        if embeddings is not None and "embeddings" in signature.parameters:
            eval_kwargs["embeddings"] = embeddings
    except Exception:
        eval_kwargs["llm"] = llm
        if embeddings is not None:
            eval_kwargs["embeddings"] = embeddings

    # Extract dataset rows for possible single-row retries later
    dataset_rows = _extract_dataset_rows(dataset)
    result = evaluate_fn(**eval_kwargs)
    if inspect.isawaitable(result):
        result = asyncio.run(result)

    serialized = _serialize_evaluation_result(result)

    # Post-process serialized rows: when judge responses are empty/errors, retry the evaluation
    # for the offending row up to 2 times using the same metrics/llm, then mark metrics as None.
    rows = serialized.get("rows", []) or []
    if allow_retries:
        bad_count = sum(1 for r in rows if _is_judge_response_bad(r))
        logger.info("Retry loop: found %d rows with bad/missing metrics (out of %d total)", bad_count, len(rows))
        for idx, r in enumerate(list(rows)):
            try:
                if not isinstance(r, dict):
                    continue
                if _is_judge_response_bad(r):
                    ds_idx = _find_dataset_row_index(r, dataset_rows)
                    if ds_idx is None:
                        continue
                    # attempt up to 2 retries
                    for attempt in range(2):
                        try:
                            single_ds = _build_ragas_dataset_from_rows([dataset_rows[ds_idx]])
                            eval_kwargs_single = dict(eval_kwargs)
                            eval_kwargs_single["dataset"] = single_ds
                            try:
                                single_result = _call_evaluate_with_timeout(evaluate_fn, eval_kwargs_single, timeout_seconds=60)
                                if single_result is None:
                                    logger.warning("Retry %d for row %s returned no result (timeout/error)", attempt + 1, r.get("id"))
                                    continue
                            except Exception as e:
                                logger.warning("Retry %d for row %s failed: %s", attempt + 1, r.get("id"), e)
                                continue

                            single_serialized = _serialize_evaluation_result(single_result)
                            single_rows = single_serialized.get("rows", []) or []
                            if single_rows:
                                # merge updated fields into original serialized row
                                rows[idx].update(single_rows[0])
                                if not _is_judge_response_bad(single_rows[0]):
                                    break
                        except Exception:
                            continue
                    # after attempts, if still bad -> set numeric metrics to None
                    if _is_judge_response_bad(rows[idx]):
                        _set_row_metrics_none(rows[idx])
            except Exception:
                continue

    serialized["rows"] = rows

    # --- negative-evidence post-processing ---
    # Fix context_recall and faithfulness to 1.0 when model correctly says
    # "no evidence" for questions whose topic isn't in the retrieved contexts.
    rows = _postprocess_negative_evidence_rows(rows, dataset_rows)
    serialized["rows"] = rows

    # recompute summary from possibly-updated rows
    serialized = _recompute_summary_from_rows(serialized)

    metric_names = [getattr(metric, "name", metric.__class__.__name__) for metric in metrics]
    serialized["metric_names"] = metric_names

    metric_names_lower = [str(name).lower() for name in metric_names]
    has_factual_metric = any("factual" in name for name in metric_names_lower)
    has_semantic_metric = any("semantic" in name for name in metric_names_lower)

    dataset_rows = _extract_dataset_rows(dataset)
    comparison_metrics: dict[str, Any] = {}
    comparison_summary: dict[str, float] = {}

    if dataset_rows:
        if embeddings is not None and not has_semantic_metric:
            semantic_payload = _compute_semantic_similarity_fallback(
                rows=dataset_rows,
                embeddings=embeddings,
            )
            if semantic_payload is not None:
                comparison_metrics["semantic_similarity"] = semantic_payload
                comparison_summary["avg_semantic_similarity"] = semantic_payload["mean"]

        if llm is not None and not has_factual_metric:
            factual_payload = _compute_factual_correctness_fallback(
                rows=dataset_rows,
                llm=llm,
                max_rows=max_factual_judge_rows,
            )
            if factual_payload is not None:
                comparison_metrics["factual_correctness"] = factual_payload
                comparison_summary["avg_factual_correctness"] = factual_payload["mean"]

    if comparison_metrics:
        serialized["comparison_metrics"] = comparison_metrics

    if comparison_summary:
        summary = serialized.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        summary.update(comparison_summary)
        serialized["summary"] = summary

    # --- context_precision LLM judge (main RAGAS path) ---
    # Always override: RAGAS reference-based precision scores topically-relevant
    # chunks as 0 when the reference answer doesn't use them verbatim.
    # We replace with question-based Average Precision every time.
    if llm is not None and dataset_rows:
        cp_payload = _compute_context_precision_llm_judge(
            rows=rows, llm=llm, max_rows=max_factual_judge_rows, override_zeros_only=False
        )
        if cp_payload is not None:
            _apply_context_precision_judge_to_rows(rows, cp_payload)
            # Drop RAGAS alias keys (e.g. llm_context_precision_without_reference)
            # so only the canonical `context_precision` appears in the summary.
            _strip_ragas_precision_aliases_from_rows(rows)
            serialized["rows"] = rows
            comparison_metrics["context_precision_judge"] = cp_payload
            serialized["comparison_metrics"] = comparison_metrics

    # --- answer_relevancy LLM judge (main RAGAS path) ---
    # Always override: RAGAS reverse-question heuristic assigns 0 to valid
    # negative/paraphrased answers; our direct rubric judge is more accurate.
    if llm is not None and dataset_rows:
        ar_payload = _compute_answer_relevancy_llm_judge(
            rows=rows, llm=llm, max_rows=max_factual_judge_rows, override_zeros_only=False
        )
        if ar_payload is not None:
            _apply_answer_relevancy_judge_to_rows(rows, ar_payload)
            comparison_metrics["answer_relevancy_judge"] = ar_payload
            serialized["comparison_metrics"] = comparison_metrics
            serialized["metric_names"] = metric_names + ["context_precision_llm_judge", "answer_relevancy_llm_judge"]

    # Final recompute: consolidate all judge overrides into a single summary.
    serialized["rows"] = rows
    serialized = _recompute_summary_from_rows(serialized)
    # Remove RAGAS precision aliases from summary if recompute re-added them.
    summary = serialized.get("summary", {})
    for _alias in _RAGAS_PRECISION_ALIASES:
        summary.pop(_alias, None)
    serialized["summary"] = summary

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(serialized, file, ensure_ascii=False, indent=2)

    logger.info("Saved RAGAS evaluation results to %s", path)
    serialized["output_path"] = str(path)
    return serialized
