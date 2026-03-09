"""
Evaluation helpers for RAG factuality and verification.

This module provides functions to decompose answers into atomic claims,
verify claims against retrieved context using NLI entailment and QA-based methods.
"""

import logging

# Core imports
from biotech_rag.generation.llm_clients import get_openrouter_llm
from biotech_rag.generation.llm_parsers import parse_llm_output
from biotech_rag.config import settings
from biotech_rag.evaluation.openrouter_verifiers import (
    nli_openrouter,
    qa_openrouter,
    find_best_supporting_chunk_openrouter,
)

# Local HF pipelines are initialised lazily so that merely importing this
# module does NOT trigger a HuggingFace model download.  When
# USE_OPENROUTER_VERIFIER is True (the default) these are never called.
_NLI_PIPELINE = None
_QA_PIPELINE = None
_TRANSFORMERS_AVAILABLE: bool | None = None  # None = not yet probed


def _get_nli_pipeline():
    """Return the cached NLI pipeline, initialising it on first call."""
    global _NLI_PIPELINE, _TRANSFORMERS_AVAILABLE
    if _NLI_PIPELINE is not None:
        return _NLI_PIPELINE
    if _TRANSFORMERS_AVAILABLE is False:
        return None
    try:
        from transformers import pipeline  # noqa: PLC0415

        _NLI_PIPELINE = pipeline(
            "text-classification", model="facebook/bart-large-mnli", return_all_scores=True
        )
        _TRANSFORMERS_AVAILABLE = True
        return _NLI_PIPELINE
    except Exception:
        logging.warning(
            "Transformers not available for local NLI. Install with: pip install transformers torch"
        )
        _TRANSFORMERS_AVAILABLE = False
        return None


def _get_qa_pipeline():
    """Return the cached QA pipeline, initialising it on first call."""
    global _QA_PIPELINE, _TRANSFORMERS_AVAILABLE
    if _QA_PIPELINE is not None:
        return _QA_PIPELINE
    if _TRANSFORMERS_AVAILABLE is False:
        return None
    try:
        from transformers import pipeline  # noqa: PLC0415

        _QA_PIPELINE = pipeline("question-answering", model="deepset/roberta-base-squad2")
        _TRANSFORMERS_AVAILABLE = True
        return _QA_PIPELINE
    except Exception:
        logging.warning(
            "Transformers not available for local QA. Install with: pip install transformers torch"
        )
        _TRANSFORMERS_AVAILABLE = False
        return None


# Backward-compat shims so any code that checks `NLI_PIPELINE is None` still works.
NLI_PIPELINE = None  # kept for API compat; real pipeline is behind _get_nli_pipeline()
QA_PIPELINE = None   # kept for API compat; real pipeline is behind _get_qa_pipeline()

logger = logging.getLogger(__name__)

# Use OpenRouter verifier by default unless explicitly disabled in settings
USE_OPENROUTER_VERIFIER: bool = True if settings is None else getattr(settings, "use_openrouter_verifier", True)

CLAIM_DECOMPOSITION_PROMPT = """
Given the following answer text, decompose it into a list of atomic, verifiable claims.
Each claim should be a single, independent statement that can be checked against evidence.
Return ONLY a JSON array of strings, like: ["claim1", "claim2", ...]

Answer text:
{answer_text}
"""


def decompose_claims(answer_text: str, llm=None) -> list[str]:
    """
    Decompose an answer into atomic claims using LLM.

    Args:
        answer_text: The full answer text to decompose.
        llm: LLM client (defaults to OpenRouter).

    Returns:
        List of atomic claims as strings.
    """
    if llm is None:
        llm = get_openrouter_llm()

    prompt = CLAIM_DECOMPOSITION_PROMPT.format(answer_text=answer_text.strip())
    response = llm.invoke(prompt)
    try:
        parsed = parse_llm_output(response)
        # Extract claims list from parsed dict; try common keys
        claims = parsed.get("claims") or parsed.get("claim_list") or list(parsed.values())[0] if parsed else []
        return claims if isinstance(claims, list) else [str(c) for c in claims] if claims else []
    except Exception as e:
        logger.error(f"Failed to parse claims from LLM response: {e}")
        return []


def verify_claim_nli(claim: str, context: str) -> dict:
    """
    Verify a claim against context using NLI entailment.

    Args:
        claim: The claim to verify.
        context: The context text to check against.

    Returns:
        Dict with 'entailment_score', 'label' (entailment/contradiction/neutral), 'confidence'.
    """
    # Short-circuit: when OpenRouter is the verifier, skip the HF pipeline entirely.
    # _get_nli_pipeline() is ONLY called when USE_OPENROUTER_VERIFIER is False to avoid
    # triggering HuggingFace model downloads during normal inference.
    if USE_OPENROUTER_VERIFIER:
        try:
            llm = get_openrouter_llm()
            nli = nli_openrouter(llm, context, claim)
            label = nli.get("label", "NEUTRAL").lower()
            confidence = float(nli.get("confidence", 0.0))
            entailment_score = confidence if label in ("entailment", "ENTAILMENT") else 0.0
            return {"entailment_score": entailment_score, "label": label, "confidence": confidence}
        except Exception as e:
            logger.error(f"OpenRouter NLI verification failed: {e}")
            return {"entailment_score": 0.0, "label": "error", "confidence": 0.0}

    nli_pipeline = _get_nli_pipeline()
    if nli_pipeline is None:
        try:
            llm = get_openrouter_llm()
            nli = nli_openrouter(llm, context, claim)
            label = nli.get("label", "NEUTRAL").lower()
            confidence = float(nli.get("confidence", 0.0))
            entailment_score = confidence if label == "ENTAILMENT" or label == "entailment" else 0.0
            return {"entailment_score": entailment_score, "label": label, "confidence": confidence}
        except Exception as e:
            logger.error(f"OpenRouter NLI verification failed: {e}")
            return {"entailment_score": 0.0, "label": "error", "confidence": 0.0}

    # Fallback to local HF pipeline (only reached when USE_OPENROUTER_VERIFIER is False)
    try:
        results = nli_pipeline(f"premise: {context}", f"hypothesis: {claim}")
        # results is list of dicts: [{'label': 'entailment', 'score': 0.8}, ...]
        entailment_score = next((r["score"] for r in results if r["label"] == "entailment"), 0.0)
        contradiction_score = next(
            (r["score"] for r in results if r["label"] == "contradiction"), 0.0
        )
        neutral_score = next((r["score"] for r in results if r["label"] == "neutral"), 0.0)

        # Label with highest score
        scores = {
            "entailment": entailment_score,
            "contradiction": contradiction_score,
            "neutral": neutral_score,
        }
        label = max(scores, key=scores.get)
        confidence = scores[label]

        return {"entailment_score": entailment_score, "label": label, "confidence": confidence}
    except Exception as e:
        logger.error(f"NLI verification failed: {e}")
        return {"entailment_score": 0.0, "label": "error", "confidence": 0.0}


def verify_claim_qa(claim: str, context: str) -> dict:
    """
    Verify a claim using QA-based evidence extraction.

    Args:
        claim: The claim (used to formulate a question).
        context: The context text.

    Returns:
        Dict with 'answer', 'confidence', 'start', 'end'.
    """
    # Short-circuit: when OpenRouter is the verifier, skip the HF pipeline entirely.
    if USE_OPENROUTER_VERIFIER:
        try:
            question = (
                f"What is {claim.lower()}?"
                if not claim.startswith(("What", "Is", "Does", "Are"))
                else claim
            )
            llm = get_openrouter_llm()
            qa = qa_openrouter(llm, question, context)
            return {
                "answer": qa.get("answer", ""),
                "confidence": qa.get("confidence", 0.0),
                "start": 0,
                "end": 0,
            }
        except Exception as e:
            logger.error(f"OpenRouter QA verification failed: {e}")
            return {"answer": "", "confidence": 0.0, "start": 0, "end": 0}

    qa_pipeline = _get_qa_pipeline()
    if qa_pipeline is None:
        try:
            # Build a simple question heuristic
            question = (
                f"What is {claim.lower()}?"
                if not claim.startswith(("What", "Is", "Does", "Are"))
                else claim
            )
            llm = get_openrouter_llm()
            qa = qa_openrouter(llm, question, context)
            return {
                "answer": qa.get("answer", ""),
                "confidence": qa.get("confidence", 0.0),
                "start": 0,
                "end": 0,
            }
        except Exception as e:
            logger.error(f"OpenRouter QA verification failed: {e}")
            return {"answer": "", "confidence": 0.0, "start": 0, "end": 0}

    # Fallback to local HF pipeline (only reached when USE_OPENROUTER_VERIFIER is False)
    # Formulate a question from the claim (simple heuristic: prepend "What is" or similar)
    question = (
        f"What is {claim.lower()}?"
        if not claim.startswith(("What", "Is", "Does", "Are"))
        else claim
    )

    try:
        result = qa_pipeline(question=question, context=context)
        return {
            "answer": result.get("answer", ""),
            "confidence": result.get("score", 0.0),
            "start": result.get("start", 0),
            "end": result.get("end", 0),
        }
    except Exception as e:
        logger.error(f"QA verification failed: {e}")
        return {"answer": "", "confidence": 0.0, "start": 0, "end": 0}


def verify_claim_hybrid(claim: str, context: str, nli_threshold: float = 0.8) -> dict:
    """
    Hybrid verification: combine NLI and QA, prioritize NLI for entailment.

    Args:
        claim: The claim to verify.
        context: The context.
        nli_threshold: Minimum entailment score to consider supported.

    Returns:
        Dict with 'supported' (bool), 'best_method', 'details' (NLI/QA results).
    """
    nli_result = verify_claim_nli(claim, context)
    qa_result = verify_claim_qa(claim, context)

    supported = nli_result["entailment_score"] >= nli_threshold
    best_method = "nli" if supported else "qa"

    return {
        "supported": supported,
        "best_method": best_method,
        "nli_result": nli_result,
        "qa_result": qa_result,
    }


# Utility to find best supporting chunk from a list
def find_best_supporting_chunk(
    claim: str, retrieved_chunks: list[str], method: str = "hybrid"
) -> tuple[int | None, dict]:
    """
    Find the best supporting chunk for a claim from retrieved chunks.

    Args:
        claim: The claim.
        retrieved_chunks: List of chunk texts.
        method: 'nli', 'qa', or 'hybrid'.

    Returns:
        (chunk_index, verification_result) or (None, {}) if none support.
    """
    best_idx = None
    best_result = {}
    best_score = 0.0

    for i, chunk in enumerate(retrieved_chunks):
        # Support chunk formats: raw text or dicts with 'text'/'content'
        chunk_text = chunk if isinstance(chunk, str) else chunk.get("text") or chunk.get("content") or str(chunk)

        if method == "nli":
            result = verify_claim_nli(claim, chunk_text)
            score = result.get("entailment_score", 0.0)
        elif method == "qa":
            result = verify_claim_qa(claim, chunk_text)
            score = result.get("confidence", 0.0)
        else:  # hybrid
            # If configured to use OpenRouter, call the faster combined openrouter finder
            if USE_OPENROUTER_VERIFIER:
                try:
                    llm = get_openrouter_llm()
                    best_idx, best_res = find_best_supporting_chunk_openrouter(claim, retrieved_chunks, llm, method=method)
                    return best_idx, best_res
                except Exception:
                    # Fall back to per-chunk hybrid checks
                    pass

            result = verify_claim_hybrid(claim, chunk_text)
            score = result.get("nli_result", {}).get("entailment_score", 0.0) if result.get("supported") else 0.0

        if score > best_score:
            best_score = score
            best_idx = i
            best_result = result

    return best_idx, best_result
