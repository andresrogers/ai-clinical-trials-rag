"""
Evaluation helpers for RAG factuality and verification.

This module provides functions to decompose answers into atomic claims,
verify claims against retrieved context using NLI entailment and QA-based methods.
"""

import logging

# Core imports
from biotech_rag.generation.llm_clients import get_openrouter_llm
from biotech_rag.generation.llm_parsers import parse_llm_output

# For NLI and QA (local models)
try:
    from transformers import pipeline

    NLI_PIPELINE = pipeline(
        "text-classification", model="facebook/bart-large-mnli", return_all_scores=True
    )
    QA_PIPELINE = pipeline("question-answering", model="deepset/roberta-base-squad2")
except ImportError:
    logging.warning(
        "Transformers not available for local NLI/QA. Install with: pip install transformers torch"
    )
    NLI_PIPELINE = None
    QA_PIPELINE = None

logger = logging.getLogger(__name__)

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
        claims = parse_llm_output(response, expected_type=list)
        return claims if isinstance(claims, list) else []
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
    if NLI_PIPELINE is None:
        return {"entailment_score": 0.0, "label": "unknown", "confidence": 0.0}

    try:
        results = NLI_PIPELINE(f"premise: {context}", f"hypothesis: {claim}")
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
    if QA_PIPELINE is None:
        return {"answer": "", "confidence": 0.0, "start": 0, "end": 0}

    # Formulate a question from the claim (simple heuristic: prepend "What is" or similar)
    question = (
        f"What is {claim.lower()}?"
        if not claim.startswith(("What", "Is", "Does", "Are"))
        else claim
    )

    try:
        result = QA_PIPELINE(question=question, context=context)
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
        if method == "nli":
            result = verify_claim_nli(claim, chunk)
            score = result["entailment_score"]
        elif method == "qa":
            result = verify_claim_qa(claim, chunk)
            score = result["confidence"]
        else:  # hybrid
            result = verify_claim_hybrid(claim, chunk)
            score = result["nli_result"]["entailment_score"] if result["supported"] else 0.0

        if score > best_score:
            best_score = score
            best_idx = i
            best_result = result

    return best_idx, best_result
