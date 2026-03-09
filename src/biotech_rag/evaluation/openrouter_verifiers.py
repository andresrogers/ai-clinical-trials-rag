"""OpenRouter-based NLI and QA verifiers used for claim verification.

These functions call the project's OpenRouter LLM wrapper to perform
lightweight NLI and extractive-QA checks, returning structured JSON-like
results suitable for downstream scoring. This avoids local HF model
downloads and torch dependency when configured.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _call_llm_text(llm: Any, prompt: str) -> str:
    """Call the provided LLM client and return a raw text response.

    The helper tries a few common interfaces used across the repo: `invoke()`
    (LangChain-style), callable LLMs, and `generate`/`chat` helpers.
    """
    # Prefer .invoke(...) -> object with .content
    try:
        if hasattr(llm, "invoke"):
            resp = llm.invoke(prompt)
            # Some wrappers return an object with `.content` attribute
            if hasattr(resp, "content"):
                return resp.content
            # Or return a raw string
            if isinstance(resp, str):
                return resp
            # Try to extract from common dict shapes
            if isinstance(resp, dict):
                # OpenRouter-like responses
                try:
                    return resp["choices"][0]["message"]["content"]
                except Exception:
                    return json.dumps(resp)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("llm.invoke failed: %s", exc)

    # Callable LLM objects (e.g., simple wrappers)
    try:
        if callable(llm):
            out = llm(prompt)
            if isinstance(out, str):
                return out
            if isinstance(out, dict):
                return out.get("text") or out.get("content") or json.dumps(out)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("callable llm failed: %s", exc)

    # Try other common attributes
    for attr in ("generate", "chat", "complete", "call"):
        fn = getattr(llm, attr, None)
        if fn:
            try:
                out = fn(prompt)
                if isinstance(out, str):
                    return out
                if isinstance(out, dict):
                    # Try to extract text-like fields
                    if "text" in out:
                        return out["text"]
                    if "output" in out:
                        return out["output"]
                    if "choices" in out and out["choices"]:
                        choice = out["choices"][0]
                        if isinstance(choice, dict):
                            return (
                                choice.get("text")
                                or (choice.get("message") or {}).get("content")
                                or json.dumps(choice)
                            )
                return str(out)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("llm.%s failed: %s", attr, exc)

    raise RuntimeError("Unable to call LLM: unsupported interface")


def nli_openrouter(llm: Any, premise: str, hypothesis: str) -> Dict[str, Any]:
    """Run NLI via remote LLM and return structured result.

    Returns a dict with keys: `label` (ENTAILMENT|CONTRADICTION|NEUTRAL),
    `confidence` (float), and `raw` (raw LLM text).
    """
    prompt = (
        "You are a strict NLI verifier. Given PREMISE and HYPOTHESIS, decide "
        "whether the PREMISE ENTAILS the HYPOTHESIS, CONTRADICTS it, or is NEUTRAL. "
        "Answer only valid JSON with keys \"label\" and \"confidence\" (0.0-1.0).\n\n"
        f"PREMISE: \"\"\"{premise}\"\"\"\n\nHYPOTHESIS: \"\"\"{hypothesis}\"\"\"\n\n"
        "Return exactly: {\"label\":\"ENTAILMENT\"|\"CONTRADICTION\"|\"NEUTRAL\",\"confidence\":0.00}"
    )

    raw = _call_llm_text(llm, prompt)
    try:
        # Try to parse JSON from the last line (robust to chatter)
        parsed = json.loads(raw.strip().splitlines()[-1])
        label = parsed.get("label", "").upper()
        confidence = float(parsed.get("confidence", 0.0))
        return {"label": label, "confidence": confidence, "raw": raw}
    except Exception:
        text = raw.strip().upper()
        for lab in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"):
            if lab in text:
                return {"label": lab, "confidence": 0.5, "raw": raw}
        return {"label": "NEUTRAL", "confidence": 0.0, "raw": raw}


def qa_openrouter(llm: Any, question: str, context: str) -> Dict[str, Any]:
    """Run extractive QA via remote LLM.

    Returns: dict with `answer` (str), `confidence` (0-1 float), `answerable` (bool), and `raw`.
    """
    prompt = (
        "You are an extractive question-answering assistant. Given CONTEXT and a QUESTION, "
        "decide whether the question can be answered from the context. If yes, return JSON with keys "
        '"answer" (short), "confidence" (0.0-1.0) and "answerable": true. If not answerable, return '
        '{"answer":"", "confidence":0.0, "answerable":false}. Return exactly JSON only.\n\n'
        f"CONTEXT: '''{context}'''\n\nQUESTION: '''{question}'''\n\nReturn JSON:"
    )

    raw = _call_llm_text(llm, prompt)
    try:
        parsed = json.loads(raw.strip().splitlines()[-1])
        return {**{"answer": "", "confidence": 0.0, "answerable": False, "raw": raw}, **parsed}
    except Exception:
        text = raw.strip()
        answerable = not any(tok in text.lower() for tok in ("cannot", "no", "not enough", "unknown"))
        return {
            "answer": text if answerable else "",
            "confidence": 0.5 if answerable else 0.0,
            "answerable": answerable,
            "raw": raw,
        }


def find_best_supporting_chunk_openrouter(
    claim: str, chunks: List[Dict[str, Any]], llm: Any, method: str = "hybrid"
) -> Tuple[int, Dict[str, Any]]:
    """Find best supporting chunk using remote QA+NLI.

    Args:
        claim: claim text
        chunks: list of chunk dicts (may contain 'text' or 'content')
        llm: OpenRouter LLM wrapper
        method: scoring method (kept for API parity)

    Returns:
        (best_idx, details) where details contains 'supported' bool, 'score', and raw outputs.
    """
    best_idx = -1
    best_score = 0.0
    best_details: Dict[str, Any] = {"supported": False}

    for i, chunk in enumerate(chunks):
        text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)
        qa_res = qa_openrouter(llm, claim, text)
        nli_res = nli_openrouter(llm, text, claim)
        score = 0.0
        if nli_res["label"] == "ENTAILMENT":
            score = max(score, nli_res.get("confidence", 0.0))
        if qa_res.get("answerable"):
            score = max(score, qa_res.get("confidence", 0.0))
        if score > best_score:
            best_score = score
            best_idx = i
            best_details = {"qa": qa_res, "nli": nli_res, "score": score, "supported": score >= 0.5}

    return best_idx, best_details
