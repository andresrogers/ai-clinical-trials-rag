"""LLM output parsing utilities for robust JSON recovery."""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_markdown(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[`*_]+", "", text)
    text = re.sub(r"^\s*[-•]+\s*", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_section(text: str | None, label: str) -> str:
    if not text:
        return ""
    label_pattern = rf"(?:{label})"
    pattern = (
        rf"(?is){label_pattern}\s*:?\s*(.*?)\s*"
        rf"(?=\n\s*(Answer|Citations|Confidence|Missing Info|Missing Information)\s*:|\Z)"
    )
    match = re.search(pattern, str(text))
    if not match:
        return ""
    value = match.group(1)
    return value.strip() if value else ""


def _coerce_citations(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        lines = [re.sub(r"^[-•]+\s*", "", line.strip()) for line in value.splitlines()]
        return [line for line in lines if line]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_parsed_output(parsed: dict[str, Any]) -> dict[str, object]:
    def _get_key(*keys: str) -> Any:
        for key in keys:
            if key in parsed:
                return parsed[key]
        # case-insensitive match
        lower_map = {k.lower(): k for k in parsed.keys()}
        for key in keys:
            lk = key.lower()
            if lk in lower_map:
                return parsed[lower_map[lk]]
        return None

    answer = _get_key("answer")
    citations = _get_key("citations", "citation", "sources", "source_chunks")
    confidence = _get_key("confidence")
    missing_info = _get_key("missing_info", "missing information", "missing", "gaps")

    answer_str = str(answer).strip() if answer is not None else ""
    confidence_str = str(confidence).strip() if confidence is not None else ""
    missing_info_str = str(missing_info).strip() if missing_info is not None else ""
    citations_list = _coerce_citations(citations)

    if answer_str and not citations_list and not missing_info_str:
        missing_info_str = "No explicit citations returned by model."

    return {
        "answer": answer_str or "N/A",
        "citations": citations_list,
        "confidence": confidence_str or "low",
        "missing_info": missing_info_str,
    }


def parse_llm_output(raw_text: Any) -> dict[str, object]:
    """Parse LLM output into DraftAnswer fields, stripping Markdown if needed.

    Returns:
        Dict with keys: answer, citations, confidence, missing_info
    """
    if raw_text is None or raw_text == "":
        return {
            "answer": "N/A",
            "citations": [],
            "confidence": "low",
            "missing_info": "Empty output.",
        }

    raw_text_str = str(raw_text)

    # Try JSON directly
    try:
        parsed = json.loads(raw_text_str)
        if isinstance(parsed, dict):
            return _normalize_parsed_output(parsed)
    except Exception:
        pass

    # Try to locate a JSON object inside text
    try:
        json_match = re.search(r"\{.*\}", raw_text_str, flags=re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                return _normalize_parsed_output(parsed)
    except Exception:
        pass

    # Fallback: strip markdown and extract sections
    cleaned = _strip_markdown(raw_text_str)
    answer = _extract_section(cleaned, "Answer")
    citations_block = _extract_section(cleaned, "Citations")
    confidence = _extract_section(cleaned, "Confidence")
    missing_info = _extract_section(cleaned, "Missing Info|Missing Information")

    citations = _coerce_citations(citations_block)

    return _normalize_parsed_output(
        {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "missing_info": missing_info,
        }
    )


def parse_structured_json(response: Any) -> dict[str, Any]:
    """Parse structured JSON from LLM response.

    Args:
        response: LangChain AIMessage or string containing JSON.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If JSON parsing fails.
    """
    # Handle LangChain AIMessage objects
    if hasattr(response, 'content'):
        text = str(response.content)
    else:
        text = str(response)

    if not text or text.strip() == "":
        raise ValueError("Empty response")

    # Strip markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text using regex
    json_match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")
