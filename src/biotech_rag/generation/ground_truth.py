"""Ground-truth generation helpers for clinical trial QA."""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from biotech_rag.generation.llm_parsers import parse_llm_output


class DraftAnswer(BaseModel):
    """Schema for extractive draft answers."""

    answer: str = Field(description="Concise, factual answer with quoted evidence")
    citations: list[str] = Field(description="Chunk citations with exact quotes")
    confidence: str = Field(description="high/medium/low")
    missing_info: str = Field(description="Any gaps in contexts?")


class RefinedAnswer(BaseModel):
    """Schema for expert-refined answers."""

    answer: str = Field(description="Expert-refined answer with quoted evidence")
    citations: list[str] = Field(description="Chunk citations with exact quotes")
    confidence: str = Field(description="high/medium/low")
    missing_info: str = Field(description="Any gaps in contexts?")


STRICT_JSON_RULES = (
    "Return ONLY a valid JSON object matching the schema. "
    "Do not use Markdown, bullet points, or code fences."
)

NEGATIVE_ANSWER_PATTERN = re.compile(
    r"\b(not mentioned|no evidence|not reported|not stated|not provided|"
    r"not described|not available|not found|absent|no indication)\b",
    re.IGNORECASE,
)
NEGATIVE_CONTEXT_PHRASE = re.compile(
    r"not mentioned in (?:these|the provided) contexts",
    re.IGNORECASE,
)


def build_ground_truth_chains(llm: Any) -> dict[str, Any]:
    """Build draft/refine prompts, parsers, and chains.

    Args:
        llm: LangChain-compatible chat LLM

    Returns:
        Dict with parsers, prompts, and runnable chains.
    """
    draft_parser = JsonOutputParser(pydantic_object=DraftAnswer)
    refine_parser = JsonOutputParser(pydantic_object=RefinedAnswer)

    draft_prompt = ChatPromptTemplate.from_template(
        "Role: Clinical trial extraction expert.\n\n"
        "Retrieved Contexts (use ONLY these):\n"
        "{retrieved_chunks}\n\n"
        "Question: {question}\n\n"
        "Output JSON:\n"
        "{format_instructions}\n\n"
        f"{STRICT_JSON_RULES}\n\n"
        "If the answer is negative, explicitly say 'not mentioned in these contexts' and cite the most relevant chunk(s). "
        "If no relevant evidence exists, return answer='N/A', confidence='low', and explain in missing_info. "
        "Never infer, summarize, or add external knowledge."
    )

    refine_prompt = ChatPromptTemplate.from_template(
        "Role: Clinical trial extraction expert.\n\n"
        "Retrieved Contexts (use ONLY these):\n"
        "{retrieved_chunks}\n\n"
        "Question: {question}\n\n"
        "Expert-Reviewed Draft (may include corrections):\n"
        "{revised_draft}\n\n"
        "Output JSON:\n"
        "{format_instructions}\n\n"
        f"{STRICT_JSON_RULES}\n\n"
        "Output only verifiable facts from provided contexts; flag uncertainties in missing_info. "
        "If the answer is negative, explicitly say 'not mentioned in these contexts' and cite the most relevant chunk(s). "
        "If no relevant evidence exists, return answer='N/A', confidence='low', and explain in missing_info. "
        "Never infer, summarize, or add external knowledge."
    )

    draft_chain = draft_prompt | llm | draft_parser
    refine_chain = refine_prompt | llm | refine_parser

    return {
        "draft_parser": draft_parser,
        "refine_parser": refine_parser,
        "draft_prompt": draft_prompt,
        "refine_prompt": refine_prompt,
        "draft_chain": draft_chain,
        "refine_chain": refine_chain,
        "draft_chain_raw": draft_prompt | llm,
        "refine_chain_raw": refine_prompt | llm,
    }


def invoke_chain_with_retries(
    chain_raw: Any,
    payload: dict[str, Any],
    max_retries: int = 2,
    min_citations: int = 0,
    allow_empty_answer: bool = True,
    allow_negative_without_citations: bool = True,
    sleep_seconds: float = 0.5,
) -> dict[str, object]:
    """Invoke a raw chain with retries and robust parsing.

    This wrapper is designed for "draft" and "refine" chains that return raw LLM
    messages. It parses the output using :func:`parse_llm_output` and retries
    when the output fails basic quality checks (e.g., missing answer or citations).

    Args:
        chain_raw: LangChain runnable (prompt | llm).
        payload: Input variables for the prompt template.
        max_retries: Number of retry attempts after the first call.
        min_citations: Minimum number of citations required when an answer is present.
        allow_empty_answer: If False, retries when answer is empty or "N/A".
        sleep_seconds: Delay between retries to reduce burstiness.

    Returns:
        Parsed output dict with keys: answer, citations, confidence, missing_info.
    """

    def _is_negative(answer_text: str) -> bool:
        return bool(NEGATIVE_ANSWER_PATTERN.search(answer_text))

    def _has_negative_phrase(answer_text: str) -> bool:
        return bool(NEGATIVE_CONTEXT_PHRASE.search(answer_text))

    def _annotate_negative_without_citations(parsed: dict[str, object]) -> dict[str, object]:
        parsed["confidence"] = "low"
        note = (
            "Negative answer without citations; model indicates 'not mentioned in these contexts'."
        )
        missing_info = str(parsed.get("missing_info") or "").strip()
        parsed["missing_info"] = f"{missing_info} | {note}" if missing_info else note
        return parsed

    def _fallback_no_evidence(reason: str) -> dict[str, object]:
        base = "No relevant evidence found in provided contexts."
        msg = f"{base} {reason}".strip()
        return {
            "answer": "N/A",
            "citations": [],
            "confidence": "low",
            "missing_info": msg,
        }

    def _needs_retry(parsed: dict[str, object]) -> tuple[bool, str, dict[str, object]]:
        answer = str(parsed.get("answer") or "").strip()
        citations = parsed.get("citations") or []
        has_answer = bool(answer) and answer.upper() != "N/A"

        if not allow_empty_answer and (not answer or answer.upper() == "N/A"):
            return True, "Empty answer.", parsed

        if min_citations > 0 and has_answer and len(citations) < min_citations:
            if (
                allow_negative_without_citations
                and _is_negative(answer)
                and _has_negative_phrase(answer)
            ):
                return False, "", _annotate_negative_without_citations(parsed)
            return True, "Missing citations for non-empty answer.", parsed

        return False, "", parsed

    last_parsed: dict[str, object] = {
        "answer": "N/A",
        "citations": [],
        "confidence": "low",
        "missing_info": "Empty output.",
    }
    last_reason = ""

    for attempt in range(max_retries + 1):
        raw = chain_raw.invoke(payload)
        raw_text = raw.content if hasattr(raw, "content") else str(raw)
        last_parsed = parse_llm_output(raw_text)
        needs_retry, reason, last_parsed = _needs_retry(last_parsed)
        last_reason = reason or last_reason

        if not needs_retry:
            return last_parsed

        if attempt < max_retries and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if last_reason:
        return _fallback_no_evidence(f"Retries exhausted. {last_reason}")
    missing_info = str(last_parsed.get("missing_info") or "").strip()
    last_parsed["missing_info"] = (
        f"{missing_info} | Retries exhausted." if missing_info else "Retries exhausted."
    )
    return last_parsed
