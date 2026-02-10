"""
Structured extraction helpers for enriching clinical trial data via RAG.

Provides schema, prompts, and chains for extracting structured fields from retrieved chunks.
Includes normalization and verification integration.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from biotech_rag.evaluation.claim_verification import verify_claim_hybrid
from biotech_rag.generation.llm_clients import get_openrouter_llm
from biotech_rag.generation.llm_parsers import parse_llm_output
from biotech_rag.indexing.embedders import Embedder
from biotech_rag.indexing.vectorstore import init_chroma
from biotech_rag.retrieval.context_retrieval import retrieve_chunks

try:
    from biotech_rag.config import settings
except ImportError:
    settings = None

try:
    from fuzzywuzzy import process

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("fuzzywuzzy not available; drug normalization will use title case fallback.")

logger = logging.getLogger(__name__)

# Cache for FDA drug names
_FDA_DRUG_NAMES = None


def _load_fda_drug_names() -> list[str]:
    """Load and cache FDA drug names from labels."""
    global _FDA_DRUG_NAMES
    if _FDA_DRUG_NAMES is not None:
        return _FDA_DRUG_NAMES

    names = set()
    fda_labels_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "fda_labels"
    if not fda_labels_dir.exists():
        logger.warning(f"FDA labels directory not found: {fda_labels_dir}")
        return []

    for json_file in fda_labels_dir.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            if "results" in data and isinstance(data["results"], list):
                for result in data["results"]:
                    # Extract brand_name, generic_name, substance_name if available
                    for key in ["brand_name", "generic_name", "substance_name"]:
                        if key in result and isinstance(result[key], str):
                            names.add(result[key].lower().strip())
                    # Also extract from indications_and_usage or description if fields not present
                    if "indications_and_usage" in result and isinstance(
                        result["indications_and_usage"], list
                    ):
                        text = " ".join(result["indications_and_usage"])
                        # Extract drug names from text (simple heuristic: capitalize words after 'is' or 'called')
                        # For now, add filename-based name
                        filename = json_file.stem.replace("_label", "")
                        names.add(filename.lower())
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    _FDA_DRUG_NAMES = sorted(list(names))
    logger.info(f"Loaded {len(_FDA_DRUG_NAMES)} unique FDA drug names.")
    return _FDA_DRUG_NAMES


def canonicalize_drug_name(extracted_name: str) -> str:
    """Canonicalize drug name using fuzzy matching against FDA labels."""
    if not extracted_name or not isinstance(extracted_name, str):
        return extracted_name

    if not FUZZY_AVAILABLE:
        return extracted_name.title()

    fda_names = _load_fda_drug_names()
    if not fda_names:
        return extracted_name.title()

    # Fuzzy match
    best_match, score = process.extractOne(extracted_name.lower(), fda_names)
    if score >= 85:
        logger.debug(f"Matched '{extracted_name}' to '{best_match}' (score: {score})")
        return best_match.title()
    else:
        logger.debug(
            f"No good match for '{extracted_name}' (best score: {score}); using title case"
        )
        return extracted_name.title()


# Extraction schema
EXTRACTION_SCHEMA = {
    "primary_outcomes": "list of dicts: [{'name': str, 'measure': str, 'timepoint': str, 'p_value': str or float}]",
    "secondary_outcomes": "same as primary_outcomes",
    "trial_status": "str: COMPLETED, TERMINATED, ACTIVE_NOT_RECRUITING, etc.",
    "success_flag": "str: LIKELY_PASS, LIKELY_FAIL, DEFINITE_FAIL, NO_RESULTS",
    "predicted_success_prob": "float: 0-1, confidence in success",
    "enrollment": "int: number of participants",
    "start_date": "str: ISO date YYYY-MM-DD",
    "completion_date": "str: ISO date YYYY-MM-DD",
    "adverse_events_summary": "str: short summary of key adverse events",
    "drug_name_normalized": "str: canonical drug name from FDA labels",
    "extraction_confidence": "float: 0-1, overall confidence",
    "provenance_chunk_ids": "list of str: chunk IDs supporting the extraction",
}

STRUCTURED_EXTRACTION_PROMPT = """
Extract structured information from the provided clinical trial chunks about the trial with NCT ID: {nct_id}.

Return ONLY valid JSON matching this schema:
{{
    "primary_outcomes": [{{ "name": "string", "measure": "string", "timepoint": "string", "p_value": "string or number" }}],
    "secondary_outcomes": [same structure],
    "trial_status": "string (e.g., COMPLETED, TERMINATED)",
    "success_flag": "string (LIKELY_PASS, LIKELY_FAIL, DEFINITE_FAIL, NO_RESULTS)",
    "predicted_success_prob": 0.0 to 1.0,
    "enrollment": integer,
    "start_date": "YYYY-MM-DD",
    "completion_date": "YYYY-MM-DD",
    "adverse_events_summary": "string summary",
    "drug_name_normalized": "string",
    "extraction_confidence": 0.0 to 1.0
}}

If a field is not available, use null or empty list/string. Be precise and cite evidence.

Retrieved Chunks:
{chunks_text}
"""


def format_chunks_for_prompt(chunks: list[str], chunk_ids: list[str]) -> str:
    """Format chunks with IDs for the prompt."""
    return "\n\n".join(f"Chunk {cid}: {text}" for cid, text in zip(chunk_ids, chunks))


def extract_structured_fields(
    nct_id: str, chunks: list[str], chunk_ids: list[str], llm=None
) -> dict[str, Any]:
    """
    Extract structured fields using LLM.

    Args:
        nct_id: The trial NCT ID.
        chunks: List of chunk texts.
        chunk_ids: Corresponding chunk IDs.
        llm: LLM client.

    Returns:
        Dict with extracted fields + provenance.
    """
    if llm is None:
        llm = get_openrouter_llm()

    chunks_text = format_chunks_for_prompt(chunks, chunk_ids)
    prompt = STRUCTURED_EXTRACTION_PROMPT.format(nct_id=nct_id, chunks_text=chunks_text)

    response = llm.invoke(prompt)
    try:
        extracted = parse_llm_output(response, expected_type=dict)
        if isinstance(extracted, dict):
            extracted["provenance_chunk_ids"] = chunk_ids
            return extracted
    except Exception as e:
        logger.error(f"Extraction failed for {nct_id}: {e}")
    return {"error": "extraction_failed", "provenance_chunk_ids": chunk_ids}


def normalize_extracted_data(extracted: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize extracted values: parse dates, numerics, canonicalize drugs.

    Args:
        extracted: Raw extracted dict.

    Returns:
        Normalized dict.
    """
    normalized = extracted.copy()

    # Parse dates
    for date_field in ["start_date", "completion_date"]:
        if date_field in normalized and isinstance(normalized[date_field], str):
            try:
                parsed = datetime.fromisoformat(normalized[date_field].replace("/", "-").split()[0])
                normalized[date_field] = parsed.date().isoformat()
            except ValueError:
                pass  # Keep as is

    # Parse numerics
    if "enrollment" in normalized:
        try:
            normalized["enrollment"] = int(normalized["enrollment"])
        except (ValueError, TypeError):
            pass

    if "predicted_success_prob" in normalized:
        try:
            normalized["predicted_success_prob"] = float(normalized["predicted_success_prob"])
        except (ValueError, TypeError):
            pass

    # Parse p-values in outcomes
    for outcome_list in ["primary_outcomes", "secondary_outcomes"]:
        if outcome_list in normalized and isinstance(normalized[outcome_list], list):
            for outcome in normalized[outcome_list]:
                if isinstance(outcome, dict) and "p_value" in outcome:
                    p_val = outcome["p_value"]
                    if isinstance(p_val, str):
                        # Extract numeric, e.g., "0.05" from "<0.05"
                        match = re.search(r"[\d.]+", p_val)
                        if match:
                            try:
                                outcome["p_value"] = float(match.group())
                            except ValueError:
                                pass

    # Drug normalization: fuzzy match to FDA labels
    if "drug_name_normalized" in normalized:
        normalized["drug_name_normalized"] = canonicalize_drug_name(
            normalized["drug_name_normalized"]
        )

    return normalized


def verify_extracted_fields(
    extracted: dict[str, Any], chunks: list[str], llm=None
) -> dict[str, Any]:
    """
    Verify extracted fields by decomposing into claims and checking support.

    Args:
        extracted: Normalized extracted data.
        chunks: Retrieved chunks.
        llm: LLM for claim decomposition.

    Returns:
        Dict with verification_score (avg support for claims).
    """
    verification_scores = []

    # Flatten extracted fields into claims
    claims = []
    for key, value in extracted.items():
        if key in ["primary_outcomes", "secondary_outcomes"] and isinstance(value, list):
            for outcome in value:
                claims.append(
                    f"{key[:-9]} outcome: {outcome.get('name', '')} with p-value {outcome.get('p_value', '')}"
                )
        elif isinstance(value, str) and value:
            claims.append(f"{key}: {value}")
        elif isinstance(value, (int, float)):
            claims.append(f"{key}: {value}")

    for claim in claims[
        : (settings.max_claims_per_answer if settings else 10)
    ]:  # Limit for efficiency
        supported = any(verify_claim_hybrid(claim, chunk)["supported"] for chunk in chunks)
        verification_scores.append(1.0 if supported else 0.0)

    avg_score = sum(verification_scores) / len(verification_scores) if verification_scores else 0.0
    extracted["verification_score"] = avg_score
    return extracted


def enrich_trial_data(
    nct_id: str, vstore_dir: Path, embedder: Embedder, top_k: int = 20
) -> dict[str, Any]:
    """
    Full enrichment pipeline for one trial.

    Args:
        nct_id: Trial NCT ID.
        vstore_dir: Vectorstore path.
        embedder: Embedder instance.
        top_k: Number of chunks to retrieve.

    Returns:
        Enriched dict for the trial.
    """
    client, collection = init_chroma(vstore_dir, collection_name="clinical_trials")

    # Retrieve chunks for this NCT
    question = f"Extract key information about clinical trial {nct_id}"
    results = retrieve_chunks(
        collection=collection, embedder=embedder, question=question, nct_id=nct_id, top_k=top_k
    )
    chunks = results.get("documents", [[]])[0]
    chunk_ids = results.get("ids", [[]])[0]

    if not chunks:
        return {"nct_id": nct_id, "error": "no_chunks_retrieved"}

    # Extract
    extracted = extract_structured_fields(nct_id, chunks, chunk_ids)

    # Normalize
    normalized = normalize_extracted_data(extracted)

    # Verify
    verified = verify_extracted_fields(normalized, chunks)

    verified["nct_id"] = nct_id
    return verified


def save_enriched_csv(enriched_records: list[dict], output_path: Path):
    """
    Save enriched records to CSV with key scalars and derived fields.
    Nested data (e.g., outcomes) is kept in JSON; CSV has summaries.

    Args:
        enriched_records: List of enriched dicts.
        output_path: Output CSV path.
    """
    # Prepare CSV rows with only key fields
    csv_rows = []
    for rec in enriched_records:
        row = {}

        # Scalars
        for key in [
            "nct_id",
            "trial_status",
            "success_flag",
            "predicted_success_prob",
            "enrollment",
            "start_date",
            "completion_date",
            "extraction_confidence",
            "verification_score",
            "drug_name_normalized",
        ]:
            row[key] = rec.get(key, None)

        # Derived counts and means
        primary_outcomes = rec.get("primary_outcomes", [])
        secondary_outcomes = rec.get("secondary_outcomes", [])

        row["primary_outcome_count"] = (
            len(primary_outcomes) if isinstance(primary_outcomes, list) else 0
        )
        row["secondary_outcome_count"] = (
            len(secondary_outcomes) if isinstance(secondary_outcomes, list) else 0
        )

        # Mean p-values
        def mean_p_value(outcomes):
            if not isinstance(outcomes, list):
                return None
            p_values = []
            for o in outcomes:
                if isinstance(o, dict) and "p_value" in o:
                    p = o["p_value"]
                    if isinstance(p, (int, float)):
                        p_values.append(p)
                    elif isinstance(p, str):
                        # Try to parse
                        try:
                            p_values.append(float(p))
                        except ValueError:
                            pass
            return sum(p_values) / len(p_values) if p_values else None

        row["mean_primary_p_value"] = mean_p_value(primary_outcomes)
        row["mean_secondary_p_value"] = mean_p_value(secondary_outcomes)

        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(csv_rows)} enriched records to {output_path}")


# Import pandas here to avoid circular
import pandas as pd
