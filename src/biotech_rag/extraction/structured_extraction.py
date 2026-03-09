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

import pandas as pd

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

logger = logging.getLogger(__name__)

try:
    from fuzzywuzzy import process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("fuzzywuzzy not available; drug normalization will use title case fallback.")

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


# Column-specific extraction prompts
# Each prompt targets one field with a focused question
COLUMN_PROMPTS = {
    "primary_outcome": (
        "In clinical trial with code {nct_id} and titled {official_title}: "
        "What is the primary outcome measure? Return only the outcome name/description as plain text."
    ),
    "primary_outcome_p_value": (
        "In clinical trial with code {nct_id} and titled {official_title}: "
        "What is the p-value for the primary outcome? Return only the numeric value (e.g., 0.042) or 'N/A' if not reported."
    ),
    "enrolled_seriously_affected": (
        "In clinical trial with code {nct_id} and titled {official_title}: "
        "How many patients experienced serious adverse events? Return only the integer count or 'N/A' if not reported."
    ),
    "enrolled_deaths": (
        "In clinical trial with code {nct_id} and titled {official_title}: "
        "How many deaths occurred during the trial? Return only the integer count or 'N/A' if not reported."
    ),
    "secondary_outcome": (
        "In clinical trial with code {nct_id} and titled {official_title}: "
        "What is the most important secondary outcome measure? Return only one outcome name/description as plain text."
    ),
    "secondary_outcome_p_value": (
        "In clinical trial with code {nct_id} and titled {official_title}: "
        "What is the p-value for the secondary outcome? Return only the numeric value or 'N/A' if not reported."
    ),
    "success_flag_extracted": (
        "In clinical trial with code {nct_id} and titled {official_title}: "
        "Based on the trial results, was this trial successful? Return one of: LIKELY_PASS, LIKELY_FAIL, DEFINITE_FAIL, NO_RESULTS."
    ),
    "intervention_name_extracted": (
        "In clinical trial with code {nct_id} and titled {official_title}: "
        "What is the primary intervention/drug name being tested? Return only the drug/intervention name."
    ),
}

# Field-specific retrieval queries (more targeted than extraction prompts)
# These are used to retrieve relevant chunks for each field
FIELD_RETRIEVAL_QUERIES = {
    "primary_outcome": "primary outcome endpoint measure",
    "primary_outcome_p_value": "primary outcome p-value statistical significance results",
    "enrolled_seriously_affected": "serious adverse events SAE grade 3 4 5",
    "enrolled_deaths": "deaths mortality adverse events fatal",
    "secondary_outcome": "secondary outcome endpoint measure",
    "secondary_outcome_p_value": "secondary outcome p-value statistical results",
    "success_flag_extracted": "trial results success efficacy met endpoint primary outcome",
    "intervention_name_extracted": "intervention treatment drug compound agent",
}

# Context template for prompts
CONTEXT_TEMPLATE = """

Relevant excerpts from trial documents:
{chunks_text}

Answer concisely based only on the provided context."""


def format_chunks_for_prompt(chunks: list[str], chunk_ids: list[str]) -> str:
    """Format chunks with IDs for the prompt."""
    return "\n\n".join(f"Chunk {cid}: {text}" for cid, text in zip(chunk_ids, chunks))


def extract_single_field(
    field_name: str,
    nct_id: str,
    official_title: str,
    llm: Any,
    hybrid_retriever: Any | None,
    embedder: Embedder,
    vstore_dir: Path,
    top_k: int = 20,
) -> str | float | int | None:
    """
    Extract a single field value using targeted retrieval and a focused prompt.

    Args:
        field_name: Column name to extract (must be in COLUMN_PROMPTS).
        nct_id: Trial NCT ID.
        official_title: Trial title.
        llm: LLM client.
        hybrid_retriever: Optional Vector+BM25 ensemble retriever.
        embedder: Embedder instance for vector-only fallback.
        vstore_dir: Vectorstore path.
        top_k: Number of chunks to retrieve per field.

    Returns:
        Extracted value (cleaned and coerced) or None on failure.
    """
    if field_name not in COLUMN_PROMPTS:
        logger.warning(f"No prompt defined for field: {field_name}")
        return None

    # --- Field-specific retrieval ---
    retrieval_query_keywords = FIELD_RETRIEVAL_QUERIES.get(field_name, "")
    retrieval_query = f"Trial {nct_id}: {retrieval_query_keywords}"
    
    logger.debug(f"Retrieving chunks for {field_name} with query: {retrieval_query}")

    if hybrid_retriever is not None:
        try:
            if hasattr(hybrid_retriever, "invoke"):
                docs = hybrid_retriever.invoke(retrieval_query)
            else:
                docs = hybrid_retriever.get_relevant_documents(retrieval_query)
        except Exception as e:
            logger.warning(f"Hybrid retriever failed for {field_name}/{nct_id}: {e} — using vector only")
            docs = []

        # Post-filter: prefer chunks for this NCT
        nct_docs = [d for d in docs if d.metadata.get("nct_id") == nct_id]
        if not nct_docs:
            nct_docs = docs  # fallback to all docs

        chunks = [d.page_content for d in nct_docs[:top_k]]
        chunk_ids = [str(d.metadata.get("chunk_id") or d.metadata.get("id") or i)
                     for i, d in enumerate(nct_docs[:top_k])]
    else:
        # Vector-only fallback
        client, collection = init_chroma(vstore_dir, collection_name="clinical_trials")
        results = retrieve_chunks(
            collection=collection, embedder=embedder, question=retrieval_query, nct_id=nct_id, top_k=top_k
        )
        chunks = results.get("documents", [[]])[0]
        chunk_ids = results.get("ids", [[]])[0]

    if not chunks:
        logger.warning(f"No chunks retrieved for {field_name}/{nct_id}")
        return None

    # Format chunks for prompt
    chunks_text = format_chunks_for_prompt(chunks, chunk_ids)

    # Build extraction prompt
    question = COLUMN_PROMPTS[field_name].format(
        nct_id=nct_id, official_title=official_title
    )
    prompt = question + CONTEXT_TEMPLATE.format(chunks_text=chunks_text)

    try:
        response = llm.invoke(prompt)
        # Extract content from LangChain AIMessage
        if hasattr(response, 'content'):
            raw_value = str(response.content).strip()
        else:
            raw_value = str(response).strip()

        # Clean response
        raw_value = raw_value.strip('"\' \n\t')

        # Handle N/A responses
        if raw_value.upper() in ['N/A', 'NA', 'NOT AVAILABLE', 'NOT REPORTED', 'UNKNOWN', '']:
            return None

        # Type coercion based on field name
        if 'p_value' in field_name:
            # Extract numeric from string like "<0.05" or "p=0.042"
            match = re.search(r'[\d.]+', raw_value)
            if match:
                try:
                    value = float(match.group())
                    # Validate p-value range
                    if 0 <= value <= 1:
                        return value
                except ValueError:
                    pass
            return None

        elif field_name in ['enrolled_seriously_affected', 'enrolled_deaths']:
            # Extract integer
            match = re.search(r'\d+', raw_value)
            if match:
                try:
                    return int(match.group())
                except ValueError:
                    pass
            return None

        elif field_name == 'success_flag_extracted':
            # Validate against allowed values
            value_upper = raw_value.upper()
            allowed = ['LIKELY_PASS', 'LIKELY_FAIL', 'DEFINITE_FAIL', 'NO_RESULTS']
            for allowed_val in allowed:
                if allowed_val in value_upper:
                    return allowed_val
            return None

        else:
            # String fields: return as-is
            return raw_value if len(raw_value) > 0 else None

    except Exception as e:
        logger.error(f"Failed to extract {field_name} for {nct_id}: {e}")
        return None


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
    nct_id: str,
    master_row: dict,
    vstore_dir: Path,
    embedder: Embedder,
    top_k: int = 20,
    hybrid_retriever: Any | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Selective enrichment pipeline: extract only missing fields using per-field retrieval.

    Args:
        nct_id: Trial NCT ID.
        master_row: Dict representing the CSV row for this NCT (with all columns).
        vstore_dir: Vectorstore path.
        embedder: Embedder instance.
        top_k: Number of chunks to retrieve per field.
        hybrid_retriever: Optional Vector+BM25 ensemble retriever.
        llm: LLM client for extraction (if None, uses default OpenRouter model).

    Returns:
        Enriched dict with nct_id + 8 target fields (only extracted if missing).
    """
    if llm is None:
        llm = get_openrouter_llm()

    # Extract official_title from master row
    official_title = str(master_row.get('official_title', 'Unknown Trial')).strip()
    if not official_title or official_title == 'nan':
        official_title = 'Unknown Trial'

    # --- Per-field extraction (only for missing fields) ---
    # Each field does its own targeted retrieval
    enriched_data = {"nct_id": nct_id}

    for field_name in COLUMN_PROMPTS.keys():
        # Check if field is missing in master row
        existing_value = master_row.get(field_name)
        
        # Consider missing if: None, NaN, empty string, or 'nan' string
        is_missing = (
            existing_value is None
            or (isinstance(existing_value, float) and pd.isna(existing_value))
            or (isinstance(existing_value, str) and existing_value.strip().lower() in ['', 'nan', 'n/a'])
        )

        if is_missing:
            logger.info(f"Extracting {field_name} for {nct_id} (missing in master CSV)")
            extracted_value = extract_single_field(
                field_name=field_name,
                nct_id=nct_id,
                official_title=official_title,
                llm=llm,
                hybrid_retriever=hybrid_retriever,
                embedder=embedder,
                vstore_dir=vstore_dir,
                top_k=top_k,
            )
            # Use 'N/A' for failed extractions (per user requirement)
            enriched_data[field_name] = extracted_value if extracted_value is not None else 'N/A'
        else:
            # Keep existing value
            logger.debug(f"Skipping {field_name} for {nct_id} (already has value: {existing_value})")
            enriched_data[field_name] = existing_value

    return enriched_data


def save_enriched_csv(enriched_records: list[dict], output_path: Path):
    """
    Merge enriched records with master CSV, updating only missing fields.

    Args:
        enriched_records: List of enriched dicts (nct_id + 8 target fields).
        output_path: Output CSV path.
    """
    # Load original master CSV
    master_csv_path = output_path.parent / 'master_ai_trials_dataset.csv'
    if not master_csv_path.exists():
        logger.warning(f"Master CSV not found at {master_csv_path}; saving enriched records only")
        df_enriched = pd.DataFrame(enriched_records)
        df_enriched.to_csv(output_path, index=False)
        return

    master_df = pd.read_csv(master_csv_path)
    logger.info(f"Loaded master CSV with {len(master_df)} rows")

    # Convert enriched records to DataFrame
    enriched_df = pd.DataFrame(enriched_records)

    # Add new columns to master if they don't exist
    new_columns = ['secondary_outcome', 'secondary_outcome_p_value', 
                   'success_flag_extracted', 'intervention_name_extracted']
    for col in new_columns:
        if col not in master_df.columns:
            master_df[col] = None

    # Merge on nct_id
    merged_df = master_df.merge(
        enriched_df,
        on='nct_id',
        how='left',
        suffixes=('', '_enriched')
    )

    # Update columns: use enriched value if original is missing
    target_columns = list(COLUMN_PROMPTS.keys())
    for col in target_columns:
        if col in merged_df.columns:
            # Column exists in master: update where enriched value is not 'N/A' or error
            enriched_col = f"{col}_enriched"
            if enriched_col in merged_df.columns:
                # Update: use enriched value when original is NaN and enriched is valid
                mask = merged_df[col].isna() & merged_df[enriched_col].notna()
                merged_df.loc[mask, col] = merged_df.loc[mask, enriched_col]
                # Drop the duplicate _enriched column
                merged_df.drop(columns=[enriched_col], inplace=True)
        else:
            # Column doesn't exist in master: just add it (already in new_columns)
            pass

    # Drop error column if present
    if 'error' in merged_df.columns:
        merged_df.drop(columns=['error'], inplace=True)
    if 'error_enriched' in merged_df.columns:
        merged_df.drop(columns=['error_enriched'], inplace=True)

    # Save merged DataFrame
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved merged CSV with {len(merged_df)} rows and {len(merged_df.columns)} columns to {output_path}")
