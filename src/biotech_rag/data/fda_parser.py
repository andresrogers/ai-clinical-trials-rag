"""Parser for FDA label JSON files.
Extract main sections like indications, dosage, safety.
"""
import json
from pathlib import Path


def parse_fda_json(path: Path) -> list[dict]:
    """Parse FDA label JSON files into sections.

    Returns list of {"filename": str, "drug_name": str, "section_title": str, "text": str}
    """
    path = Path(path)
    # Extract drug name from filename (e.g., "Aspirin.json" -> "Aspirin")
    drug_name = path.stem

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    out = []
    # Common keys
    keys = ['indications_and_usage', 'dosage_and_administration', 'warnings_and_precautions', 'adverse_reactions', 'clinical_studies']
    # Try normalized keys
    for k in keys:
        v = data.get(k) or data.get(k.upper()) or data.get(k.replace('_', ' '))
        if v:
            if isinstance(v, list):
                text = '\n\n'.join([str(x) for x in v])
            else:
                text = str(v)
            out.append({"filename": path.name, "drug_name": drug_name, "section_title": k, "text": text})

    # Fallback: flatten text fields
    if not out:
        # collect string values
        parts = []
        def collect(d):
            if isinstance(d, dict):
                for val in d.values():
                    collect(val)
            elif isinstance(d, list):
                for v in d:
                    collect(v)
            else:
                parts.append(str(d))

        collect(data)
        if parts:
            out.append({"filename": path.name, "drug_name": drug_name, "section_title": 'full_label', "text": '\n\n'.join(parts)})

    return out
