"""PDF processing utilities.
Extract text with page numbers and simple section heuristics.
"""
import re
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None


def extract_pages(path: Path) -> list[dict]:
    """Extract pages of text from a PDF file.

    Returns a list of dicts: {"page": int, "text": str}
    """
    path = Path(path)
    if fitz:
        doc = fitz.open(path.as_posix())
        pages = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text")
            pages.append({"page": i + 1, "text": text})
        return pages

    if pdfplumber:
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, p in enumerate(pdf.pages):
                text = p.extract_text() or ""
                pages.append({"page": i + 1, "text": text})
        return pages

    raise RuntimeError("No PDF backend available (install PyMuPDF or pdfplumber)")


# Better scientific section detection:
# ALL CAPS, numbered (1. Introduction), or case-sensitive common headers
HEADING_RE = re.compile(
    r"^("
    r"[A-Z][A-Z0-9 \-]{3,}" # ALL CAPS
    r"|\d+\.?\s+[A-Z][a-z]+" # 1. Introduction
    r"|Abstract|Introduction|Methods|Results|Discussion|References|Conclusion|Supplementary|Synopsis"
    r")$",
    re.M
)


def split_into_sections(page_text: str) -> list[dict]:
    """Split a page's text into coarse sections using heading heuristics.

    Returns list of {"section_title": str, "text": str}
    """
    # Find candidate headings: lines in ALL CAPS or starting with 'Section' or numbered
    lines = page_text.splitlines()
    sections = []
    current_title = ""
    buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append("")
            continue

        if HEADING_RE.match(stripped) or stripped.lower().startswith("section") or re.match(r"^\d+\.", stripped):
            # flush previous
            if buffer:
                sections.append({"section_title": current_title or "body", "text": "\n".join(buffer).strip()})
            current_title = stripped
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections.append({"section_title": current_title or "body", "text": "\n".join(buffer).strip()})

    return sections


def pdf_to_sections(path: Path) -> list[dict]:
    """Full PDF -> list of sections with metadata (filename, page, section_title, text, nct_id)."""
    path = Path(path)
    # Extract NCT ID from filename (e.g., NCT03088813.pdf -> NCT03088813)
    nct_match = re.search(r"(NCT\d{8})", path.name, re.IGNORECASE)
    nct_id = nct_match.group(1).upper() if nct_match else None

    # Use pdfplumber for better table and layout extraction if available
    # Preferred for scientific papers to catch tabular data
    pages = []
    if pdfplumber:
        with pdfplumber.open(path) as pdf:
            for i, p in enumerate(pdf.pages):
                # Extract text
                text = p.extract_text() or ""

                # Simple Table extraction: Concatenate tables as markdown-like text
                tables = p.extract_tables()
                if tables:
                    table_text = "\n\n--- TABLE START ---\n"
                    for table in tables:
                        for row in table:
                            # Filter None and join
                            row_str = " | ".join([str(cell or "").strip() for cell in row])
                            table_text += row_str + "\n"
                        table_text += "---\n"
                    text += table_text

                pages.append({"page": i + 1, "text": text})
    else:
        pages = extract_pages(path)

    out = []
    for p in pages:
        page_no = p["page"]
        secs = split_into_sections(p["text"]) or [{"section_title": "body", "text": p["text"]}]
        for s in secs:
            out.append({
                "filename": path.name,
                "nct_id": nct_id,
                "page": page_no,
                "section_title": s.get("section_title", "body"),
                "text": s.get("text", ""),
            })
    return out
