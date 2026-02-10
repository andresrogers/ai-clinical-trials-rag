"""Parsers for PubMed XML files (PMC / PubMed)."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_pubmed_xml(path: Path) -> list[dict]:
    """Parse a PubMed/PMC XML file into sections.

    Returns list of dicts: {"filename": str, "nct_id": str, "pmid": str, "pmcid": str, "section_title": str, "text": str}
    """
    path = Path(path)
    # Extract NCT ID from filename pattern {nct_id}_{nums}.xml
    nct_match = re.search(r"(NCT\d{8})", path.name, re.IGNORECASE)
    filename_nct = nct_match.group(1).upper() if nct_match else None

    tree = ET.parse(path)
    root = tree.getroot()

    # Determine if this is a PubmedArticleSet or a single article
    articles = root.findall('.//PubmedArticle')
    if not articles and root.tag == 'PubmedArticle':
        articles = [root]

    # If it's PMC XML, it might have <article> tags instead
    if not articles:
        articles = root.findall('.//article') or [root]

    sections = []

    for article in articles:
        pmid = None
        pmcid = None

        # Try common tags for PMID
        pmid_el = article.find('.//PMID')
        if pmid_el is not None:
            pmid = pmid_el.text

        # article-id for PMC
        pmcid_el = article.find('.//article-id[@pub-id-type="pmc"]')
        if pmcid_el is None:
            pmcid_el = article.find('.//article-id[@pub-id-type="pmcid"]')
        if pmcid_el is not None:
            pmcid = pmcid_el.text

        # 1. Search for <sec> tags (PMC style)
        article_has_secs = False
        for sec in article.findall('.//sec'):
            article_has_secs = True
            title_el = sec.find('title')
            title = title_el.text if title_el is not None else 'section'
            texts = []
            for p in sec.findall('.//p'):
                # Extract all text, including children
                p_text = "".join(p.itertext()).strip()
                if p_text:
                    texts.append(p_text)

            if texts:
                sections.append({
                    "filename": path.name,
                    "nct_id": filename_nct,
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "section_title": title,
                    "text": '\n\n'.join(texts)
                })

        # 2. Search for <AbstractText> tags (PubMed/Medline style)
        if not article_has_secs:
            abstract_els = article.findall('.//AbstractText')
            for abs_el in abstract_els:
                label = abs_el.get('Label') or 'Abstract'
                # Extract text recursively to handle bold/italic tags
                text = "".join(abs_el.itertext()).strip()
                if text:
                    # Look for NCT IDs in the text if not in filename
                    current_nct = filename_nct
                    if not current_nct:
                        nct_in_text = re.search(r"(NCT\d{8})", text, re.IGNORECASE)
                        if nct_in_text:
                            current_nct = nct_in_text.group(1).upper()

                    sections.append({
                        "filename": path.name,
                        "nct_id": current_nct,
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "section_title": label,
                        "text": text
                    })

            # Fallback for <abstract><p> (some PMC or other variants)
            if not abstract_els:
                for a in article.findall('.//abstract//p'):
                    p_text = "".join(a.itertext()).strip()
                    if p_text:
                        sections.append({
                            "filename": path.name,
                            "nct_id": filename_nct,
                            "pmid": pmid,
                            "pmcid": pmcid,
                            "section_title": 'Abstract',
                            "text": p_text
                        })

        # 3. Last Resort: Just the Title
        if not sections:
            title_el = article.find('.//ArticleTitle')
            if title_el is not None:
                title_text = "".join(title_el.itertext()).strip()
                if title_text:
                    sections.append({
                        "filename": path.name,
                        "nct_id": filename_nct,
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "section_title": "Title",
                        "text": title_text
                    })

    return sections
