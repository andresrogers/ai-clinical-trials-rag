import json

import pytest

from biotech_rag.data.fda_parser import parse_fda_json
from biotech_rag.data.pubmed_parser import parse_pubmed_xml
from biotech_rag.indexing.chunkers import SectionAwareChunker
from biotech_rag.indexing.embedders import Embedder


def test_chunker_basic():
    text = """This is a sample paragraph. """ * 200
    chunker = SectionAwareChunker(min_tokens=64, overlap_tokens=8)
    chunks = chunker.chunk_text(text, section_title="body")
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all("text" in c and "metadata" in c for c in chunks)


def test_pubmed_parser_abstract(tmp_path):
    xml = tmp_path / "sample.xml"
    xml.write_text(
        "<article><abstract><p>Abstract here.</p></abstract></article>", encoding="utf-8"
    )
    sections = parse_pubmed_xml(xml)
    assert isinstance(sections, list)
    assert any(
        s.get("section_title") == "abstract" or "abstract" in s.get("section_title", "").lower()
        for s in sections
    )


def test_fda_parser_basic(tmp_path):
    j = tmp_path / "label.json"
    data = {"indications_and_usage": "Use for X", "adverse_reactions": ["nausea"]}
    j.write_text(json.dumps(data), encoding="utf-8")
    sections = parse_fda_json(j)
    assert isinstance(sections, list)
    assert any(s.get("section_title") == "indications_and_usage" for s in sections)


def test_embedder_openrouter_no_key(monkeypatch):
    eb = Embedder(backend="openrouter")
    with pytest.raises(RuntimeError):
        eb.embed(["test"])


def test_embedder_local_if_available():
    # Skip if sentence-transformers not available
    pytest.importorskip("sentence_transformers")
    eb = Embedder(backend="local")
    out = eb.embed(["hello world"])
    assert isinstance(out, list)
    assert len(out) == 1
