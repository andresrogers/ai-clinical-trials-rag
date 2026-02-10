#!/usr/bin/env python
"""CLI to run ingestion for PDFs, PubMed XML, and FDA JSON into Chroma."""
import argparse
import logging
from pathlib import Path

from biotech_rag.data.pdf_processor import pdf_to_sections
from biotech_rag.data.pubmed_parser import parse_pubmed_xml
from biotech_rag.data.fda_parser import parse_fda_json
from biotech_rag.indexing.chunkers import SectionAwareChunker
from biotech_rag.indexing.embedders import Embedder
from biotech_rag.indexing.vectorstore import init_chroma, upsert_documents


def ingest_sections(sections, source_tag, chunker, embedder, collection):
    docs = []
    for s in sections:
        text = (s.get('text') or '').strip()
        if not text:
            continue
        meta = dict(s)
        meta.pop('text', None)
        chunks = chunker.chunk_text(text, section_title=meta.get('section_title'), metadata=meta)
        for c in chunks:
            chunk_meta = dict(meta)
            chunk_meta.update(c.get('metadata', {}))
            doc_id = f"{source_tag}-{chunk_meta.get('filename','unknown')}-p{chunk_meta.get('page',0)}-c{chunk_meta.get('chunk_id')}"
            docs.append({'id': doc_id, 'text': c['text'], 'metadata': chunk_meta})

    if not docs:
        return 0

    texts = [d['text'] for d in docs]
    embeddings = embedder.embed(texts)
    upsert_documents(collection, docs, embeddings=embeddings)
    return len(docs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['pdf', 'pubmed', 'fda', 'all'], default='all')
    parser.add_argument('--path', type=str, default='data/raw')
    parser.add_argument('--persist', type=str, default='data/processed/vectorstore/chroma_db')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--backend', type=str, default=None, help='Embedder backend override')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    data_root = Path(args.path)
    pdf_dir = data_root / 'pdf_studies'
    xml_dir = data_root / 'pubmed_semantic_data' / 'xml_files'
    fda_dir = data_root / 'fda_labels'

    chunker = SectionAwareChunker(min_tokens=512, overlap_tokens=50)
    embedder = Embedder(backend=args.backend, batch_size=args.batch_size)
    client, collection = init_chroma(Path(args.persist), collection_name='clinical_trials')

    if args.source in ('pdf', 'all'):
        pdf_files = sorted(pdf_dir.glob('*.pdf')) if pdf_dir.exists() else []
        for p in pdf_files:
            logging.info(f'Processing PDF: {p.name}')
            secs = pdf_to_sections(p)
            n = ingest_sections(secs, 'pdf', chunker, embedder, collection)
            logging.info(f'Ingested {n} chunks from {p.name}')

    if args.source in ('pubmed', 'all'):
        xml_files = sorted(xml_dir.glob('*.xml')) if xml_dir.exists() else []
        for x in xml_files:
            logging.info(f'Processing XML: {x.name}')
            secs = parse_pubmed_xml(x)
            n = ingest_sections(secs, 'pubmed', chunker, embedder, collection)
            logging.info(f'Ingested {n} chunks from {x.name}')

    if args.source in ('fda', 'all'):
        fda_files = sorted(fda_dir.glob('*.json')) if fda_dir.exists() else []
        for f in fda_files:
            logging.info(f'Processing FDA file: {f.name}')
            secs = parse_fda_json(f)
            n = ingest_sections(secs, 'fda', chunker, embedder, collection)
            logging.info(f'Ingested {n} chunks from {f.name}')

    logging.info('Ingestion completed')


if __name__ == '__main__':
    main()
