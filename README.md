# 🧬 ai-clinical-trials-rag — Phase III Clinical Trials RAG Portfolio

End-to-end Retrieval‑Augmented Generation (RAG) pipeline for extracting, grounding, and answering clinical‑trial questions and predicting Phase III trial outcomes using multi‑source clinical data. This repository demonstrates data engineering, semantic chunking, vector search (Chroma), LLM prompting/chaining (OpenRouter by default), and a defensible ground‑truth workflow — designed as a data‑science / AI RAG project focused on biotech.

Highlights
- Lightweight RAG stack: section‑aware chunking, OpenRouter/Qwen embeddings (fallbacks supported), Chroma vectorstore, hybrid retrieval + reranking, and LLM answer‑drafting + refinement.
- Example dataset: master clinical trials dataset (127 Phase III trials) and a RAG‑ready subset for retrieval experiments.
- Reproducible notebooks that walk through data loading, ingestion, and retrieval/ground‑truth construction.

See `src/biotech_rag/` for implementation details and `notebooks/` for the guided experiments.

---

## Project Goal

Provide a reproducible, production‑minded RAG pipeline that:
- Ingests multi‑source clinical trial materials (AACT metadata, PubMed XML, PMC PDFs, FDA labels).
- Performs section‑aware chunking and metadata preservation for precise citations.
- Indexes semantic embeddings into a persistent Chroma vectorstore for fast lookup.
- Runs a defensible ground‑truth pipeline (extractive draft → human review → refined answers) to produce evaluation‑grade annotations.

This repo is intended as to showcase data engineering, retrieval systems, and LLM orchestration in a scientific domain (biotech/clinical trials).

---

## Data Sources & Key Files

- Primary sources: AACT (ClinicalTrials.gov) metadata, PubMed / PMC publications (XML & PDFs), and FDA drug labels (JSON).
- Notable processed files (stored under `data/processed/`):
  - `master_ai_trials_dataset.csv` — master table (~127 trials × ~53 columns) with target `success_flag` and `data_richness_score`.
  - `rag_ready_trials.csv` — subset of trials with PDFs/XML available for RAG (~70–75 trials depending on preprocessing).
  - `dataset_summary.json` — dataset stats and creation metadata.
  - `ground_truth_template.csv` — sample question template used to build ground truth (200 rows in the demo template).

Practical notes from the current run:
- Vectorstore (Chroma) contains 5,532 indexed chunks (PDF ≈ 1,411; PubMed XML ≈ 305; FDA labels ≈ 3,816).
- Embedding dimension observed with the default embedder: 4096 (OpenRouter Qwen embedding in this demo).

---

## Notebooks (run in order)

- `notebooks/01_clinical_data_loading.ipynb` — Data loading & inspection
  - Loads the master dataset and RAG‑ready subset, prints schema and summary stats, and documents dataset characteristics (missingness, class imbalance, richness score).
  - Useful artifacts: confirms `master_ai_trials_dataset.csv` (127 trials) and `rag_ready` counts.

- `notebooks/02_ingest_to_chroma.ipynb` — Document parsing, chunking, embedding, and upsert
  - Section‑aware PDF parsing, PubMed XML parsing, and FDA JSON parsing.
  - Uses `HybridScientificChunker` to create semantically coherent chunks with overlap and strong metadata (filename, page, section_title, nct_id, pmid).
  - Embeddings are produced by the centralized `Embedder` (OpenRouter/Qwen first, with local fallback) and persisted into Chroma at `data/processed/vectorstore/chroma_db`.
  - Output example: ingestion totals and a sample collection preview (documents + metadata).

- `notebooks/03_retrieval_strategies.ipynb` — Retrieval, ground truth, and LLM drafting
  - Builds a defensible ground‑truth pipeline: retrieve top‑K (K=5) chunks per question, draft extractive answers with citations (Draft 1), support human review and refinement (Draft 2 → final JSON).
  - Implements hybrid retrieval options (dense embeddings + BM25) and supports reranking (cross‑encoder) for evaluation experiments.
  - Produces artifacts such as `retrieved_contexts.json`, `draft1_answers.json`, and downstream ground‑truth JSONs.

---

## Quick Start (developer)

1. Create a virtual environment and install dev dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```

2. Provide API keys in a `.env` file for optional remote LLM/embedding providers (e.g. `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`). The code falls back to local models when keys are missing.

3. Launch Jupyter and run notebooks in sequence: 01 → 02 → 03. Notebooks include diagnostic cells and example outputs to verify successful ingestion and retrieval.

   ```bash
   jupyter lab
   ```

4. Tests / CI (project recommended commands):

   - Lint: `ruff check src/ tests/`
   - Format: `black src/ tests/` (line length 100)
   - Run tests: `pytest tests/ -v --cov=src/biotech_rag`

---

## Implementation notes & caveats

- The demo uses OpenRouter/Qwen embedding by default (embedding dim 4096). For reproducibility, lock the same embedding model at ingestion and query time or reindex if you change the model.
- The ingestion pipeline detects duplicate chunk IDs and deduplicates batches before upsert.
- Some LLM draft runs in the example notebooks show minor parsing edge cases (e.g. `NoneType.strip()` in a few template rows). These are easy fixes by validating inputs before calling the LLM chains.

---

## Contact & Next steps

- This project is intended as a portfolio showcase. If you'd like, I can:
  1. Add a small CLI script to run ingestion end‑to‑end and produce a reproducible demo dataset.
  2. Harden the ground‑truth pipeline with additional input validation and retries for LLM calls.

Find the core code under `src/biotech_rag/` and the notebooks under `notebooks/`.
