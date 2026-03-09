from src.biotech_rag.evaluation.ragas_runner import build_evaluation_rows


def test_build_evaluation_rows_normalizes_contexts_and_reference() -> None:
    records = [
        {
            "row_id": 1,
            "question": "What is the primary endpoint?",
            "draft_answer": "The primary endpoint is progression-free survival.",
            "retrieved_chunks": [
                {"text": "Primary endpoint: progression-free survival."},
                {"content": "Secondary endpoint: overall survival."},
                "Randomized, double blind, placebo controlled.",
            ],
            "ground_truth_answer": "Progression-free survival is the primary endpoint.",
        }
    ]

    rows = build_evaluation_rows(records=records)

    assert len(rows) == 1
    assert rows[0]["user_input"] == "What is the primary endpoint?"
    assert rows[0]["response"].startswith("The primary endpoint")
    assert rows[0]["retrieved_contexts"][0] == "Primary endpoint: progression-free survival."
    assert rows[0]["reference"].startswith("Progression-free survival")
    assert rows[0]["row_id"] == 1


def test_build_evaluation_rows_applies_sample_size_and_skips_empty_records() -> None:
    records = [
        {
            "row_id": 1,
            "question": "Q1",
            "draft_answer": "A1",
            "retrieved_chunks": ["C1"],
        },
        {
            "row_id": 2,
            "question": "Q2",
            "draft_answer": "",
            "retrieved_chunks": ["C2"],
        },
        {
            "row_id": 3,
            "question": "Q3",
            "draft_answer": "A3",
            "retrieved_chunks": ["C3"],
        },
    ]

    rows = build_evaluation_rows(records=records, sample_n=2)

    assert len(rows) == 1
    assert rows[0]["row_id"] == 1


def test_build_evaluation_rows_normalizes_reference_variants() -> None:
    records = [
        {
            "row_id": 10,
            "nct_id": "NCT-XYZ",
            "question": "Q",
            "draft_answer": "A",
            "retrieved_chunks": ["C"],
            "ground_truth": ["Reference", "text"],
        }
    ]

    rows = build_evaluation_rows(records=records)

    assert len(rows) == 1
    assert rows[0]["reference"] == "Reference text"
    assert rows[0]["nct_id"] == "NCT-XYZ"
