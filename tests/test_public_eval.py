from __future__ import annotations

import csv
from pathlib import Path

from src.evaluation import run_eval
from src.evaluation.run_public_eval import (
    PUBLIC_EVAL_SET_PATH,
    PUBLIC_SOURCES_PATH,
    load_public_sources,
    run_public_evaluation,
)
from src.evaluation.runner import REQUIRED_EVAL_COLUMNS, load_examples


def test_public_sources_yaml_schema() -> None:
    required_fields = {
        "source_id",
        "title",
        "institution",
        "year",
        "document_type",
        "official_source_url",
        "local_filename",
        "notes",
    }

    sources = load_public_sources(PUBLIC_SOURCES_PATH)

    assert len(sources) >= 5
    for source in sources:
        assert required_fields.issubset(source)
        assert source["source_id"]
        assert source["local_filename"].endswith((".pdf", ".docx", ".txt"))
        official_url = source["official_source_url"]
        assert official_url == "TODO_OFFICIAL_URL" or official_url.startswith("https://")


def test_public_eval_csv_schema_question_types_and_no_answer_cases() -> None:
    with PUBLIC_EVAL_SET_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert REQUIRED_EVAL_COLUMNS.issubset(reader.fieldnames or [])
        rows = list(reader)

    examples = load_examples(PUBLIC_EVAL_SET_PATH)
    question_types = {example.question_type for example in examples}
    no_answer_examples = [example for example in examples if not example.answerable]

    assert len(rows) >= 20
    assert {
        "fact_lookup",
        "policy_lookup",
        "definition_lookup",
        "numeric_or_date_lookup",
        "cross_document",
        "no_answer",
    }.issubset(question_types)
    assert len(no_answer_examples) >= 2


def test_public_eval_runner_exits_cleanly_when_public_docs_are_missing(tmp_path: Path, capsys) -> None:
    public_docs_dir = tmp_path / "public_docs"
    public_docs_dir.mkdir()
    results_path = tmp_path / "public_eval_results.csv"
    report_path = tmp_path / "public_evaluation_report.md"

    exit_code = run_public_evaluation(
        public_docs_dir=public_docs_dir,
        results_path=results_path,
        report_path=report_path,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No supported public documents found" in captured.out
    assert "Public evaluation was not run" in captured.out
    assert "ucl_chapter_4_assessment_framework_2025_26.pdf" in captured.out
    assert not results_path.exists()
    assert not report_path.exists()


def test_synthetic_evaluation_still_loads_documents_and_examples() -> None:
    chunks, document_names = run_eval.load_sample_document_chunks(
        Path("data/sample_docs"),
        {"chunking": {"chunk_size": 700, "chunk_overlap": 120}},
    )
    examples = load_examples(run_eval.EVAL_SET_PATH)

    assert len(document_names) == 6
    assert chunks
    assert len(examples) >= 40
