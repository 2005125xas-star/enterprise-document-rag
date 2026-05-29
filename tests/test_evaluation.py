from __future__ import annotations

from pathlib import Path

from src.evaluation.metrics import hit_at_k, reciprocal_rank
import pytest

from src.evaluation.run_eval import load_sample_document_chunks
from src.evaluation.runner import (
    determine_failure_reason,
    evaluate,
    load_examples,
    save_evaluation_outputs,
    validate_eval_csv_schema,
)
from src.indexing.embeddings import HashingEmbeddingModel
from src.llm.providers import MockProvider
from src.models import DocumentChunk, EvaluationExample, RetrievalResult
from src.qa.pipeline import QAPipeline
from src.retrieval.hybrid import HybridRetriever


def _result(chunk_id: str, document_id: str) -> RetrievalResult:
    chunk = DocumentChunk(chunk_id, document_id, f"{document_id}.txt", 1, "text", 0, 4)
    return RetrievalResult(chunk, 0.5, 0.5, 0.5, rank=1)


def test_retrieval_metrics_support_chunk_and_document_relevance() -> None:
    results = [_result("doc-a:p1:c1", "doc-a"), _result("doc-b:p1:c1", "doc-b")]

    assert hit_at_k(results, ["doc-b:p1:c1"], [], 1) == 0.0
    assert hit_at_k(results, ["doc-b:p1:c1"], [], 2) == 1.0
    assert reciprocal_rank(results, [], ["doc-b"]) == 0.5


def test_evaluate_computes_metrics_and_outputs_files(tmp_path: Path) -> None:
    chunk = DocumentChunk(
        "policy-doc:p1:c1",
        "policy-doc",
        "policy.txt",
        1,
        "Customer records are retained for seven years.",
        0,
        46,
    )
    retriever = HybridRetriever(HashingEmbeddingModel(dimensions=64), min_score=0.0)
    retriever.build([chunk])
    pipeline = QAPipeline(retriever, MockProvider(), min_score=0.0)
    examples = [
        EvaluationExample(
            "How long are customer records retained?",
            True,
            question_id="Q001",
            question_type="fact_lookup",
            difficulty="easy",
            expected_document_ids=["policy-doc"],
            expected_keywords=["seven years"],
            expected_page=1,
        ),
        EvaluationExample("quantum asteroid orbit", False, question_id="Q002", question_type="no_answer"),
    ]

    result = evaluate(pipeline, examples)

    assert result.total_examples == 2
    assert result.hit_at_3 == 1.0
    assert result.hit_at_5 == 1.0
    assert result.mrr == 1.0
    assert result.no_answer_accuracy == 1.0
    assert result.answerable_count == 1
    assert result.unanswerable_count == 1
    assert result.citation_rate == 1.0
    assert result.metrics_by_question_type["fact_lookup"]["questions"] == 1
    assert result.metrics_by_question_type["no_answer"]["no_answer_accuracy"] == 1.0
    assert result.metrics_by_difficulty["easy"]["questions"] == 1

    csv_path = tmp_path / "eval_results.csv"
    report_path = tmp_path / "evaluation_report.md"
    save_evaluation_outputs(result, csv_path, report_path, document_names=["policy.txt"])
    report_text = report_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "Enterprise Document RAG QA System" in report_text
    assert "Number of evaluation questions: 2" in report_text
    assert "Question Distribution" in report_text
    assert "Metrics By Question Type" in report_text
    assert "Failure Or Weak Examples" in report_text
    assert csv_text.startswith(
        "question_id,question,question_type,difficulty,answerable,expected_doc,expected_page,"
        "expected_keywords,retrieved_chunk_ids,retrieved_docs,top_score,hit_at_3,hit_at_5,"
        "reciprocal_rank,returned_no_answer,citation_present,latency,answer_preview,failure_reason"
    )


def test_load_examples_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "examples.csv"
    path.write_text(
        "question_id,question,question_type,expected_doc,expected_page,expected_keywords,answerable,difficulty,notes\n"
        "Q1,Question?,fact_lookup,doc-1,4,chunk-1|Answer,true,easy,Note\n",
        encoding="utf-8",
    )

    examples = load_examples(path)

    assert examples[0].question_id == "Q1"
    assert examples[0].question_type == "fact_lookup"
    assert examples[0].expected_document_ids == ["doc-1"]
    assert examples[0].expected_page == 4
    assert examples[0].expected_keywords == ["chunk-1", "Answer"]
    assert examples[0].notes == "Note"


def test_eval_csv_schema_validation_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="question_type"):
        validate_eval_csv_schema(["question_id", "question"])


def test_failure_reason_generation() -> None:
    example = EvaluationExample(
        "Question?",
        True,
        expected_document_ids=["expected-doc"],
        expected_keywords=["required keyword"],
        expected_page=1,
    )
    item = {
        "returned_no_answer": False,
        "retrieved_docs": ["other-doc"],
        "retrieved_pages": [1],
        "answer": "Some answer [1]",
        "citation_present": True,
    }

    assert determine_failure_reason(example, item) == "missed_expected_doc"

    no_answer_example = EvaluationExample("Unsupported?", False)
    answered_item = {
        "returned_no_answer": False,
        "retrieved_docs": ["doc"],
        "retrieved_pages": [1],
        "answer": "Unsupported answer [1]",
        "citation_present": True,
    }
    assert determine_failure_reason(no_answer_example, answered_item) == "should_have_refused_but_answered"


def test_loads_multiple_sample_documents() -> None:
    chunks, document_names = load_sample_document_chunks(
        Path("data/sample_docs"),
        {"chunking": {"chunk_size": 700, "chunk_overlap": 120}},
    )

    assert len(document_names) == 6
    assert len(chunks) > len(document_names)
    assert "bank_recruitment_policy.txt" in document_names


def test_qa_eval_set_schema_and_distribution() -> None:
    examples = load_examples("data/eval/qa_eval_set.csv")
    by_type: dict[str, int] = {}
    for example in examples:
        by_type[example.question_type] = by_type.get(example.question_type, 0) + 1

    assert len(examples) >= 40
    assert sum(example.answerable for example in examples) >= 35
    assert by_type["cross_document"] >= 5
    assert by_type["numeric_threshold"] >= 5
    assert by_type["role_responsibility"] >= 5
    assert by_type["no_answer"] >= 5
