from __future__ import annotations

from pathlib import Path

from src.evaluation.metrics import hit_at_k, reciprocal_rank
from src.evaluation.runner import evaluate, load_examples, save_evaluation_outputs
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
            expected_document_ids=["policy-doc"],
            expected_page=1,
        ),
        EvaluationExample("quantum asteroid orbit", False),
    ]

    result = evaluate(pipeline, examples)

    assert result.total_examples == 2
    assert result.hit_at_3 == 1.0
    assert result.hit_at_5 == 1.0
    assert result.mrr == 1.0
    assert result.no_answer_accuracy == 1.0

    csv_path = tmp_path / "eval_results.csv"
    report_path = tmp_path / "evaluation_report.md"
    save_evaluation_outputs(result, csv_path, report_path)
    report_text = report_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "Enterprise Document RAG QA System" in report_text
    assert "Number of evaluation questions: 2" in report_text
    assert "Known Limitations" in report_text
    assert csv_text.startswith(
        "question,answerable,expected_doc,expected_page,retrieved_chunk_ids,top_score,"
        "hit_at_3,hit_at_5,reciprocal_rank,returned_no_answer,latency"
    )


def test_load_examples_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "examples.csv"
    path.write_text(
        "question,answerable,expected_chunk_ids,expected_document_ids,expected_page,expected_answer\n"
        "Question?,true,chunk-1|chunk-2,doc-1,4,Answer\n",
        encoding="utf-8",
    )

    examples = load_examples(path)

    assert examples[0].expected_chunk_ids == ["chunk-1", "chunk-2"]
    assert examples[0].expected_document_ids == ["doc-1"]
    assert examples[0].expected_page == 4
