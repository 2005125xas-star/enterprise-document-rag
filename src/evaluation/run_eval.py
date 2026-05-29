from __future__ import annotations

from pathlib import Path

from src.evaluation.runner import evaluate, save_evaluation_outputs
from src.indexing.embeddings import HashingEmbeddingModel
from src.ingestion.chunker import chunk_pages
from src.ingestion.parsers import parse_document
from src.llm.providers import MockProvider
from src.models import EvaluationExample
from src.qa.pipeline import QAPipeline
from src.retrieval.hybrid import HybridRetriever
from src.utils.config import DEFAULT_CONFIG, deep_merge
from src.utils.query_logger import QueryLogger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DOC_PATH = PROJECT_ROOT / "data" / "eval" / "sample_policy.txt"
EVAL_RESULTS_PATH = PROJECT_ROOT / "outputs" / "eval_results.csv"
EVAL_REPORT_PATH = PROJECT_ROOT / "outputs" / "evaluation_report.md"
EVAL_LOG_PATH = PROJECT_ROOT / "data" / "logs" / "eval_query_logs.sqlite3"


def main() -> None:
    EVAL_DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not EVAL_DOC_PATH.exists():
        EVAL_DOC_PATH.write_text(_sample_document_text(), encoding="utf-8")

    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "chunking": {"chunk_size": 500, "chunk_overlap": 80},
            "retrieval": {"min_score": 0.05},
            "logging": {"sqlite_path": str(EVAL_LOG_PATH)},
        },
    )
    pages = parse_document(EVAL_DOC_PATH, document_id="sample-policy")
    chunks = chunk_pages(
        pages,
        chunk_size=int(config["chunking"]["chunk_size"]),
        chunk_overlap=int(config["chunking"]["chunk_overlap"]),
    )

    retriever = HybridRetriever(
        embedding_model=HashingEmbeddingModel(),
        semantic_weight=float(config["retrieval"]["semantic_weight"]),
        keyword_weight=float(config["retrieval"]["keyword_weight"]),
        min_score=float(config["retrieval"]["min_score"]),
    )
    retriever.build(chunks)
    pipeline = QAPipeline(
        retriever=retriever,
        provider=MockProvider(),
        max_context_chunks=int(config["qa"]["max_context_chunks"]),
        min_score=float(config["retrieval"]["min_score"]),
        semantic_evidence_threshold=0.8,
        insufficient_evidence_message=config["qa"]["insufficient_evidence_message"],
        logger=QueryLogger(EVAL_LOG_PATH),
    )
    examples = [
        EvaluationExample(
            question="How long are customer records retained?",
            answerable=True,
            expected_document_ids=["sample-policy"],
            expected_page=1,
            expected_answer="seven years",
        ),
        EvaluationExample(
            question="How are backups protected?",
            answerable=True,
            expected_document_ids=["sample-policy"],
            expected_page=1,
            expected_answer="encrypted",
        ),
        EvaluationExample(
            question="quantum asteroid orbit",
            answerable=False,
        ),
    ]

    result = evaluate(pipeline, examples, top_k=int(config["retrieval"]["top_k"]))
    save_evaluation_outputs(result, EVAL_RESULTS_PATH, EVAL_REPORT_PATH)

    print(f"Evaluation complete: {result.total_examples} questions")
    print(f"Hit@3: {result.hit_at_3:.4f}")
    print(f"Hit@5: {result.hit_at_5:.4f}")
    print(f"MRR: {result.mrr:.4f}")
    print(f"No-answer accuracy: {result.no_answer_accuracy:.4f}")
    print(f"Average latency ms: {result.average_latency_ms:.2f}")
    print(f"Wrote {EVAL_RESULTS_PATH}")
    print(f"Wrote {EVAL_REPORT_PATH}")


def _sample_document_text() -> str:
    return (
        "Enterprise Customer Records Policy\n\n"
        "Customer records must be retained for seven years after account closure. "
        "The retention schedule applies to contracts, invoices, and support tickets.\n\n"
        "Backup Security Standard\n\n"
        "Operational backups are encrypted at rest and reviewed every quarter. "
        "Access to backup archives requires approval from the security team."
    )


if __name__ == "__main__":
    main()
