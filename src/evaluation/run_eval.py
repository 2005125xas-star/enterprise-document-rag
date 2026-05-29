from __future__ import annotations

from pathlib import Path

from src.evaluation.runner import evaluate, load_examples, save_evaluation_outputs
from src.indexing.embeddings import HashingEmbeddingModel
from src.ingestion.chunker import chunk_pages
from src.ingestion.parsers import parse_document
from src.llm.providers import MockProvider
from src.models import DocumentChunk
from src.qa.pipeline import QAPipeline
from src.retrieval.hybrid import HybridRetriever
from src.utils.config import DEFAULT_CONFIG, deep_merge
from src.utils.query_logger import QueryLogger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval" / "qa_eval_set.csv"
EVAL_RESULTS_PATH = PROJECT_ROOT / "outputs" / "eval_results.csv"
EVAL_REPORT_PATH = PROJECT_ROOT / "outputs" / "evaluation_report.md"
EVAL_LOG_PATH = PROJECT_ROOT / "data" / "logs" / "eval_query_logs.sqlite3"
DATASET_DESCRIPTION = (
    "Synthetic banking and enterprise operations benchmark with policy, product, "
    "risk, security, branch operations, and marketing documents."
)


def main() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "chunking": {"chunk_size": 700, "chunk_overlap": 120},
            "retrieval": {"min_score": 0.05, "top_k": 5},
            "logging": {"sqlite_path": str(EVAL_LOG_PATH)},
        },
    )
    chunks, document_names = load_sample_document_chunks(SAMPLE_DOCS_DIR, config)
    examples = load_examples(EVAL_SET_PATH)

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
        semantic_evidence_threshold=0.82,
        insufficient_evidence_message=config["qa"]["insufficient_evidence_message"],
        logger=QueryLogger(EVAL_LOG_PATH),
    )

    result = evaluate(pipeline, examples, top_k=int(config["retrieval"]["top_k"]))
    save_evaluation_outputs(
        result,
        EVAL_RESULTS_PATH,
        EVAL_REPORT_PATH,
        document_names=document_names,
        dataset_description=DATASET_DESCRIPTION,
    )

    print(f"Evaluation complete: {result.total_examples} questions")
    print(f"Answerable: {result.answerable_count}")
    print(f"Unanswerable: {result.unanswerable_count}")
    print(f"Hit@3: {result.hit_at_3:.4f}")
    print(f"Hit@5: {result.hit_at_5:.4f}")
    print(f"MRR: {result.mrr:.4f}")
    print(f"No-answer accuracy: {result.no_answer_accuracy:.4f}")
    print(f"Citation rate: {result.citation_rate:.4f}")
    print(f"Average latency ms: {result.average_latency_ms:.2f}")
    print(f"Wrote {EVAL_RESULTS_PATH}")
    print(f"Wrote {EVAL_REPORT_PATH}")


def load_sample_document_chunks(sample_docs_dir: Path, config: dict) -> tuple[list[DocumentChunk], list[str]]:
    paths = sorted(sample_docs_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No sample TXT documents found in {sample_docs_dir}")

    pages = []
    for path in paths:
        pages.extend(parse_document(path, document_id=path.stem))

    chunks = chunk_pages(
        pages,
        chunk_size=int(config["chunking"]["chunk_size"]),
        chunk_overlap=int(config["chunking"]["chunk_overlap"]),
    )
    return chunks, [path.name for path in paths]


if __name__ == "__main__":
    main()
