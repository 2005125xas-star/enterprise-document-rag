from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from src.evaluation.runner import evaluate, load_examples, save_evaluation_outputs
from src.indexing.embeddings import HashingEmbeddingModel
from src.ingestion.chunker import chunk_pages
from src.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_document
from src.llm.providers import MockProvider
from src.models import DocumentChunk, DocumentPage
from src.qa.pipeline import QAPipeline
from src.retrieval.hybrid import HybridRetriever
from src.utils.config import DEFAULT_CONFIG, deep_merge
from src.utils.query_logger import QueryLogger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DOCS_DIR = PROJECT_ROOT / "data" / "public_docs"
PUBLIC_EVAL_DIR = PROJECT_ROOT / "data" / "public_eval"
PUBLIC_SOURCES_PATH = PUBLIC_EVAL_DIR / "public_sources.yaml"
PUBLIC_EVAL_SET_PATH = PUBLIC_EVAL_DIR / "university_policy_eval_set.csv"
PUBLIC_RESULTS_PATH = PROJECT_ROOT / "outputs" / "public_eval_results.csv"
PUBLIC_REPORT_PATH = PROJECT_ROOT / "outputs" / "public_evaluation_report.md"
PUBLIC_LOG_PATH = PROJECT_ROOT / "data" / "logs" / "public_eval_query_logs.sqlite3"
DATASET_DESCRIPTION = (
    "Public university policy document demo path. This evaluation runs only after "
    "official public documents are manually placed in data/public_docs/."
)


def main() -> int:
    try:
        return run_public_evaluation()
    except Exception as exc:  # pragma: no cover - debug path is environment-dependent
        print(f"Public evaluation failed: {exc}", file=sys.stderr)
        if os.environ.get("APP_DEBUG") == "1":
            raise
        return 1


def run_public_evaluation(
    public_docs_dir: Path = PUBLIC_DOCS_DIR,
    eval_set_path: Path = PUBLIC_EVAL_SET_PATH,
    sources_path: Path = PUBLIC_SOURCES_PATH,
    results_path: Path = PUBLIC_RESULTS_PATH,
    report_path: Path = PUBLIC_REPORT_PATH,
) -> int:
    public_docs_dir = Path(public_docs_dir)
    sources = load_public_sources(sources_path)
    supported_paths = find_supported_documents(public_docs_dir)

    if not supported_paths:
        print(_missing_docs_message(public_docs_dir, sources))
        return 0

    missing = missing_registered_files(sources, public_docs_dir)
    if missing:
        print("Some registered public source files are missing from data/public_docs/:")
        for filename in missing:
            print(f"- {filename}")
        print("Continuing with the supported documents that are present.")

    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "chunking": {"chunk_size": 700, "chunk_overlap": 120},
            "retrieval": {
                "min_score": 0.05,
                "top_k": 5,
                "vector_store": "memory",
                "reranker": {"enabled": False},
            },
            "logging": {"sqlite_path": str(PUBLIC_LOG_PATH)},
        },
    )
    chunks, document_names = load_public_document_chunks(supported_paths, sources, config)
    examples = load_examples(eval_set_path)

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
        logger=QueryLogger(PUBLIC_LOG_PATH),
    )

    result = evaluate(pipeline, examples, top_k=int(config["retrieval"]["top_k"]))
    save_evaluation_outputs(
        result,
        results_path,
        report_path,
        document_names=document_names,
        dataset_description=DATASET_DESCRIPTION,
    )

    print(f"Public evaluation complete: {result.total_examples} questions")
    print(f"Answerable: {result.answerable_count}")
    print(f"Unanswerable: {result.unanswerable_count}")
    print(f"Hit@3: {result.hit_at_3:.4f}")
    print(f"Hit@5: {result.hit_at_5:.4f}")
    print(f"MRR: {result.mrr:.4f}")
    print(f"No-answer accuracy: {result.no_answer_accuracy:.4f}")
    print(f"Citation rate: {result.citation_rate:.4f}")
    print(f"Average latency ms: {result.average_latency_ms:.2f}")
    print(f"Wrote {results_path}")
    print(f"Wrote {report_path}")
    return 0


def load_public_sources(path: str | Path = PUBLIC_SOURCES_PATH) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency behavior
        raise RuntimeError("PyYAML is required to load public_sources.yaml. Install requirements.txt.") from exc

    source_path = Path(path)
    data = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict):
        sources = data.get("sources", [])
    elif isinstance(data, list):
        sources = data
    else:
        sources = []
    if not isinstance(sources, list):
        raise ValueError("public_sources.yaml must contain a 'sources' list.")
    return [dict(item) for item in sources]


def find_supported_documents(public_docs_dir: Path) -> list[Path]:
    if not public_docs_dir.exists():
        return []
    return sorted(
        path
        for path in public_docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def missing_registered_files(sources: list[dict[str, Any]], public_docs_dir: Path) -> list[str]:
    missing: list[str] = []
    for source in sources:
        filename = str(source.get("local_filename") or "").strip()
        if filename and not (public_docs_dir / filename).exists():
            missing.append(filename)
    return missing


def load_public_document_chunks(
    paths: list[Path],
    sources: list[dict[str, Any]],
    config: dict,
) -> tuple[list[DocumentChunk], list[str]]:
    source_id_by_filename = {
        str(source.get("local_filename")): str(source.get("source_id"))
        for source in sources
        if source.get("local_filename") and source.get("source_id")
    }
    pages: list[DocumentPage] = []
    for path in paths:
        document_id = source_id_by_filename.get(path.name, path.stem)
        pages.extend(parse_document(path, document_id=document_id))

    chunks = chunk_pages(
        pages,
        chunk_size=int(config["chunking"]["chunk_size"]),
        chunk_overlap=int(config["chunking"]["chunk_overlap"]),
    )
    return chunks, [path.name for path in paths]


def _missing_docs_message(public_docs_dir: Path, sources: list[dict[str, Any]]) -> str:
    expected_files = [str(source.get("local_filename")) for source in sources if source.get("local_filename")]
    lines = [
        f"No supported public documents found in {public_docs_dir}.",
        "",
        "To run the public real-document demo:",
        "1. Verify the official source URLs in data/public_eval/public_sources.yaml.",
        "2. Download only publicly available PDF, DOCX, or TXT policy documents.",
        "3. Place them in data/public_docs/ using these local filenames:",
    ]
    lines.extend(f"- {filename}" for filename in expected_files)
    lines.extend(
        [
            "",
            "Only place publicly available documents here. Do not upload private coursework,",
            "VLE/Moodle/Learning Mall/Canvas files, internal university documents, or",
            "copyrighted materials without permission.",
            "",
            "Public evaluation was not run.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
