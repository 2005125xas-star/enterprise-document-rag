from __future__ import annotations

from pathlib import Path

from src.indexing.embeddings import HashingEmbeddingModel
from src.llm.providers import MockProvider
from src.qa.pipeline import validate_citations
from src.qa.service import EnterpriseRAGSystem
from src.utils.config import DEFAULT_CONFIG, deep_merge
from src.utils.query_logger import QueryLogger


def test_end_to_end_document_rag_smoke_test(tmp_path: Path) -> None:
    document_path = tmp_path / "enterprise_policy.txt"
    document_path.write_text(
        "Enterprise retention policy\n\n"
        "Customer records must be retained for seven years after account closure. "
        "Operational backups are encrypted at rest.",
        encoding="utf-8",
    )
    log_path = tmp_path / "query_logs.sqlite3"
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "chunking": {"chunk_size": 160, "chunk_overlap": 30},
            "retrieval": {"min_score": 0.0},
            "logging": {"sqlite_path": str(log_path)},
        },
    )
    system = EnterpriseRAGSystem(
        config,
        embedding_model=HashingEmbeddingModel(dimensions=64),
        provider=MockProvider(),
        logger=QueryLogger(log_path),
    )

    chunks = system.ingest_paths([document_path])
    retrieval_results = system.retriever.search("How long are customer records retained?", top_k=3)
    answer = system.answer("How long are customer records retained?", top_k=3, session_id="e2e")
    logs = system.recent_logs()

    assert chunks
    assert retrieval_results
    assert retrieval_results[0].chunk.file_name == "enterprise_policy.txt"
    assert retrieval_results[0].chunk.page_number == 1
    assert answer.is_answered is True
    assert "[1]" in answer.answer
    assert validate_citations(answer.answer, answer.citations) is True
    assert answer.citations[0]["file_name"] == "enterprise_policy.txt"
    assert answer.citations[0]["page_number"] == 1
    assert logs[0]["question"] == "How long are customer records retained?"
    assert logs[0]["session_id"] == "e2e"
    assert logs[0]["sources"][0]["file_name"] == "enterprise_policy.txt"
