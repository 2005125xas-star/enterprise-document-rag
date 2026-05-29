from __future__ import annotations

from pathlib import Path

from src.indexing.embeddings import HashingEmbeddingModel
from src.llm.providers import MockProvider
from src.qa.service import EnterpriseRAGSystem
from src.utils.config import DEFAULT_CONFIG, deep_merge
from src.utils.query_logger import QueryLogger


def test_service_ingests_documents_and_answers(tmp_path: Path) -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "chunking": {"chunk_size": 200, "chunk_overlap": 20},
            "retrieval": {"min_score": 0.0},
            "logging": {"sqlite_path": str(tmp_path / "logs.sqlite3")},
        },
    )
    document_path = tmp_path / "policy.txt"
    document_path.write_text("Customer records are retained for seven years.", encoding="utf-8")
    service = EnterpriseRAGSystem(
        config,
        embedding_model=HashingEmbeddingModel(dimensions=64),
        provider=MockProvider(),
        logger=QueryLogger(tmp_path / "logs.sqlite3"),
    )

    chunks = service.ingest_paths([document_path])
    answer = service.answer("How long are customer records retained?")

    assert len(chunks) == 1
    assert answer.is_answered is True
    assert service.recent_logs()[0]["question"] == "How long are customer records retained?"

