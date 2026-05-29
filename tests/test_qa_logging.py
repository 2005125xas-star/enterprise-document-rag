from __future__ import annotations

from pathlib import Path

from src.indexing.embeddings import HashingEmbeddingModel
import pytest

from src.llm.providers import MockProvider, ProviderConfigurationError, create_llm_provider
from src.models import DocumentChunk
from src.qa.pipeline import DEFAULT_INSUFFICIENT_EVIDENCE_MESSAGE, QAPipeline, validate_citations
from src.retrieval.hybrid import HybridRetriever
from src.utils.query_logger import QueryLogger


def _retriever() -> HybridRetriever:
    chunks = [
        DocumentChunk(
            chunk_id="doc1:p1:c1",
            document_id="doc1",
            file_name="policy.txt",
            page_number=1,
            text="Customer records must be retained for seven years. Backups are encrypted.",
            start_char=0,
            end_char=75,
        )
    ]
    retriever = HybridRetriever(HashingEmbeddingModel(dimensions=64), min_score=0.0)
    retriever.build(chunks)
    return retriever


def test_create_provider_uses_mock_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    provider = create_llm_provider({"provider": "auto"})

    assert isinstance(provider, MockProvider)


def test_explicit_openai_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        create_llm_provider({"provider": "openai"})


def test_qa_pipeline_returns_cited_answer_and_logs_query(tmp_path: Path) -> None:
    logger = QueryLogger(tmp_path / "logs.sqlite3")
    pipeline = QAPipeline(_retriever(), MockProvider(), logger=logger, min_score=0.0)

    result = pipeline.answer_question("How long are customer records retained?", session_id="test-session")

    assert result.is_answered is True
    assert "[1]" in result.answer
    assert result.citations[0]["file_name"] == "policy.txt"
    assert result.citations[0]["page_range"] == "1-1"
    assert validate_citations(result.answer, result.citations) is True
    assert validate_citations("Unsupported citation [2]", result.citations) is False
    rows = logger.list_recent()
    assert len(rows) == 1
    assert rows[0]["question"] == "How long are customer records retained?"
    assert rows[0]["session_id"] == "test-session"
    assert rows[0]["is_answered"] is True
    assert rows[0]["top_score"] is not None
    assert rows[0]["sources"][0]["chunk_id"] == "doc1:p1:c1"


def test_qa_pipeline_returns_no_answer_when_evidence_is_missing() -> None:
    retriever = HybridRetriever(HashingEmbeddingModel(dimensions=64), min_score=0.0)
    retriever.build([])
    pipeline = QAPipeline(retriever, MockProvider())

    result = pipeline.answer_question("What is the travel policy?")

    assert result.is_answered is False
    assert result.answer == DEFAULT_INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.citations == []


def test_qa_pipeline_returns_no_answer_when_score_is_below_threshold() -> None:
    retriever = _retriever()
    pipeline = QAPipeline(retriever, MockProvider(), min_score=0.99)

    result = pipeline.answer_question("How long are customer records retained?")

    assert result.is_answered is False
    assert result.answer == DEFAULT_INSUFFICIENT_EVIDENCE_MESSAGE
