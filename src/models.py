from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentPage:
    """A parsed page or page-like unit from an uploaded document."""

    document_id: str
    file_name: str
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """A retrieval-ready text chunk with source metadata preserved."""

    chunk_id: str
    document_id: str
    file_name: str
    page_number: int
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """A scored retrieval result from hybrid search."""

    chunk: DocumentChunk
    semantic_score: float
    keyword_score: float
    score: float
    rank: int
    hybrid_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class AnswerResult:
    """A generated answer, its citations, and retrieval/debug metadata."""

    answer: str
    citations: list[dict[str, Any]]
    retrieval_results: list[RetrievalResult]
    is_answered: bool
    latency_ms: float
    provider: str


@dataclass(frozen=True)
class EvaluationExample:
    """One labeled QA evaluation item."""

    question: str
    answerable: bool
    question_id: str = ""
    question_type: str = "fact_lookup"
    difficulty: str = "medium"
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_document_ids: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    expected_page: int | None = None
    expected_answer: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate retrieval, no-answer, and latency metrics."""

    hit_at_3: float
    hit_at_5: float
    mrr: float
    no_answer_accuracy: float
    average_latency_ms: float
    total_examples: int
    answerable_count: int = 0
    unanswerable_count: int = 0
    citation_rate: float = 0.0
    metrics_by_question_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics_by_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)
