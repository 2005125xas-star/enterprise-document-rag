from __future__ import annotations

import pytest

from src.llm.providers import MockProvider
from src.models import DocumentChunk, RetrievalResult
from src.qa.pipeline import QAPipeline, build_citations
from src.retrieval.reranker import (
    CrossEncoderReranker,
    NoOpReranker,
    RerankerConfigurationError,
    RERANKER_FALLBACK_WARNING,
    create_reranker,
)


class FakeCrossEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs):
        scores = []
        for _, text in pairs:
            if "highest priority" in text:
                scores.append(3.0)
            elif "medium priority" in text:
                scores.append(2.0)
            else:
                scores.append(1.0)
        return scores


class RecordingRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.requested_top_k: int | None = None

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.requested_top_k = top_k
        return self.results[:top_k]


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split(":")[0],
        file_name=f"{chunk_id.split(':')[0]}.txt",
        page_number=3,
        text=text,
        start_char=4,
        end_char=4 + len(text),
    )


def _result(chunk_id: str, text: str, score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        chunk=_chunk(chunk_id, text),
        semantic_score=0.9,
        keyword_score=0.6,
        score=score,
        rank=1,
        hybrid_score=score,
    )


def test_noop_reranker_preserves_existing_order() -> None:
    candidates = [
        _result("doc1:p3:c1", "first result"),
        _result("doc2:p3:c1", "second result"),
    ]

    reranked = NoOpReranker().rerank("question", candidates, final_k=2)

    assert [result.chunk.chunk_id for result in reranked] == ["doc1:p3:c1", "doc2:p3:c1"]
    assert [result.rank for result in reranked] == [1, 2]


def test_cross_encoder_reranker_reorders_candidates_and_returns_final_k() -> None:
    candidates = [
        _result("doc1:p3:c1", "low priority answer"),
        _result("doc2:p3:c1", "highest priority answer"),
        _result("doc3:p3:c1", "medium priority answer"),
    ]
    reranker = CrossEncoderReranker(model_name="fake-model", model_factory=FakeCrossEncoder)

    reranked = reranker.rerank("question", candidates, final_k=2)

    assert [result.chunk.chunk_id for result in reranked] == ["doc2:p3:c1", "doc3:p3:c1"]
    assert [result.rank for result in reranked] == [1, 2]
    assert reranked[0].hybrid_score == candidates[1].score
    assert reranked[0].rerank_score == 3.0


def test_reranker_preserves_metadata_and_citation_inputs() -> None:
    candidate = _result("policy:p3:c9", "highest priority customer retention evidence")
    reranker = CrossEncoderReranker(model_name="fake-model", model_factory=FakeCrossEncoder)

    reranked = reranker.rerank("customer retention", [candidate], final_k=1)
    citation = build_citations(reranked)[0]

    assert reranked[0].chunk.chunk_id == "policy:p3:c9"
    assert reranked[0].chunk.file_name == "policy.txt"
    assert reranked[0].chunk.page_number == 3
    assert reranked[0].chunk.start_char == 4
    assert citation["chunk_id"] == "policy:p3:c9"
    assert citation["file_name"] == "policy.txt"


def test_pipeline_fetches_top_n_candidates_before_returning_final_k() -> None:
    candidates = [
        _result("doc1:p3:c1", "low priority customer records evidence"),
        _result("doc2:p3:c1", "highest priority customer records evidence"),
        _result("doc3:p3:c1", "medium priority customer records evidence"),
    ]
    retriever = RecordingRetriever(candidates)
    reranker = CrossEncoderReranker(model_name="fake-model", model_factory=FakeCrossEncoder)
    pipeline = QAPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        provider=MockProvider(),
        min_score=0.0,
        reranker=reranker,
        reranker_top_n=3,
        reranker_final_k=2,
    )

    answer = pipeline.answer_question("What customer records evidence is available?")

    assert retriever.requested_top_k == 3
    assert len(answer.retrieval_results) == 2
    assert answer.retrieval_results[0].chunk.chunk_id == "doc2:p3:c1"
    assert "[1]" in answer.answer


def test_reranker_fallback_uses_noop_when_model_load_fails() -> None:
    def failing_factory(model_name: str):
        raise RuntimeError("download unavailable")

    reranker = create_reranker(
        {"enabled": True, "model": "fake-model", "allow_fallback": True},
        environ={},
        model_factory=failing_factory,
    )

    assert isinstance(reranker, NoOpReranker)
    assert reranker.warning == RERANKER_FALLBACK_WARNING


def test_reranker_can_fail_cleanly_when_fallback_is_disabled() -> None:
    def failing_factory(model_name: str):
        raise RuntimeError("download unavailable")

    with pytest.raises(RerankerConfigurationError, match="CrossEncoder reranker could not be loaded"):
        create_reranker(
            {"enabled": True, "model": "fake-model", "allow_fallback": False},
            environ={},
            model_factory=failing_factory,
        )
