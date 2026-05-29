from __future__ import annotations

from dataclasses import asdict

from src.models import AnswerResult, DocumentChunk, EvaluationExample, EvaluationResult, RetrievalResult


def test_core_models_are_dataclass_serializable() -> None:
    chunk = DocumentChunk("doc:p1:c1", "doc", "file.txt", 1, "hello", 0, 5)
    retrieval = RetrievalResult(chunk=chunk, semantic_score=0.7, keyword_score=0.2, score=0.5, rank=1)
    answer = AnswerResult("hello", [{"chunk_id": chunk.chunk_id}], [retrieval], True, 12.3, "mock")
    example = EvaluationExample("question?", True, expected_chunk_ids=[chunk.chunk_id])
    result = EvaluationResult(1.0, 1.0, 1.0, 0.0, 12.3, 1, details=[asdict(example)])

    assert asdict(answer)["retrieval_results"][0]["chunk"]["chunk_id"] == chunk.chunk_id
    assert asdict(result)["details"][0]["question"] == "question?"

