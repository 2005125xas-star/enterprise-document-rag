from __future__ import annotations

from src.models import RetrievalResult


def is_relevant(result: RetrievalResult, expected_chunk_ids: list[str], expected_document_ids: list[str]) -> bool:
    return result.chunk.chunk_id in expected_chunk_ids or result.chunk.document_id in expected_document_ids


def hit_at_k(results: list[RetrievalResult], expected_chunk_ids: list[str], expected_document_ids: list[str], k: int) -> float:
    if not expected_chunk_ids and not expected_document_ids:
        return 0.0
    return float(any(is_relevant(result, expected_chunk_ids, expected_document_ids) for result in results[:k]))


def reciprocal_rank(
    results: list[RetrievalResult],
    expected_chunk_ids: list[str],
    expected_document_ids: list[str],
) -> float:
    if not expected_chunk_ids and not expected_document_ids:
        return 0.0
    for index, result in enumerate(results, start=1):
        if is_relevant(result, expected_chunk_ids, expected_document_ids):
            return 1.0 / index
    return 0.0

