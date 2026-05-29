from __future__ import annotations

import math

from src.indexing.embeddings import EmbeddingModel
from src.models import DocumentChunk


class VectorIndex:
    """In-memory vector index for local semantic retrieval."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.chunks: list[DocumentChunk] = []
        self.embeddings: list[list[float]] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.embeddings = self.embedding_model.encode([chunk.text for chunk in self.chunks]) if chunks else []

    def search_scores(self, query: str) -> list[float]:
        if not self.chunks:
            return []
        query_vector = self.embedding_model.encode([query])[0]
        return [cosine_similarity(query_vector, embedding) for embedding in self.embeddings]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)

