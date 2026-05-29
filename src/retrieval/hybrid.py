from __future__ import annotations

from src.indexing.embeddings import EmbeddingModel
from src.indexing.vector_index import VectorIndex
from src.models import DocumentChunk, RetrievalResult
from src.retrieval.bm25 import BM25Index, min_max_normalize


class HybridRetriever:
    """Combines semantic cosine similarity and BM25 keyword evidence."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        semantic_weight: float = 0.65,
        keyword_weight: float = 0.35,
        min_score: float = 0.12,
    ) -> None:
        if semantic_weight < 0 or keyword_weight < 0:
            raise ValueError("Retrieval weights cannot be negative.")
        if semantic_weight + keyword_weight == 0:
            raise ValueError("At least one retrieval weight must be positive.")
        total = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total
        self.min_score = min_score
        self.vector_index = VectorIndex(embedding_model)
        self.keyword_index = BM25Index()
        self.chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        seen_chunk_ids: set[str] = set()
        self.chunks = []
        for chunk in chunks:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            self.chunks.append(chunk)
        self.vector_index.build(self.chunks)
        self.keyword_index.build([chunk.text for chunk in self.chunks])

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        if not self.chunks:
            return []

        semantic_raw = self.vector_index.search_scores(query)
        semantic_scores = [max(0.0, score) for score in semantic_raw]
        keyword_scores = min_max_normalize(self.keyword_index.score(query))
        combined: list[RetrievalResult] = []

        for index, chunk in enumerate(self.chunks):
            score = self.semantic_weight * semantic_scores[index] + self.keyword_weight * keyword_scores[index]
            if score < self.min_score:
                continue
            combined.append(
                RetrievalResult(
                    chunk=chunk,
                    semantic_score=semantic_scores[index],
                    keyword_score=keyword_scores[index],
                    score=score,
                    rank=0,
                )
            )

        combined.sort(key=lambda result: result.score, reverse=True)
        return [
            RetrievalResult(
                chunk=result.chunk,
                semantic_score=result.semantic_score,
                keyword_score=result.keyword_score,
                score=result.score,
                rank=rank,
            )
            for rank, result in enumerate(combined[:top_k], start=1)
        ]
