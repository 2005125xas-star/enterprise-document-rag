from __future__ import annotations

from src.indexing.embeddings import EmbeddingModel
from src.indexing.vector_store import MemoryVectorStore, VectorStore
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
        vector_store: VectorStore | None = None,
    ) -> None:
        if semantic_weight < 0 or keyword_weight < 0:
            raise ValueError("Retrieval weights cannot be negative.")
        if semantic_weight + keyword_weight == 0:
            raise ValueError("At least one retrieval weight must be positive.")
        total = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total
        self.min_score = min_score
        self.embedding_model = embedding_model
        self.vector_store = vector_store or MemoryVectorStore()
        self.keyword_index = BM25Index()
        self.chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        new_chunks = _deduplicate_chunks(chunks)
        embeddings = self.embedding_model.encode([chunk.text for chunk in new_chunks]) if new_chunks else []
        if self.vector_store.name == "memory":
            self.vector_store.clear()
        self.vector_store.add_chunks(new_chunks, embeddings)
        self.chunks = _deduplicate_chunks(self.vector_store.get_chunks())
        self.keyword_index.build([chunk.text for chunk in self.chunks])

    def load_persisted(self) -> list[DocumentChunk]:
        self.chunks = _deduplicate_chunks(self.vector_store.get_chunks())
        self.keyword_index.build([chunk.text for chunk in self.chunks])
        return self.chunks

    def clear(self) -> None:
        self.vector_store.clear()
        self.chunks = []
        self.keyword_index.build([])

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        if not self.chunks:
            return []

        query_embedding = self.embedding_model.encode([query])[0]
        semantic_results = self.vector_store.search(query_embedding, top_k=len(self.chunks))
        semantic_by_chunk_id = {result.chunk.chunk_id: max(0.0, result.semantic_score) for result in semantic_results}
        keyword_scores = min_max_normalize(self.keyword_index.score(query))
        combined: list[RetrievalResult] = []

        for index, chunk in enumerate(self.chunks):
            semantic_score = semantic_by_chunk_id.get(chunk.chunk_id, 0.0)
            score = self.semantic_weight * semantic_score + self.keyword_weight * keyword_scores[index]
            if score < self.min_score:
                continue
            combined.append(
                RetrievalResult(
                    chunk=chunk,
                    semantic_score=semantic_score,
                    keyword_score=keyword_scores[index],
                    score=score,
                    rank=0,
                    hybrid_score=score,
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
                hybrid_score=result.hybrid_score if result.hybrid_score is not None else result.score,
                rerank_score=result.rerank_score,
            )
            for rank, result in enumerate(combined[:top_k], start=1)
        ]


def _deduplicate_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    seen_chunk_ids: set[str] = set()
    deduplicated: list[DocumentChunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        deduplicated.append(chunk)
    return deduplicated
