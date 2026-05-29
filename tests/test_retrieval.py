from __future__ import annotations

from src.indexing.embeddings import HashingEmbeddingModel
from src.indexing.vector_index import cosine_similarity
from src.models import DocumentChunk
from src.retrieval.bm25 import BM25Index, min_max_normalize
from src.retrieval.hybrid import HybridRetriever


class StaticEmbeddingModel:
    name = "static"

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if lowered == "alpha" or "semantic" in lowered:
                vectors.append([1.0, 0.0])
            elif "alpha" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.0, 0.0])
        return vectors


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split(":")[0],
        file_name=f"{chunk_id.split(':')[0]}.txt",
        page_number=1,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def test_bm25_scores_keyword_matches_highest() -> None:
    index = BM25Index()
    index.build(["vacation policy and benefits", "database backup rotation", "security incident response"])

    scores = index.score("backup database")

    assert scores[1] == max(scores)
    assert min_max_normalize(scores)[1] == 1.0


def test_hashing_embedding_model_returns_normalized_vectors() -> None:
    model = HashingEmbeddingModel(dimensions=32)
    first, second = model.encode(["data retention", "data retention policy"])

    assert len(first) == 32
    assert cosine_similarity(first, second) > 0


def test_hybrid_retriever_returns_ranked_results_with_metadata() -> None:
    chunks = [
        _chunk("doc1:p1:c1", "Employees must complete security training every year."),
        _chunk("doc2:p1:c1", "Customer records follow a seven year data retention policy."),
        _chunk("doc3:p1:c1", "Travel expenses require manager approval."),
    ]
    retriever = HybridRetriever(HashingEmbeddingModel(dimensions=64), min_score=0.0)
    retriever.build(chunks)

    results = retriever.search("What is the data retention policy?", top_k=2)

    assert [result.rank for result in results] == [1, 2]
    assert results[0].chunk.chunk_id == "doc2:p1:c1"
    assert results[0].chunk.file_name == "doc2.txt"
    assert results[0].score >= results[1].score


def test_semantic_retrieval_result_preserves_metadata() -> None:
    chunks = [
        _chunk("doc1:p3:c1", "semantic evidence for onboarding controls"),
        _chunk("doc2:p1:c1", "unrelated travel expenses"),
    ]
    retriever = HybridRetriever(StaticEmbeddingModel(), semantic_weight=1.0, keyword_weight=0.0, min_score=0.0)
    retriever.build(chunks)

    results = retriever.search("alpha", top_k=1)

    assert results[0].chunk.chunk_id == "doc1:p3:c1"
    assert results[0].chunk.file_name == "doc1.txt"
    assert results[0].chunk.page_number == 1
    assert results[0].semantic_score == 1.0


def test_bm25_retrieval_result_preserves_metadata() -> None:
    chunks = [
        _chunk("doc1:p1:c1", "backup rotation schedule and restore testing"),
        _chunk("doc2:p1:c1", "travel reimbursement policy"),
    ]
    retriever = HybridRetriever(StaticEmbeddingModel(), semantic_weight=0.0, keyword_weight=1.0, min_score=0.0)
    retriever.build(chunks)

    results = retriever.search("backup restore", top_k=1)

    assert results[0].chunk.chunk_id == "doc1:p1:c1"
    assert results[0].keyword_score == 1.0
    assert results[0].chunk.file_name == "doc1.txt"


def test_hybrid_score_merging_and_deduplication() -> None:
    chunks = [
        _chunk("doc1:p1:c1", "alpha keyword evidence"),
        _chunk("doc2:p1:c1", "semantic evidence"),
        _chunk("doc2:p1:c1", "semantic evidence duplicated"),
    ]
    retriever = HybridRetriever(StaticEmbeddingModel(), semantic_weight=0.7, keyword_weight=0.3, min_score=0.0)
    retriever.build(chunks)

    results = retriever.search("alpha", top_k=10)

    assert len({result.chunk.chunk_id for result in results}) == len(results)
    assert [chunk.chunk_id for chunk in retriever.chunks] == ["doc1:p1:c1", "doc2:p1:c1"]
    assert results[0].chunk.chunk_id == "doc2:p1:c1"
    assert results[0].score == 0.7
    assert results[1].score == 0.3
