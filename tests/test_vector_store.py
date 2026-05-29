from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.indexing.embeddings import HashingEmbeddingModel
from src.indexing.vector_store import (
    ChromaVectorStore,
    MemoryVectorStore,
    VectorStoreConfigurationError,
    create_vector_store,
)
from src.models import DocumentChunk
from src.retrieval.hybrid import HybridRetriever
from src.utils.config import DEFAULT_CONFIG


class FakeChromaClient:
    def __init__(self):
        self.collections: dict[str, FakeChromaCollection] = {}

    def get_or_create_collection(self, name: str, metadata: dict | None = None):
        if name not in self.collections:
            self.collections[name] = FakeChromaCollection()
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)


class FakeChromaCollection:
    def __init__(self):
        self.items: dict[str, dict] = {}

    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        for item_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            self.items[item_id] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

    def query(self, query_embeddings, n_results, include):
        query_embedding = query_embeddings[0]
        scored = []
        for item_id, item in self.items.items():
            score = _cosine(query_embedding, item["embedding"])
            scored.append((item_id, item, 1.0 - score))
        scored.sort(key=lambda item: item[2])
        selected = scored[:n_results]
        return {
            "ids": [[item_id for item_id, _, _ in selected]],
            "documents": [[item["document"] for _, item, _ in selected]],
            "metadatas": [[item["metadata"] for _, item, _ in selected]],
            "distances": [[distance for _, _, distance in selected]],
        }

    def get(self, include):
        ordered = sorted(self.items.items())
        return {
            "ids": [item_id for item_id, _ in ordered],
            "documents": [item["document"] for _, item in ordered],
            "metadatas": [item["metadata"] for _, item in ordered],
        }

    def count(self) -> int:
        return len(self.items)


def _chunk(chunk_id: str, text: str = "Customer records are retained for seven years.") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split(":")[0],
        file_name=f"{chunk_id.split(':')[0]}.txt",
        page_number=2,
        text=text,
        start_char=10,
        end_char=10 + len(text),
    )


def test_memory_vector_store_adds_and_searches_chunks() -> None:
    store = MemoryVectorStore()
    chunks = [_chunk("doc1:p2:c1", "alpha policy"), _chunk("doc2:p2:c1", "beta policy")]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    store.add_chunks(chunks, embeddings)
    results = store.search([1.0, 0.0], top_k=1)

    assert store.count() == 2
    assert results[0].chunk.chunk_id == "doc1:p2:c1"
    assert results[0].semantic_score == 1.0


def test_chroma_vector_store_preserves_metadata_for_citations(tmp_path: Path) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path,
        collection_name="test_chunks",
        client=FakeChromaClient(),
    )
    chunk = _chunk("policy:p2:c7")

    store.add_chunks([chunk], [[1.0, 0.0]])
    result = store.search([1.0, 0.0], top_k=1)[0]
    loaded = store.get_chunks()[0]

    assert store.count() == 1
    assert result.chunk.chunk_id == chunk.chunk_id
    assert result.chunk.document_id == chunk.document_id
    assert result.chunk.file_name == chunk.file_name
    assert result.chunk.page_number == 2
    assert result.chunk.start_char == 10
    assert result.chunk.end_char == chunk.end_char
    assert loaded == result.chunk


def test_vector_store_config_defaults_to_memory() -> None:
    store = create_vector_store(DEFAULT_CONFIG["retrieval"], environ={})

    assert isinstance(store, MemoryVectorStore)


def test_chroma_without_chromadb_reports_clean_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "chromadb", None)

    with pytest.raises(VectorStoreConfigurationError, match="Chroma is not installed"):
        ChromaVectorStore(persist_directory=tmp_path, collection_name="missing_chroma")


def test_hybrid_retrieval_still_works_with_memory_backend() -> None:
    chunks = [
        _chunk("doc1:p2:c1", "Employees complete annual security training."),
        _chunk("doc2:p2:c1", "Customer records follow a seven year retention policy."),
    ]
    retriever = HybridRetriever(HashingEmbeddingModel(dimensions=64), min_score=0.0, vector_store=MemoryVectorStore())

    retriever.build(chunks)
    results = retriever.search("What is the retention policy?", top_k=1)

    assert results[0].chunk.chunk_id == "doc2:p2:c1"
    assert results[0].chunk.file_name == "doc2.txt"


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
