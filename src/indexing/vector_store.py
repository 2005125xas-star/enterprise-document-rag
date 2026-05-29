from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from src.indexing.vector_index import cosine_similarity
from src.models import DocumentChunk, RetrievalResult


DEFAULT_CHROMA_PERSIST_DIRECTORY = "data/vector_store"
DEFAULT_CHROMA_COLLECTION_NAME = "enterprise_document_chunks"
CHROMA_MISSING_MESSAGE = "Chroma is not installed. Please run: python -m pip install -r requirements.txt"


class VectorStore(Protocol):
    name: str

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Store chunks and precomputed embeddings."""

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """Return semantic retrieval results with higher scores representing better matches."""

    def count(self) -> int:
        """Return the number of stored chunks."""

    def clear(self) -> None:
        """Remove stored chunks from this vector store."""

    def get_chunks(self) -> list[DocumentChunk]:
        """Load stored chunks with citation metadata."""


class VectorStoreConfigurationError(RuntimeError):
    """Raised when a selected vector store backend cannot be initialized."""


class MemoryVectorStore:
    """In-memory vector store preserving the current local retrieval behavior."""

    name = "memory"
    persist_directory = None
    collection_name = None

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._embeddings: list[list[float]] = []

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self._chunks = list(chunks)
        self._embeddings = list(embeddings)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        if top_k <= 0 or not self._chunks:
            return []
        scored = [
            RetrievalResult(
                chunk=chunk,
                semantic_score=max(0.0, cosine_similarity(query_embedding, embedding)),
                keyword_score=0.0,
                score=max(0.0, cosine_similarity(query_embedding, embedding)),
                rank=0,
            )
            for chunk, embedding in zip(self._chunks, self._embeddings)
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return [
            RetrievalResult(
                chunk=result.chunk,
                semantic_score=result.semantic_score,
                keyword_score=0.0,
                score=result.score,
                rank=rank,
            )
            for rank, result in enumerate(scored[:top_k], start=1)
        ]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = []

    def get_chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)


class ChromaVectorStore:
    """Persistent Chroma vector store using project-computed embeddings."""

    name = "chroma"

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_CHROMA_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        if client is None:
            if client_factory is None:
                try:
                    import chromadb
                except ImportError as exc:
                    raise VectorStoreConfigurationError(CHROMA_MISSING_MESSAGE) from exc
                client_factory = chromadb.PersistentClient
            client = client_factory(path=self.persist_directory)

        self._client = client
        self._collection = self._get_or_create_collection()

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length.")
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_metadata_from_chunk(chunk) for chunk in chunks],
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        stored_count = self.count()
        if top_k <= 0 or stored_count == 0:
            return []
        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, stored_count),
            include=["documents", "metadatas", "distances"],
        )
        ids = _first_row(response.get("ids", []))
        documents = _first_row(response.get("documents", []))
        metadatas = _first_row(response.get("metadatas", []))
        distances = _first_row(response.get("distances", []))

        results: list[RetrievalResult] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            chunk = _chunk_from_chroma(chunk_id=chunk_id, text=text, metadata=metadata or {})
            score = _score_from_distance(distance)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    semantic_score=score,
                    keyword_score=0.0,
                    score=score,
                    rank=rank,
                )
            )
        return results

    def count(self) -> int:
        return int(self._collection.count())

    def clear(self) -> None:
        if hasattr(self._client, "delete_collection"):
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
        self._collection = self._get_or_create_collection()

    def get_chunks(self) -> list[DocumentChunk]:
        if self.count() == 0:
            return []
        response = self._collection.get(include=["documents", "metadatas"])
        ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])
        chunks = [
            _chunk_from_chroma(chunk_id=chunk_id, text=text, metadata=metadata or {})
            for chunk_id, text, metadata in zip(ids, documents, metadatas)
        ]
        return sorted(chunks, key=lambda chunk: chunk.chunk_id)

    def _get_or_create_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def create_vector_store(
    settings: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> VectorStore:
    import os

    env = environ if environ is not None else os.environ
    backend = str(env.get("VECTOR_STORE") or settings.get("vector_store", "memory")).strip().lower()

    if backend == "memory":
        return MemoryVectorStore()
    if backend == "chroma":
        return ChromaVectorStore(
            persist_directory=env.get("CHROMA_PERSIST_DIRECTORY")
            or settings.get("persist_directory", DEFAULT_CHROMA_PERSIST_DIRECTORY),
            collection_name=env.get("CHROMA_COLLECTION_NAME")
            or settings.get("collection_name", DEFAULT_CHROMA_COLLECTION_NAME),
        )
    raise VectorStoreConfigurationError(f"Unsupported vector store backend '{backend}'. Expected memory or chroma.")


def _metadata_from_chunk(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "file_name": chunk.file_name,
        "page_number": int(chunk.page_number),
        "start_char": int(chunk.start_char),
        "end_char": int(chunk.end_char),
    }


def _chunk_from_chroma(chunk_id: str, text: str, metadata: Mapping[str, Any]) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=str(metadata.get("chunk_id") or chunk_id),
        document_id=str(metadata.get("document_id") or ""),
        file_name=str(metadata.get("file_name") or ""),
        page_number=int(metadata.get("page_number") or 1),
        text=text or "",
        start_char=int(metadata.get("start_char") or 0),
        end_char=int(metadata.get("end_char") or len(text or "")),
    )


def _score_from_distance(distance: object) -> float:
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - value))


def _first_row(value: list) -> list:
    if not value:
        return []
    first = value[0]
    return first if isinstance(first, list) else value
