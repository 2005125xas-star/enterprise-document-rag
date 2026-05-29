from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.evaluation.runner import evaluate
from src.indexing.embeddings import EmbeddingModel, create_embedding_model
from src.indexing.vector_store import VectorStore, create_vector_store
from src.ingestion.chunker import chunk_pages
from src.ingestion.parsers import parse_document
from src.llm.providers import LLMProvider, create_llm_provider
from src.models import AnswerResult, DocumentChunk, DocumentPage, EvaluationExample, EvaluationResult
from src.qa.pipeline import QAPipeline
from src.retrieval.hybrid import HybridRetriever
from src.utils.query_logger import QueryLogger


class EnterpriseRAGSystem:
    """Reusable service used by Streamlit, tests, and evaluation scripts."""

    def __init__(
        self,
        config: dict,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        provider: LLMProvider | None = None,
        logger: QueryLogger | None = None,
    ) -> None:
        self.config = config
        retrieval_config = config["retrieval"]
        qa_config = config["qa"]
        logging_config = config["logging"]

        self.embedding_model = embedding_model or create_embedding_model(
            retrieval_config["embedding_model"],
            allow_fallback=True,
        )
        self.provider = provider or create_llm_provider(qa_config)
        self.logger = logger or QueryLogger(logging_config["sqlite_path"])
        self.vector_store = vector_store or create_vector_store(retrieval_config)
        self.retriever = HybridRetriever(
            embedding_model=self.embedding_model,
            semantic_weight=float(retrieval_config["semantic_weight"]),
            keyword_weight=float(retrieval_config["keyword_weight"]),
            min_score=float(retrieval_config["min_score"]),
            vector_store=self.vector_store,
        )
        self.pipeline = QAPipeline(
            retriever=self.retriever,
            provider=self.provider,
            max_context_chunks=int(qa_config["max_context_chunks"]),
            min_score=float(retrieval_config["min_score"]),
            insufficient_evidence_message=qa_config["insufficient_evidence_message"],
            logger=self.logger,
        )
        self.pages: list[DocumentPage] = []
        self.chunks: list[DocumentChunk] = self.retriever.load_persisted()

    def ingest_paths(self, paths: Iterable[str | Path]) -> list[DocumentChunk]:
        chunking_config = self.config["chunking"]
        pages: list[DocumentPage] = []
        for path in paths:
            pages.extend(parse_document(path))
        chunks = chunk_pages(
            pages,
            chunk_size=int(chunking_config["chunk_size"]),
            chunk_overlap=int(chunking_config["chunk_overlap"]),
        )
        self.pages = pages
        self.chunks = chunks
        self.retriever.build(chunks)
        self.chunks = self.retriever.chunks
        return chunks

    def answer(self, question: str, top_k: int | None = None, session_id: str | None = None) -> AnswerResult:
        return self.pipeline.answer_question(question=question, top_k=top_k, session_id=session_id)

    def evaluate(self, examples: list[EvaluationExample], top_k: int = 5) -> EvaluationResult:
        return evaluate(self.pipeline, examples, top_k=top_k)

    def recent_logs(self, limit: int = 50) -> list[dict]:
        return self.logger.list_recent(limit=limit)

    def vector_store_count(self) -> int:
        return self.vector_store.count()

    def clear_vector_store(self) -> None:
        self.retriever.clear()
        self.pages = []
        self.chunks = []
