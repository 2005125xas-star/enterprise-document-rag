from __future__ import annotations

import re
import time

from src.llm.providers import LLMProvider
from src.models import AnswerResult, RetrievalResult
from src.retrieval.hybrid import HybridRetriever
from src.utils.query_logger import QueryLogger


DEFAULT_INSUFFICIENT_EVIDENCE_MESSAGE = "I could not find enough evidence in the uploaded documents."


class QAPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        provider: LLMProvider,
        max_context_chunks: int = 5,
        min_score: float = 0.12,
        semantic_evidence_threshold: float = 0.28,
        insufficient_evidence_message: str = DEFAULT_INSUFFICIENT_EVIDENCE_MESSAGE,
        logger: QueryLogger | None = None,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.max_context_chunks = max_context_chunks
        self.min_score = min_score
        self.semantic_evidence_threshold = semantic_evidence_threshold
        self.insufficient_evidence_message = insufficient_evidence_message
        self.logger = logger

    def answer_question(self, question: str, top_k: int | None = None, session_id: str | None = None) -> AnswerResult:
        started = time.perf_counter()
        retrieval_results = self.retriever.search(question, top_k=top_k or self.max_context_chunks)
        evidence = retrieval_results[: self.max_context_chunks]

        if not self._has_sufficient_evidence(evidence):
            result = AnswerResult(
                answer=self.insufficient_evidence_message,
                citations=[],
                retrieval_results=retrieval_results,
                is_answered=False,
                latency_ms=_elapsed_ms(started),
                provider=self.provider.name,
            )
            self._log(question, result, session_id)
            return result

        citations = build_citations(evidence)
        answer = self.provider.generate(question, evidence, self.insufficient_evidence_message)
        if answer.strip() == self.insufficient_evidence_message:
            citations = []
            is_answered = False
        else:
            answer = ensure_inline_citation(answer, citations)
            is_answered = True

        result = AnswerResult(
            answer=answer,
            citations=citations,
            retrieval_results=retrieval_results,
            is_answered=is_answered,
            latency_ms=_elapsed_ms(started),
            provider=self.provider.name,
        )
        self._log(question, result, session_id)
        return result

    def _has_sufficient_evidence(self, evidence: list[RetrievalResult]) -> bool:
        if not evidence:
            return False
        best = evidence[0]
        if best.score < self.min_score:
            return False
        return best.keyword_score > 0.0 or best.semantic_score >= self.semantic_evidence_threshold

    def _log(self, question: str, result: AnswerResult, session_id: str | None) -> None:
        if self.logger is not None:
            self.logger.log_query(question=question, result=result, session_id=session_id)


def build_citations(evidence: list[RetrievalResult]) -> list[dict]:
    citations: list[dict] = []
    for index, result in enumerate(evidence, start=1):
        citations.append(
            {
                "index": index,
                "chunk_id": result.chunk.chunk_id,
                "document_id": result.chunk.document_id,
                "file_name": result.chunk.file_name,
                "page_number": result.chunk.page_number,
                "page_range": f"{result.chunk.page_number}-{result.chunk.page_number}",
                "score": round(result.score, 4),
                "preview": result.chunk.text[:240],
            }
        )
    return citations


def ensure_inline_citation(answer: str, citations: list[dict]) -> str:
    if not citations:
        return answer
    if any(f"[{citation['index']}]" in answer for citation in citations):
        return answer
    return f"{answer.rstrip()} [1]"


def validate_citations(answer: str, citations: list[dict]) -> bool:
    citation_indexes = {citation["index"] for citation in citations}
    referenced_indexes = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
    return bool(referenced_indexes) and referenced_indexes.issubset(citation_indexes)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
