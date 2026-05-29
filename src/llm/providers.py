from __future__ import annotations

import os
import re
from typing import Protocol

from src.models import RetrievalResult
from src.utils.text import tokenize


class LLMProvider(Protocol):
    name: str

    def generate(self, question: str, evidence: list[RetrievalResult], insufficient_message: str) -> str:
        """Generate a source-grounded answer from retrieved evidence."""


class MockProvider:
    """Deterministic provider used when no OpenAI API key is available."""

    name = "mock"

    def generate(self, question: str, evidence: list[RetrievalResult], insufficient_message: str) -> str:
        if not evidence:
            return insufficient_message

        query_terms = set(tokenize(question))
        best = evidence[0]
        sentence = _best_sentence(best.chunk.text, query_terms)
        return f"Based on the uploaded documents, {sentence} [1]"


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, api_key: str | None = None) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency behavior
            raise RuntimeError("openai is required for OpenAIProvider. Install requirements.txt.") from exc

        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=resolved_key)

    def generate(self, question: str, evidence: list[RetrievalResult], insufficient_message: str) -> str:
        context = "\n\n".join(
            f"[{index}] {result.chunk.file_name}, page {result.chunk.page_number}, chunk {result.chunk.chunk_id}\n"
            f"{result.chunk.text}"
            for index, result in enumerate(evidence, start=1)
        )
        system_prompt = (
            "You are an enterprise document QA assistant. Answer only from the provided context. "
            "Cite sources inline with bracket numbers like [1]. "
            f"If the context is insufficient, respond exactly: {insufficient_message}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
            ],
        )
        return response.choices[0].message.content or insufficient_message


def create_llm_provider(settings: dict) -> LLMProvider:
    provider_name = str(settings.get("provider", "auto")).lower()
    model = settings.get("model", "gpt-4o-mini")
    temperature = float(settings.get("temperature", 0.0))

    if provider_name == "mock":
        return MockProvider()
    if provider_name == "openai":
        return OpenAIProvider(model=model, temperature=temperature)
    if provider_name == "auto" and os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider(model=model, temperature=temperature)
    return MockProvider()


def _best_sentence(text: str, query_terms: set[str]) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if not sentences:
        return text.strip()
    if not query_terms:
        return sentences[0]
    return max(sentences, key=lambda sentence: len(query_terms.intersection(tokenize(sentence))))

