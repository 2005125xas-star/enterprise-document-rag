from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from urllib.parse import urlparse

from src.models import RetrievalResult
from src.utils.text import tokenize


OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
MISSING_COMPATIBLE_KEY_MESSAGE = "LLM_API_KEY is not set for openai_compatible provider."
MISSING_OPENAI_KEY_MESSAGE = "OPENAI_API_KEY is not set."
LLM_REQUEST_FAILED_MESSAGE = (
    "LLM request failed. Please check your API key, base URL, model name, provider quota, and network connection."
)


class LLMProvider(Protocol):
    name: str

    def generate(self, question: str, evidence: list[RetrievalResult], insufficient_message: str) -> str:
        """Generate a source-grounded answer from retrieved evidence."""


class ProviderConfigurationError(RuntimeError):
    """Raised when a selected LLM provider is missing required configuration."""


class LLMProviderRequestError(RuntimeError):
    """Raised when a configured LLM provider cannot complete a request."""


class MockProvider:
    """Deterministic provider used when no API key is available."""

    name = "mock"
    model = "mock"
    base_url = None
    backend_label = "local mock"

    def generate(self, question: str, evidence: list[RetrievalResult], insufficient_message: str) -> str:
        if not evidence:
            return insufficient_message

        query_terms = set(tokenize(question))
        best = evidence[0]
        sentence = _best_sentence(best.chunk.text, query_terms)
        return f"Based on the uploaded documents, {sentence} [1]"


class OpenAICompatibleProvider:
    """LLM provider for OpenAI-compatible chat completion APIs."""

    name = "openai-compatible"

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        client_factory: Callable[..., Any] | None = None,
        missing_key_message: str = MISSING_COMPATIBLE_KEY_MESSAGE,
    ) -> None:
        if not _present(api_key):
            raise ProviderConfigurationError(missing_key_message)

        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency behavior
                raise ProviderConfigurationError(
                    "openai is required for OpenAI-compatible providers. Install requirements.txt."
                ) from exc
            client_factory = OpenAI

        self.api_key = api_key
        self.base_url = _clean(base_url) or OPENAI_DEFAULT_BASE_URL
        self.model = model
        self.temperature = temperature
        self._client = client_factory(api_key=api_key, base_url=self.base_url)

    @property
    def backend_label(self) -> str:
        return _base_url_domain(self.base_url)

    def generate(self, question: str, evidence: list[RetrievalResult], insufficient_message: str) -> str:
        if not evidence:
            return insufficient_message

        context = "\n\n".join(
            f"[{index}] {result.chunk.file_name}, page {result.chunk.page_number}, chunk {result.chunk.chunk_id}\n"
            f"{result.chunk.text}"
            for index, result in enumerate(evidence, start=1)
        )
        system_prompt = (
            "You are an enterprise document QA assistant. Answer only from the provided context. "
            "Do not use outside knowledge. Cite sources inline with bracket numbers like [1]. "
            f"If the context is insufficient, respond exactly: {insufficient_message}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
                ],
            )
        except Exception as exc:  # pragma: no cover - SDK/provider exceptions vary
            raise LLMProviderRequestError(LLM_REQUEST_FAILED_MESSAGE) from exc

        return response.choices[0].message.content or insufficient_message


class OpenAIProvider(OpenAICompatibleProvider):
    """Backward-compatible OpenAI provider wrapper."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or OPENAI_DEFAULT_BASE_URL,
            model=model,
            temperature=temperature,
            client_factory=client_factory,
            missing_key_message=MISSING_OPENAI_KEY_MESSAGE,
        )


def create_llm_provider(
    settings: dict,
    environ: Mapping[str, str] | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> LLMProvider:
    env = environ if environ is not None else os.environ
    provider_name = _provider_name(_first_present(env.get("LLM_PROVIDER"), settings.get("provider"), "auto"))
    model = str(_first_present(env.get("LLM_MODEL"), settings.get("model"), "gpt-4o-mini"))
    temperature = float(settings.get("temperature", 0.0))
    config_base_url = _clean(settings.get("base_url"))

    llm_api_key = _clean(env.get("LLM_API_KEY"))
    openai_api_key = _clean(env.get("OPENAI_API_KEY"))
    llm_base_url = _clean(env.get("LLM_BASE_URL"))
    openai_base_url = _clean(env.get("OPENAI_BASE_URL"))

    if provider_name == "mock":
        return MockProvider()

    if provider_name == "auto":
        if llm_api_key:
            return OpenAICompatibleProvider(
                api_key=llm_api_key,
                base_url=llm_base_url or config_base_url,
                model=model,
                temperature=temperature,
                client_factory=client_factory,
            )
        if openai_api_key:
            return OpenAICompatibleProvider(
                api_key=openai_api_key,
                base_url=llm_base_url or openai_base_url or config_base_url or OPENAI_DEFAULT_BASE_URL,
                model=model,
                temperature=temperature,
                client_factory=client_factory,
                missing_key_message=MISSING_OPENAI_KEY_MESSAGE,
            )
        return MockProvider()

    if provider_name == "openai_compatible":
        return OpenAICompatibleProvider(
            api_key=llm_api_key,
            base_url=llm_base_url or config_base_url,
            model=model,
            temperature=temperature,
            client_factory=client_factory,
            missing_key_message=MISSING_COMPATIBLE_KEY_MESSAGE,
        )

    if provider_name == "openai":
        return OpenAICompatibleProvider(
            api_key=openai_api_key or llm_api_key,
            base_url=llm_base_url or openai_base_url or config_base_url or OPENAI_DEFAULT_BASE_URL,
            model=model,
            temperature=temperature,
            client_factory=client_factory,
            missing_key_message=MISSING_OPENAI_KEY_MESSAGE,
        )

    raise ValueError(f"Unsupported LLM provider '{provider_name}'. Expected mock, auto, openai, or openai_compatible.")


def _provider_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _first_present(*values: object) -> object | None:
    for value in values:
        if _present(value):
            return value
    return None


def _present(value: object) -> bool:
    return bool(value is not None and str(value).strip())


def _clean(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _base_url_domain(base_url: str | None) -> str:
    if not base_url:
        return "api.openai.com"
    parsed = urlparse(base_url)
    return parsed.netloc or parsed.path or base_url


def _best_sentence(text: str, query_terms: set[str]) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if not sentences:
        return text.strip()
    if not query_terms:
        return sentences[0]
    return max(sentences, key=lambda sentence: len(query_terms.intersection(tokenize(sentence))))
