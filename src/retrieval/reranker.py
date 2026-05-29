from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from src.models import RetrievalResult


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_FALLBACK_WARNING = "Reranker unavailable. Falling back to hybrid retrieval ranking."
RERANKER_LOAD_ERROR = "CrossEncoder reranker could not be loaded. Install dependencies or disable reranker."


class Reranker(Protocol):
    name: str
    enabled: bool
    model_name: str | None
    warning: str | None

    def rerank(self, query: str, candidates: list[RetrievalResult], final_k: int) -> list[RetrievalResult]:
        """Rerank retrieval candidates and return final evidence results."""


class RerankerConfigurationError(RuntimeError):
    """Raised when an enabled reranker cannot be initialized."""


class NoOpReranker:
    """Reranker that preserves the incoming hybrid retrieval order."""

    name = "disabled"
    enabled = False
    model_name = None

    def __init__(self, warning: str | None = None) -> None:
        self.warning = warning

    def rerank(self, query: str, candidates: list[RetrievalResult], final_k: int) -> list[RetrievalResult]:
        return _with_ranks(candidates[:final_k])


class CrossEncoderReranker:
    """Second-stage reranker backed by sentence-transformers CrossEncoder."""

    name = "cross-encoder"
    enabled = True
    warning = None

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        if model_factory is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - dependency behavior
                raise RerankerConfigurationError(RERANKER_LOAD_ERROR) from exc
            model_factory = CrossEncoder
        try:
            self._model = model_factory(model_name)
        except Exception as exc:
            raise RerankerConfigurationError(RERANKER_LOAD_ERROR) from exc

    def rerank(self, query: str, candidates: list[RetrievalResult], final_k: int) -> list[RetrievalResult]:
        if final_k <= 0 or not candidates:
            return []
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self._model.predict(pairs)
        scored = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
        reranked = [
            RetrievalResult(
                chunk=result.chunk,
                semantic_score=result.semantic_score,
                keyword_score=result.keyword_score,
                score=result.score,
                rank=rank,
                hybrid_score=result.hybrid_score if result.hybrid_score is not None else result.score,
                rerank_score=float(score),
            )
            for rank, (result, score) in enumerate(scored[:final_k], start=1)
        ]
        return reranked


def create_reranker(
    settings: Mapping[str, Any] | None,
    environ: Mapping[str, str] | None = None,
    model_factory: Callable[[str], Any] | None = None,
) -> Reranker:
    import os

    env = environ if environ is not None else os.environ
    config = dict(settings or {})
    enabled = _env_bool(env.get("RERANKER_ENABLED"), default=_to_bool(config.get("enabled", False)))
    if not enabled:
        return NoOpReranker()

    model_name = str(env.get("RERANKER_MODEL") or config.get("model") or DEFAULT_RERANKER_MODEL)
    allow_fallback = _to_bool(config.get("allow_fallback", True))
    try:
        return CrossEncoderReranker(model_name=model_name, model_factory=model_factory)
    except RerankerConfigurationError:
        if allow_fallback:
            fallback = NoOpReranker(warning=RERANKER_FALLBACK_WARNING)
            fallback.model_name = model_name
            return fallback
        raise


def resolve_reranker_settings(settings: Mapping[str, Any] | None, environ: Mapping[str, str] | None = None) -> dict:
    import os

    env = environ if environ is not None else os.environ
    config = dict(settings or {})
    enabled = _env_bool(env.get("RERANKER_ENABLED"), default=_to_bool(config.get("enabled", False)))
    final_k = _positive_int(env.get("RERANKER_FINAL_K") or config.get("final_k"), default=5)
    top_n = _positive_int(env.get("RERANKER_TOP_N") or config.get("top_n"), default=20)
    return {
        "enabled": enabled,
        "model": str(env.get("RERANKER_MODEL") or config.get("model") or DEFAULT_RERANKER_MODEL),
        "top_n": max(top_n, final_k),
        "final_k": final_k,
        "allow_fallback": _to_bool(config.get("allow_fallback", True)),
    }


def _with_ranks(results: list[RetrievalResult]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=result.chunk,
            semantic_score=result.semantic_score,
            keyword_score=result.keyword_score,
            score=result.score,
            rank=rank,
            hybrid_score=result.hybrid_score,
            rerank_score=result.rerank_score,
        )
        for rank, result in enumerate(results, start=1)
    ]


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return _to_bool(value)


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
