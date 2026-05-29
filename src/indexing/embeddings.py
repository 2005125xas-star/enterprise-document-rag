from __future__ import annotations

import hashlib
import math
from typing import Protocol

from src.utils.text import tokenize


class EmbeddingModel(Protocol):
    name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per text."""


class SentenceTransformerEmbeddingModel:
    """sentence-transformers backed embedder used by the real retrieval path."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency behavior
            raise RuntimeError("sentence-transformers is required for semantic retrieval.") from exc

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


class HashingEmbeddingModel:
    """Small deterministic fallback embedder for tests and offline smoke runs."""

    def __init__(self, dimensions: int = 384) -> None:
        self.name = f"hashing-{dimensions}"
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def create_embedding_model(model_name: str, allow_fallback: bool = True) -> EmbeddingModel:
    """Create the production embedder, optionally falling back for local demos/tests.

    The default production path is sentence-transformers. The hashing model is only
    used when fallback is allowed and the production model cannot be imported or
    loaded in the current environment.
    """

    try:
        return SentenceTransformerEmbeddingModel(model_name)
    except Exception:
        if allow_fallback:
            return HashingEmbeddingModel()
        raise
