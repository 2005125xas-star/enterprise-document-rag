from __future__ import annotations

import math
from collections import Counter

from src.utils.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: list[list[str]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_doc_length = 0.0

    def build(self, documents: list[str]) -> None:
        self._documents = [tokenize(document) for document in documents]
        self._term_frequencies = [Counter(document) for document in self._documents]
        self._document_frequency = Counter()
        for document in self._documents:
            self._document_frequency.update(set(document))
        self._average_doc_length = (
            sum(len(document) for document in self._documents) / len(self._documents) if self._documents else 0.0
        )

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        if not self._documents or not query_terms:
            return [0.0] * len(self._documents)

        scores: list[float] = []
        total_documents = len(self._documents)
        for document, term_frequency in zip(self._documents, self._term_frequencies):
            doc_length = len(document)
            score = 0.0
            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequency.get(term, 0)
                idf = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / (self._average_doc_length or 1.0)
                )
                score += idf * (frequency * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [1.0 if maximum > 0 else 0.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]

