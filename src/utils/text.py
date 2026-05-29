from __future__ import annotations

import re
import unicodedata


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def clean_text(text: str) -> str:
    """Normalize extracted document text without destroying paragraph boundaries."""

    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\x00", " ")
    normalized = re.sub(r"[ \t\r\f\v]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    lines = [line.strip() for line in normalized.splitlines()]
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    return "\n".join(cleaned_lines).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if token not in STOPWORDS]
