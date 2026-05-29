from __future__ import annotations

from src.models import DocumentChunk, DocumentPage
from src.utils.text import clean_text


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[DocumentChunk] = []
    for page in pages:
        page_text = clean_text(page.text)
        if not page_text:
            continue
        for local_index, (start, end, text) in enumerate(_iter_chunk_spans(page_text, chunk_size, chunk_overlap), start=1):
            chunk_id = f"{page.document_id}:p{page.page_number}:c{local_index}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    file_name=page.file_name,
                    page_number=page.page_number,
                    text=text,
                    start_char=start,
                    end_char=end,
                    metadata={**page.metadata, "chunk_index_on_page": local_index},
                )
            )
    return chunks


def _iter_chunk_spans(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    text_length = len(text)
    start = 0
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            boundary = text.rfind(" ", start, end)
            if boundary > start + max(50, chunk_size // 3):
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            stripped_start = start + len(text[start:end]) - len(text[start:end].lstrip())
            stripped_end = end - (len(text[start:end]) - len(text[start:end].rstrip()))
            spans.append((stripped_start, stripped_end, chunk))
        if end >= text_length:
            break
        start = max(0, end - chunk_overlap)
        while start < text_length and text[start].isspace():
            start += 1
    return spans

