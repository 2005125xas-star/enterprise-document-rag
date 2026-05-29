from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_pages
from src.ingestion.parsers import build_document_id, parse_document
from src.models import DocumentPage


def test_parse_txt_preserves_metadata_and_cleans_text(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("  Enterprise   policy\n\n\nRetention is 7 years.  ", encoding="utf-8")

    pages = parse_document(path)

    assert len(pages) == 1
    assert pages[0].file_name == "policy.txt"
    assert pages[0].page_number == 1
    assert pages[0].text == "Enterprise policy\n\nRetention is 7 years."
    assert pages[0].document_id == build_document_id(path)
    assert pages[0].metadata["source_path"] == str(path)


def test_parse_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"not actually an image")

    with pytest.raises(ValueError, match="Unsupported document type"):
        parse_document(path)


def test_chunk_pages_preserves_document_metadata() -> None:
    page = DocumentPage(
        document_id="doc-1",
        file_name="handbook.txt",
        page_number=2,
        text="alpha beta gamma delta epsilon zeta eta theta",
        metadata={"department": "hr"},
    )

    chunks = chunk_pages([page], chunk_size=24, chunk_overlap=5)

    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "doc-1:p2:c1"
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].file_name == "handbook.txt"
    assert chunks[0].page_number == 2
    assert chunks[0].metadata["department"] == "hr"
    assert chunks[0].start_char < chunks[0].end_char


def test_chunk_overlap_and_chunk_ids_are_unique() -> None:
    pages = [
        DocumentPage("doc-1", "handbook.txt", 1, "abcdefghijklmnopqrstuvwxyz"),
        DocumentPage("doc-1", "handbook.txt", 2, "0123456789abcdef"),
    ]

    chunks = chunk_pages(pages, chunk_size=10, chunk_overlap=3)

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert any(chunk.chunk_id.startswith("doc-1:p2:") for chunk in chunks)


def test_chunk_pages_validates_overlap() -> None:
    page = DocumentPage("doc", "file.txt", 1, "text")

    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_pages([page], chunk_size=100, chunk_overlap=100)
