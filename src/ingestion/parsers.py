from __future__ import annotations

import hashlib
from pathlib import Path

from src.models import DocumentPage
from src.utils.text import clean_text


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def build_document_id(file_path: str | Path, content: bytes | None = None) -> str:
    path = Path(file_path)
    payload = content if content is not None else path.read_bytes()
    digest = hashlib.sha256(path.name.encode("utf-8") + b"\0" + payload).hexdigest()
    return digest[:16]


def parse_document(file_path: str | Path, document_id: str | None = None) -> list[DocumentPage]:
    path = Path(file_path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type '{extension}'. Expected PDF, DOCX, or TXT.")

    content = path.read_bytes()
    doc_id = document_id or build_document_id(path, content)

    if extension == ".txt":
        return _parse_txt(path, doc_id, content)
    if extension == ".pdf":
        return _parse_pdf(path, doc_id)
    return _parse_docx(path, doc_id)


def _parse_txt(path: Path, document_id: str, content: bytes) -> list[DocumentPage]:
    text = content.decode("utf-8", errors="replace")
    return [
        DocumentPage(
            document_id=document_id,
            file_name=path.name,
            page_number=1,
            text=clean_text(text),
            metadata={"source_path": str(path)},
        )
    ]


def _parse_pdf(path: Path, document_id: str) -> list[DocumentPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency behavior
        raise RuntimeError("pypdf is required to parse PDF files. Install requirements.txt.") from exc

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(
            DocumentPage(
                document_id=document_id,
                file_name=path.name,
                page_number=index,
                text=clean_text(page.extract_text() or ""),
                metadata={"source_path": str(path)},
            )
        )
    return pages


def _parse_docx(path: Path, document_id: str) -> list[DocumentPage]:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - dependency behavior
        raise RuntimeError("python-docx is required to parse DOCX files. Install requirements.txt.") from exc

    document = Document(str(path))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    return [
        DocumentPage(
            document_id=document_id,
            file_name=path.name,
            page_number=1,
            text=clean_text(text),
            metadata={"source_path": str(path), "page_note": "DOCX page numbers are approximated as 1."},
        )
    ]

