# Enterprise Document RAG QA System

## 3-Sentence Summary

Enterprise Document RAG QA System is a Streamlit-based RAG portfolio prototype for source-grounded question answering over enterprise-style documents. It supports a reproducible synthetic banking benchmark and a public-document demo path for official university policy documents supplied locally by the user. The project demonstrates ingestion, metadata-preserving chunking, hybrid retrieval, citations, no-answer handling, SQLite logging, optional Chroma persistence, optional CrossEncoder reranking, and configurable OpenAI-compatible LLM backends.

## Problem Statement

Real document QA systems need traceability, refusal behavior, repeatable evaluation, and careful source handling. This project shows how to build and test a small but realistic RAG workflow without relying on private documents, API keys, or fabricated results.

## Technical Highlights

- PDF, DOCX, and TXT ingestion with page, file, document, chunk, and character-offset metadata.
- Hybrid semantic/BM25 retrieval with deterministic fallback embeddings for offline tests.
- MockProvider for CI and local pipeline testing, plus OpenAI-compatible provider support for OpenAI, DeepSeek, and Qwen-style APIs.
- Optional local Chroma vector persistence and optional CrossEncoder reranking, both disabled by default for reproducibility.
- Source-grounded citations, insufficient-evidence refusal, and SQLite query logging.
- Streamlit app that orchestrates backend modules without embedding retrieval or QA logic directly in UI code.

## Evaluation Highlights

- Synthetic benchmark: 6 synthetic banking/enterprise documents and 46 evaluation questions.
- Public demo path: source registry and 20-question public university policy evaluation set for official documents placed in `data/public_docs/`.
- Metrics include Hit@3, Hit@5, MRR, no-answer accuracy, citation rate, latency, grouped metrics, and per-question failure reasons.
- Public real-document evaluation is intentionally not claimed complete unless official public documents are actually present locally.

## Main Limitations

- This is a portfolio prototype, not a production enterprise deployment.
- Synthetic documents are useful for reproducibility but do not represent the full complexity of enterprise document collections.
- Public university policy demo documents are not committed by default; users must verify official URLs and place public documents locally.
- MockProvider validates pipeline behavior but does not measure real LLM answer quality.
- The public university policy demo is for real-document validation, not legal, academic, admissions, or student policy advice.

## Suggested Resume Bullet Points

- Built a Streamlit-based RAG portfolio prototype supporting synthetic benchmark evaluation and a public-document demo path with source-grounded retrieval, citations, no-answer handling, optional Chroma persistence, and configurable OpenAI-compatible LLM backends.
- Implemented metadata-preserving PDF/DOCX/TXT ingestion, hybrid semantic/BM25 retrieval, SQLite query logging, and reproducible offline tests using MockProvider and deterministic embeddings.
- Designed evaluation workflows measuring Hit@3, Hit@5, MRR, no-answer accuracy, citation rate, latency, and per-question failure reasons across answerable, cross-document, and no-answer scenarios.

## Suggested Interview Talking Points

- Why hybrid retrieval is useful for policy documents with exact terms, codes, dates, thresholds, and institutional language.
- How no-answer handling and citation validation reduce unsupported answers.
- Why MockProvider remains valuable for CI even when real LLM providers are supported.
- How the public-document path improves credibility without committing copyrighted, private, or non-public files.
- What would change for production: access controls, managed vector database, OCR, monitoring, permission-aware retrieval, larger human-labeled evaluation sets, and deployment hardening.
