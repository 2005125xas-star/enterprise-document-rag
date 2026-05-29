# Enterprise Document RAG QA System

## 3-Sentence Summary

Enterprise Document RAG QA System is a local, source-grounded question-answering prototype for synthetic banking and enterprise operations documents. It ingests PDF, DOCX, and TXT files, preserves metadata, chunks documents, performs hybrid semantic/BM25 retrieval, generates cited answers, refuses unsupported questions, logs queries, and evaluates retrieval/no-answer behavior. The project is built to demonstrate practical AI engineering judgment rather than a toy single-PDF chatbot.

## Problem Statement

Enterprise teams need answers from policy, product, risk, security, branch operations, and marketing documents, but they also need traceability and refusal behavior when evidence is missing. This project shows how to build and evaluate a small but realistic document QA workflow with citations and audit logs.

## Technical Highlights

- Metadata-preserving ingestion for PDF, DOCX, and TXT documents.
- Dataclass-based core models for pages, chunks, retrieval results, answers, and evaluation records.
- Hybrid retrieval with sentence-transformers as the production path and BM25 keyword scoring.
- Deterministic hashing embedder and MockProvider for API-free testing and demos.
- Evidence gate for no-answer handling.
- SQLite query logging.
- Streamlit interface separated from backend retrieval and QA logic.

## Evaluation Highlights

- Six synthetic banking/enterprise documents in `data/sample_docs/`.
- Forty-six evaluation questions in `data/eval/qa_eval_set.csv`.
- Question types include fact lookup, policy lookup, product lookup, numeric thresholds, role responsibility, cross-document, and no-answer.
- Reports include Hit@3, Hit@5, MRR, no-answer accuracy, citation rate, latency, grouped metrics, and failure reasons.

## Main Limitations

- Sample documents are synthetic and contain no confidential data.
- MockProvider is deterministic and does not measure real LLM answer quality.
- Evaluation uses local hashing embeddings if sentence-transformers is unavailable.
- TXT files only expose page 1, so page-level evaluation is limited.
- This is a portfolio prototype, not a production enterprise system.

## Suggested Resume Bullet Points

- Built an enterprise-style document RAG QA system with metadata-preserving ingestion, hybrid semantic/BM25 retrieval, source-grounded answers, no-answer handling, and SQLite query logging.
- Designed a synthetic banking benchmark with 6 documents and 46 evaluation questions covering retrieval quality, citation behavior, and unsupported-question refusal.
- Implemented evaluation reporting for Hit@3, Hit@5, MRR, no-answer accuracy, citation rate, latency, grouped metrics, and per-question failure analysis.

## Suggested Interview Talking Points

- Why hybrid retrieval is useful for enterprise documents with exact terms, codes, thresholds, and policy language.
- How no-answer handling reduces unsupported answers.
- Why evaluation should include no-answer cases and cross-document questions.
- How MockProvider keeps the project reproducible while OpenAI mode can be used for stronger answer generation.
- What would change for production: persistent vector store, reranker, access controls, real PDFs/OCR, deployment, monitoring, and human-labeled evaluation.

