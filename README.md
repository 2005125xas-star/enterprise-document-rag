# Enterprise Document RAG QA System

Enterprise Document RAG QA System is a portfolio-quality AI/data science project that demonstrates how to build a source-grounded question-answering system over internal business documents. It is designed to be realistic enough for job applications while staying stable and runnable on a local machine.

The project works without an OpenAI API key by using a deterministic `MockProvider`, and it can switch to OpenAI when `OPENAI_API_KEY` is configured.

## Business Motivation

Organizations store policy, compliance, operations, and support knowledge across PDFs, Word files, and text documents. Employees need fast answers, but the system must show where each answer came from and avoid guessing when evidence is missing.

This project models that workflow: ingest documents, preserve source metadata, retrieve relevant chunks, generate an answer with citations, log the query, and evaluate retrieval and no-answer behavior.

## Why This Is Not Just a PDF Chatbot

This is not a single-file prompt wrapper. The implementation separates document ingestion, chunking, indexing, retrieval, answer generation, evaluation, logging, and UI orchestration. It includes:

- Metadata-preserving document chunks.
- Hybrid retrieval using semantic embeddings and BM25 keyword scoring.
- Explicit no-answer handling when evidence is insufficient.
- Source citations for every answered response.
- SQLite query logging for auditability.
- An evaluation pipeline with retrieval and no-answer metrics.
- Tests that run without external model downloads or API keys.

## Key Features

- Upload and parse PDF, DOCX, and TXT documents.
- Preserve `file_name`, `document_id`, `page_number`, `chunk_id`, and character offsets.
- Clean and chunk document text with overlap.
- Build a local in-memory vector index.
- Use sentence-transformers as the production semantic retrieval path.
- Use a deterministic hashing embedder in tests and fallback environments.
- Score keyword evidence with a local BM25 implementation.
- Merge semantic and keyword scores into hybrid retrieval results.
- Generate answers through OpenAI or MockProvider.
- Return: `I could not find enough evidence in the uploaded documents.` when evidence is weak.
- Save query logs to SQLite.
- Run evaluation and generate CSV plus Markdown reports.
- Provide a Streamlit UI for upload, asking, evaluation, and logs.

## System Architecture

```text
Documents
  -> src.ingestion.parsers
  -> src.ingestion.chunker
  -> src.indexing.vector_index + src.retrieval.bm25
  -> src.retrieval.hybrid
  -> src.qa.pipeline
  -> src.llm.providers
  -> SQLite query logs + Streamlit UI + evaluation outputs
```

Core application logic lives under `src/`. The Streamlit app in `app/streamlit_app.py` orchestrates these modules but does not implement retrieval or QA logic directly.

## Folder Structure

```text
enterprise-document-rag/
├── app/
│   └── streamlit_app.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw_docs/
│   ├── processed/
│   ├── eval/
│   └── logs/
├── src/
│   ├── ingestion/
│   ├── indexing/
│   ├── retrieval/
│   ├── llm/
│   ├── qa/
│   ├── evaluation/
│   └── utils/
├── tests/
├── outputs/
│   ├── eval_results.csv
│   └── evaluation_report.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── Makefile
```

## Setup

Python 3.11+ is recommended.

```bash
make install
cp .env.example .env
```

Run tests:

```bash
python -m pytest -q
```

Or, if you use the project virtual environment:

```bash
.venv/bin/python -m pytest -q
```

## Run In Mock Mode

Mock mode requires no API key and no external LLM call.

```bash
unset OPENAI_API_KEY
streamlit run app/streamlit_app.py
```

If `sentence-transformers` is not installed or the configured model cannot load, the app falls back to the deterministic hashing embedder so the demo can still run locally. Tests always inject a local test embedder to avoid model downloads.

## Run With OpenAI API

Set your key in the environment or in a local `.env` file that you load before starting the app. Do not commit real keys.

```bash
export OPENAI_API_KEY="your-key-here"
streamlit run app/streamlit_app.py
```

With `qa.provider: auto` in `configs/config.yaml`, the system uses OpenAI when the key is present and MockProvider otherwise. If `qa.provider: openai` is explicitly selected, startup requires `OPENAI_API_KEY`.

## Run Evaluation

The evaluation CLI uses MockProvider and hashing embeddings so it runs without an OpenAI key or model download.

```bash
python -m src.evaluation.run_eval
```

Outputs:

- `outputs/eval_results.csv`
- `outputs/evaluation_report.md`

The CSV includes question, answerability, expected document/page, retrieved chunk IDs, top score, Hit@3, Hit@5, reciprocal rank, no-answer flag, latency, and answer text.

## Example Questions

After indexing the included evaluation sample document or your own enterprise documents, try:

- How long are customer records retained?
- How are backups protected?
- Who approves access to backup archives?
- quantum asteroid orbit

The last question is intentionally unsupported and should return the no-answer message.

## Evaluation Design

The evaluation pipeline measures:

- `Hit@3`: whether a relevant chunk or document appears in the top 3 retrieved results.
- `Hit@5`: whether a relevant chunk or document appears in the top 5 retrieved results.
- `MRR`: reciprocal rank of the first relevant retrieval result.
- `No-answer accuracy`: whether unanswerable examples return the no-answer response.
- `Average latency`: average pipeline latency in milliseconds.

The default evaluation fixture is a smoke test, not a benchmark. It verifies that ingestion, retrieval, QA, no-answer handling, output writing, and logging are wired correctly.

## Current Limitations

- The local vector index is in memory and intended for small to medium demo datasets.
- DOCX page numbers are approximated as page `1` because DOCX files do not store stable rendered page boundaries.
- MockProvider is deterministic and useful for testing, but it does not represent final LLM answer quality.
- The default evaluation dataset is intentionally small.
- No CrossEncoder reranker, Docker setup, or external vector database is included in this first stable version.

## Future Improvements

- Add a persistent vector store such as FAISS or Chroma.
- Add richer evaluation examples with human-labeled relevant chunks.
- Add optional reranking after the base retriever is stable.
- Add document-level metadata filters such as department, date, or access tier.
- Add batch ingestion and incremental re-indexing.
- Add authentication and role-aware retrieval for a more enterprise-like demo.

## Suggested Resume Bullet Points

- Built an enterprise-style document RAG QA system with PDF, DOCX, and TXT ingestion, metadata-preserving chunking, hybrid semantic/BM25 retrieval, source-grounded answer generation, and SQLite query logging.
- Implemented no-answer handling and citation validation to reduce unsupported answers in document question answering workflows.
- Developed an evaluation pipeline measuring Hit@3, Hit@5, MRR, no-answer accuracy, and latency, with reproducible CSV and Markdown reports.
- Created a Streamlit demo that separates UI orchestration from backend retrieval and QA modules, supporting both OpenAI and deterministic mock execution.

