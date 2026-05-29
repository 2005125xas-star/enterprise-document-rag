# Enterprise Document RAG QA System

![CI](https://github.com/2005125xas-star/enterprise-document-rag/actions/workflows/ci.yml/badge.svg)

Source-grounded question answering for synthetic enterprise banking documents, with document ingestion, hybrid retrieval, citations, no-answer handling, query logging, evaluation, optional Chroma persistence, optional CrossEncoder reranking, and OpenAI-compatible LLM support.

**Python 3.11+ | Streamlit | RAG | Hybrid Retrieval | Chroma Optional | Reranker Optional | Evaluation | SQLite Logging**

## Why This Project Matters

Enterprise teams often need answers from policy, risk, product, operations, and compliance documents. A credible RAG system should retrieve evidence, cite sources, log questions, evaluate retrieval quality, and refuse unsupported questions instead of guessing. This project is a portfolio prototype that demonstrates those engineering patterns locally and reproducibly.

## More Than A Basic PDF Chatbot

This is not a single PDF prompt wrapper. Core logic is separated under `src/`, while Streamlit orchestrates ingestion, indexing, retrieval, QA, evaluation, and logs. The system includes metadata-preserving chunking, semantic plus BM25 hybrid retrieval, source-grounded answer generation, explicit no-answer behavior, SQLite logging, a 46-question synthetic benchmark, and tests that do not require real API keys or model downloads.

## Recommended Local Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_env.py
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

If `python3.11` is unavailable, use the active Python 3.11+ interpreter:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The preferred launch command is:

```bash
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

`python -m streamlit` helps ensure Streamlit uses the active virtual environment. `--server.fileWatcherType none` can avoid Streamlit watcher issues with optional ML packages, including transformer or torchvision-related import errors.

## Common Run Modes

Minimal local demo mode uses the mock provider, memory vector store, and no reranker. It requires no API key and is the safest first run:

```bash
export LLM_PROVIDER=mock
export VECTOR_STORE=memory
export RERANKER_ENABLED=false
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

Real LLM mode uses an OpenAI-compatible provider. OpenAI, DeepSeek, and Qwen/DashScope-style APIs are selected through `LLM_BASE_URL` and `LLM_MODEL`.

DeepSeek plus Chroma, with reranker disabled:

```bash
export LLM_PROVIDER=openai_compatible
export LLM_API_KEY=your_deepseek_api_key
export LLM_BASE_URL=https://api.deepseek.com
export LLM_MODEL=deepseek-v4-flash
export VECTOR_STORE=chroma
export RERANKER_ENABLED=false
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

Reranker enabled:

```bash
export RERANKER_ENABLED=true
export RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
export RERANKER_TOP_N=20
export RERANKER_FINAL_K=5
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

## Key Features

- Upload and parse PDF, DOCX, and TXT files.
- Preserve `file_name`, `document_id`, `page_number`, `chunk_id`, and character offsets.
- Clean and chunk document text with overlap.
- Use sentence-transformers for semantic retrieval when available.
- Use a deterministic hashing embedder as a fallback and in tests.
- Score keyword evidence with BM25.
- Merge semantic and BM25 scores for hybrid retrieval.
- Optionally persist semantic vectors and chunk metadata with Chroma.
- Optionally rerank hybrid candidates with a CrossEncoder.
- Generate answers through `MockProvider` or OpenAI-compatible chat completion APIs.
- Return `I could not find enough evidence in the uploaded documents.` when evidence is weak.
- Include source citations in answered responses.
- Save query logs to SQLite.
- Run evaluation and generate CSV plus Markdown reports.
- Provide a Streamlit UI for upload, asking, evaluation, and logs.

## System Architecture

```text
Documents
-> Parser
-> Cleaner
-> Chunker
-> Embedding Index (memory or Chroma) + BM25 Index
-> Hybrid Retriever
-> Optional CrossEncoder Reranker
-> Evidence Gate / No-answer Check
-> LLM Provider
-> Source-grounded Answer
-> Logs + Evaluation
```

## Folder Structure

```text
enterprise-document-rag/
├── app/
│   └── streamlit_app.py
├── configs/
│   └── config.yaml
├── data/
│   ├── sample_docs/
│   ├── raw_docs/
│   ├── processed/
│   ├── eval/
│   ├── logs/
│   └── vector_store/
├── src/
│   ├── ingestion/
│   ├── indexing/
│   ├── retrieval/
│   ├── llm/
│   ├── qa/
│   ├── evaluation/
│   └── utils/
├── scripts/
│   └── check_env.py
├── tests/
├── outputs/
│   ├── eval_results.csv
│   ├── evaluation_report.md
│   └── demo_transcript.md
├── Dockerfile
├── README.md
├── PORTFOLIO_SUMMARY.md
├── requirements.txt
├── .env.example
├── .dockerignore
├── .gitignore
└── Makefile
```

`data/vector_store/` is generated local state and is gitignored.

## Use A Real LLM Provider

The app supports OpenAI-compatible chat completion APIs. Configuration priority is environment variables, then `configs/config.yaml`, then internal defaults.

Preferred variables:

- `LLM_PROVIDER`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`

Backward-compatible OpenAI variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

OpenAI:

```bash
export LLM_PROVIDER=openai_compatible
export LLM_API_KEY=your_openai_api_key
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

DeepSeek:

```bash
export LLM_PROVIDER=openai_compatible
export LLM_API_KEY=your_deepseek_api_key
export LLM_BASE_URL=https://api.deepseek.com
export LLM_MODEL=deepseek-v4-flash
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

Qwen / DashScope:

```bash
export LLM_PROVIDER=openai_compatible
export LLM_API_KEY=your_dashscope_api_key
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_MODEL=qwen-plus
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

Do not commit API keys. Do not paste API keys into screenshots, README files, GitHub issues, or commits. Official provider APIs are recommended over third-party shared or unknown API key resellers.

`MockProvider` is for pipeline testing only. It extracts relevant-looking sentences from retrieved chunks and adds citations, so it does not represent real LLM answer quality.

## Persistent Vector Store With Chroma

The default vector store backend is `memory`, which preserves the simplest local behavior: uploaded documents are indexed in the running app process and are lost after restart.

Chroma is optional. It persists indexed chunks, embeddings, and citation metadata under `data/vector_store/`.

Enable Chroma in `configs/config.yaml`:

```yaml
retrieval:
  vector_store: chroma
  persist_directory: data/vector_store
  collection_name: enterprise_document_chunks
```

Or use environment variables:

```bash
export VECTOR_STORE=chroma
export CHROMA_PERSIST_DIRECTORY=data/vector_store
export CHROMA_COLLECTION_NAME=enterprise_document_chunks
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

The app sidebar shows the active vector store, persist directory, collection name, and stored chunk count. It also provides a clear button for the persistent vector store.

Chroma persistence is local. It is not a managed production vector database. Multi-user or cloud deployment would need a managed vector DB or server-backed Chroma deployment.

## Optional CrossEncoder Reranking

Hybrid retrieval remains the default retrieval path. An optional second-stage reranker can improve precision by scoring `(question, chunk_text)` pairs after broad hybrid candidate recall.

Default mode is disabled for speed and offline reproducibility:

```yaml
retrieval:
  reranker:
    enabled: false
    model: cross-encoder/ms-marco-MiniLM-L-6-v2
    top_n: 20
    final_k: 5
    allow_fallback: true
```

`top_n` is the number of hybrid candidates sent to the reranker. `final_k` is the number of final evidence chunks returned to the LLM.

The recommended starter model is `cross-encoder/ms-marco-MiniLM-L-6-v2`. First use may download the model, and CPU inference can be slower. If model loading fails and `allow_fallback: true`, the app falls back to hybrid ranking and shows a warning.

## Docker

Build:

```bash
docker build -t enterprise-document-rag .
```

Run mock mode:

```bash
docker run --rm -p 8501:8501 enterprise-document-rag
```

Run with DeepSeek:

```bash
docker run --rm -p 8501:8501 \
  -e LLM_PROVIDER=openai_compatible \
  -e LLM_API_KEY=your_deepseek_api_key \
  -e LLM_BASE_URL=https://api.deepseek.com \
  -e LLM_MODEL=deepseek-v4-flash \
  enterprise-document-rag
```

Run with persistent Chroma volume:

```bash
docker run --rm -p 8501:8501 \
  -e VECTOR_STORE=chroma \
  -v "$(pwd)/data/vector_store:/app/data/vector_store" \
  enterprise-document-rag
```

The Docker image defaults to safe offline mode: `LLM_PROVIDER=mock`, `VECTOR_STORE=memory`, and `RERANKER_ENABLED=false`.

## Run Evaluation

The evaluation CLI uses the synthetic benchmark in `data/sample_docs/` and `data/eval/qa_eval_set.csv`. It uses mock mode and local embeddings so it can run without a real LLM API key.

```bash
python -m src.evaluation.run_eval
```

Outputs:

- `outputs/eval_results.csv`
- `outputs/evaluation_report.md`
- `outputs/demo_transcript.md`

The CSV includes question ID, question type, difficulty, answerability, expected document/page, expected keywords, retrieved chunk IDs, retrieved documents, top score, Hit@3, Hit@5, reciprocal rank, no-answer flag, citation flag, latency, answer preview, and failure reason.

## Evaluation Benchmark

The benchmark uses six synthetic banking and enterprise operations documents. These files are demo documents created for evaluation and contain no confidential, proprietary, or customer data.

Documents:

- `bank_recruitment_policy.txt`
- `retail_banking_product_guide.txt`
- `credit_risk_policy_summary.txt`
- `data_security_policy.txt`
- `branch_operations_manual.txt`
- `customer_marketing_campaign_guide.txt`

The benchmark contains 46 questions across fact lookup, policy lookup, product lookup, numeric thresholds, role responsibilities, cross-document questions, and no-answer cases.

| Question type | Count |
| --- | ---: |
| fact_lookup | 4 |
| policy_lookup | 6 |
| product_lookup | 4 |
| numeric_threshold | 14 |
| role_responsibility | 7 |
| cross_document | 6 |
| no_answer | 5 |

Recent local benchmark metrics:

| Metric | Value |
| --- | ---: |
| Questions | 46 |
| Answerable questions | 41 |
| Unanswerable questions | 5 |
| Hit@3 | 1.0000 |
| Hit@5 | 1.0000 |
| MRR | 0.9878 |
| No-answer accuracy | 1.0000 |
| Citation rate | 1.0000 |

Evaluation matters because a RAG project should be judged on retrieval quality, citation behavior, and refusal behavior, not only on fluent-looking answers.

## Demo Screenshots

This is a portfolio prototype using synthetic sample documents. When no LLM API key is configured, the app runs with `MockProvider`.

**Upload and indexing interface**

![Upload page](outputs/screenshots/upload_page.png)

**Question answering with source-grounded retrieval**

![Ask page](outputs/screenshots/ask_page.png)

**Evaluation dashboard with retrieval metrics**

![Evaluation page](outputs/screenshots/evaluation_page.png)

## Example Questions

- What cap applies to employee referral bonuses?
- What direct deposit amount waives the Everyday Checking monthly fee?
- Who approves customer-facing claims before a campaign launches?
- Compare the direct deposit amount that waives checking fees with the bonus campaign deposit total.
- What is the bank's cryptocurrency custody product code?

The last question is intentionally unsupported and should return the no-answer message.

## Troubleshooting

Missing `pypdf`:

```bash
python -m pip install -r requirements.txt
python -m pip show pypdf
```

Missing `docx` / `python-docx`:

```bash
python -m pip install -r requirements.txt
python -m pip show python-docx
```

Conda `base` and `.venv` both active:

```bash
conda deactivate
source .venv/bin/activate
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

Provider stays as `mock`: check that the app was launched from the shell where `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` are exported. Also confirm `LLM_PROVIDER` is not set to `mock`.

Embedding falls back to `hashing-384`: install requirements and allow the configured sentence-transformers model to load. Hashing embeddings are a fallback mode for reproducible local execution.

Streamlit watcher error involving transformers or torchvision:

```bash
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

API key works in a standalone OpenAI SDK test but not in the app: confirm the app is launched from the same shell where the variables are exported, or use a local `.env` file that is never committed.

Safe DeepSeek-style standalone test:

```bash
export LLM_API_KEY=your_deepseek_api_key

python - <<'PY'
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("LLM_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."}
    ],
    stream=False
)

print(response.choices[0].message.content)
PY
```

Model name or base URL errors: verify the provider's documented base URL and exact model name. The app sidebar shows the active provider, backend domain, and model, but never displays API keys.

## What Not To Commit

- `.env`
- Real API keys or access tokens.
- `data/vector_store/`
- Local logs that may contain user questions or private document names.
- Generated cache directories such as `__pycache__/` and `.pytest_cache/`.
- Private PDFs, customer documents, or internal enterprise files.

## Current Limitations

- Sample documents are synthetic and contain no confidential data.
- MockProvider is deterministic and useful for testing, but it does not represent real LLM answer quality.
- Evaluation uses local or fallback embeddings when sentence-transformers is unavailable.
- TXT files expose only page `1`, so page-level evaluation is limited for the current benchmark.
- DOCX page numbers are approximated because DOCX files do not store stable rendered page boundaries.
- Chroma persistence is local, not a managed production vector database.
- CrossEncoder reranking can be slower on CPU and may require model download on first use.
- This is a portfolio prototype, not a production enterprise system.

## Future Improvements

- Real PDF benchmark with stable page-level citations.
- OCR support for scanned documents.
- Managed vector store option for cloud deployment.
- Server-backed Chroma deployment.
- Ollama local model backend for offline answer generation.
- Real OpenAI answer quality evaluation with human-reviewed examples.
- Hosted demo deployment.
- Document-level metadata filters such as department, date, or access tier.
- Batch ingestion and incremental re-indexing.
- Authentication and role-aware retrieval.

## Suggested Resume Bullet Points

- Built an enterprise-style document RAG QA system with PDF, DOCX, and TXT ingestion, metadata-preserving chunking, hybrid semantic/BM25 retrieval, source-grounded answer generation, and SQLite query logging.
- Added no-answer handling and citation validation to reduce unsupported answers in document question answering workflows.
- Implemented optional Chroma persistence and optional CrossEncoder reranking while preserving an offline mock mode for reproducible tests.
- Developed an evaluation pipeline measuring Hit@3, Hit@5, MRR, no-answer accuracy, citation rate, and latency with reproducible CSV and Markdown reports.
