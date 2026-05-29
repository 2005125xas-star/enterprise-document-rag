from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.runner import load_examples, save_evaluation_outputs
from src.qa.service import EnterpriseRAGSystem
from src.utils.config import load_config


RAW_DOCS_DIR = ROOT / "data" / "raw_docs"
EVAL_DIR = ROOT / "data" / "eval"
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def get_system() -> EnterpriseRAGSystem:
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = EnterpriseRAGSystem(load_config(CONFIG_PATH))
    return st.session_state.rag_system


def save_uploads(uploaded_files) -> list[Path]:
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        target = RAW_DOCS_DIR / uploaded_file.name
        target.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target)
    return saved_paths


def display_answer(result) -> None:
    st.subheader("Answer")
    st.write(result.answer)
    if result.citations:
        st.subheader("Citations")
        st.dataframe(pd.DataFrame(result.citations), use_container_width=True, hide_index=True)

    st.subheader("Retrieved Sources")
    for retrieval in result.retrieval_results:
        page_range = f"{retrieval.chunk.page_number}-{retrieval.chunk.page_number}"
        label = (
            f"#{retrieval.rank} {retrieval.chunk.file_name} "
            f"pages {page_range} score {retrieval.score:.3f}"
        )
        with st.expander(label):
            st.code(retrieval.chunk.text)
            st.json(
                {
                    "chunk_id": retrieval.chunk.chunk_id,
                    "document_id": retrieval.chunk.document_id,
                    "file_name": retrieval.chunk.file_name,
                    "page_range": page_range,
                    "semantic_score": retrieval.semantic_score,
                    "keyword_score": retrieval.keyword_score,
                    "final_score": retrieval.score,
                }
            )


def display_evaluation(result) -> None:
    metrics = {
        "Hit@3": result.hit_at_3,
        "Hit@5": result.hit_at_5,
        "MRR": result.mrr,
        "No-answer accuracy": result.no_answer_accuracy,
        "Avg latency ms": result.average_latency_ms,
    }
    st.subheader("Evaluation")
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, value)
    st.dataframe(pd.DataFrame(result.details), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Enterprise Document RAG QA", layout="wide")
    st.title("Enterprise Document RAG QA System")
    system = get_system()

    with st.sidebar:
        st.caption(f"Embedding: {system.embedding_model.name}")
        st.caption(f"Provider: {system.provider.name}")
        top_k = st.slider("Retrieved sources", min_value=1, max_value=10, value=5)

    upload_tab, qa_tab, eval_tab, logs_tab = st.tabs(["Documents", "Ask", "Evaluate", "Logs"])

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        if st.button("Index", type="primary", disabled=not uploaded_files):
            paths = save_uploads(uploaded_files)
            chunks = system.ingest_paths(paths)
            st.success(f"Indexed {len(paths)} document(s) into {len(chunks)} chunk(s).")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "chunk_id": chunk.chunk_id,
                            "file_name": chunk.file_name,
                            "page_number": chunk.page_number,
                            "characters": len(chunk.text),
                        }
                        for chunk in chunks
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    with qa_tab:
        question = st.text_input("Question")
        if st.button("Ask", type="primary", disabled=not question.strip()):
            result = system.answer(question.strip(), top_k=top_k, session_id=st.session_state.get("session_id"))
            display_answer(result)

    with eval_tab:
        eval_upload = st.file_uploader("Evaluation CSV or JSON", type=["csv", "json"])
        if st.button("Run evaluation", disabled=eval_upload is None):
            EVAL_DIR.mkdir(parents=True, exist_ok=True)
            eval_path = EVAL_DIR / f"uploaded_eval_{uuid4().hex}{Path(eval_upload.name).suffix}"
            eval_path.write_bytes(eval_upload.getbuffer())
            result = system.evaluate(load_examples(eval_path), top_k=top_k)
            save_evaluation_outputs(result, ROOT / "outputs" / "eval_results.csv", ROOT / "outputs" / "evaluation_report.md")
            display_evaluation(result)

    with logs_tab:
        rows = system.recent_logs(limit=100)
        st.subheader("Query Logs")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No queries logged yet.")


if __name__ == "__main__":
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex
    main()
