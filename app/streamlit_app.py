from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.runner import load_examples, save_evaluation_outputs
from src.indexing.vector_store import VectorStoreConfigurationError
from src.llm.providers import LLMProviderRequestError, ProviderConfigurationError
from src.qa.service import EnterpriseRAGSystem
from src.retrieval.reranker import RerankerConfigurationError
from src.utils.config import load_config


RAW_DOCS_DIR = ROOT / "data" / "raw_docs"
EVAL_DIR = ROOT / "data" / "eval"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
MISSING_PROVIDER_KEY_MESSAGE = (
    "No API key found for the selected LLM provider. Set LLM_API_KEY / OPENAI_API_KEY, "
    "or switch to mock mode with LLM_PROVIDER=mock."
)
LLM_REQUEST_FAILED_MESSAGE = (
    "LLM request failed. Please check your API key, base URL, model name, provider quota, and network connection."
)


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
                    "hybrid_score": retrieval.hybrid_score if retrieval.hybrid_score is not None else retrieval.score,
                    "reranker_score": retrieval.rerank_score,
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


def app_debug_enabled() -> bool:
    return os.getenv("APP_DEBUG", "0").strip() == "1"


def clean_error_message(exc: Exception, fallback_message: str) -> str:
    if isinstance(exc, ProviderConfigurationError):
        if "API_KEY" in str(exc):
            return MISSING_PROVIDER_KEY_MESSAGE
        return str(exc)
    if isinstance(exc, VectorStoreConfigurationError):
        return str(exc)
    if isinstance(exc, RerankerConfigurationError):
        return str(exc)
    if isinstance(exc, LLMProviderRequestError):
        return LLM_REQUEST_FAILED_MESSAGE

    raw_message = str(exc)
    if "OPENAI_API_KEY is not set" in raw_message or "LLM_API_KEY is not set" in raw_message:
        return MISSING_PROVIDER_KEY_MESSAGE
    if "pypdf is required" in raw_message:
        return "PDF parsing dependency is missing. Please run: python -m pip install -r requirements.txt"
    if "python-docx is required" in raw_message or "docx is required" in raw_message:
        return "DOCX parsing dependency is missing. Please run: python -m pip install -r requirements.txt"
    return fallback_message


def show_clean_exception(exc: Exception, fallback_message: str) -> None:
    st.error(clean_error_message(exc, fallback_message))
    if app_debug_enabled():
        st.exception(exc)


def display_sidebar_status(system: EnterpriseRAGSystem) -> None:
    provider = system.provider
    vector_store = system.vector_store
    st.caption(f"Embedding: {system.embedding_model.name}")
    st.caption(f"Provider: {getattr(provider, 'name', 'unknown')}")
    st.caption(f"Model: {getattr(provider, 'model', 'unknown')}")
    backend_label = getattr(provider, "backend_label", None)
    if backend_label:
        st.caption(f"Backend: {backend_label}")

    if getattr(provider, "name", "") == "mock":
        st.warning("MockProvider is for pipeline testing only and does not represent real LLM answer quality.")

    st.caption(f"Vector store: {getattr(vector_store, 'name', 'unknown')}")
    st.caption(f"Stored chunks: {system.vector_store_count()}")
    if getattr(vector_store, "name", "") == "chroma":
        st.caption(f"Chroma directory: {getattr(vector_store, 'persist_directory', 'unknown')}")
        st.caption(f"Collection: {getattr(vector_store, 'collection_name', 'unknown')}")
        confirm_clear = st.checkbox("Confirm clear persistent vector store")
        if st.button("Clear persistent vector store", disabled=not confirm_clear):
            system.clear_vector_store()
            st.success("Persistent vector store cleared.")
            st.rerun()

    reranker_status = "enabled" if system.reranker_settings["enabled"] else "disabled"
    st.caption(f"Reranker: {reranker_status}")
    st.caption(f"Reranker model: {system.reranker_settings['model']}")
    st.caption(f"Reranker candidates: {system.reranker_settings['top_n']}")
    st.caption(f"Final sources: {system.reranker_settings['final_k']}")
    if system.reranker_warning:
        st.warning(system.reranker_warning)


def main() -> None:
    st.set_page_config(page_title="Enterprise Document RAG QA", layout="wide")
    st.title("Enterprise Document RAG QA System")
    try:
        system = get_system()
    except Exception as exc:
        show_clean_exception(exc, "Application startup failed. Please check your configuration and dependencies.")
        st.stop()

    with st.sidebar:
        display_sidebar_status(system)
        top_k = st.slider("Retrieved sources", min_value=1, max_value=10, value=5)

    upload_tab, qa_tab, eval_tab, logs_tab = st.tabs(["Documents", "Ask", "Evaluate", "Logs"])

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        if st.button("Index", type="primary", disabled=not uploaded_files):
            try:
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
            except Exception as exc:
                show_clean_exception(exc, "Document indexing failed. Please check the file type and dependencies.")

    with qa_tab:
        question = st.text_input("Question")
        if st.button("Ask", type="primary", disabled=not question.strip()):
            try:
                result = system.answer(question.strip(), top_k=top_k, session_id=st.session_state.get("session_id"))
                display_answer(result)
            except Exception as exc:
                show_clean_exception(exc, "Question answering failed. Please check the indexed documents and LLM settings.")

    with eval_tab:
        eval_upload = st.file_uploader("Evaluation CSV or JSON", type=["csv", "json"])
        if st.button("Run evaluation", disabled=eval_upload is None):
            try:
                EVAL_DIR.mkdir(parents=True, exist_ok=True)
                eval_path = EVAL_DIR / f"uploaded_eval_{uuid4().hex}{Path(eval_upload.name).suffix}"
                eval_path.write_bytes(eval_upload.getbuffer())
                result = system.evaluate(load_examples(eval_path), top_k=top_k)
                save_evaluation_outputs(
                    result,
                    ROOT / "outputs" / "eval_results.csv",
                    ROOT / "outputs" / "evaluation_report.md",
                )
                display_evaluation(result)
            except Exception as exc:
                show_clean_exception(exc, "Evaluation failed. Please check the evaluation file schema and current index.")

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
