from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.env_check import run_environment_check


def main() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    result = run_environment_check(config)
    print(f"Python executable: {result.python_executable}")
    print(f"Python version: {result.python_version}")
    print(f"pypdf installed: {_yes_no(result.package_status['pypdf'])}")
    print(f"python-docx / docx installed: {_yes_no(result.package_status['docx'])}")
    print(f"sentence-transformers installed: {_yes_no(result.package_status['sentence_transformers'])}")
    print(f"openai installed: {_yes_no(result.package_status['openai'])}")
    print(f"Chroma installed: {_yes_no(result.package_status['chromadb'])}")
    print(f"Configured vector store: {result.vector_store_backend}")
    print(f"Chroma persist directory: {result.chroma_persist_directory}")
    print(f"Reranker enabled: {_yes_no(result.reranker_enabled)}")
    print(f"Reranker model: {result.reranker_model}")
    print("Recommended launch command: python -m streamlit run app/streamlit_app.py --server.fileWatcherType none")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
