from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class EnvironmentCheckResult:
    python_executable: str
    python_version: str
    package_status: dict[str, bool]
    vector_store_backend: str
    chroma_persist_directory: str
    reranker_enabled: bool
    reranker_model: str


def run_environment_check(config: Mapping | None = None, environ: Mapping[str, str] | None = None) -> EnvironmentCheckResult:
    env = environ if environ is not None else os.environ
    retrieval_config = dict((config or {}).get("retrieval", {}) if config else {})
    reranker_config = dict(retrieval_config.get("reranker", {}))
    backend = str(env.get("VECTOR_STORE") or retrieval_config.get("vector_store", "memory"))
    persist_directory = str(
        env.get("CHROMA_PERSIST_DIRECTORY") or retrieval_config.get("persist_directory", "data/vector_store")
    )
    packages = {
        "pypdf": _is_importable("pypdf"),
        "docx": _is_importable("docx"),
        "sentence_transformers": _is_importable("sentence_transformers"),
        "openai": _is_importable("openai"),
        "chromadb": _is_importable("chromadb"),
    }
    return EnvironmentCheckResult(
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        package_status=packages,
        vector_store_backend=backend,
        chroma_persist_directory=persist_directory,
        reranker_enabled=_env_bool(env.get("RERANKER_ENABLED"), _to_bool(reranker_config.get("enabled", False))),
        reranker_model=str(
            env.get("RERANKER_MODEL")
            or reranker_config.get("model")
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
    )


def _is_importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return _to_bool(value)
