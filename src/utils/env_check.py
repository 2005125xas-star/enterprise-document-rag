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


def run_environment_check(config: Mapping | None = None, environ: Mapping[str, str] | None = None) -> EnvironmentCheckResult:
    env = environ if environ is not None else os.environ
    retrieval_config = dict((config or {}).get("retrieval", {}) if config else {})
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
    )


def _is_importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None
