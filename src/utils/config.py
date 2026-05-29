from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "chunking": {
        "chunk_size": 900,
        "chunk_overlap": 150,
    },
    "retrieval": {
        "top_k": 5,
        "semantic_weight": 0.65,
        "keyword_weight": 0.35,
        "min_score": 0.12,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "qa": {
        "provider": "auto",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_context_chunks": 5,
        "insufficient_evidence_message": "I could not find enough evidence in the uploaded documents.",
    },
    "logging": {
        "sqlite_path": "data/logs/query_logs.sqlite3",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return deepcopy(DEFAULT_CONFIG)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
        raise RuntimeError("PyYAML is required to load config.yaml. Install requirements.txt.") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return deep_merge(DEFAULT_CONFIG, loaded)

