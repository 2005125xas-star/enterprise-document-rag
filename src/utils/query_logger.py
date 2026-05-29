from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models import AnswerResult


class QueryLogger:
    def __init__(self, sqlite_path: str | Path = "data/logs/query_logs.sqlite3") -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    is_answered INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    top_score REAL,
                    citation_count INTEGER NOT NULL,
                    sources_json TEXT NOT NULL
                )
                """
            )

    def log_query(self, question: str, result: AnswerResult, session_id: str | None = None) -> None:
        top_score = result.retrieval_results[0].score if result.retrieval_results else None
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                INSERT INTO query_logs (
                    timestamp, session_id, question, answer, is_answered, provider,
                    latency_ms, top_score, citation_count, sources_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    session_id,
                    question,
                    result.answer,
                    int(result.is_answered),
                    result.provider,
                    result.latency_ms,
                    top_score,
                    len(result.citations),
                    json.dumps(result.citations),
                ),
            )

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT * FROM query_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["is_answered"] = bool(value["is_answered"])
        value["sources"] = json.loads(value.pop("sources_json"))
        return value

