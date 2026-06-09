from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def document_to_context(document: Any) -> dict[str, Any]:
    score = getattr(document, "score", None)
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
    return {
        "content": getattr(document, "content", "") or "",
        "metadata": dict(getattr(document, "meta", {}) or {}),
        "score": score,
    }


def append_run_log(
    path: Path,
    *,
    run_id: str,
    phase: str,
    duration_seconds: float,
    num_documents: int,
    num_chunks: int,
    embedding_model: str,
) -> None:
    append_jsonl(
        path,
        {
            "run_id": run_id,
            "system": "haystack_naive_rag",
            "phase": phase,
            "duration_seconds": round(duration_seconds, 4),
            "num_documents": num_documents,
            "num_chunks": num_chunks,
            "embedding_model": embedding_model,
            "timestamp": utc_timestamp(),
        },
    )


def append_query_result(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, record)
