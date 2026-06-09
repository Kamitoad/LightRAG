from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - handled by runtime message in scripts
    load_dotenv = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv_if_available() -> None:
    if load_dotenv is None:
        return
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path if env_path.exists() else None)


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw_value!r}.") from exc


def _get_path(name: str, default: str) -> Path:
    raw_value = os.getenv(name, default)
    path = Path(raw_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class RagConfig:
    project_name: str
    processed_data_dir: Path
    evaluation_dir: Path
    index_dir: Path
    index_file: Path
    results_file: Path
    run_log_file: Path
    embedding_provider: str
    embedding_model: str
    openai_embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    request_timeout_seconds: int

    @property
    def llm_model_name(self) -> str:
        if self.llm_provider == "ollama":
            return self.ollama_model
        if self.llm_provider in {"openai", "openai_compatible"}:
            return self.openai_model
        return ""


def load_config() -> RagConfig:
    _load_dotenv_if_available()

    processed_data_dir = _get_path("PROCESSED_DATA_DIR", "data/processed")
    evaluation_dir = _get_path("EVALUATION_DIR", "evaluation")
    index_dir = _get_path("HAYSTACK_INDEX_DIR", "data/indexes")

    chunk_size = _get_int("HAYSTACK_CHUNK_SIZE", 1000)
    chunk_overlap = _get_int("HAYSTACK_CHUNK_OVERLAP", 150)
    if chunk_size <= 0:
        raise ValueError("HAYSTACK_CHUNK_SIZE must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("HAYSTACK_CHUNK_OVERLAP must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("HAYSTACK_CHUNK_OVERLAP must be smaller than HAYSTACK_CHUNK_SIZE.")

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()
    if embedding_provider not in {"sentence_transformers", "openai"}:
        raise ValueError(
            "EMBEDDING_PROVIDER must be either 'sentence_transformers' or 'openai'."
        )
    openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    local_embedding_model = os.getenv(
        "HAYSTACK_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_model = openai_embedding_model if embedding_provider == "openai" else local_embedding_model
    index_file = index_dir / f"naive_haystack_{_safe_fragment(embedding_provider)}_{_safe_fragment(embedding_model)}.json"

    return RagConfig(
        project_name=os.getenv("PROJECT_NAME", "CourseWiki-RAG"),
        processed_data_dir=processed_data_dir,
        evaluation_dir=evaluation_dir,
        index_dir=index_dir,
        index_file=index_file,
        results_file=evaluation_dir / "results_naive_rag.jsonl",
        run_log_file=evaluation_dir / "run_log.jsonl",
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        openai_embedding_model=openai_embedding_model,
        top_k=_get_int("HAYSTACK_TOP_K", 5),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        request_timeout_seconds=_get_int("LLM_REQUEST_TIMEOUT_SECONDS", 120),
    )


def _safe_fragment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()
