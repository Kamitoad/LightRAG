from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naive_rag_haystack.config import RagConfig, load_config
from naive_rag_haystack.documents import build_haystack_documents, count_source_files, require_haystack_document
from naive_rag_haystack.embeddings import assign_openai_embeddings
from naive_rag_haystack.logging_utils import append_run_log, timestamp_id


def _require_haystack_indexing_components() -> tuple[Any, Any]:
    try:
        from haystack.document_stores.in_memory import InMemoryDocumentStore
    except ImportError as exc:
        raise RuntimeError(
            "Haystack indexing dependencies are missing. Run `pip install -r requirements.txt`."
        ) from exc
    return None, InMemoryDocumentStore


def _require_sentence_transformers_document_embedder() -> Any:
    try:
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    except ImportError as exc:
        raise RuntimeError(
            "SentenceTransformers Haystack embedder is missing. Run `pip install -r requirements.txt`."
        ) from exc
    return SentenceTransformersDocumentEmbedder


def _embedding_to_json(embedding: Any) -> list[float] | None:
    if embedding is None:
        return None
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return [float(value) for value in embedding]


def save_index_documents(path: Path, documents: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for document in documents:
        payload.append(
            {
                "id": getattr(document, "id", None),
                "content": getattr(document, "content", "") or "",
                "meta": dict(getattr(document, "meta", {}) or {}),
                "embedding": _embedding_to_json(getattr(document, "embedding", None)),
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index_documents(path: Path) -> list[Any]:
    document_cls = require_haystack_document()
    raw_documents = json.loads(path.read_text(encoding="utf-8"))
    documents: list[Any] = []
    for item in raw_documents:
        documents.append(
            document_cls(
                id=item.get("id"),
                content=item.get("content", ""),
                meta=item.get("meta") or {},
                embedding=item.get("embedding"),
            )
        )
    return documents


def create_document_store(documents: list[Any]) -> Any:
    _, document_store_cls = _require_haystack_indexing_components()
    document_store = document_store_cls(embedding_similarity_function="cosine")
    document_store.write_documents(documents)
    return document_store


def index_documents(config: RagConfig | None = None, *, persist: bool = True, log_run: bool = True) -> tuple[Any, list[Any], float]:
    config = config or load_config()

    started = perf_counter()
    documents = build_haystack_documents(
        config.processed_data_dir,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    if config.embedding_provider == "openai":
        embedded_documents = assign_openai_embeddings(documents, config)
    else:
        document_embedder_cls = _require_sentence_transformers_document_embedder()
        document_embedder = document_embedder_cls(model=config.embedding_model, progress_bar=True)
        document_embedder.warm_up()
        embedded_documents = document_embedder.run(documents=documents)["documents"]
    document_store = create_document_store(embedded_documents)
    duration_seconds = perf_counter() - started

    if persist:
        save_index_documents(config.index_file, embedded_documents)

    if log_run:
        append_run_log(
            config.run_log_file,
            run_id=f"naive_small_index_{timestamp_id()}",
            phase="indexing",
            duration_seconds=duration_seconds,
            num_documents=count_source_files(config.processed_data_dir),
            num_chunks=len(embedded_documents),
            embedding_model=config.embedding_model,
        )

    return document_store, embedded_documents, duration_seconds


def load_or_create_document_store(config: RagConfig | None = None) -> tuple[Any, list[Any]]:
    config = config or load_config()
    if config.index_file.exists():
        documents = load_index_documents(config.index_file)
        return create_document_store(documents), documents

    document_store, documents, _ = index_documents(config, persist=True, log_run=True)
    return document_store, documents


def main() -> None:
    config = load_config()
    document_store, documents, duration_seconds = index_documents(config, persist=True, log_run=True)
    _ = document_store
    print(
        f"Indexed {len(documents)} chunks from {count_source_files(config.processed_data_dir)} Markdown file(s) "
        f"in {duration_seconds:.2f}s. Index: {config.index_file}"
    )


if __name__ == "__main__":
    main()
