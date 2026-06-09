from __future__ import annotations

from typing import Any

from naive_rag_haystack.config import RagConfig


def _require_requests() -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("The `requests` package is missing. Run `pip install -r requirements.txt`.") from exc
    return requests


def embed_texts_openai(texts: list[str], config: RagConfig) -> list[list[float]]:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai.")

    requests = _require_requests()
    response = requests.post(
        f"{config.openai_base_url}/embeddings",
        headers={
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.openai_embedding_model,
            "input": texts,
        },
        timeout=config.request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"OpenAI embeddings response has no data list: {payload}")

    sorted_items = sorted(data, key=lambda item: item.get("index", 0))
    embeddings: list[list[float]] = []
    for item in sorted_items:
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"OpenAI embeddings response item has no embedding list: {item}")
        embeddings.append([float(value) for value in embedding])

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"OpenAI returned {len(embeddings)} embeddings for {len(texts)} input texts."
        )
    return embeddings


def assign_openai_embeddings(documents: list[Any], config: RagConfig) -> list[Any]:
    texts = [getattr(document, "content", "") or "" for document in documents]
    embeddings = embed_texts_openai(texts, config)
    for document, embedding in zip(documents, embeddings, strict=True):
        document.embedding = embedding
    return documents


def embed_query_openai(question: str, config: RagConfig) -> list[float]:
    return embed_texts_openai([question], config)[0]
