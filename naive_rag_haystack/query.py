from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naive_rag_haystack.config import RagConfig, load_config
from naive_rag_haystack.embeddings import embed_query_openai
from naive_rag_haystack.indexing import load_or_create_document_store
from naive_rag_haystack.logging_utils import append_query_result, document_to_context, timestamp_id, utc_timestamp
from naive_rag_haystack.prompts import SYSTEM_PROMPT, build_user_prompt


def _require_haystack_query_components() -> tuple[Any, Any]:
    try:
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    except ImportError as exc:
        raise RuntimeError(
            "Haystack query dependencies are missing. Run `pip install -r requirements.txt`."
        ) from exc
    return None, InMemoryEmbeddingRetriever


def _require_sentence_transformers_text_embedder() -> Any:
    try:
        from haystack.components.embedders import SentenceTransformersTextEmbedder
    except ImportError as exc:
        raise RuntimeError(
            "SentenceTransformers Haystack embedder is missing. Run `pip install -r requirements.txt`."
        ) from exc
    return SentenceTransformersTextEmbedder


def retrieve_documents(question: str, config: RagConfig, document_store: Any) -> list[Any]:
    _, retriever_cls = _require_haystack_query_components()
    if config.embedding_provider == "openai":
        query_embedding = embed_query_openai(question, config)
    else:
        text_embedder_cls = _require_sentence_transformers_text_embedder()
        text_embedder = text_embedder_cls(model=config.embedding_model)
        text_embedder.warm_up()
        query_embedding = text_embedder.run(text=question)["embedding"]

    retriever = retriever_cls(document_store=document_store, top_k=config.top_k)
    result = retriever.run(query_embedding=query_embedding)
    return result["documents"]


def _require_requests() -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("The `requests` package is missing. Run `pip install -r requirements.txt`.") from exc
    return requests


def _call_ollama(user_prompt: str, config: RagConfig) -> str:
    requests = _require_requests()
    response = requests.post(
        f"{config.ollama_base_url}/api/generate",
        json={
            "model": config.ollama_model,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=config.request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    answer = payload.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"Ollama returned no usable response: {payload}")
    return answer.strip()


def _call_openai_compatible(user_prompt: str, config: RagConfig) -> str:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM_PROVIDER=openai.")
    if not config.openai_model:
        raise RuntimeError("OPENAI_MODEL is required for LLM_PROVIDER=openai.")

    requests = _require_requests()
    response = requests.post(
        f"{config.openai_base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.openai_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=config.request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"OpenAI-compatible provider returned no usable response: {payload}") from exc


def _fallback_answer(generation_error: str | None) -> str:
    if generation_error:
        return (
            "LLM-Generierung fehlgeschlagen. Retrieval wurde ausgeführt und die gefundenen Kontexte wurden geloggt. "
            f"Fehler: {generation_error}"
        )
    return "LLM_PROVIDER=none gesetzt. Retrieval wurde ausgeführt; es wurde keine generative Antwort erstellt."


def generate_answer(user_prompt: str, config: RagConfig) -> tuple[str, str | None]:
    provider = config.llm_provider
    if provider == "ollama":
        try:
            return _call_ollama(user_prompt, config), None
        except Exception as exc:  # keep retrieval logs even when the local LLM is not running
            return _fallback_answer(str(exc)), str(exc)
    if provider in {"openai", "openai_compatible"}:
        try:
            return _call_openai_compatible(user_prompt, config), None
        except Exception as exc:
            return _fallback_answer(str(exc)), str(exc)
    if provider in {"none", "retrieval_only"}:
        return _fallback_answer(None), None
    raise ValueError(f"Unsupported LLM_PROVIDER={config.llm_provider!r}.")


def answer_question(
    question: str,
    *,
    config: RagConfig | None = None,
    run_id: str | None = None,
    question_id: str | None = None,
    category: str | None = None,
    document_store: Any | None = None,
    log_result: bool = True,
) -> dict[str, Any]:
    config = config or load_config()
    run_id = run_id or f"naive_small_{timestamp_id()}"

    started = perf_counter()
    if document_store is None:
        document_store, _ = load_or_create_document_store(config)
    retrieved_documents = retrieve_documents(question, config, document_store)
    retrieved_contexts = [document_to_context(document) for document in retrieved_documents]
    user_prompt = build_user_prompt(question, retrieved_contexts)
    answer, generation_error = generate_answer(user_prompt, config)
    latency_seconds = perf_counter() - started

    record: dict[str, Any] = {
        "run_id": run_id,
        "system": "haystack_naive_rag",
        "question_id": question_id,
        "category": category,
        "question": question,
        "answer": answer,
        "retrieved_contexts": retrieved_contexts,
        "latency_seconds": round(latency_seconds, 4),
        "embedding_model": config.embedding_model,
        "embedding_provider": config.embedding_provider,
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model_name,
        "top_k": config.top_k,
        "estimated_input_tokens": None,
        "estimated_output_tokens": None,
        "estimated_cost_usd": None,
        "timestamp": utc_timestamp(),
    }
    if generation_error:
        record["generation_error"] = generation_error

    if log_result:
        append_query_result(config.results_file, record)
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Haystack NaiveRAG baseline for one question.")
    parser.add_argument("question", nargs="*", help="Question to answer. If omitted, interactive input is used.")
    parser.add_argument("--run-id", default="naive_small_001", help="Run ID written to the JSONL result.")
    parser.add_argument("--question-id", default=None, help="Optional question ID for later evaluation.")
    parser.add_argument("--category", default=None, help="Optional question category for later evaluation.")
    parser.add_argument("--no-log", action="store_true", help="Do not append the result to evaluation/results_naive_rag.jsonl.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    question = " ".join(args.question).strip()
    if not question:
        question = input("Frage: ").strip()
    if not question:
        raise SystemExit("No question provided.")

    result = answer_question(
        question,
        run_id=args.run_id,
        question_id=args.question_id,
        category=args.category,
        log_result=not args.no_log,
    )
    print("\nAntwort:\n")
    print(result["answer"])
    print("\nRetrieved contexts:\n")
    for index, context in enumerate(result["retrieved_contexts"], start=1):
        metadata = context.get("metadata", {})
        print(
            f"{index}. {metadata.get('source_file')} | {metadata.get('section')} | "
            f"chunk={metadata.get('chunk_index')} | score={context.get('score')}"
        )


if __name__ == "__main__":
    main()
