from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import shutil
import sys
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naive_rag_haystack.config import PROJECT_ROOT, RagConfig, load_config
from naive_rag_haystack.logging_utils import append_jsonl, timestamp_id, utc_timestamp
from naive_rag_haystack.run_questions import DEFAULT_QUESTIONS_PATH, load_questions


RESULTS_FILE = PROJECT_ROOT / "evaluation" / "results_lightrag.jsonl"
RUN_LOG_FILE = PROJECT_ROOT / "evaluation" / "run_log.jsonl"
DEFAULT_WORKING_DIR = PROJECT_ROOT / "lightrag_storage" / "coursewiki"


def _embedding_dim(model: str) -> int:
    if model == "text-embedding-3-large":
        return 3072
    return 1536


def _clear_working_dir(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (PROJECT_ROOT / "lightrag_storage").resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to clear LightRAG directory outside {allowed_root}: {resolved}")
    if resolved == allowed_root:
        raise ValueError(f"Refusing to clear LightRAG root directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _read_processed_documents(config: RagConfig) -> list[tuple[Path, str]]:
    documents: list[tuple[Path, str]] = []
    for path in sorted(config.processed_data_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append((path, content))
    if not documents:
        raise FileNotFoundError(f"No Markdown files found in {config.processed_data_dir}")
    return documents


def _to_jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(item) for item in value]
        return str(value)


def _require_lightrag_components() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
    except ImportError as exc:
        raise RuntimeError(
            "LightRAG dependencies are missing. Install this repository's LightRAG dependencies "
            "before running LightRAG experiments, for example with `uv sync` or `pip install -e .`."
        ) from exc
    return LightRAG, QueryParam, openai_complete_if_cache, openai_embed, EmbeddingFunc


def build_lightrag(config: RagConfig, working_dir: Path) -> tuple[Any, Any]:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LightRAG OpenAI comparison runs.")

    LightRAG, QueryParam, openai_complete_if_cache, openai_embed, EmbeddingFunc = (
        _require_lightrag_components()
    )

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        return await openai_complete_if_cache(
            config.openai_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=_embedding_dim(config.openai_embedding_model),
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=config.openai_embedding_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        ),
    )

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_name=config.openai_model,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )
    return rag, QueryParam


async def ensure_indexed(rag: Any, config: RagConfig, run_id: str) -> float:
    started = perf_counter()
    documents = _read_processed_documents(config)
    await rag.ainsert(
        [content for _, content in documents],
        file_paths=[path.as_posix() for path, _ in documents],
    )
    duration_seconds = perf_counter() - started
    append_jsonl(
        RUN_LOG_FILE,
        {
            "run_id": f"{run_id}_index",
            "system": "lightrag",
            "phase": "indexing",
            "duration_seconds": round(duration_seconds, 4),
            "num_documents": len(documents),
            "num_chunks": None,
            "embedding_model": config.openai_embedding_model,
            "llm_model": config.openai_model,
            "timestamp": utc_timestamp(),
        },
    )
    return duration_seconds


async def run_lightrag_questions(
    *,
    config: RagConfig,
    questions_path: Path,
    modes: tuple[str, ...],
    run_id: str,
    working_dir: Path,
    clear_storage: bool,
    clear_results: bool,
) -> list[dict[str, Any]]:
    if clear_storage:
        _clear_working_dir(working_dir)
    if clear_results and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

    working_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(questions_path)
    rag, query_param_cls = build_lightrag(config, working_dir)
    await rag.initialize_storages()

    results: list[dict[str, Any]] = []
    try:
        await ensure_indexed(rag, config, run_id)

        for mode in modes:
            for index, question_row in enumerate(questions, start=1):
                question = question_row["question"]
                assert question is not None
                system = f"lightrag_{mode}"
                print(f"[{system} {index}/{len(questions)}] {question_row.get('question_id')}: {question}")
                started = perf_counter()
                try:
                    response = await rag.aquery_llm(
                        question,
                        param=query_param_cls(mode=mode, stream=False, top_k=config.top_k),
                    )
                    llm_response = response.get("llm_response", {})
                    answer = llm_response.get("content") or ""
                    if inspect.isasyncgen(answer):
                        answer = ""
                    record = {
                        "run_id": run_id,
                        "system": system,
                        "question_id": question_row.get("question_id"),
                        "category": question_row.get("category"),
                        "question": question,
                        "answer": answer,
                        "retrieved_contexts": [],
                        "retrieval_data": _to_jsonable(response.get("data", {})),
                        "metadata": _to_jsonable(response.get("metadata", {})),
                        "latency_seconds": round(perf_counter() - started, 4),
                        "embedding_provider": "openai",
                        "embedding_model": config.openai_embedding_model,
                        "llm_provider": "openai",
                        "llm_model": config.openai_model,
                        "top_k": config.top_k,
                        "estimated_input_tokens": None,
                        "estimated_output_tokens": None,
                        "estimated_cost_usd": None,
                        "timestamp": utc_timestamp(),
                    }
                    print("  OK")
                except Exception as exc:
                    record = {
                        "run_id": run_id,
                        "system": system,
                        "question_id": question_row.get("question_id"),
                        "category": question_row.get("category"),
                        "question": question,
                        "answer": "",
                        "retrieved_contexts": [],
                        "latency_seconds": round(perf_counter() - started, 4),
                        "embedding_provider": "openai",
                        "embedding_model": config.openai_embedding_model,
                        "llm_provider": "openai",
                        "llm_model": config.openai_model,
                        "top_k": config.top_k,
                        "estimated_input_tokens": None,
                        "estimated_output_tokens": None,
                        "estimated_cost_usd": None,
                        "timestamp": utc_timestamp(),
                        "error": str(exc),
                    }
                    print(f"  ERROR: {exc}")
                append_jsonl(RESULTS_FILE, record)
                results.append(record)
    finally:
        await rag.finalize_storages()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightRAG naive/mix over questions.csv.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR)
    parser.add_argument("--run-id", default=f"lightrag_{timestamp_id()}")
    parser.add_argument(
        "--mode",
        choices=["naive", "mix", "both"],
        default="both",
        help="LightRAG query mode to evaluate.",
    )
    parser.add_argument("--clear-storage", action="store_true")
    parser.add_argument("--clear-results", action="store_true")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    config = load_config()
    modes = ("naive", "mix") if args.mode == "both" else (args.mode,)
    results = await run_lightrag_questions(
        config=config,
        questions_path=args.questions,
        modes=modes,
        run_id=args.run_id,
        working_dir=args.working_dir,
        clear_storage=args.clear_storage,
        clear_results=args.clear_results,
    )
    errors = sum(1 for result in results if result.get("error"))
    print(f"Done. Wrote {len(results)} result(s) to {RESULTS_FILE}. Errors: {errors}.")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
