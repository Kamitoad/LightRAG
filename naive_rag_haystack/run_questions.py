from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naive_rag_haystack.config import RagConfig, load_config
from naive_rag_haystack.indexing import load_or_create_document_store
from naive_rag_haystack.logging_utils import append_query_result, timestamp_id, utc_timestamp
from naive_rag_haystack.query import answer_question


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "data" / "testset" / "questions.csv"


def load_questions(path: Path) -> list[dict[str, str | None]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"question_id", "category", "question"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Questions CSV is missing required columns: {missing}")

        questions: list[dict[str, str | None]] = []
        for row in reader:
            question = (row.get("question") or "").strip()
            if not question:
                continue
            questions.append(
                {
                    "question_id": (row.get("question_id") or "").strip() or None,
                    "category": (row.get("category") or "").strip() or None,
                    "question": question,
                }
            )
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def _error_record(
    *,
    config: RagConfig,
    run_id: str,
    question_row: dict[str, str | None],
    error: Exception,
    latency_seconds: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "system": "haystack_naive_rag",
        "question_id": question_row.get("question_id"),
        "category": question_row.get("category"),
        "question": question_row.get("question"),
        "answer": "",
        "retrieved_contexts": [],
        "latency_seconds": round(latency_seconds, 4),
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model_name,
        "top_k": config.top_k,
        "estimated_input_tokens": None,
        "estimated_output_tokens": None,
        "estimated_cost_usd": None,
        "timestamp": utc_timestamp(),
        "error": str(error),
    }


def run_questions(
    *,
    config: RagConfig,
    questions_path: Path,
    run_id: str,
    clear_results: bool,
) -> list[dict[str, Any]]:
    if clear_results and config.results_file.exists():
        config.results_file.unlink()

    questions = load_questions(questions_path)
    document_store, _ = load_or_create_document_store(config)
    results: list[dict[str, Any]] = []

    for index, question_row in enumerate(questions, start=1):
        question = question_row["question"]
        assert question is not None
        print(f"[{index}/{len(questions)}] {question_row.get('question_id')}: {question}")

        started = perf_counter()
        try:
            result = answer_question(
                question,
                config=config,
                run_id=run_id,
                question_id=question_row.get("question_id"),
                category=question_row.get("category"),
                document_store=document_store,
                log_result=True,
            )
        except Exception as exc:
            result = _error_record(
                config=config,
                run_id=run_id,
                question_row=question_row,
                error=exc,
                latency_seconds=perf_counter() - started,
            )
            append_query_result(config.results_file, result)
            print(f"  ERROR: {exc}")
        else:
            print(f"  OK: {len(result.get('retrieved_contexts', []))} context(s)")
        results.append(result)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Haystack NaiveRAG over questions.csv.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--run-id", default=f"haystack_naive_{timestamp_id()}")
    parser.add_argument(
        "--clear-results",
        action="store_true",
        help="Delete evaluation/results_naive_rag.jsonl before this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    results = run_questions(
        config=config,
        questions_path=args.questions,
        run_id=args.run_id,
        clear_results=args.clear_results,
    )
    errors = sum(1 for result in results if result.get("error"))
    print(
        f"Done. Wrote {len(results)} result(s) to {config.results_file}. "
        f"Errors: {errors}."
    )


if __name__ == "__main__":
    main()
