from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORING_PATH = PROJECT_ROOT / "evaluation" / "scoring_template.csv"

SCORE_COLUMNS = [
    "correctness",
    "completeness",
    "citation_accuracy",
    "uncertainty_handling",
]


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scoring file not found: {path}")

    frame = pd.read_csv(path)
    required_columns = {"question_id", "system", *SCORE_COLUMNS, "hallucination"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Scoring file is missing required columns: {missing}")

    for column in SCORE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def summarize_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.dropna(subset=SCORE_COLUMNS, how="all").copy()
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "system",
                "num_scored_questions",
                "correctness_mean",
                "completeness_mean",
                "citation_accuracy_mean",
                "uncertainty_handling_mean",
                "hallucination_yes_count",
            ]
        )

    hallucination_yes = (
        scored.assign(
            hallucination_yes=scored["hallucination"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"yes", "ja", "true", "1"})
        )
        .groupby("system")["hallucination_yes"]
        .sum()
    )

    summary = (
        scored.groupby("system")
        .agg(
            num_scored_questions=("question_id", "nunique"),
            correctness_mean=("correctness", "mean"),
            completeness_mean=("completeness", "mean"),
            citation_accuracy_mean=("citation_accuracy", "mean"),
            uncertainty_handling_mean=("uncertainty_handling", "mean"),
        )
        .join(hallucination_yes.rename("hallucination_yes_count"))
        .reset_index()
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize manually filled CourseWiki-RAG evaluation scores."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORING_PATH,
        help="Path to the manually filled scoring CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    summary = summarize_scores(scores)

    if summary.empty:
        print("No manual scores found yet.")
        return

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
