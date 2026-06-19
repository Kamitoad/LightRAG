"""Shared query definitions for the later LightRAG comparison."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    category: str
    question: str


LIGHTRAG_EVALUATION_MODES: tuple[str, ...] = ("naive", "mix")
EVALUATION_SYSTEMS: tuple[str, ...] = (
    "lightrag_naive",
    "lightrag_mix",
    "haystack_naive_rag",
)


SMALL_TEST_QUESTIONS: tuple[EvaluationQuestion, ...] = (
    EvaluationQuestion(
        question_id="Q001",
        category="fact",
        question="Wie viele CrP hat Konzepte des Deep Learning?",
    ),
    EvaluationQuestion(
        question_id="Q002",
        category="content",
        question="Welche Inhalte behandelt Konzepte des Deep Learning?",
    ),
    EvaluationQuestion(
        question_id="Q003",
        category="uncertainty",
        question="Welche Voraussetzungen hat Secure Software Design?",
    ),
    EvaluationQuestion(
        question_id="Q004",
        category="fact",
        question="Zu welchem Bereich gehört Secure Software Design?",
    ),
    EvaluationQuestion(
        question_id="Q005",
        category="rule",
        question="Was sagt die Beispiel-Regel zu Wahlpflichtmodulen?",
    ),
    EvaluationQuestion(
        question_id="Q006",
        category="uncertainty",
        question="Wird Secure Software Design im Beispiel-Stundenplan eindeutig angeboten?",
    ),
)
