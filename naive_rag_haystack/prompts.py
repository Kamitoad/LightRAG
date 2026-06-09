from __future__ import annotations

from collections.abc import Mapping, Sequence


SYSTEM_PROMPT = """
Du beantwortest Fragen zu Studiengangsinformationen ausschließlich anhand der bereitgestellten Kontexte.

Regeln:

* Antworte auf Deutsch.
* Nutze nur Informationen aus den Kontexten.
* Wenn die Antwort nicht eindeutig in den Kontexten steht, sage: "In den bereitgestellten Quellen nicht eindeutig gefunden."
* Nenne relevante Quellen mit Dokumentname und Abschnitt/Chunk, soweit vorhanden.
* Unterscheide zwischen "nicht gefunden" und "gilt nicht".
* Formuliere vorsichtig und nicht rechtsverbindlich.
* Erfinde keine Module, CrP, Voraussetzungen, Dozierenden, Semesterangebote oder Prüfungsordnungsregeln.
""".strip()


def build_user_prompt(question: str, retrieved_contexts: Sequence[Mapping[str, object]]) -> str:
    context_blocks: list[str] = []
    for index, context in enumerate(retrieved_contexts, start=1):
        metadata = context.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        source_file = metadata.get("source_file", "unbekannt")
        section = metadata.get("section", "unbekannt")
        chunk_index = metadata.get("chunk_index", "unbekannt")
        score = context.get("score")
        score_text = f"{score:.4f}" if isinstance(score, float) else "n/a"
        content = str(context.get("content", "")).strip()
        context_blocks.append(
            "\n".join(
                [
                    f"[Kontext {index}]",
                    f"Quelle: {source_file}",
                    f"Abschnitt: {section}",
                    f"Chunk: {chunk_index}",
                    f"Retriever-Score: {score_text}",
                    "Inhalt:",
                    content,
                ]
            )
        )

    contexts = "\n\n---\n\n".join(context_blocks) if context_blocks else "Keine Kontexte gefunden."
    return f"""
Frage:
{question}

Bereitgestellte Kontexte:
{contexts}

Aufgabe:
Beantworte die Frage anhand der bereitgestellten Kontexte. Wenn die Kontexte keine eindeutige Antwort enthalten, sage genau das und erfinde keine Details.
""".strip()


def build_prompt(question: str, retrieved_contexts: Sequence[Mapping[str, object]]) -> str:
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(question, retrieved_contexts)}"
