from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from haystack import Document
except ImportError:  # pragma: no cover - dependency availability is checked at runtime
    Document = None  # type: ignore[assignment]


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class MarkdownSource:
    path: Path
    content: str


def require_haystack_document() -> Any:
    if Document is None:
        raise RuntimeError(
            "Haystack is not installed. Run `pip install -r requirements.txt` before indexing or querying."
        )
    return Document


def load_markdown_sources(processed_data_dir: Path) -> list[MarkdownSource]:
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"Processed data directory does not exist: {processed_data_dir}")

    sources: list[MarkdownSource] = []
    for path in sorted(processed_data_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            sources.append(MarkdownSource(path=path, content=content))
    if not sources:
        raise FileNotFoundError(f"No non-empty Markdown files found in {processed_data_dir}")
    return sources


def _extract_sections(content: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(content))
    if not matches:
        return [("document", content.strip())]

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()
        body_without_heading = content[match.end() : end].strip()
        if len(body_without_heading) < 20:
            continue
        sections.append((title, section_text))
    return sections or [("document", content.strip())]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [(0, len(cleaned), cleaned)]

    chunks: list[tuple[int, int, str]] = []
    start = 0
    text_length = len(cleaned)
    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = hard_end
        if hard_end < text_length:
            newline_cut = cleaned.rfind("\n", start, hard_end)
            if newline_cut > start + (chunk_size // 2):
                end = newline_cut
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _metadata_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def build_haystack_documents(
    processed_data_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Any]:
    document_cls = require_haystack_document()
    haystack_documents: list[Any] = []
    chunk_index = 0

    for source in load_markdown_sources(processed_data_dir):
        for section_title, section_text in _extract_sections(source.content):
            for char_start, char_end, chunk in _split_text(section_text, chunk_size, chunk_overlap):
                metadata = {
                    "source_file": source.path.name,
                    "source_path": _metadata_path(source.path),
                    "section": section_title,
                    "chunk_index": chunk_index,
                    "char_start": char_start,
                    "char_end": char_end,
                }
                haystack_documents.append(document_cls(content=chunk, meta=metadata))
                chunk_index += 1

    if not haystack_documents:
        raise RuntimeError("No chunks were created from processed Markdown files.")
    return haystack_documents


def count_source_files(processed_data_dir: Path) -> int:
    return len(load_markdown_sources(processed_data_dir))
