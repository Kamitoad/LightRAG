from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_pymupdf():
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed. Run `pip install -r requirements.txt`.") from exc
    return fitz


def extract_pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Path:
    fitz = _require_pymupdf()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}.md"

    markdown_parts = [f"# {pdf_path.name}", ""]
    with fitz.open(pdf_path) as document:
        for page_index in range(document.page_count):
            page_number = page_index + 1
            page = document.load_page(page_index)
            text = page.get_text("text").strip()
            markdown_parts.extend(
                [
                    f"<!-- source: {pdf_path.name} page: {page_number} -->",
                    "",
                    f"## Seite {page_number}",
                    "",
                    text if text else "[Kein extrahierbarer Text gefunden]",
                    "",
                ]
            )

    output_path.write_text("\n".join(markdown_parts).strip() + "\n", encoding="utf-8")
    return output_path


def update_sources_metadata(metadata_path: Path, extracted: list[tuple[Path, Path]]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for source_path, processed_path in extracted:
        records.append(
            {
                "source_file": source_path.name,
                "source_path": str(source_path.as_posix()),
                "processed_file": processed_path.name,
                "processed_path": str(processed_path.as_posix()),
                "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "extractor": "pymupdf",
            }
        )
    metadata_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_all(raw_dir: Path, output_dir: Path, metadata_path: Path | None = None) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found in {raw_dir}")
        return []

    extracted_pairs: list[tuple[Path, Path]] = []
    for pdf_path in pdf_paths:
        output_path = extract_pdf_to_markdown(pdf_path, output_dir)
        extracted_pairs.append((pdf_path, output_path))
        print(f"Extracted {pdf_path.name} -> {output_path}")

    if metadata_path is not None:
        update_sources_metadata(metadata_path, extracted_pairs)
    return [processed_path for _, processed_path in extracted_pairs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract data/raw/*.pdf to stable Markdown files in data/processed/.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "metadata" / "sources.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = extract_all(args.raw_dir, args.output_dir, args.metadata)
    print(f"Done. Extracted {len(output_paths)} PDF file(s).")


if __name__ == "__main__":
    main()
