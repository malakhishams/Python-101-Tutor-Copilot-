"""
Text chunking for the Python 101 course pack.

The chunker keeps chunks reasonably small for retrieval while preserving
page/source/chapter metadata for grounded citations.
"""

from dataclasses import dataclass
import re

from ingestion.loader import PDFPage


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


@dataclass
class TextChunk:
    """A searchable chunk of textbook content."""

    text: str
    source: str
    page: int
    chapter: str
    chunk_id: str


def _clean_text(text: str) -> str:
    """Normalize whitespace without destroying paragraph boundaries."""
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse excessive spaces while preserving newlines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _detect_chapter(text: str, current_chapter: str = "") -> str:
    """
    Try to detect a chapter/section heading.

    This is intentionally conservative. If no obvious heading is found,
    the previous chapter value is retained.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:8]:
        lowered = line.lower()

        if (
            lowered.startswith("chapter ")
            or lowered.startswith("part ")
            or lowered.startswith("section ")
        ):
            return line

    return current_chapter


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Character-based splitting keeps the implementation dependency-light
    and is sufficient for this MVP.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer ending at a paragraph/sentence boundary.
        if end < len(text):
            boundary_candidates = [
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]

            best_boundary = max(boundary_candidates)

            if best_boundary > start + chunk_size // 2:
                end = best_boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap

        # Safety against an infinite loop.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def chunk_pages(
    pages: list[PDFPage],
    source: str = "course_pack.pdf",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Convert PDF pages into searchable chunks.

    Metadata preserved:
      - source
      - page
      - chapter
      - chunk_id
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    current_chapter = ""

    for page in pages:
        text = _clean_text(page.text)

        if not text:
            continue

        current_chapter = _detect_chapter(
            text,
            current_chapter=current_chapter,
        )

        page_chunks = _split_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_number, chunk_text in enumerate(page_chunks):
            chunk_id = (
                f"page_{page.page_number}_chunk_{chunk_number}"
            )

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    source=source,
                    page=page.page_number,
                    chapter=current_chapter,
                    chunk_id=chunk_id,
                )
            )

    return chunks