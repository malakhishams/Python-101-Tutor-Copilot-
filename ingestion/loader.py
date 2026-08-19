"""
PDF loader.

Loads the course-pack PDF page by page so page numbers can be preserved
as metadata for retrieval and citations.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class PDFPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str


def load_pdf(pdf_path: str | Path) -> list[PDFPage]:
    """
    Load a PDF and return its pages as PDFPage objects.

    Page numbers are 1-based so they match the PDF/textbook pages.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Course-pack PDF not found: {pdf_path}\n"
            "Place the licensed PDF at data/course_pack.pdf."
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    pages: list[PDFPage] = []

    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        # Ignore completely empty pages.
        if not text:
            continue

        pages.append(
            PDFPage(
                page_number=index + 1,
                text=text,
            )
        )

    if not pages:
        raise ValueError(
            f"No extractable text was found in {pdf_path}."
        )

    return pages