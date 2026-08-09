from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


def get_pdf_info(pdf_path: str) -> Dict:
    """
    Return basic information about a PDF.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path))

    return {
        "filename": pdf_path.name,
        "total_pages": len(reader.pages)
    }


def extract_pdf_text(pdf_path: str) -> List[Dict]:
    """
    Extract text from every PDF page.

    Each page contains the document name
    so metadata is preserved through the
    RAG pipeline.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path))

    pages = []

    document_name = pdf_path.name

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text is None:
            text = ""

        pages.append(
            {
                "page_number": page_number,
                "document_name": document_name,
                "text": text.strip()
            }
        )

    return pages


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Backward-compatible wrapper.

    Existing scripts can continue using
    extract_text_from_pdf().
    """

    return extract_pdf_text(pdf_path)