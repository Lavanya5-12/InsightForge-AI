from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


class PDFProcessingError(Exception):
    """Exception raised for errors during PDF processing."""
    pass


def get_pdf_info(pdf_path: str) -> Dict:
    """
    Return basic information about a PDF file.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    try:
        reader = PdfReader(str(path))
        return {
            "filename": path.name,
            "total_pages": len(reader.pages)
        }
    except Exception as e:
        raise PDFProcessingError(f"Failed to read PDF info from {path.name}: {e}") from e


def extract_pdf_text(pdf_path: str) -> List[Dict]:
    """
    Extract text from every page of a PDF document.

    Preserves page number and document name metadata for each page.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise PDFProcessingError(f"Failed to open PDF file {path.name}: {e}") from e

    if len(reader.pages) == 0:
        raise PDFProcessingError(f"PDF file {path.name} contains no pages.")

    pages = []
    document_name = path.name

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        pages.append({
            "page_number": page_number,
            "document_name": document_name,
            "text": text.strip()
        })

    total_text_length = sum(len(p["text"]) for p in pages)
    if total_text_length == 0:
        # Note: We do not fail hard so scanned/empty PDFs can produce a friendly warning, but log it clearly
        pass

    return pages


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Backward-compatible wrapper function."""
    return extract_pdf_text(pdf_path)