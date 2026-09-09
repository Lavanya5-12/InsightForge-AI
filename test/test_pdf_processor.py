import pytest
from pathlib import Path
from src.pdf_processor import extract_pdf_text, get_pdf_info, PDFProcessingError

FIXTURE_PATH = "data/fixtures/insightforge_test.pdf"


def test_get_pdf_info():
    info = get_pdf_info(FIXTURE_PATH)
    assert info["filename"] == "insightforge_test.pdf"
    assert info["total_pages"] == 15


def test_extract_pdf_text_structure():
    pages = extract_pdf_text(FIXTURE_PATH)
    assert len(pages) == 15

    first_page = pages[0]
    assert first_page["page_number"] == 1
    assert first_page["document_name"] == "insightforge_test.pdf"
    assert "Artificial Intelligence" in first_page["text"]


def test_nonexistent_pdf_raises():
    with pytest.raises(FileNotFoundError):
        extract_pdf_text("data/fixtures/non_existent_file.pdf")
