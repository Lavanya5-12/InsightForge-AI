import pytest
from src.chunker import chunk_text


def test_chunk_text_basic():
    pages = [
        {
            "page_number": 1,
            "document_name": "test_doc.pdf",
            "text": "Word " * 600  # 600 words
        }
    ]

    chunks = chunk_text(pages, chunk_size=500, overlap=100)
    assert len(chunks) == 2

    # Check metadata preservation
    for idx, chunk in enumerate(chunks):
        assert chunk["metadata"]["document_name"] == "test_doc.pdf"
        assert chunk["metadata"]["page_number"] == 1
        assert chunk["metadata"]["chunk_number"] == idx
        assert len(chunk["text"]) > 0


def test_chunk_text_invalid_params():
    pages = [{"page_number": 1, "document_name": "doc.pdf", "text": "sample text"}]

    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        chunk_text(pages, chunk_size=0)

    with pytest.raises(ValueError, match="overlap cannot be negative"):
        chunk_text(pages, chunk_size=100, overlap=-10)

    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_text(pages, chunk_size=100, overlap=100)


def test_chunk_text_empty_input():
    assert chunk_text([]) == []
