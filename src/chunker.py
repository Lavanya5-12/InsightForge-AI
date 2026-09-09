from typing import List, Dict
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def chunk_text(
    pages: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict]:
    """
    Split extracted PDF pages into overlapping text chunks while preserving metadata.

    Every chunk retains:
    - document_name
    - page_number
    - chunk_number
    - text
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not pages:
        return []

    chunks = []
    global_chunk_count = 0

    for page in pages:
        page_number = page.get("page_number", 1)
        document_name = page.get("document_name", "Unknown document")
        text = page.get("text", "").strip()

        if not text:
            continue

        words = text.split()
        if not words:
            continue

        start = 0
        page_chunk_index = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text_str = " ".join(chunk_words).strip()

            if chunk_text_str:
                chunks.append({
                    "text": chunk_text_str,
                    "metadata": {
                        "document_name": document_name,
                        "page_number": page_number,
                        "chunk_number": page_chunk_index,
                        "global_chunk_id": global_chunk_count
                    }
                })
                page_chunk_index += 1
                global_chunk_count += 1

            if end == len(words):
                break

            start = end - overlap

    return chunks