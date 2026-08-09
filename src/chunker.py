from typing import List, Dict


def chunk_text(
    pages: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Every chunk inherits the document name from its source page.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    for page in pages:

        page_number = page.get(
            "page_number",
            "Unknown"
        )

        document_name = page.get(
            "document_name",
            "Unknown document"
        )

        text = page.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        words = text.split()

        start = 0
        chunk_number = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk_words = words[start:end]

            chunk = " ".join(
                chunk_words
            ).strip()

            if chunk:

                chunks.append(
                    {
                        "text": chunk,

                        "metadata": {
                            "document_name": document_name,
                            "page_number": page_number,
                            "chunk_number": chunk_number
                        }
                    }
                )

            if end == len(words):
                break

            start = end - overlap

            chunk_number += 1

    return chunks