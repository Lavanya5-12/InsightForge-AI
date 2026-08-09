import requests


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "mxbai-embed-large:latest"


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Ollama.
    """

    if not texts:
        return []

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"]


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text.
    """

    embeddings = generate_embeddings([text])

    return embeddings[0]