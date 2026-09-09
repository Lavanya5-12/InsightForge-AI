from typing import List
from src.ollama_client import default_ollama_client, OllamaError
from src.config import EMBEDDING_MODEL


def generate_embeddings(
    texts: List[str],
    model: str = EMBEDDING_MODEL
) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings using OllamaClient.
    """
    if not texts:
        return []

    return default_ollama_client.generate_embeddings(texts, model=model)


def generate_embedding(
    text: str,
    model: str = EMBEDDING_MODEL
) -> List[float]:
    """
    Generate an embedding vector for a single text string.
    """
    if not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")

    embeddings = generate_embeddings([text], model=model)
    if not embeddings:
        raise OllamaError("Failed to generate embedding for input text.")

    return embeddings[0]