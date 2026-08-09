import os
import json
import faiss
import numpy as np


FAISS_PATH = "faiss_db"
INDEX_FILE = os.path.join(FAISS_PATH, "index.faiss")
METADATA_FILE = os.path.join(FAISS_PATH, "metadata.json")


class FAISSVectorStore:

    def __init__(self):
        self.index = None
        self.metadata = []

        os.makedirs(FAISS_PATH, exist_ok=True)

        self._load()

    def _load(self):
        """
        Load an existing FAISS index and metadata.
        """

        if os.path.exists(INDEX_FILE):
            self.index = faiss.read_index(INDEX_FILE)

        if os.path.exists(METADATA_FILE):
            with open(
                METADATA_FILE,
                "r",
                encoding="utf-8"
            ) as file:
                self.metadata = json.load(file)

    def _save(self):
        """
        Save FAISS index and metadata.
        """

        if self.index is not None:
            faiss.write_index(
                self.index,
                INDEX_FILE
            )

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

    def add_documents(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ):
        """
        Add document embeddings to FAISS.
        """

        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        vectors = np.array(
            embeddings,
            dtype="float32"
        )

        dimension = vectors.shape[1]

        # Create index if it doesn't exist
        if self.index is None:
            self.index = faiss.IndexFlatL2(
                dimension
            )

        # Add vectors
        self.index.add(vectors)

        # Store corresponding metadata
        for chunk in chunks:

            self.metadata.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })

        self._save()

        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5
    ):
        """
        Search FAISS for the most similar chunks.
        """

        if self.index is None:
            return []

        if self.index.ntotal == 0:
            return []

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        n_results = min(
            n_results,
            self.index.ntotal
        )

        distances, indices = self.index.search(
            query_vector,
            n_results
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index < 0:
                continue

            item = self.metadata[index]

            # Convert L2 distance into a similarity-like score
            similarity = 1 / (1 + float(distance))

            results.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "score": similarity,
                "distance": float(distance)
            })

        return results

    def count(self):
        """
        Return number of stored vectors.
        """

        if self.index is None:
            return 0

        return self.index.ntotal


# Global store instance
_store = None


def get_vector_store():
    global _store

    if _store is None:
        _store = FAISSVectorStore()

    return _store


def add_documents(
    chunks: list[dict],
    embeddings: list[list[float]]
):
    store = get_vector_store()

    return store.add_documents(
        chunks,
        embeddings
    )


def search_documents(
    query_embedding: list[float],
    n_results: int = 5
):
    store = get_vector_store()

    return store.search(
        query_embedding,
        n_results
    )


def get_document_count():
    store = get_vector_store()

    return store.count()