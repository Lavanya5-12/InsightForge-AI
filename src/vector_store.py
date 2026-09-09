import os
import json
import faiss
import numpy as np
from typing import List, Dict, Optional
from src.config import FAISS_PATH, INDEX_FILE, METADATA_FILE


class FAISSVectorStore:
    """
    FAISS-backed vector store for document embeddings and chunk metadata.
    """

    def __init__(self, db_path: str = FAISS_PATH, load_existing: bool = False):
        self.db_path = db_path
        self.index_file = os.path.join(self.db_path, "index.faiss")
        self.metadata_file = os.path.join(self.db_path, "metadata.json")
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []

        os.makedirs(self.db_path, exist_ok=True)

        if load_existing:
            self._load()

    def _load(self):
        """Load an existing FAISS index and metadata from disk."""
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
            except Exception as e:
                self.index = None

        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = []

    def _save(self):
        """Save FAISS index and metadata to disk."""
        os.makedirs(self.db_path, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, self.index_file)

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def clear(self):
        """
        Safely clear the current vector store lifecycle (memory & disk).
        Ensures new document uploads do not mix with stale indices.
        """
        self.index = None
        self.metadata = []

        if os.path.exists(self.index_file):
            try:
                os.remove(self.index_file)
            except OSError:
                pass

        if os.path.exists(self.metadata_file):
            try:
                os.remove(self.metadata_file)
            except OSError:
                pass

    def add_documents(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add chunk embeddings to the FAISS index and save corresponding metadata.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match."
            )

        vectors = np.array(embeddings, dtype="float32")
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, axis=0)

        dimension = vectors.shape[1]

        # Create FAISS L2 index if missing
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        elif self.index.d != dimension:
            # Re-initialize index if vector dimensions differ
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = []

        self.index.add(vectors)

        for chunk in chunks:
            self.metadata.append({
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {})
            })

        self._save()
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search FAISS index for most similar document chunks.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        if not query_embedding:
            return []

        query_vector = np.array([query_embedding], dtype="float32")

        # Cap n_results to total available vectors
        top_k = min(n_results, self.index.ntotal)

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            item = self.metadata[idx]
            # Convert L2 distance to similarity score [0, 1]
            similarity = 1.0 / (1.0 + float(distance))

            results.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "score": similarity,
                "distance": float(distance)
            })

        return results

    def count(self) -> int:
        """Return number of stored vectors in FAISS index."""
        if self.index is None:
            return 0
        return self.index.ntotal


# Global singleton instance
_store_instance: Optional[FAISSVectorStore] = None


def get_vector_store(reset: bool = False) -> FAISSVectorStore:
    """
    Get or create the global FAISS vector store instance.
    If reset is True, clears the existing vector store for a new session/upload.
    """
    global _store_instance

    if _store_instance is None:
        _store_instance = FAISSVectorStore(load_existing=not reset)

    if reset:
        _store_instance.clear()

    return _store_instance


def add_documents(chunks: List[Dict], embeddings: List[List[float]]) -> int:
    store = get_vector_store()
    return store.add_documents(chunks, embeddings)


def search_documents(query_embedding: List[float], n_results: int = 5) -> List[Dict]:
    store = get_vector_store()
    return store.search(query_embedding, n_results=n_results)


def get_document_count() -> int:
    store = get_vector_store()
    return store.count()