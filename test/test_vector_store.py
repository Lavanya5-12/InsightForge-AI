import os
import shutil
import pytest
import numpy as np
from src.vector_store import FAISSVectorStore


@pytest.fixture
def temp_vector_store(tmp_path):
    store_dir = str(tmp_path / "faiss_test")
    store = FAISSVectorStore(db_path=store_dir)
    yield store
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)


def test_vector_store_add_and_search(temp_vector_store):
    store = temp_vector_store
    assert store.count() == 0

    chunks = [
        {"text": "Artificial Intelligence system", "metadata": {"doc": "a.pdf", "page": 1}},
        {"text": "Database indexing algorithms", "metadata": {"doc": "b.pdf", "page": 2}}
    ]

    # Generate dummy embeddings (dimension=4)
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ]

    count = store.add_documents(chunks, embeddings)
    assert count == 2
    assert store.count() == 2
    assert store.index.ntotal == 2

    # Query vector close to first chunk
    query = [0.9, 0.1, 0.0, 0.0]
    results = store.search(query, n_results=2)

    assert len(results) == 2
    assert results[0]["text"] == "Artificial Intelligence system"
    assert results[0]["metadata"]["doc"] == "a.pdf"
    assert results[0]["score"] > results[1]["score"]


def test_vector_store_clear(temp_vector_store):
    store = temp_vector_store
    chunks = [{"text": "Hello world", "metadata": {"page": 1}}]
    embeddings = [[0.5, 0.5]]
    store.add_documents(chunks, embeddings)

    assert store.count() == 1
    store.clear()
    assert store.count() == 0
    assert store.metadata == []
