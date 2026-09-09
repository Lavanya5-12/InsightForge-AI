import os
import shutil
import pytest
from src.sparse_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.vector_store import FAISSVectorStore


@pytest.fixture
def temp_retriever_setup(tmp_path):
    store_dir = str(tmp_path / "faiss_retriever_test")
    vector_store = FAISSVectorStore(db_path=store_dir)

    chunks = [
        {
            "text": "Artificial Intelligence is a branch of computer science focused on building smart machines.",
            "metadata": {"document_name": "ai_doc.pdf", "page_number": 1, "chunk_number": 0}
        },
        {
            "text": "Types of Artificial Intelligence include Narrow AI, General AI, and Superintelligent AI.",
            "metadata": {"document_name": "ai_doc.pdf", "page_number": 2, "chunk_number": 0}
        },
        {
            "text": "PostgreSQL relational database indexing and SQL query optimization strategies.",
            "metadata": {"document_name": "db_doc.pdf", "page_number": 1, "chunk_number": 0}
        }
    ]

    dummy_embeddings = [
        [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 0.0, 1.0]
    ]

    vector_store.add_documents(chunks, dummy_embeddings)

    yield chunks, vector_store

    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)


def test_bm25_retriever(temp_retriever_setup):
    chunks, _ = temp_retriever_setup
    bm25 = BM25Retriever(chunks)

    results = bm25.search("types of artificial intelligence", n_results=2)
    assert len(results) > 0
    assert "Narrow AI" in results[0]["text"]
    assert results[0]["metadata"]["page_number"] == 2


def test_hybrid_retriever_normalization_and_scores(temp_retriever_setup):
    chunks, vector_store = temp_retriever_setup
    hybrid = HybridRetriever(
        chunks=chunks,
        vector_store=vector_store,
        dense_weight=0.6,
        sparse_weight=0.4
    )

    # Mock query embedding generation for deterministic search test
    def mock_gen_embed(query):
        if "database" in query.lower():
            return [0.0, 0.0, 1.0]
        return [1.0, 0.0, 0.0]

    import src.hybrid_retriever
    monkeypatch_embed = mock_gen_embed
    old_gen = src.hybrid_retriever.generate_embedding
    src.hybrid_retriever.generate_embedding = monkeypatch_embed

    try:
        results = hybrid.retrieve("Artificial Intelligence smart machines", n_results=3)
        assert len(results) > 0

        first_res = results[0]
        assert "hybrid_score" in first_res
        assert "dense_score" in first_res
        assert "sparse_score" in first_res
        assert first_res["metadata"]["document_name"] == "ai_doc.pdf"
        assert first_res["hybrid_score"] >= results[-1]["hybrid_score"]
    finally:
        src.hybrid_retriever.generate_embedding = old_gen
