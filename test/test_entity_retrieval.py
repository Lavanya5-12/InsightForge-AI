import os
import shutil
import pytest
from src.vector_store import FAISSVectorStore
from src.hybrid_retriever import HybridRetriever
from src.rag_pipeline import RAGPipeline
from src.entity_utils import (
    detect_entity_types,
    expand_query_for_bm25,
    has_entity_evidence,
    normalize_query
)

ASTATECH_TEXT = """AstaTech, Inc.
2525 Pearl Buck Road, Bristol, PA 19007, USA
www.astatechinc.com Tel: +1-215-785-3197 Fax: +1-215-785-2656

Safety Data Sheet

1. Product and Company Identification

Company: AstaTech, Inc.
2525 Pearl Buck Road
Bristol, PA 19007
USA
Tel: +1-215-785-3197
Fax: +1-215-785-2656
"""

CHEMICAL_NO_PHONE_TEXT = """Safety Data Sheet

1. Product and Company Identification
Product Name: ABC Chemical
CAS Number: 12345-67-8
Molecular Weight: 100.20
Chemical Formula: C5H10O2
"""


@pytest.fixture
def mock_embedding_fn(monkeypatch):
    """Mock generate_embedding and generate_embeddings for fast deterministic testing."""
    def dummy_generate_embedding(text, model=None):
        lower = text.lower()
        # Document chunks stored with [1.0, 0.0, 0.0]
        if "astatech" in lower or "chemical" in lower or "cas number:" in lower or "company:" in lower:
            return [1.0, 0.0, 0.0]
        
        # Dense similarity low for entity queries / out of scope queries in tests
        # so BM25 / entity evidence controls retrieval
        return [0.0, 5.0, 0.0]

    def dummy_generate_embeddings(texts, model=None):
        return [dummy_generate_embedding(t, model) for t in texts]

    monkeypatch.setattr("src.hybrid_retriever.generate_embedding", dummy_generate_embedding)
    monkeypatch.setattr("src.embeddings.generate_embedding", dummy_generate_embedding)
    monkeypatch.setattr("src.embeddings.generate_embeddings", dummy_generate_embeddings)
    monkeypatch.setattr("src.rag_pipeline.generate_embeddings", dummy_generate_embeddings)


@pytest.fixture
def astatech_retriever(tmp_path, mock_embedding_fn):
    store_dir = str(tmp_path / "faiss_astatech_test")
    vector_store = FAISSVectorStore(db_path=store_dir)

    chunks = [
        {
            "text": ASTATECH_TEXT,
            "metadata": {"document_name": "AstaTech_SDS_P18528.pdf", "page_number": 1, "chunk_number": 0}
        }
    ]
    dummy_embeddings = [[1.0, 0.0, 0.0]]
    vector_store.add_documents(chunks, dummy_embeddings)

    retriever = HybridRetriever(chunks=chunks, vector_store=vector_store)
    yield retriever, chunks, vector_store

    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)


def test_1_telephone_query_matches_tel_label(astatech_retriever):
    retriever, chunks, _ = astatech_retriever
    question = "What is the telephone number?"
    results = retriever.retrieve(question, n_results=1)

    assert len(results) > 0
    top_chunk = results[0]
    assert "Tel:" in top_chunk["text"]
    assert "+1-215-785-3197" in top_chunk["text"]
    assert top_chunk["metadata"]["document_name"] == "AstaTech_SDS_P18528.pdf"
    assert top_chunk["metadata"]["page_number"] == 1


def test_2_tele_query(astatech_retriever):
    retriever, _, _ = astatech_retriever
    question = "what is tele number?"
    results = retriever.retrieve(question, n_results=1)

    assert len(results) > 0
    top_chunk = results[0]
    assert "+1-215-785-3197" in top_chunk["text"]
    assert top_chunk.get("entity_evidence") is True


def test_3_phone_number_query(astatech_retriever):
    retriever, _, _ = astatech_retriever
    question = "What is the phone number?"
    results = retriever.retrieve(question, n_results=1)

    assert len(results) > 0
    assert "+1-215-785-3197" in results[0]["text"]


def test_4_case_and_punctuation():
    # Test label matching case-insensitively and with various punctuation styles
    sample_texts = [
        "TEL: +1-215-785-3197",
        "Telephone: +1-215-785-3197",
        "Phone No: +1-215-785-3197",
        "Contact Number: +1-215-785-3197",
        "tel. +1-215-785-3197"
    ]
    for text in sample_texts:
        assert has_entity_evidence(text, ["telephone"]) is True, f"Failed for text: {text}"


def test_5_negative_case(tmp_path, mock_embedding_fn):
    store_dir = str(tmp_path / "faiss_negative_test")
    vector_store = FAISSVectorStore(db_path=store_dir)

    chunks = [
        {
            "text": CHEMICAL_NO_PHONE_TEXT,
            "metadata": {"document_name": "chemical_sds.pdf", "page_number": 1, "chunk_number": 0}
        }
    ]
    vector_store.add_documents(chunks, [[1.0, 0.0, 0.0]])

    pipeline = RAGPipeline(chunks=chunks, vector_store=vector_store, reset_vector_store=False)
    question = "What is the telephone number?"
    res = pipeline.ask(question)

    assert res["grounded"] is False
    assert "couldn't find sufficient information" in res["answer"].lower()

    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)


def test_6_cas_number(tmp_path, mock_embedding_fn):
    store_dir = str(tmp_path / "faiss_cas_test")
    vector_store = FAISSVectorStore(db_path=store_dir)

    chunks = [
        {
            "text": "CAS Number: 2173991-59-6\nProduct Name: Example Compound",
            "metadata": {"document_name": "compound.pdf", "page_number": 1, "chunk_number": 0}
        }
    ]
    vector_store.add_documents(chunks, [[1.0, 0.0, 0.0]])

    retriever = HybridRetriever(chunks=chunks, vector_store=vector_store)
    results = retriever.retrieve("What is the CAS number?", n_results=1)

    assert len(results) > 0
    assert "2173991-59-6" in results[0]["text"]
    assert results[0].get("entity_evidence") is True

    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)


def test_7_address(tmp_path, mock_embedding_fn):
    store_dir = str(tmp_path / "faiss_address_test")
    vector_store = FAISSVectorStore(db_path=store_dir)

    chunks = [
        {
            "text": "Company: AstaTech, Inc.\n2525 Pearl Buck Road\nBristol, PA 19007\nUSA",
            "metadata": {"document_name": "address_doc.pdf", "page_number": 1, "chunk_number": 0}
        }
    ]
    vector_store.add_documents(chunks, [[1.0, 0.0, 0.0]])

    retriever = HybridRetriever(chunks=chunks, vector_store=vector_store)
    results = retriever.retrieve("What is the company address?", n_results=1)

    assert len(results) > 0
    assert "2525 Pearl Buck Road" in results[0]["text"]

    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)


def test_8_existing_out_of_scope_behavior(tmp_path, mock_embedding_fn):
    store_dir = str(tmp_path / "faiss_oos_test")
    vector_store = FAISSVectorStore(db_path=store_dir)

    chunks = [
        {
            "text": ASTATECH_TEXT,
            "metadata": {"document_name": "AstaTech_SDS.pdf", "page_number": 1, "chunk_number": 0}
        }
    ]
    vector_store.add_documents(chunks, [[1.0, 0.0, 0.0]])
    pipeline = RAGPipeline(chunks=chunks, vector_store=vector_store, reset_vector_store=False)

    unrelated_question = "What is the secret recipe for baking a traditional chocolate fudge cake?"
    res = pipeline.ask(unrelated_question)

    assert res["grounded"] is False
    assert "couldn't find sufficient information" in res["answer"].lower()

    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)


def test_integration_rag_pipeline_telephone_retrieval(tmp_path, mock_embedding_fn, monkeypatch):
    """Integration level test verifying query -> retrieval -> context contains telephone number."""
    store_dir = str(tmp_path / "faiss_rag_integration_test")
    vector_store = FAISSVectorStore(db_path=store_dir)

    chunks = [
        {
            "text": ASTATECH_TEXT,
            "metadata": {"document_name": "AstaTech SDS P18528.pdf", "page_number": 1, "chunk_number": 0}
        }
    ]
    vector_store.add_documents(chunks, [[1.0, 0.0, 0.0]])
    pipeline = RAGPipeline(chunks=chunks, vector_store=vector_store, reset_vector_store=False)

    # Mock LLM response for deterministic integration test
    def mock_llm_answer(question, context):
        assert "+1-215-785-3197" in context
        return "The telephone number is +1-215-785-3197."

    monkeypatch.setattr("src.rag_pipeline.generate_answer", mock_llm_answer)

    res = pipeline.ask("What is the telephone number?")
    assert res["grounded"] is True
    assert "+1-215-785-3197" in res["answer"]
    assert "AstaTech SDS P18528.pdf" in res["answer"]
    assert "Page 1" in res["answer"]

    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
