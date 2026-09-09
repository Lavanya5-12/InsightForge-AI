import pytest
from src.pdf_processor import extract_pdf_text
from src.chunker import chunk_text
from src.rag_pipeline import RAGPipeline
from src.ollama_client import default_ollama_client
from src.config import RETRIEVAL_THRESHOLD

FIXTURE_PATH = "data/fixtures/insightforge_test.pdf"


@pytest.fixture(scope="module")
def rag_pipeline_fixture():
    pages = extract_pdf_text(FIXTURE_PATH)
    chunks = chunk_text(pages)
    pipeline = RAGPipeline(chunks, reset_vector_store=True)
    return pipeline


def test_rag_ingestion_faiss_populated(rag_pipeline_fixture):
    pipeline = rag_pipeline_fixture
    assert pipeline.vector_store.count() > 0
    assert pipeline.vector_store.count() == len(pipeline.chunks)


def test_out_of_scope_question_rejection(rag_pipeline_fixture):
    pipeline = rag_pipeline_fixture

    # Question completely unrelated to document content
    unrelated_question = "What is the secret recipe for baking a traditional chocolate fudge cake?"
    res = pipeline.ask(unrelated_question, threshold=RETRIEVAL_THRESHOLD)

    assert "question" in res
    assert "answer" in res
    assert "couldn't find sufficient information" in res["answer"].lower() or "not find" in res["answer"].lower()
    assert res.get("grounded") is False


def test_supported_question_with_citations(rag_pipeline_fixture):
    online, _ = default_ollama_client.check_health()
    if not online:
        pytest.skip("Ollama service offline. Skipping live LLM generation test.")

    pipeline = rag_pipeline_fixture
    question = "What is Artificial Intelligence?"
    res = pipeline.ask(question, threshold=RETRIEVAL_THRESHOLD)

    assert res["question"] == question
    assert len(res["answer"]) > 0
    assert "couldn't find" not in res["answer"].lower()
    assert "Sources:" in res["answer"] or len(res["sources"]) > 0
