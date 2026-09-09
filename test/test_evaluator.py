from src.evaluator import RAGEvaluator


def test_evaluator_basic():
    evaluator = RAGEvaluator()

    results = [
        {
            "metadata": {"document_name": "insightforge_test.pdf", "page_number": 1, "chunk_number": 0},
            "hybrid_score": 0.85
        },
        {
            "metadata": {"document_name": "insightforge_test.pdf", "page_number": 2, "chunk_number": 0},
            "hybrid_score": 0.40
        }
    ]

    metrics = evaluator.evaluate(results, expected_pages=[1])

    assert metrics["source_count"] == 2
    assert metrics["top_score"] == 0.85
    assert metrics["confidence"] == "HIGH"
    assert metrics["relevant_sources"] == 1
    assert metrics["retrieval_precision"] == 0.5


def test_evaluator_empty():
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate([])

    assert metrics["source_count"] == 0
    assert metrics["top_score"] == 0.0
    assert metrics["confidence"] == "LOW"
    assert metrics["relevant_sources"] == 0
    assert metrics["retrieval_precision"] is None