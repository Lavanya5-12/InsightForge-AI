from src.evaluator import RAGEvaluator


def main():

    print("=" * 50)
    print("INSIGHTFORGE AI - EVALUATOR TEST")
    print("=" * 50)

    results = [
        {
            "metadata": {
                "document_name": "UNIT 1 notes.pdf",
                "page_number": 1,
                "chunk_number": 0
            },
            "hybrid_score": 1.0
        },
        {
            "metadata": {
                "document_name": "UNIT 1 notes.pdf",
                "page_number": 2,
                "chunk_number": 0
            },
            "hybrid_score": 0.3608
        },
        {
            "metadata": {
                "document_name": "UNIT 1 notes.pdf",
                "page_number": 5,
                "chunk_number": 0
            },
            "hybrid_score": 0.2551
        }
    ]

    evaluator = RAGEvaluator()

    evaluation = evaluator.evaluate(
        results,
        expected_pages=[1]
    )

    print()
    print("Evaluation Results")
    print("-" * 30)

    print(
        f"Sources Retrieved: "
        f"{evaluation['source_count']}"
    )

    print(
        f"Top Hybrid Score: "
        f"{evaluation['top_score']}"
    )

    print(
        f"Average Hybrid Score: "
        f"{evaluation['average_score']}"
    )

    print(
        f"Confidence: "
        f"{evaluation['confidence']}"
    )

    print(
        f"Relevant Sources: "
        f"{evaluation['relevant_sources']}"
    )

    print(
        f"Retrieval Precision: "
        f"{evaluation['retrieval_precision']:.2f}"
    )

    print()
    print("Evaluator test completed successfully!")


if __name__ == "__main__":

    main()