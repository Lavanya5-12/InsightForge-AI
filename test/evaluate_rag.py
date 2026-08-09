from src.rag_pipeline import RAGPipeline
from src.pdf_processor import extract_pdf_text
from src.chunker import chunk_text
from src.evaluator import RAGEvaluator
from test.evaluation_dataset import EVALUATION_DATASET


PDF_PATH = r"data\uploads\UNIT 1 notes.pdf"


def main():

    print("=" * 60)
    print("INSIGHTFORGE AI - RAG EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------

    print("\n1. Loading PDF...")

    pages = extract_pdf_text(PDF_PATH)

    print(f"   Pages: {len(pages)}")

    # --------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------

    print("\n2. Creating chunks...")

    chunks = chunk_text(pages)

    print(f"   Chunks: {len(chunks)}")

    # --------------------------------------------------
    # 3. Create RAG pipeline
    # --------------------------------------------------

    print("\n3. Creating RAG pipeline...")

    rag = RAGPipeline(chunks)

    print("   RAG pipeline ready!")

    # --------------------------------------------------
    # 4. Evaluate questions
    # --------------------------------------------------

    evaluator = RAGEvaluator()

    total_questions = len(EVALUATION_DATASET)
    total_relevant = 0
    total_retrieved = 0

    print("\n4. Running evaluation...")
    print("-" * 60)

    for index, item in enumerate(
        EVALUATION_DATASET,
        start=1
    ):

        question = item["question"]
        expected_pages = item["expected_pages"]

        print(f"\nQuestion {index}/{total_questions}")
        print(f"Question: {question}")
        print(f"Expected Pages: {expected_pages}")

        # Retrieve sources
        results = rag.retriever.retrieve(
            question,
            n_results=5
        )

        # Evaluate retrieval
        evaluation = evaluator.evaluate(
            results,
            expected_pages=expected_pages
        )

        total_relevant += evaluation[
            "relevant_sources"
        ]

        total_retrieved += evaluation[
            "source_count"
        ]

        print(
            f"Retrieved Sources: "
            f"{evaluation['source_count']}"
        )

        print(
            f"Relevant Sources: "
            f"{evaluation['relevant_sources']}"
        )

        print(
            f"Top Score: "
            f"{evaluation['top_score']:.4f}"
        )

        print(
            f"Precision: "
            f"{evaluation['retrieval_precision']:.2f}"
        )

    # --------------------------------------------------
    # 5. Overall evaluation
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("OVERALL EVALUATION")
    print("=" * 60)

    overall_precision = 0.0

    if total_retrieved > 0:
        overall_precision = (
            total_relevant / total_retrieved
        )

    print(
        f"Questions Evaluated: "
        f"{total_questions}"
    )

    print(
        f"Total Sources Retrieved: "
        f"{total_retrieved}"
    )

    print(
        f"Total Relevant Sources: "
        f"{total_relevant}"
    )

    print(
        f"Overall Retrieval Precision: "
        f"{overall_precision:.2f}"
    )

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()