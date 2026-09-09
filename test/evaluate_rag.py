import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pdf_processor import extract_pdf_text
from src.chunker import chunk_text
from src.rag_pipeline import RAGPipeline
from src.evaluator import RAGEvaluator
from test.evaluation_dataset import EVALUATION_DATASET

FIXTURE_PATH = os.path.join("data", "fixtures", "insightforge_test.pdf")
EVAL_DIR = "evaluation"


def main():
    print("=" * 60)
    print("INSIGHTFORGE AI - REPRODUCIBLE RAG EVALUATION")
    print("=" * 60)

    # 1. Ensure test fixture exists
    fixture_path = Path(FIXTURE_PATH)
    if not fixture_path.exists():
        print(f"Creating test fixture at {FIXTURE_PATH}...")
        from scripts.generate_fixture import generate_fixture_pdf
        generate_fixture_pdf()

    # 2. Extract PDF text
    print(f"\n1. Loading fixture PDF ({FIXTURE_PATH})...")
    pages = extract_pdf_text(str(fixture_path))
    print(f"   Extracted Pages: {len(pages)}")

    # 3. Create Chunks
    print("\n2. Chunking document text...")
    chunks = chunk_text(pages)
    print(f"   Total Chunks: {len(chunks)}")

    # 4. Initialize RAG Pipeline
    print("\n3. Initializing RAG Pipeline & Ingesting Embeddings into FAISS...")
    rag = RAGPipeline(chunks, reset_vector_store=True)
    print(f"   FAISS Vectors Indexed: {rag.vector_store.count()}")

    # 5. Run Evaluation Loop
    evaluator = RAGEvaluator()
    results_detail = []
    total_questions = len(EVALUATION_DATASET)
    total_retrieved = 0
    total_relevant = 0
    hits = 0
    top_scores = []

    print("\n4. Running evaluation benchmark...")
    print("-" * 60)

    for idx, item in enumerate(EVALUATION_DATASET, start=1):
        question = item["question"]
        expected_pages = item["expected_pages"]

        # Retrieve chunks
        retrieved_results = rag.retriever.retrieve(question, n_results=5)
        eval_metrics = evaluator.evaluate(retrieved_results, expected_pages=expected_pages)

        source_count = eval_metrics["source_count"]
        relevant_count = eval_metrics["relevant_sources"]
        top_score = eval_metrics["top_score"]
        precision = eval_metrics["retrieval_precision"] or 0.0

        hit = relevant_count > 0
        if hit:
            hits += 1

        total_retrieved += source_count
        total_relevant += relevant_count
        top_scores.append(top_score)

        retrieved_pages = [r.get("metadata", {}).get("page_number") for r in retrieved_results]

        results_detail.append({
            "id": idx,
            "question": question,
            "expected_pages": expected_pages,
            "retrieved_pages": retrieved_pages,
            "source_count": source_count,
            "relevant_sources": relevant_count,
            "top_score": top_score,
            "precision": precision,
            "hit": hit
        })

        print(f"Q{idx}: {question}")
        print(f"   Expected: {expected_pages} | Retrieved: {retrieved_pages}")
        print(f"   Top Score: {top_score:.4f} | Precision: {precision:.2f} | Hit: {hit}")
        print("-" * 60)

    overall_precision = (total_relevant / total_retrieved) if total_retrieved > 0 else 0.0
    hit_rate = (hits / total_questions) if total_questions > 0 else 0.0
    avg_top_score = (sum(top_scores) / len(top_scores)) if top_scores else 0.0

    print("\n" + "=" * 60)
    print("OVERALL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Questions Evaluated: {total_questions}")
    print(f"Hit Rate (Recall@K >= 1): {hit_rate * 100:.1f}% ({hits}/{total_questions})")
    print(f"Overall Retrieval Precision: {overall_precision:.4f}")
    print(f"Average Top Similarity Score: {avg_top_score:.4f}")
    print("=" * 60)

    # 6. Save Evaluation Output
    os.makedirs(EVAL_DIR, exist_ok=True)
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "fixture_path": str(fixture_path),
        "total_questions": total_questions,
        "total_chunks": len(chunks),
        "faiss_vectors": rag.vector_store.count(),
        "hit_rate": hit_rate,
        "overall_precision": overall_precision,
        "avg_top_score": avg_top_score,
        "details": results_detail
    }

    json_path = os.path.join(EVAL_DIR, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nEvaluation summary saved to: {json_path}")

    md_path = os.path.join(EVAL_DIR, "results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# InsightForge AI — RAG Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Fixture Document:** `{fixture_path}` ({len(pages)} pages, {len(chunks)} chunks)\n\n")
        f.write(f"## Summary Metrics\n\n")
        f.write(f"| Metric | Score |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Questions Evaluated | {total_questions} |\n")
        f.write(f"| Hit Rate (Recall@5 >= 1) | **{hit_rate * 100:.1f}%** ({hits}/{total_questions}) |\n")
        f.write(f"| Overall Precision@5 | **{overall_precision:.4f}** |\n")
        f.write(f"| Average Top Hybrid Score | **{avg_top_score:.4f}** |\n\n")
        f.write(f"## Question Breakdown\n\n")
        f.write(f"| # | Question | Expected Pages | Retrieved Pages | Top Score | Precision | Hit |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        for d in results_detail:
            f.write(
                f"| {d['id']} | {d['question']} | {d['expected_pages']} | {d['retrieved_pages']} | "
                f"{d['top_score']:.4f} | {d['precision']:.2f} | {'✅' if d['hit'] else '❌'} |\n"
            )
    print(f"Evaluation report saved to: {md_path}")


if __name__ == "__main__":
    main()