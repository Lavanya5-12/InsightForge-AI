from src.hybrid_retriever import HybridRetriever
from src.llm import generate_answer
from src.evaluator import RAGEvaluator


class RAGPipeline:

    def __init__(self, chunks: list[dict]):

        self.retriever = HybridRetriever(
            chunks
        )

        self.evaluator = RAGEvaluator()


    def build_context(
        self,
        results: list[dict]
    ) -> str:
        """
        Convert retrieved chunks into LLM context.
        """

        context_parts = []

        for i, result in enumerate(
            results,
            start=1
        ):

            metadata = result.get(
                "metadata",
                {}
            )

            document_name = metadata.get(
                "document_name",
                "Unknown document"
            )

            page = metadata.get(
                "page_number",
                "Unknown"
            )

            chunk_number = metadata.get(
                "chunk_number",
                "Unknown"
            )

            text = result.get(
                "text",
                ""
            )

            context_parts.append(
                f"""
SOURCE {i}
DOCUMENT: {document_name}
PAGE: {page}
CHUNK: {chunk_number}

{text}
"""
            )

        return "\n".join(
            context_parts
        )


    def ask(
        self,
        question: str,
        n_results: int = 5
    ) -> dict:

        # ==================================================
        # 1. Retrieve relevant chunks
        # ==================================================

        results = self.retriever.retrieve(
            question,
            n_results=n_results
        )


        # ==================================================
        # 2. Evaluate retrieval
        # ==================================================

        evaluation = self.evaluator.evaluate(
            results
        )


        # ==================================================
        # 3. Build context
        # ==================================================

        context = self.build_context(
            results
        )


        # ==================================================
        # 4. Generate answer
        # ==================================================

        answer = generate_answer(
            question,
            context
        )


        # ==================================================
        # 5. Return complete RAG response
        # ==================================================

        return {
            "question": question,
            "answer": answer,
            "sources": results,
            "evaluation": evaluation
        }