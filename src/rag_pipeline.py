from typing import List, Dict, Optional
from src.embeddings import generate_embeddings
from src.vector_store import FAISSVectorStore, get_vector_store
from src.hybrid_retriever import HybridRetriever
from src.llm import generate_answer
from src.evaluator import RAGEvaluator
from src.config import RETRIEVAL_THRESHOLD, DEFAULT_TOP_K
from src.entity_utils import detect_entity_types, has_entity_evidence


class RAGPipeline:
    """
    Complete end-to-end Hybrid RAG Pipeline.
    Manages document embedding ingestion, FAISS indexing, hybrid retrieval,
    relevance thresholding, LLM answer generation, dynamic citations, and evaluation.
    """

    def __init__(
        self,
        chunks: List[Dict],
        vector_store: Optional[FAISSVectorStore] = None,
        reset_vector_store: bool = True
    ):
        if not chunks:
            raise ValueError("RAGPipeline cannot be initialized with empty chunks.")

        self.chunks = chunks

        # ----------------------------------------------------
        # 1. Initialize Vector Store Lifecycle
        # ----------------------------------------------------
        if vector_store is not None:
            self.vector_store = vector_store
            if reset_vector_store:
                self.vector_store.clear()
        else:
            self.vector_store = get_vector_store(reset=reset_vector_store)

        # ----------------------------------------------------
        # 2. DOCUMENT INGESTION WORKFLOW
        # Connect: Chunks -> generate_embeddings -> add_documents -> FAISS
        # ----------------------------------------------------
        chunk_texts = [chunk.get("text", "") for chunk in self.chunks]
        embeddings = generate_embeddings(chunk_texts)

        if len(embeddings) != len(self.chunks):
            raise RuntimeError(
                f"Embedding count mismatch: generated {len(embeddings)} vectors for {len(self.chunks)} chunks."
            )

        self.vector_store.add_documents(self.chunks, embeddings)

        # ----------------------------------------------------
        # 3. Initialize Hybrid Retriever & Evaluator
        # ----------------------------------------------------
        self.retriever = HybridRetriever(
            chunks=self.chunks,
            vector_store=self.vector_store
        )
        self.evaluator = RAGEvaluator()

    @staticmethod
    def build_context(results: List[Dict]) -> str:
        """Format retrieved chunks into clean LLM context."""
        context_parts = []
        for i, res in enumerate(results, start=1):
            meta = res.get("metadata", {})
            doc_name = meta.get("document_name", "Unknown Document")
            page = meta.get("page_number", "Unknown")
            chunk_num = meta.get("chunk_number", "Unknown")
            text = res.get("text", "")

            context_parts.append(
                f"SOURCE {i}\n"
                f"DOCUMENT: {doc_name}\n"
                f"PAGE: {page}\n"
                f"CHUNK: {chunk_num}\n\n"
                f"{text}\n"
            )
        return "\n".join(context_parts)

    @staticmethod
    def build_citations(results: List[Dict]) -> str:
        """
        Build clear, deduplicated page citations from retrieved sources.
        """
        if not results:
            return ""

        seen = set()
        citations = []

        for res in results:
            meta = res.get("metadata", {})
            doc_name = meta.get("document_name", "Document")
            page = meta.get("page_number")

            if page is not None:
                key = (doc_name, page)
                if key not in seen:
                    seen.add(key)
                    citations.append(f"- **{doc_name}** — Page {page}")

        if not citations:
            return ""

        return "\n\n**Sources:**\n" + "\n".join(citations)

    def ask(
        self,
        question: str,
        n_results: int = DEFAULT_TOP_K,
        threshold: float = RETRIEVAL_THRESHOLD
    ) -> Dict:
        """
        Execute RAG question-answering workflow:
        1. Retrieve hybrid chunks
        2. Evaluate quality & threshold
        3. Reject out-of-scope / low-relevance queries without document evidence
        4. Generate grounded LLM response
        5. Attach dynamic citations
        """
        question_clean = question.strip()
        if not question_clean:
            return {
                "question": question,
                "answer": "Please provide a non-empty question.",
                "sources": [],
                "evaluation": self.evaluator.evaluate([])
            }

        # 1. Hybrid Retrieval
        results = self.retriever.retrieve(question_clean, n_results=n_results)

        # 2. Evaluate retrieval quality
        evaluation = self.evaluator.evaluate(results)
        top_score = evaluation.get("top_score", 0.0)

        # 3. Retrieval Safety / Threshold Check with Entity Fallback
        entity_types = detect_entity_types(question_clean)
        has_valid_entity_fallback = False

        if results and top_score < threshold and entity_types:
            for res in results:
                if res.get("entity_evidence") or has_entity_evidence(res.get("text", ""), entity_types):
                    has_valid_entity_fallback = True
                    res["entity_promoted"] = True
                    break

        if not results or (top_score < threshold and not has_valid_entity_fallback):
            fallback_answer = "I couldn't find sufficient information about this question in the uploaded documents."
            return {
                "question": question_clean,
                "answer": fallback_answer,
                "sources": results,
                "evaluation": evaluation,
                "grounded": False
            }

        # 4. Build context
        context = self.build_context(results)

        # 5. Generate LLM answer
        raw_answer = generate_answer(question_clean, context)

        # 6. Check if LLM produced standard fallback
        if "could not find" in raw_answer.lower() or "insufficient information" in raw_answer.lower():
            final_answer = raw_answer
        else:
            # Attach dynamic citations to grounded answer
            citations_str = self.build_citations(results)
            final_answer = f"{raw_answer}{citations_str}"

        return {
            "question": question_clean,
            "answer": final_answer,
            "sources": results,
            "evaluation": evaluation,
            "grounded": True
        }