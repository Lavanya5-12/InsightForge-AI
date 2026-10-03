from typing import List, Dict, Optional
from src.embeddings import generate_embedding
from src.vector_store import FAISSVectorStore, get_vector_store
from src.sparse_retriever import BM25Retriever
from src.config import DENSE_WEIGHT, SPARSE_WEIGHT, DEFAULT_TOP_K
from src.entity_utils import detect_entity_types, expand_query_for_bm25, has_entity_evidence


class HybridRetriever:
    """
    Combines dense FAISS vector retrieval and BM25 sparse keyword retrieval.
    """

    def __init__(
        self,
        chunks: List[Dict],
        vector_store: Optional[FAISSVectorStore] = None,
        dense_weight: float = DENSE_WEIGHT,
        sparse_weight: float = SPARSE_WEIGHT
    ):
        if not chunks:
            raise ValueError("Chunks list cannot be empty for HybridRetriever.")

        if abs((dense_weight + sparse_weight) - 1.0) > 1e-6:
            raise ValueError("dense_weight + sparse_weight must sum to 1.0")

        self.chunks = chunks
        self.vector_store = vector_store or get_vector_store()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.bm25 = BM25Retriever(chunks)

    def dense_search(self, query: str, n_results: int = DEFAULT_TOP_K) -> List[Dict]:
        """Retrieve top chunks using dense vector embeddings in FAISS."""
        if not query.strip():
            return []

        query_embedding = generate_embedding(query)
        raw_results = self.vector_store.search(query_embedding, n_results=n_results)

        dense_results = []
        for res in raw_results:
            dense_results.append({
                "text": res["text"],
                "metadata": res["metadata"],
                "score": res["score"]
            })

        return dense_results

    def sparse_search(self, query: str, n_results: int = DEFAULT_TOP_K) -> List[Dict]:
        """Retrieve top chunks using BM25 sparse keyword matching."""
        if not query.strip():
            return []

        return self.bm25.search(query, n_results=n_results)

    @staticmethod
    def normalize_bm25_scores(results: List[Dict]) -> List[Dict]:
        """
        Normalize BM25 keyword scores to [0.0, 1.0] by dividing by max score.
        If all BM25 scores are 0, returns 0.0 for all items.
        """
        if not results:
            return results

        scores = [res["score"] for res in results]
        max_score = max(scores)

        for res in results:
            if max_score > 0:
                res["normalized_score"] = res["score"] / max_score
            else:
                res["normalized_score"] = 0.0

        return results

    def retrieve(self, query: str, n_results: int = DEFAULT_TOP_K) -> List[Dict]:
        """
        Perform hybrid retrieval combining dense FAISS search and sparse BM25 search.
        Preserves absolute FAISS similarity semantics to enforce retrieval safety thresholds.
        Supports query expansion and candidate re-ranking for factual/entity questions.
        """
        if not query or not query.strip():
            return []

        # 1. Dense retrieval (FAISS score is already absolute similarity 1/(1+dist) in [0.0, 1.0])
        dense_results = self.dense_search(query, n_results=n_results)

        # 2. Sparse retrieval (BM25 normalized by max score)
        sparse_results = self.sparse_search(query, n_results=n_results)

        # 3. Entity-aware query expansion for BM25
        entity_types = detect_entity_types(query)
        if entity_types:
            expanded_query = expand_query_for_bm25(query, entity_types)
            expanded_sparse = self.sparse_search(expanded_query, n_results=n_results * 2)

            # Merge expanded sparse results with standard sparse results
            sparse_map = {
                (
                    r["metadata"].get("document_name", ""),
                    r["metadata"].get("page_number", 0),
                    r["metadata"].get("chunk_number", 0)
                ): r
                for r in sparse_results
            }
            for exp_res in expanded_sparse:
                key = (
                    exp_res["metadata"].get("document_name", ""),
                    exp_res["metadata"].get("page_number", 0),
                    exp_res["metadata"].get("chunk_number", 0)
                )
                if key not in sparse_map or exp_res["score"] > sparse_map[key]["score"]:
                    sparse_map[key] = exp_res
            sparse_results = list(sparse_map.values())

        sparse_results = self.normalize_bm25_scores(sparse_results)

        # 4. Combine results by unique key (document_name, page_number, chunk_number)
        combined = {}

        for res in dense_results:
            key = (
                res["metadata"].get("document_name", ""),
                res["metadata"].get("page_number", 0),
                res["metadata"].get("chunk_number", 0)
            )
            combined[key] = {
                "text": res["text"],
                "metadata": res["metadata"],
                "dense_score": res["score"],  # Absolute similarity
                "sparse_score": 0.0,
                "retrieval_reason": "hybrid"
            }

        for res in sparse_results:
            key = (
                res["metadata"].get("document_name", ""),
                res["metadata"].get("page_number", 0),
                res["metadata"].get("chunk_number", 0)
            )
            if key not in combined:
                combined[key] = {
                    "text": res["text"],
                    "metadata": res["metadata"],
                    "dense_score": 0.0,
                    "sparse_score": res["normalized_score"],
                    "retrieval_reason": "hybrid"
                }
            else:
                combined[key]["sparse_score"] = res["normalized_score"]
                combined[key]["metadata"] = {
                    **combined[key]["metadata"],
                    **res["metadata"]
                }

        # 5. Compute weighted hybrid score
        final_results = []
        for item in combined.values():
            hybrid_score = (
                self.dense_weight * item["dense_score"] +
                self.sparse_weight * item["sparse_score"]
            )
            item["hybrid_score"] = float(hybrid_score)

            if entity_types and has_entity_evidence(item["text"], entity_types):
                item["entity_evidence"] = True
                item["retrieval_reason"] = "entity/keyword match"
            else:
                item["entity_evidence"] = False

            final_results.append(item)

        # 6. Sort descending by hybrid_score
        final_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return final_results[:n_results]