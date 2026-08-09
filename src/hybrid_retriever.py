from src.embeddings import generate_embedding
from src.vector_store import search_documents
from src.sparse_retriever import BM25Retriever


class HybridRetriever:
    """
    Combines dense FAISS retrieval and BM25 sparse retrieval.
    """

    def __init__(
        self,
        chunks: list[dict],
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ):

        if not chunks:
            raise ValueError(
                "Chunks cannot be empty."
            )

        if abs(
            (dense_weight + sparse_weight) - 1.0
        ) > 1e-6:

            raise ValueError(
                "dense_weight + sparse_weight must equal 1.0"
            )

        self.chunks = chunks

        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.bm25 = BM25Retriever(
            chunks
        )


    # ========================================================
    # DENSE SEARCH
    # ========================================================

    def dense_search(
        self,
        query: str,
        n_results: int = 5
    ) -> list[dict]:
        """
        Retrieve documents using dense vector similarity.
        """

        query_embedding = generate_embedding(
            query
        )

        results = search_documents(
            query_embedding,
            n_results=n_results
        )

        dense_results = []

        for result in results:

            dense_results.append(
                {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "score": result["score"]
                }
            )

        return dense_results


    # ========================================================
    # SCORE NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_scores(
        results: list[dict]
    ) -> list[dict]:
        """
        Normalize retrieval scores to the range 0-1.
        """

        if not results:
            return results

        scores = [
            result["score"]
            for result in results
        ]

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:

            for result in results:

                result["normalized_score"] = 1.0

            return results

        for result in results:

            result["normalized_score"] = (
                result["score"] - min_score
            ) / (
                max_score - min_score
            )

        return results


    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query: str,
        n_results: int = 5
    ) -> list[dict]:
        """
        Perform hybrid retrieval using:

        Dense retrieval + BM25 retrieval
        """

        # ----------------------------------------------------
        # 1. Dense Retrieval
        # ----------------------------------------------------

        dense_results = self.dense_search(
            query,
            n_results=n_results
        )


        # ----------------------------------------------------
        # 2. Sparse Retrieval
        # ----------------------------------------------------

        sparse_results = self.bm25.search(
            query,
            n_results=n_results
        )


        # ----------------------------------------------------
        # 3. Normalize Scores
        # ----------------------------------------------------

        dense_results = self.normalize_scores(
            dense_results
        )

        sparse_results = self.normalize_scores(
            sparse_results
        )


        # ----------------------------------------------------
        # 4. Combine Results
        # ----------------------------------------------------

        combined = {}


        # ----------------------------------------------------
        # Add Dense Results
        # ----------------------------------------------------

        for result in dense_results:

            key = (
                result["metadata"]["page_number"],
                result["metadata"]["chunk_number"]
            )

            combined[key] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "dense_score": result["normalized_score"],
                "sparse_score": 0.0
            }


        # ----------------------------------------------------
        # Add Sparse Results
        # ----------------------------------------------------

        for result in sparse_results:

            key = (
                result["metadata"]["page_number"],
                result["metadata"]["chunk_number"]
            )


            # ------------------------------------------------
            # New result found only by BM25
            # ------------------------------------------------

            if key not in combined:

                combined[key] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "dense_score": 0.0,
                    "sparse_score": result["normalized_score"]
                }


            # ------------------------------------------------
            # Result exists in both Dense + BM25
            # ------------------------------------------------

            else:

                combined[key]["sparse_score"] = (
                    result["normalized_score"]
                )

                # --------------------------------------------
                # IMPORTANT FIX
                #
                # Use the metadata from the current chunks.
                # This preserves document_name.
                # --------------------------------------------

                combined[key]["metadata"] = {
                    **combined[key]["metadata"],
                    **result["metadata"]
                }


                # --------------------------------------------
                # Also use sparse text if available
                # --------------------------------------------

                if result.get("text"):

                    combined[key]["text"] = result["text"]


        # ----------------------------------------------------
        # 5. Calculate Hybrid Score
        # ----------------------------------------------------

        final_results = []


        for result in combined.values():

            hybrid_score = (
                self.dense_weight
                * result["dense_score"]
                +
                self.sparse_weight
                * result["sparse_score"]
            )

            result["hybrid_score"] = hybrid_score

            final_results.append(
                result
            )


        # ----------------------------------------------------
        # 6. Sort by Hybrid Score
        # ----------------------------------------------------

        final_results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )


        # ----------------------------------------------------
        # 7. Return Top Results
        # ----------------------------------------------------

        return final_results[:n_results]