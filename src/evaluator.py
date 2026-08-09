from typing import List, Dict, Optional


class RAGEvaluator:
    """
    Evaluates the quality of retrieved RAG sources.

    Metrics:
    - Number of retrieved sources
    - Top hybrid score
    - Average hybrid score
    - Retrieval confidence
    - Relevant sources
    - Retrieval precision
    """

    def __init__(self):
        pass

    def evaluate(
        self,
        results: List[Dict],
        expected_pages: Optional[List[int]] = None
    ) -> Dict:
        """
        Evaluate retrieved RAG results.

        Args:
            results:
                Retrieved RAG source dictionaries.

            expected_pages:
                Optional list of page numbers that are expected
                to be relevant.

        Returns:
            Dictionary containing evaluation metrics.
        """

        # --------------------------------------------------
        # No results
        # --------------------------------------------------

        if not results:

            return {
                "source_count": 0,
                "top_score": 0.0,
                "average_score": 0.0,
                "confidence": "LOW",
                "relevant_sources": 0,
                "retrieval_precision": None
            }

        # --------------------------------------------------
        # Extract hybrid scores
        # --------------------------------------------------

        scores = []

        for result in results:

            score = result.get(
                "hybrid_score",
                0.0
            )

            scores.append(
                float(score)
            )

        # --------------------------------------------------
        # Calculate basic metrics
        # --------------------------------------------------

        source_count = len(results)

        top_score = max(scores)

        average_score = (
            sum(scores) / len(scores)
        )

        # --------------------------------------------------
        # Determine confidence
        # --------------------------------------------------

        if top_score >= 0.75:

            confidence = "HIGH"

        elif top_score >= 0.45:

            confidence = "MEDIUM"

        else:

            confidence = "LOW"

        # --------------------------------------------------
        # Retrieval relevance evaluation
        # --------------------------------------------------

        relevant_sources = 0
        retrieval_precision = None

        if expected_pages is not None:

            expected_pages_set = set(
                expected_pages
            )

            for result in results:

                metadata = result.get(
                    "metadata",
                    {}
                )

                page_number = metadata.get(
                    "page_number"
                )

                if page_number in expected_pages_set:

                    relevant_sources += 1

            retrieval_precision = (
                relevant_sources / source_count
            )

        # --------------------------------------------------
        # Return evaluation
        # --------------------------------------------------

        return {
            "source_count": source_count,
            "top_score": top_score,
            "average_score": average_score,
            "confidence": confidence,
            "relevant_sources": relevant_sources,
            "retrieval_precision": retrieval_precision
        }