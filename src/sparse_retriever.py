from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Sparse retrieval using BM25.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        self.documents = [
            chunk["text"]
            for chunk in chunks
        ]

        self.tokenized_documents = [
            self.tokenize(text)
            for text in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Basic tokenization for BM25.
        """

        return text.lower().split()

    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks using BM25.
        """

        if not query.strip():
            return []

        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:n_results]:
            results.append({
                "text": self.chunks[index]["text"],
                "metadata": self.chunks[index]["metadata"],
                "score": float(scores[index])
            })

        return results