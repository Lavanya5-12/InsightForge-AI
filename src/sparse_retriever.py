from typing import List, Dict
from rank_bm25 import BM25Okapi

ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class BM25Retriever:
    """
    Sparse keyword retrieval using BM25Okapi with stop word filtering.
    """

    def __init__(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("Chunks list cannot be empty for BM25Retriever.")

        self.chunks = chunks
        self.documents = [chunk.get("text", "") for chunk in chunks]
        self.tokenized_documents = [self.tokenize(text) for text in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_documents)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text by lowercasing, splitting, and removing stop words."""
        if not text:
            return []
        
        words = text.lower().split()
        filtered = [w.strip(".,!?:;\"'()") for w in words]
        return [w for w in filtered if w and w not in ENGLISH_STOP_WORDS]

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant chunks using BM25 ranking.
        """
        if not query or not query.strip():
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        
        # Check if max score is 0
        if max(scores) == 0:
            return []

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        top_k = min(n_results, len(self.chunks))
        results = []

        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:
                results.append({
                    "text": self.chunks[idx]["text"],
                    "metadata": self.chunks[idx]["metadata"],
                    "score": float(scores[idx])
                })

        return results