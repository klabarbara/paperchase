from sentence_transformers import CrossEncoder
from typing import List, Tuple

"""
cross encoder reranker

takesa top n BM25 hits, applies a transformer relevance model to produce refined relevance scores, and returns a list sorted by those scores (still keeping the original BM25 score for inspection)

"""
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        using small, 65M parameter cross encoder fine tuned on MS Marco for mvp
        """
        self.model = CrossEncoder(model_name)
        
    def __call__(
        self, 
        query: str, 
        docs: List[Tuple[str, str, float]],
    ) -> List[Tuple[str, float, float]]:
        """
        parameters
        query - user's search query
        docs - list of tuples. each tuple: (doc_id, text_to_score, bm25_score)

        returns
        list of tuples (doc_id, bm25 score, cross_score) sorted descending by cross-encoder score
        """
        pairs = [(query, d[1]) for d in docs]

        # vectorised relevance prediction (returns 1d numpy array)
        cross_scores = self.model.predict(pairs, convert_to_numpy=True)


        reranked = sorted(zip(docs, cross_scores), key=lambda x: x[1], reverse=True)
        return [(doc_id, bm25, float(score)) for ((doc_id, _text, bm25), score) in reranked]