from pyserini.search.lucene import LuceneSearcher

"""
lightweight BM25 retriever wrapper. (using 3rd party pyserini's LuceneSearcher)

single call returns a list of
(doc_id, bm25_score) tuples for the given query.
"""
class Retriever:
    def __init__(self, index_dir, top_k=100):
        self.searcher = LuceneSearcher(str(index_dir))
        self.top_k = top_k
        
    def __call__(self, query):
        """
        executes a BM25 search.

        returns: 
        List[Tuple[str, float]]
            each tuple is (doc_id, bm25_score).  scores are not normalised,
            but ordering is monotonic. higher is better!
        """
        hits = self.searcher.search(query, self.top_k)
        return [(h.docid, h.score) for h in hits]