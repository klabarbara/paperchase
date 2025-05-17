import json
from pathlib import Path
from typing import List, Dict

from .retriever import Retriever
from .reranker import Reranker
from .reader import Reader

"""
end to end pipeline for a single query. 

retriever -> reranker -> reader

returns structured JSON-compatible dict list
"""

retriever = Retriever(index_dir="index/bm25", top_k=100)
reranker = Reranker()
reader = Reader(cache_path="cache.jsonl")

def _load_doc_text(paper_id: str) -> str:
    # helper to loads doc full text from jsonl file on disk
    path = Path("data/processed") / f"{paper_id}.json"
    return Path(path).read_text()

def answer(query: str, k: int = 5) -> List[Dict]:
    """
    one shot query handler

    parameters
    query : user question
    k : number of final results to return post rerank

    returns
    list of dicts : each dict contains paper metadata,  BM25, and cross scores, and the cached "why this paper?" commentary 
    """
    
    #retrieves list of (paper_id, bm25_score)s
    bm25_hits = retriever(query)

    # loads top n texts for rerank
    docs_for_rerank = [
        (pid, _load_doc_text(pid), bm25) for pid, bm25 in bm25_hits
    ]
    
    reranked = reranker(query, docs_for_rerank)[:k]

    results = []
    for paper_id, bm25_score, cross_score in reranked:
        doc_json = json.loads(_load_doc_text(paper_id))
        llm_note = reader(query, doc_json) # reads/summarizes

        results.append(
            {
                "paper_id": paper_id,
                "title": doc_json["title"],
                "score_bm25": bm25_score,
                "score_cross": cross_score,
                "llm_note": llm_note,
            }
        )

    return results
