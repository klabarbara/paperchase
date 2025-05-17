import json, tempfile, shutil
from pathlib import Path

import pytest
from app.retriever import Retriever
from app.reranker import Reranker
from app.reader import Reader
from app.pipeline import answer

"""
small integration test to run the full pipeline end-to-end on
a small in-memory corpus. Reader is monkeypatched to avoid LLM calls.
"""

# ------------------------------------------------------------------ #
# fixtures
# ------------------------------------------------------------------ #
@pytest.fixture(scope="session")
def mini_corpus(tmp_path_factory):
    """
    build a toy corpus + BM25 index in a temp dir and return its root path
    """
    root = tmp_path_factory.mktemp("mini_rag")
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)

    docs = [
        {
            "id": "d1",
            "title": "Retrieval Augmented Generation",
            "contents": "We propose combining dense and sparse retrieval, wee woo wee woo.",
            "abstract": "Since the dawn of humanity, we have struggled over retrieval...",
            "year": 2020,
        },
        {
            "id": "d2",
            "title": "BM25 Ranking Explained",
            "contents": "Sometimes, sparse is best.",
            "abstract": "A survey of BM25 variations.",
            "text": "Full text of BM25 survey.",
            "year": 2018,
        },
    ]
    for doc in docs:
        (processed / f"{doc['id']}.json").write_text(json.dumps(doc))

    # build a tiny BM25 index
    from pyserini.index.lucene import LuceneIndexer

    index_dir = root / "index" / "bm25"
    indexer = LuceneIndexer(str(index_dir))
    for p in processed.iterdir():
        indexer.add_doc_raw(p.read_text())
    indexer.close()

    return root


@pytest.fixture(autouse=True)
def patch_app_paths(monkeypatch, mini_corpus):
    """
    redirect the app.* modules to temporary corpus/index locations
    """
    # patch Retriever default index dir
    monkeypatch.setattr(
        "app.pipeline.retriever",
        Retriever(index_dir=str(mini_corpus / "index" / "bm25"), top_k=10),
        raising=False,
    )

    # patch helper that loads doc text
    def _load_doc_text(paper_id: str) -> str:
        return Path(mini_corpus / "data" / "processed" / f"{paper_id}.json").read_text()

    monkeypatch.setattr("app.pipeline._load_doc_text", _load_doc_text, raising=True)

    # patch Reader to a dummy that returns static text
    class DummyReader:
        def __call__(self, query, doc):
            return "dummy-llm-note"

    monkeypatch.setattr("app.pipeline.reader", DummyReader(), raising=False)


# ------------------------------------------------------------------ #
# tests
# ------------------------------------------------------------------ #
def test_pipeline_runs():
    res = answer("retrieval augmented generation")
    assert len(res) > 0, "No results returned"

    first = res[0]
    # schema sanity
    assert {"paper_id", "title", "score_bm25", "score_cross", "llm_note"} <= first.keys()
    # patched Reader should echo dummy text
    assert first["llm_note"] == "dummy-llm-note"
