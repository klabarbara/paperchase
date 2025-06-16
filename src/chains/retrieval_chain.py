import re
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import ArxivLoader

from .keyword_chain import build_keyword_chain
from ..config import settings

CHROMA_DIR = Path(".chroma_full")
    
def make_doc_id(title: str, published: str) -> str:
    key = f"{title.strip()}|{published.strip()}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


# full doc loader has metadata to directly access
# api_wrapper variant removed now that spend is no longer
# tied to token count
def _docs_from_loader(query: str, k: int) -> list[Document]:
    raw_docs = ArxivLoader(query=query, load_max_docs=k).load()

    docs =[]
    for d in raw_docs:
        arxiv_id = d.metadata.get("arxiv_id") or make_doc_id(d.metadata.get("Title", ""), d.metadata.get("Published", ""))

        docs.append(
            Document(
                page_content=d.page_content,
                metadata={
                    "title": d.metadata.get("Title"),
                    "url": d.metadata.get("Entry ID"), # entry_id?
                    "published": d.metadata.get("Published"),
                    "arxiv_id": arxiv_id
                },
                id=arxiv_id 
            )
        )
    return docs

# checks doc ids against ids in vectordb before adding them 
# (and using azure embedding tokens)
def upsert_docs(docs, vectordb: Chroma):
    ids = [d.id for d in docs]

    existing = set(vectordb._collection.get(
        ids=ids, include=[]
    )["ids"])

    new_docs = [d for d in docs if d.id not in existing]

    if new_docs:
        vectordb.add_documents(new_docs)  

def retrieve(user_query: str, vectordb: Chroma) -> list[Document]:
    docs = fetch_docs(user_query)
    
    if not docs:
        raise ValueError("no documents returned from fetch_docs")

    upsert_docs(docs, vectordb)
    ids_this_run = [d.metadata["arxiv_id"] for d in docs]

    return vectordb.similarity_search(user_query, 
                                      k=5, 
                                      filter={"arxiv_id": {"$in":ids_this_run}})

def fetch_docs(user_query: str, use_full_docs: bool = False) -> list[Document]:
    msg = build_keyword_chain().invoke({"query": user_query})
    keywords = clean_keywords(raw=msg)
    return _docs_from_loader(query=keywords, k=20)

# fetch_docs's keywords are in markdown, needs to be cleaned
def clean_keywords(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = [re.sub(r"\*\*(.*?)\*\*", r"\1", line).strip() for line in lines]
    cleaned = [re.sub(r"^\d+\.\s*", "", line) for line in cleaned]
    return ", ".join(filter(None, cleaned))


def build_retrieval_chain(use_full_docs: bool = False):
    """
    open-source retrieval chain
    embeddings: instructor-base (local) or HF endpoint (remote)
    keywords: via build_keyword_chain()
    """
    
    if settings.oss_mode == "remote":
        emb = HuggingFaceEmbeddings(
            endpoint_url=settings.embed_endpoint,
            huggingface_api_token=settings.embed_token,
            task="feature-extraction",
        )
    else:
        emb = HuggingFaceEmbeddings(
            model_name="hkunlp/instructor-base",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # vectordb used within upsert_docs() and retrieve()
    vectordb = Chroma(
        collection_name="arxiv_tmp",
        embedding_function=emb,
        persist_directory=str(CHROMA_DIR) # local cacheing to avoid excessive calls
    )
    # chain now user_query to retrieve to docs
    return RunnableParallel({"docs": lambda parms: retrieve(parms["docs"], vectordb)})