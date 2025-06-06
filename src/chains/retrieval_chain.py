from langchain_core.documents import Document
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.document_loaders import ArxivLoader
from pathlib import Path
import re
import hashlib

from .keyword_chain import build_keyword_chain
from ..config import settings

CHROMA_DIR = Path(".chroma_full")
    
def make_doc_id(title: str, published: str, summary: str) -> str:
    key = f"{title.strip()}|{published.strip()}|{summary.strip()}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


"""
Option to utilize ArxivAPIWrapper to avoid ArxivLoader's behavior
of retrieving full papers. 

Results are returned in one string, with each doc separated by \n\n
and metadata labels embedded within the string. Since arxiv id is not
included in these results, unique id is formed by hashing the document
contents. 
"""
def _docs_from_api_wrapper(query: str, k: int) -> list[Document]:

    wrapper = ArxivAPIWrapper(load_max_docs=k)
    raw_output = wrapper.run(query)

    blocks = raw_output.strip().split("\n\n") # each paper is a block
    docs = []

    for block in blocks:
        published = title = summary = None
        lines = block.strip().splitlines()

        for line in lines:
            if line.startswith("Published:"):
                published = line.replace("Published:", "").strip()
            elif line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            else:
                summary = (summary or "") + line.replace("Summary:", "").strip() + " "
        
        # stable id generated
        arxiv_id = make_doc_id(title=title, published=published, summary=summary)

        if title and summary:
            docs.append(Document(
                page_content=summary.strip(),
                metadata={
                    "title": title,
                    "published": published or "unknown",
                    "arxiv_id": arxiv_id
                },
                id=arxiv_id
            ))

    return docs

# full doc loader has metadata to directly access
def _docs_from_loader(query: str, k: int) -> list[Document]:
    raw_docs = ArxivLoader(query=query, load_max_docs=k).load()

    # arxivloader already includes metadata keys 'Title' and 'Published'
    # TODO: what is the field for its Arxiv ID? 
    # goal is to set these Documents' unique IDs to their arxiv id,
    # which will diferentiate fully-loaded documents from the summaries
    # pulled using ArxivAPIWrapper
    docs =[]
    for d in raw_docs:
        docs.append(
            Document(
                page_content=d.page_content,
                metadata={
                    "title": d.metadata.get("Title"),
                    "url": d.metadata.get("Entry ID"), # entry_id?
                    "published": d.metadata.get("Published"),
                    "arxiv_id": d.metadata.get("arxiv_id") # this will break if the field is wrong
                },
                id=d.metadata.get("arxiv_id") # this will break if the field is wrong
            )
        )
    return docs

# this is a mess. TODO: refactor when unifying id methods/hooking in open source models
def build_retrieval_chain(use_full_docs: bool = False):

    # fetch_docs's keywords are in markdown, needs to be cleaned
    def clean_keywords(raw: str) -> str:
        lines = raw.splitlines()
        cleaned = [re.sub(r"\*\*(.*?)\*\*", r"\1", line).strip() for line in lines]
        cleaned = [re.sub(r"^\d+\.\s*", "", line) for line in cleaned]
        return ", ".join(filter(None, cleaned))

    def fetch_docs(user_query: str) -> list[Document]:
        kw_chain = build_keyword_chain()
        msg = kw_chain.invoke({"query": user_query})
        print("LLM response: ", msg)

        raw_keywords = msg.content.strip()
        print("extracted keywords: ", raw_keywords)

        keywords = clean_keywords(raw=raw_keywords)
        print("cleaned keywords: ", keywords)

        docs = _docs_from_loader(keywords, 20) if use_full_docs else _docs_from_api_wrapper(keywords, 20)
        print(f"retrieved {len(docs)} documents")
        
        return docs
    
    # build ad-hoc vector store for docs
    emb = AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_endpoint,
        api_key=settings.azure_key,
        azure_deployment=settings.embed_deployment,
        model=settings.embed_deployment,
        api_version=settings.embed_api_version, 
        chunk_size=512,
    )
    
    vectordb = Chroma(
        collection_name="arxiv_tmp",
        embedding_function=emb,
        persist_directory=".chroma_tmp" # local cacheing to avoid excessive calls
    )

    # checks doc ids against ids in vectordb before adding them 
    # (and using azure embedding tokens)
    def upsert_docs(docs):
        ids = [d.id for d in docs]

        existing = set(vectordb._collection.get(
            ids=ids, include=[]
        )["ids"])

        new_docs = [d for d in docs if d.id not in existing]

        if new_docs:
            vectordb.add_documents(new_docs)  

    def retrieve(user_query: str) -> list[Document]:
        docs = fetch_docs(user_query)
        
        if not docs:
            raise ValueError("no documents returned from fetch_docs")

        upsert_docs(docs)
        ids_this_run = [d.metadata["arxiv_id"] for d in docs]

        return vectordb.similarity_search(user_query, 
                                          k=5,
                                          filter={"arxiv_id": {"$in": ids_this_run}})
    
    # chain now user_query to retrieve to docs
    return RunnableParallel({"docs": retrieve})