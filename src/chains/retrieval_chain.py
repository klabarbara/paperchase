from langchain_core.documents import Document
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.document_loaders import ArxivLoader
from pathlib import Path
import re

from .keyword_chain import build_keyword_chain
from ..config import settings

CHROMA_DIR = Path(".chroma_full")

def _docs_from_api_wrapper(query: str, k: int) -> list[Document]:
    wrapper = ArxivAPIWrapper(load_max_docs=k)
    # wrapper.run returns one string, so split on "\n\n" between papers
    summaries = wrapper.run(query).split("\n\n")
    docs = []
    for block in summaries:
        # first line is title, second is URL
        parts = block.strip().splitlines()
        if len(parts) >= 3:
            docs.append(
                Document(
                    page_content="\n".join(parts[2:]), # summary paragraph
                    metadata={"title": parts[0], "url": parts[1]}
                )
            )
    return docs

def _docs_from_loader(query: str, k: int) -> list[Document]:
    return ArxivLoader(query=query, load_max_docs=k).load()

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
        api_version=settings.embed_api_version, 
        chunk_size=512,
    )

    # choose your fighter
    if CHROMA_DIR.exists():
        # loads prebuilt index for papersum dataset
        vectordb = Chroma(
            collection_name="cs_papersum",
            embedding_function=emb,
            persist_directory=str(CHROMA_DIR),
        )
    else:
        # prototyping fallback - build ad hoc temp index 
        vectordb = Chroma(
            collection_name="scratch",
            embedding_function=emb,
            persist_directory=None, # mem only
        )
    
    def retrieve(user_query: str) -> list[Document]:
        docs = fetch_docs(user_query)
        
        if not docs:
            raise ValueError("no documents returned from fetch_docs")
        
        vectordb = Chroma(
            collection_name="arxiv_tmp",
            embedding_function=AzureOpenAIEmbeddings(
                azure_endpoint=settings.azure_endpoint,
                api_key=settings.azure_key,
                azure_deployment=settings.embed_deployment,
                api_version=settings.embed_api_version,
            ),
            persist_directory=None # avoiding persisting in memory to start
        )

        vectordb.reset_collection()
        vectordb.add_documents(docs)
        return vectordb.similarity_search(user_query, k=5)
    
    # chain now user_query to retrieve to docs
    return RunnableParallel({"docs": retrieve})