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

# api results returned in one string, with each doc separated by \n\n
# and metadata labels embedded within the string
def _docs_from_api_wrapper(query: str, k: int) -> list[Document]:
    wrapper = ArxivAPIWrapper(load_max_docs=k)
    raw_output = wrapper.run(query)

    blocks = raw_output.strip().split("\n\n") # each paper is a block
    docs = []

    for block in blocks:
        published = title = summary = url = None # lol
        lines = block.strip().splitlines()

        for line in lines:
            if line.startswith("Published:"):
                published = line.replace("Published:", "").strip()
            elif line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Authors:"):
                continue
            elif line.startswith("Link:"):
                url = line.replace("Link:", "").strip()
            else:
                summary = (summary or "") + line.replace("Summary:", "").strip() + " "
        
        if title and summary:
            docs.append(Document(
                page_content=summary.strip(),
                metadata={
                    "title": title,
                    "published": published or "unknown",
                    "url": url or "",
                }
            ))

    return docs

# full doc loader has metadata to directly access
def _docs_from_loader(query: str, k: int) -> list[Document]:
    raw_docs = ArxivLoader(query=query, load_max_docs=k).load()

    # arxivloader already includes metadata keys 'Title' and 'Published'
    docs =[]
    for d in raw_docs:
        docs.append(
            Document(
                page_content=d.page_content,
                metadata={
                    "title": d.metadata.get("Title"),
                    "url": d.metadata.get("Entry ID"), # entry_id?
                    "published": d.metadata.get("Published"),
                }
            )
        )
    return docs

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
    
    def retrieve(user_query: str) -> list[Document]:
        docs = fetch_docs(user_query)
        
        if not docs:
            raise ValueError("no documents returned from fetch_docs")
        
        vectordb = Chroma(
            collection_name="arxiv_tmp",
            embedding_function=emb,
            persist_directory=".chroma_tmp" # avoiding persisting in memory to start
        )

        vectordb.reset_collection()
        vectordb.add_documents(docs)
        return vectordb.similarity_search(user_query, k=5)
    
    # chain now user_query to retrieve to docs
    return RunnableParallel({"docs": retrieve})