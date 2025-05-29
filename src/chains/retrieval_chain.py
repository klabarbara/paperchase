from langchain_core.documents import Document
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.document_loaders import ArxivLoader
from pathlib import Path

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
    kw_chain = build_keyword_chain()

    def fetch_docs(user_query: str) -> list[Document]:
        keywords = kw_chain.invoke({"query": user_query})["text"]
        return (
            _docs_from_loader(keywords, 20) if use_full_docs else _docs_from_api_wrapper(keywords, 10)
        )
    
    # build ad-hoc vector store for docs
    emb = AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_endpoint,
        api_key=settings.azure_key,
        azure_deployment=settings.embed_deployment,
    )

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
        vectordb.delete_collection() # reset db b/t users (for now?)
        vectordb.add_documents(docs)
        return vectordb.similarity_search(user_query, k=5)
    
    # chain now user_query to retrieve to docs
    return RunnableParallel({"docs": retrieve})