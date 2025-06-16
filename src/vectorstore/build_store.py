"""
NOTE: THIS SHOULD NOT BE RUN IN PROTOTYPING AND SHOULD ONLY BE RUN ONCE (IE: NOT PART OF A PIPELINE)

prebuilds a Chroma index for entire CS-PaperSum csv.

assumes columns 'paper_id', 'title', and 'abstract' for now (as i know for a fact there are missing abstracts)
"""

import csv, pathlib, tqdm
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

from ..config import settings

CSV_PATH = pathlib.Path("data/cs_papersum.csv")
CHROMA_DIR = pathlib.Path(".chroma_full")

def main():
    if settings.oss_mode == "remote":
        emb = HuggingFaceEndpointEmbeddings(
            endpoint_url=settings.embed_endpoint,
            huggingface_api_token=settings.huggingface_token,
            task="feature-extraction",
        )
    else:
        emb = HuggingFaceEmbeddings(
            model_name="hkunlp/instructor-base",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    
    vectordb = Chroma(
        collection_name="cs_papersum",
        embedding_function=emb,
        persist_directory=str(CHROMA_DIR),
    )

    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        docs = []
        for row in tqdm.tqdm(reader):
            docs.append(
                Document(
                    page_content=row["title"] + "\n" + row["abstract"],
                    metadata={
                        "arxiv_id": row["paper_id"],
                        "title": row["title"],
                    },
                )
            )
    ids = vectordb.add_documents(docs)
    print(f"indexed {len(ids)} papers to {CHROMA_DIR}")
    vectordb.persist()

if __name__ == "__main__":
    main()