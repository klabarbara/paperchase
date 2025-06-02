"""
NOTE: THIS SHOULD NOT BE RUN IN PROTOTYPING AND SHOULD ONLY BE RUN ONCE (IE: NOT PART OF A PIPELINE)

prebuilds a Chroma index for entire CS-PaperSum csv.

assumes columns 'paper_id', 'title', and 'abstract' for now (as i know for a fact there are missing abstracts)
"""

import csv, pathlib, tqdm
from langchain.docstore.document import Document
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from ..config import settings

CSV_PATH = pathlib.Path("data/cs_papersum.csv")
CHROMA_DIR = pathlib.Path(".chroma_full")

def main():
    emb = AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_endpoint,
        api_key=settings.azure_key,
        azure_deployment=settings.embed_deployment,
        model=settings.embed_deployment,
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