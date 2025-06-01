import typer, rich
from .chains.retrieval_chain import build_retrieval_chain
from .chains.summary_chain import build_summary_chain

"""
typer function to enable cli for accessing rag
"""


app = typer.Typer(help="Query CS arXiv with RAG")

@app.command()
def query(q: str, top: int = 5):
    retr_chain = build_retrieval_chain()
    result = retr_chain.invoke(q)
    docs = result["docs"][:top]

    summ_chain = build_summary_chain()
    summaries = summ_chain.invoke(docs)
    summaries = summaries['output_text']

    for doc, summ in zip(docs, summaries):
        rich.print(f"[bold]{doc.metadata.get('title','(no title)')}[/bold]")
        rich.print(doc.metadata.get("url",""))
        rich.print(doc.metadata.get("summary",""))
        rich.print(summ)
        rich.print("-"*60)

if __name__ == "__main__":
    app()