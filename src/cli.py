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
    single_summaries = []

    for d in docs:
        res = summ_chain.invoke({"input_documents": [d]})
        single_summaries.append(res["output_text"])

    for d, summ in zip(docs, single_summaries):
        rich.print(f"[bold]{d.metadata.get('title', '(no title)')}[/bold]")
        rich.print(f"Published: {d.metadata.get('published', 'unknown')}\n")
        rich.print("[green]Summary:[/green]")
        rich.print(summ)
        rich.print("-" * 30)

if __name__ == "__main__":
    app()