import typer, rich
from .chains.retrieval_chain import build_retrieval_chain
from .chains.summary_chain import build_summary_chain

"""
typer function to enable cli for accessing rag
"""


app = typer.Typer(help="Query CS arXiv with RAG")

@app.command()
def query(
    q: str, 
    top: int = 5,
    summary: bool = typer.Option(False, "--summary/--no-summary",
                                 help="Generate summary with LLM"),
    ):
    retr_chain = build_retrieval_chain()
    result = retr_chain.invoke({"docs": q})
    docs = result["docs"][:top]

    if summary:    
        summ_chain = build_summary_chain()
        single_summaries = []

        for d in docs:
            res = summ_chain.invoke({"input_documents": [d]})
            single_summaries.append(res["output_text"])

    for idx, d in enumerate(docs):
        rich.print(f"[bold]{d.metadata.get('title', '(no title)')}[/bold]")
        rich.print(f"Published: {d.metadata.get('published', 'unknown')}\n")
        if summary:
            rich.print("[green]Summary:[/green]")
            rich.print(single_summaries[idx])
        rich.print("-" * 30)

if __name__ == "__main__":
    app()