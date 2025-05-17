from rich import print  
from app.pipeline import answer

while True:
    q = input("\nQuery (ENTER to quit): ").strip()
    if not q:
        break
    for i, hit in enumerate(answer(q, k=3), 1):
        print(f"[bold cyan]{i}. {hit['title']}[/bold cyan]")
        print(f"   BM25 {hit['score_bm25']:.2f} | X-enc {hit['score_cross']:.2f}")
        print(f"   {hit['llm_note'][:120]}…\n")
