import rich, statistics

from .datasets import EVAL_SET
from .rag_metrics import retrieval_scores, summary_scores
from ..chains.retrieval_chain import build_retrieval_chain
from ..chains.summary_chain import build_summary_chain

def run():
    retr_chain = build_retrieval_chain()
    read_chain = build_summary_chain()

    retrieval_metrics, faith_metrics = [], []

    for example in EVAL_SET:
        q = example["query"]
        gold = example["gold_ids"]

        # retrieval
        docs = retr_chain.run(q)
        pred_ids = [d.metadata.get("arxiv_id") for d in docs]
        retrieval_metrics.append(retrieval_scores(pred_ids, gold))

        # reader
        summary = read_chain.invoke({"input_documents": docs})["output_text"]
        faith = summary_scores(q, docs, summary)
        faith_metrics.append(faith["faithfulness"])

    # aggregate
    agg = {
        k: statistics.mean(m[k] for m in retrieval_metrics)
        for k in retrieval_metrics[0]
    }
    agg["faithfulness"] = statistics.mean(faith_metrics)

    rich.print("[bold yellow]Evaluation results[/bold yellow]")
    for k, v in agg.items():
        rich.print(f"{k:<15} {v:.3f}")

if __name__ == "__main__":
    run()