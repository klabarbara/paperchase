from typing import List, Dict

from langchain.evaluation import load_evaluator
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document

from ..config import settings

# note: standard retrieval metrics. recall included in spite of dubious usefulness
# when dealing with small k size in a larger corpus. please refer to readme for more information
precision_at5 = load_evaluator("retrieval_precision", k=5)
recall_at5 = load_evaluator("retrieval_recall", k=5)
mrr_at10 = load_evaluator("retrieval_mrr", k=10)

def retireval_scores(pred_ids: List[str], gold_ids: List[str]) -> Dict[str, float]:
    return {
        "precision@5": precision_at5.evaluate(pred_ids, gold_ids),
        "recall@5": recall_at5.evaluate(pred_ids, gold_ids),
        "mrr@10": mrr_at10.evaluate(pred_ids, gold_ids),
    }

# generative (reader) metrics
_FAITHFUL_LLM = AzureChatOpenAI(
    azure_endpoint=settings.azure_endpoint,
    api_key=settings.azure_key,
    deployment_name=settings.chat_deployment,
    model_name=settings.chat_deployment,
    temperature=0.0,
)
faithfulness = load_evaluator("faithfulness", llm=_FAITHFUL_LLM)

def summary_scores(query: str, docs: List[Document], summary: str) -> Dict[str, float]:
    """
    docs = the chunks fed to reader (ground truth for faithfulness)
    summary = reader output
    """
    res = faithfulness.evaluate(
        question=query,
        contexts=[d.page_content for d in docs],
        answer=summary,
    )

    # return format {"faithfulness": [0,1], "explaination": str}
    return res