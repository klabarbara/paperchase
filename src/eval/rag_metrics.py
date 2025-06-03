from typing import List, Dict

from langchain.evaluation import load_evaluator
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document

from ..config import settings

# DIY metrics since I can't find where tf they are in langchain/smith (TODO)
def _precision_at_k(pred: List[str], gold: List[str], k: int) -> float:
    return sum(1 for p in pred[:k] if p in gold) / k

def _recall_at_k(pred: List[str], gold: List[str], k: int) -> float:
    if not gold:               
        return 1.0
    return sum(1 for p in pred[:k] if p in gold) / len(gold)

def _mrr_at_k(pred: List[str], gold: List[str], k: int) -> float:
    for rank, p in enumerate(pred[:k], 1):
        if p in gold:
            return 1.0 / rank
    return 0.0

def retrieval_scores(pred_ids: List[str], gold_ids: List[str]) -> Dict[str, float]:
    return {
        "precision@5": _precision_at_k(pred_ids, gold_ids, 5),
        "recall@5":    _recall_at_k(pred_ids, gold_ids, 5),
        "mrr@10":      _mrr_at_k(pred_ids,   gold_ids, 10),
    }

# generative (reader) metrics
_FAITHFUL_LLM = AzureChatOpenAI(
    azure_endpoint=settings.azure_endpoint,
    api_key=settings.azure_key,
    api_version=settings.chat_api_version,
    deployment_name=settings.chat_deployment,
    model_name=settings.chat_deployment,
    temperature=0.0,
)



def summary_scores(query: str, docs: List[Document], summary: str) -> Dict[str, float]:
    """
    docs = the chunks fed to reader (ground truth for faithfulness)
    summary = reader output
    """
    faithfulness = load_evaluator("faithfulness", llm=_FAITHFUL_LLM) # move outside once summary included?

    res = faithfulness.evaluate(
        question=query,
        contexts=[d.page_content for d in docs],
        answer=summary,
    )

    # return format {"faithfulness": [0,1], "explaination": str}
    return res