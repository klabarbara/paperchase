from typing import List, Dict
import torch
from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
from langchain.evaluation import load_evaluator
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

try:
    if settings.oss_mode == "remote":
        _FAITHFUL_LLM = HuggingFaceEndpoint(
            endpoint_url=settings.faithful_endpoint,
            huggingface_api_toke=settings.huggingface_token,
            model_kwargs={"task": "text-generation", "max_new_tokens": 512, "temperature": 0.0},
            timeout=60,
        )
    else:
        hf_pipe = pipeline(
            task="text-generation",
            model="meta-llama/Llama-2-7b-chat-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            max_new_tokens=512,
            temperature=0.0
        )
        _FAITHFUL_LLM = HuggingFacePipeline(pipeline=hf_pipe)
except Exception as e:
    _FAITHFUL_LLM = None
    print(f"Faithfulness LLM could not load: {e}")

def summary_scores(query: str, docs: List[Document], summary: str) -> Dict[str, float]:
    """
    docs = the chunks fed to reader (ground truth for faithfulness)
    summary = reader output
    """
    if _FAITHFUL_LLM is None:
        return {"faithfulness": None, "explanation": "Faithfulness evaluation skipped"}
    
    evaluator = load_evaluator("faithfulness", llm=_FAITHFUL_LLM)
    res = evaluator.evaluate(
        question=query,
        contexts=[d.page_content for d in docs],
        answer=summary,
    )

    # return format {"faithfulness": [0,1], "explaination": str}
    return res