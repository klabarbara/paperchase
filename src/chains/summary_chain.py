from transformers import pipeline
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint

from ..config import settings

def build_summary_chain():
    if settings.oss_mode == "remote":
        llm = HuggingFaceEndpoint(
            endpoint_url=settings.summary_endpoint,
            huggingface_api_token=settings.summary_token,
            model_kwargs={"task": "summarization", "max_new_tokens": 256, "temperature": 0.3},
            timeout=60,
        )
    else:
        hf_pipe = pipeline(
            task="summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device_map="auto",
            max_new_tokens=256,
            temperature=0.3,
        )
        breakpoint()
        llm = HuggingFacePipeline(pipeline=hf_pipe)

    return load_summarize_chain(llm, chain_type="map_reduce")