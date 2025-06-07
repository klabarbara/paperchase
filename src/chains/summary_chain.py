from langchain.chains.summarize import load_summarize_chain
from langchain_openai.chat_models import AzureChatOpenAI

from ..config import settings

def build_summary_chain():
    llm = AzureChatOpenAI(
        azure_endpoint=settings.azure_endpoint,
        api_key=settings.azure_key,
        deployment_name=settings.chat_deployment,
        model_name=settings.chat_deployment,
        api_version=settings.chat_api_version, 
        temperature=0.3,
    )

    return load_summarize_chain(llm, chain_type="map_reduce")

from langchain.chains.summarize import load_summarize_chain
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers.pipelines import pipeline
from ..config import settings
#TODO: unify local vs remote to use for both keyword and summary chains?

def _local_llm() -> HuggingFacePipeline:
    pipe = pipeline(
        task="summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device_map="auto",
        max_new_tokens=256,
        temperature=0.3,
    )

    return HuggingFacePipeline(pipeline=pipe)

def _remote_llm() -> HuggingFacePipeline:
    return HuggingFaceEndpoint(
        endpoint_url=settings.summary_endpointe,
        huggingface_api_token=settings.summary_token,
        model_kwargs={
            "max_new_tokens": 256,
            "temperature": 0.3,
        },
        timeout=60,
    )

USE_REMOTE = False

def build_summary_chain():
    llm = _remote_llm() if USE_REMOTE else _local_llm()
    return load_summarize_chain(llm, chain_type="map_reduce")