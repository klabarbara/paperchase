from langchain.chains.summarize import load_summarize_chain
from langchain_openai import AzureChatOpenAI

from ..config import settings

def build_summary_chain():
    llm = AzureChatOpenAI(
        azure_endpoint=settings.azure_endpoint,
        api_key=settings.azure_key,
        deployment_name=settings.chat_deployment,
        temperature=0.3,
    )

    return load_summarize_chain(llm, chain_type="map_reduce")