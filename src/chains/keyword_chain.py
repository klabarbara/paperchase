from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from ..config import settings

_PROMPT = ChatPromptTemplate.from_template(
    "List 3–6 short technical keywords for searching arXiv based on the user query: {query}"
)

def build_keyword_chain():
    llm = AzureChatOpenAI(
        azure_endpoint=settings.azure_endpoint,
        api_key=settings.azure_key,
        deployment_name=settings.chat_deployment,
        temperature=0.1,
    )
    return LLMChain(prompt=_PROMPT, llm=llm)