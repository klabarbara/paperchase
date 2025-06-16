from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint

from ..config import settings

_PROMPT = PromptTemplate(
    input_variables=["query"],
    template=(
        "Extract 3-6 concise technical keywords from the following query:\n\n"
        "{query}\n\nKeywords:"
    )
)

def build_keyword_chain():
    """
    open source keyword extraction
    in-process HF pipeline when settings.oss_mode == "local"
    azure ML HF pipeline when settings.oss_mode == "remote"
    """
    if settings.oss_mode == "remote":
        llm = HuggingFaceEndpoint(
            endpoint_url=settings.keyword_endpoint,
            huggingface_api_token=settings.keyword_token,
            model_kwargs={
                "task": "text2text-generation",
                "max_new_tokens": 32,
            },
            timeout=30,
        )
    else:
        hf_pipe = pipeline(
            task="text2text-generation",
            model="google/flan-t5-small",
            device_map="auto",
            max_new_tokens=32,
        )
        llm = HuggingFacePipeline(pipeline=hf_pipe)

    return _PROMPT | llm