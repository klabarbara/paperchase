from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings

class BackendType(str, Enum):
    azure = "azure"
    opensource = "opensource"

class OSSMode(str, Enum):
    local = "local"
    remote = "remote"

class Settings(BaseSettings):
    arxiv_email: str = Field(..., alias="ARXIV_EMAIL")

    # top-level backend switch
    backend: BackendType = Field(
        BackendType.opensource,
        env="BACKEND",
        description="Choose 'azure' or 'opensource' backend. azure disabled for development.",
    )

    # only used when backend == 'opensource'
    oss_mode: OSSMode = Field(
        OSSMode.local,
        env="OSS_MODE",
        description="For opensource: 'local' or 'remote' model execution",
    )

    # # azure fields (unused in opensource-only version)
    # azure_endpoint: str = Field("", alias="AZURE_OPENAI_ENDPOINT")
    # azure_key: str = Field("", alias="AZURE_OPENAI_KEY")
    # chat_deployment: str = Field("gpt-4o", alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    # embed_deployment: str = Field("text-embedding-3-small", alias="AZURE_OPENAI_EMBED_DEPLOYMENT")
    # chat_api_version: str = Field("", alias="AZURE_OPENAI_CHAT_API_VERSION")
    # embed_api_version: str = Field("", alias="AZURE_OPENAI_EMBED_API_VERSION")

    # open-source HF endpoints (only if oss_mode == 'remote')
    embed_endpoint: str = Field("", alias="EMBED_ENDPOINT")
    embed_token: str = Field("", alias="EMBED_TOKEN")
    keyword_endpoint: str = Field("", alias="KEYWORD_ENDPOINT")
    keyword_token: str = Field("", alias="KEYWORD_TOKEN")
    summary_endpoint: str = Field("", alias="SUMMARY_ENDPOINT")
    summary_token: str = Field("", alias="SUMMARY_TOKEN")

    class Config:
        case_sensitive = False
        env_file = ".env"

settings = Settings()
