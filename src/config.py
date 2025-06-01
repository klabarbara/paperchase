from pydantic import Field
from pydantic_settings import BaseSettings
"""
assigns and loads all environmental values for use in app.
call 'settings.[value_name]' to use
"""

class Settings(BaseSettings):
    arxiv_email: str = Field(..., alias="ARXIV_EMAIL")

    
    azure_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    azure_key: str = Field(..., alias="AZURE_OPENAI_KEY")
    chat_deployment: str = Field("gpt-4o", alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    embed_deployment: str = Field("text-embedding-3-small", alias="AZURE_OPENAI_EMBED_DEPLOYMENT")
    chat_api_version: str = Field(..., alias="AZURE_OPENAI_CHAT_API_VERSION")
    embed_api_version: str = Field(..., alias="AZURE_OPENAI_EMBED_API_VERSION")
    
    model_config = {
        "case_sensitive": False,   
        "env_file": ".env",
    }

settings = Settings()
