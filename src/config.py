from pydantic import BaseSettings, Field

"""
assigns and loads all environmental values for use in app.
call 'settings.[value_name]' to use
"""

class Settings(BaseSettings):
    arxiv_email: str = Field(..., env="ARXIV_EMAIL")
    azure_endpoint: str = Field(...,
                                env="AZURE_OPENAI_ENDPOINT")
    azure_key: str = Field(..., env="AZURE_OPENAI_KEY")
    chat_deployment: str = Field("gpt-4o-mini", env="AZURE_OPENAI_CHAT_DEPLOYMENT")
    embed_deployment: str = Field("text-embedding-3-small", env="AZURE_OPENAI_EMBED_DEPLOYMENT")

    class Config:
        env_file = ".env"

settings = Settings()
