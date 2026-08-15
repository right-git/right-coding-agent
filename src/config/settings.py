from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = Field(..., description="Active runtime environment name.")
    llm_api_key: str = Field(
        ...,
        description="LLM API Key",
    )
    llm_api_base: str = Field(
        ...,
        description="LLM API Base",
    )


settings = Settings()
