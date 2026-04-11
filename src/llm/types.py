from collections.abc import Callable
from typing import Any, TypeVar

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)
AgentTool = BaseTool | Callable[..., Any] | dict[str, Any]
MiddlewareFactory = Callable[["LLMProvider"], AgentMiddleware]


class LLMProvider(BaseModel):
    model_name: str | None = Field(
        default=None, description='The model name to use, for example "gpt-4.1-mini".'
    )
    provider_name: str = Field(
        description='The provider name, for example "openai" or "azure_openai".'
    )
    api_key: str = Field(
        description="The API key used to authenticate to the LLM provider."
    )
    api_base: str | None = Field(
        default=None,
        description="Optional base URL or endpoint for the provider.",
    )
    api_version: str | None = Field(
        default=None,
        description="Optional API version for providers such as Azure OpenAI.",
    )
    deployment_name: str | None = Field(
        default=None,
        description="Optional deployment name for Azure OpenAI.",
    )
    additional_headers: dict[str, str] | None = Field(
        default=None,
        description="Optional extra headers sent with provider requests.",
    )
