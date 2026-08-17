from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)
AgentTool = BaseTool | Callable[..., Any] | dict[str, Any]


@dataclass(frozen=True)
class TurnUsage:
    """Token and call counts of one chat turn (summed in `src.llm.statistics.usage`)."""

    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    calls: int = 0
    tool_calls: int = 0
    script_tool_calls: int = 0
    # Provider-reported cache reads (`input_token_details.cache_read`) — a
    # subset of `input_tokens` billed at the cache-read price, not full price.
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(BaseModel):
    model_name: str | None = Field(default=None, description='The model name to use, for example "gpt-4.1-mini".')
    provider_name: str = Field(description='The provider name, for example "openai" or "azure_openai".')
    api_key: str = Field(description="The API key used to authenticate to the LLM provider.")
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
