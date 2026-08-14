from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import (
    LLMToolSelectorMiddleware,
    SummarizationMiddleware,
)
from .meta_tools import META_TOOLS
from src.config.prompts import Prompts
from .base import LLMClient


class Agents(LLMClient):
    async def right_coding_agent(
        self,
        messages: list[HumanMessage | AIMessage | ToolMessage],
        model: str,
        thread_id: str | None = None,
    ):
        # The agent's whole tool surface is the three meta tools: every other
        # capability is discovered through search_tools / get_tool and driven
        # from run_tools scripts (see src/llm/meta_tools.py).
        tools = [
            *META_TOOLS,
        ]

        response = await self.ask_agent(
            system_prompt=Prompts.right_coding_agent_sys,
            model_name=model,
            agent_input={"messages": messages},
            tools=tools,
            thread_id=thread_id,
            middlewares=[
                SummarizationMiddleware(
                    model=self.build_chat_model(
                        model_name="openai/gpt-4.1-mini",
                        provider=self.providers[0],
                    ),
                    trigger=("tokens", 40000),
                    keep=("messages", 10),
                ),
                # LLMToolSelectorMiddleware(
                #     model=self.build_chat_model(
                #         model_name="openai/gpt-4.1",
                #         provider=self.providers[0],
                #     ),
                #     max_tools=3
                # ),
            ],
        )

        return response

    async def resume(
        self,
        resume_value: Any,
        thread_id: str,
    ) -> dict[str, Any]:
        return await self.resume_agent(resume_value, thread_id)
