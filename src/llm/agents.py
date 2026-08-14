import os
from typing import Any

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import (
    LLMToolSelectorMiddleware,
    SummarizationMiddleware,
)
from deepagents.backends import LocalShellBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from .computer_tools import COMPUTER_TOOLS
from .tools import web_search
from src.config.prompts import Prompts
from .base import LLMClient


class Agents(LLMClient):
    async def right_coding_agent(
        self,
        messages: list[HumanMessage | AIMessage | ToolMessage],
        model: str,
        thread_id: str | None = None,
    ):
        filesystem_backend = LocalShellBackend(
            root_dir=os.getcwd(),
            virtual_mode=False,
            inherit_env=True,
        )
        
        tools = [
            # web_search,
            # DuckDuckGoSearchRun(),
            *COMPUTER_TOOLS,
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
                FilesystemMiddleware(
                    backend=filesystem_backend,
                    custom_tool_descriptions={
                        "execute": (
                            "Run a shell command in the current project workspace. "
                            "Use this for file deletion, moves, renames, git, tests, "
                            "and diagnostics when no more specific tool exists."
                        )
                    },
                ), # type: ignore
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
