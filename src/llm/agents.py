from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import SummarizationMiddleware
from src.config.prompts import Prompts

from .client import LLMClient
from .middlewares import AttachedImagesMiddleware, MessageLogMiddleware
from .tools import META_TOOLS


class Agents(LLMClient):
    async def right_coding_agent(
        self,
        messages: list[HumanMessage | AIMessage | ToolMessage],
        model: str,
        thread_id: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        voice_mode: bool = False,
        on_message=None,
        on_token=None,
    ):
        # The agent's whole tool surface is the single run_tools meta tool:
        # every other capability is discovered via the in-script
        # search_tools()/get_tool() functions and driven from run_tools
        # scripts (see src/llm/tools/meta/tool.py).
        tools = [
            *META_TOOLS,
        ]

        response = await self.ask_agent(
            system_prompt=Prompts.coding_system(voice_mode),
            model_name=model,
            agent_input={"messages": messages},
            tools=tools,
            thread_id=thread_id,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            on_message=on_message,
            on_token=on_token,
            middlewares=[
                # Runs first so screenshots captured by tools become a vision
                # message before summarization ever touches the tool tail.
                AttachedImagesMiddleware(),
                SummarizationMiddleware(
                    model=self.build_chat_model(
                        model_name="openai/gpt-4.1-mini",
                        provider=self.providers[0],
                    ),
                    # Deliberately high: every mid-turn summarization rewrites
                    # history, which breaks the provider prompt-cache prefix
                    # (cached re-reads cost ~10% of fresh input, so a long
                    # tool tail is cheap to keep) and loses task detail while
                    # the task is still running. Summarize only when the
                    # window genuinely needs the room.
                    trigger=("tokens", 100000),
                    keep=("messages", 20),
                ),
                MessageLogMiddleware(),
            ],
        )

        return response

    async def resume(
        self,
        resume_value: Any,
        thread_id: str,
    ) -> dict[str, Any]:
        return await self.resume_agent(resume_value, thread_id)
