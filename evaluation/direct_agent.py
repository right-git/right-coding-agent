"""The direct-wiring agent: identical to the production agent except tools.

Everything that could bias the comparison is kept the same — LLM client,
retry/failover, summarization threshold, image surfacing, request logging.
The only difference is the tool layer: DIRECT_TOOLS in the schema instead of
the three meta tools.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents.middleware import SummarizationMiddleware

from src.llm.agents import Agents
from src.llm.middlewares import AttachedImagesMiddleware, MessageLogMiddleware

from .direct_tools import DIRECT_TOOLS

DIRECT_CODING_AGENT_SYS = """\
You are right_coding_agent, a code assistant agent working inside the current project.

Your tools for the web and the user's screen are wired in directly — call \
them as needed. Screenshots captured by screen tools are attached to the \
conversation as images you can see.
"""


class DirectAgents(Agents):
    """Same agent, classic tool wiring: every tool schema goes to the model."""

    async def direct_coding_agent(
        self,
        messages: list[HumanMessage | AIMessage | ToolMessage],
        model: str,
        thread_id: str | None = None,
        on_message=None,
        on_token=None,
    ):
        # Mirrors Agents.right_coding_agent so only the tool wiring differs.
        return await self.ask_agent(
            system_prompt=DIRECT_CODING_AGENT_SYS,
            model_name=model,
            agent_input={"messages": messages},
            tools=[*DIRECT_TOOLS],
            thread_id=thread_id,
            on_message=on_message,
            on_token=on_token,
            middlewares=[
                AttachedImagesMiddleware(),
                SummarizationMiddleware(
                    model=self.build_chat_model(
                        model_name="openai/gpt-4.1-mini",
                        provider=self.providers[0],
                    ),
                    trigger=("tokens", 40000),
                    keep=("messages", 10),
                ),
                MessageLogMiddleware(),
            ],
        )

    # Same entry-point name as the production agent, so src.main's turn loop
    # (and its usage footer) drives either architecture unchanged.
    async def right_coding_agent(
        self,
        messages: list[HumanMessage | AIMessage | ToolMessage],
        model: str,
        thread_id: str | None = None,
        on_message=None,
        on_token=None,
    ):
        return await self.direct_coding_agent(
            messages=messages,
            model=model,
            thread_id=thread_id,
            on_message=on_message,
            on_token=on_token,
        )
