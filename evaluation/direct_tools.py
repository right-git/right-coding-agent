"""The registry tools wired directly into the agent's schema.

This is the control group for the meta-tool architecture: the exact same
tools (`web_search` + the `screen_*` family), but every schema is exposed to
the model and every call is a model round-trip. `as_direct_tool` wraps each
tool so screenshots still reach the model — the attachments channel is
normally opened by `run_tools`, so here it is opened around each direct call
and the collected images ride out as the ToolMessage artifact for
`AttachedImagesMiddleware` to surface.

`schema_token_estimate` sizes what an architecture's tool schemas add to
every single model request, which is the fixed cost being compared.
"""

import json
from collections.abc import Sequence

from langchain_core.tools import BaseTool, StructuredTool

from src.llm.tools import COMPUTER_TOOLS, FILE_TOOLS, bash, collecting_images, web_fetch, web_search


def as_direct_tool(tool_obj: BaseTool) -> BaseTool:
    """The same tool, invokable directly by the model, with image attachment."""

    async def call(**kwargs):
        with collecting_images() as images:
            content = await tool_obj.ainvoke(kwargs)
        return str(content), list(images)

    return StructuredTool(
        name=tool_obj.name,
        description=tool_obj.description,
        args_schema=tool_obj.args_schema,
        coroutine=call,
        response_format="content_and_artifact",
    )


DIRECT_TOOLS = [as_direct_tool(tool_obj) for tool_obj in [web_fetch, web_search, *FILE_TOOLS, bash, *COMPUTER_TOOLS]]


def schema_token_estimate(tools: Sequence[BaseTool]) -> int:
    """Rough tokens the tool schemas add to every model request (~4 chars/token)."""
    payload = json.dumps(
        [
            {
                "name": tool_obj.name,
                "description": tool_obj.description,
                "parameters": tool_obj.args,
            }
            for tool_obj in tools
        ],
        ensure_ascii=False,
    )
    return max(1, len(payload) // 4)
