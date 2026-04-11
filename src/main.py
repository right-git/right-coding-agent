import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.config.base import settings
from src.config.logging import logger
from src.llm.types import LLMProvider
from src.ui import ChatUI

available_models = [
    "openai/gpt-5.1-codex-mini",
]

def trim_incomplete_tool_calls(
    messages: list[HumanMessage | AIMessage | ToolMessage],
) -> list[HumanMessage | AIMessage | ToolMessage]:
    """Drop a trailing assistant tool-call block if it has missing tool outputs."""
    trimmed: list[HumanMessage | AIMessage | ToolMessage] = []
    pending_tool_calls: set[str] = set()
    pending_start_index: int | None = None

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            if pending_tool_calls and pending_start_index is not None:
                trimmed = trimmed[:pending_start_index]
            pending_start_index = len(trimmed)
            pending_tool_calls = {
                tool_call["id"]
                for tool_call in message.tool_calls
                if tool_call.get("id")
            }
            trimmed.append(message)
            continue

        if isinstance(message, ToolMessage):
            if message.tool_call_id in pending_tool_calls:
                pending_tool_calls.remove(message.tool_call_id)
            trimmed.append(message)
            continue

        if pending_tool_calls and pending_start_index is not None:
            trimmed = trimmed[:pending_start_index]
            pending_tool_calls = set()
            pending_start_index = None

        trimmed.append(message)

    if pending_tool_calls and pending_start_index is not None:
        trimmed = trimmed[:pending_start_index]

    return trimmed


async def process_user_turn(
    *,
    agents,
    ui: ChatUI,
    messages: list[HumanMessage | AIMessage | ToolMessage],
    model: str,
    user_content: str,
) -> list[HumanMessage | AIMessage | ToolMessage]:
    base_messages = trim_incomplete_tool_calls(messages)
    working_messages = [*base_messages, HumanMessage(user_content)]
    shown_count = len(working_messages)
    logger.info(
        "Starting user turn model [{}] message_chars [{}] history_messages [{}]",
        model,
        len(user_content),
        len(base_messages),
    )

    try:
        with ui.loading("thinking"):
            response = await agents.right_coding_agent(
                messages=working_messages,
                model=model,
            )
        raw_messages = response["messages"]
        trimmed_messages = trim_incomplete_tool_calls(raw_messages)
        new_messages = trimmed_messages[shown_count:]
        visible_output = ui.has_visible_output(new_messages)

        if len(raw_messages) != len(trimmed_messages):
            logger.warning(
                "Trimmed incomplete tool-call tail model [{}] raw_messages [{}] "
                "trimmed_messages [{}]",
                model,
                len(raw_messages),
                len(trimmed_messages),
            )

        logger.info(
            "Completed user turn model [{}] total_messages [{}] raw_messages [{}] "
            "new_messages [{}] visible_output [{}]",
            model,
            len(trimmed_messages),
            len(raw_messages),
            len(new_messages),
            visible_output,
        )
        if not visible_output:
            logger.warning(
                "Assistant completed turn without visible output model [{}] "
                "history_messages [{}] total_messages [{}]",
                model,
                shown_count,
                len(trimmed_messages),
            )
            ui.print_warning(
                "assistant completed the turn but produced no visible output"
            )
            return trimmed_messages

        ui.print_response(new_messages)
        return trimmed_messages
    except Exception as e:
        logger.exception(
            "User turn failed model [{}] message_chars [{}]",
            model,
            len(user_content),
        )
        ui.print_error(e)
        return base_messages


async def main():
    try:
        from src.llm.agents import Agents
    except Exception:
        logger.exception("Failed to import agent implementation")
        raise

    agents = Agents([
        LLMProvider(
            provider_name="openai",
            api_key=settings.llm_api_key,
            api_base=settings.llm_api_base,
        )
    ])

    messages = []
    model = available_models[0]
    ui = ChatUI(model=model, available_models=available_models)
    ui.print_welcome()

    while True:
        user_content = await ui.get_input()

        if not user_content.strip():
            continue

        if user_content.startswith("/"):
            result = ui.handle_command(user_content)
            if result == "clear":
                messages = []
            model = ui.model
            continue

        messages = await process_user_turn(
            agents=agents,
            ui=ui,
            messages=messages,
            model=model,
            user_content=user_content,
        )


if __name__ == "__main__":
    asyncio.run(main())
