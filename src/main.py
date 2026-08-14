import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.config.base import settings
from src.config.logging import logger
from src.llm.openrouter import OpenRouterCatalog
from src.llm.types import LLMProvider
from src.llm.usage import (
    SessionUsage,
    collect_message_ids,
    turn_usage_from_messages,
)
from src.ui import ChatUI
from src.utils.functions import trim_incomplete_tool_calls

available_models = [
    "google/gemini-3.7-flash",
]


def preload_vision_model() -> None:
    """Load the LocateAnything vision model so the first screen query is fast.

    Runs in a worker thread at startup; a failure only costs the warm start,
    the locator will retry lazily on first use.
    """
    try:
        from src.llm.computer_tools import warm_up_computer

        logger.info("Preloading the vision locator model")
        warm_up_computer()
        logger.info("Vision locator model is ready")
    except Exception:
        logger.exception("Vision model preload failed")


async def report_usage(
    *,
    ui: ChatUI,
    catalog: OpenRouterCatalog | None,
    session_usage: SessionUsage | None,
    model: str,
    response_messages,
    previous_ids: frozenset[str],
) -> None:
    """Print the context/token/cost footer for one finished turn.

    Reporting must never break a turn that already succeeded, so every
    failure here is logged and swallowed.
    """
    if catalog is None or session_usage is None:
        return
    try:
        turn = turn_usage_from_messages(response_messages, previous_ids)
        model_info = await catalog.get(model)
        cost = (
            model_info.cost_of(turn.input_tokens, turn.output_tokens)
            if model_info is not None and turn.calls
            else None
        )
        session_usage.add(turn, cost)
        ui.print_usage(turn, model_info, cost, session_usage)
        logger.info(
            "Turn usage model [{}] input [{}] output [{}] context [{}] "
            "cost [{}] session_tokens [{}]",
            model,
            turn.input_tokens,
            turn.output_tokens,
            turn.context_tokens,
            cost,
            session_usage.total_tokens,
        )
    except Exception:
        logger.exception("Failed to report token usage for model [{}]", model)


async def process_user_turn(
    *,
    agents,
    ui: ChatUI,
    messages: list[HumanMessage | AIMessage | ToolMessage],
    model: str,
    user_content: str,
    catalog: OpenRouterCatalog | None = None,
    session_usage: SessionUsage | None = None,
) -> list[HumanMessage | AIMessage | ToolMessage]:
    base_messages = trim_incomplete_tool_calls(messages)
    working_messages = [*base_messages, HumanMessage(user_content)]
    shown_count = len(working_messages)
    previous_ids = collect_message_ids(working_messages)
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
            await report_usage(
                ui=ui,
                catalog=catalog,
                session_usage=session_usage,
                model=model,
                response_messages=raw_messages,
                previous_ids=previous_ids,
            )
            return trimmed_messages

        ui.print_response(new_messages)
        await report_usage(
            ui=ui,
            catalog=catalog,
            session_usage=session_usage,
            model=model,
            response_messages=raw_messages,
            previous_ids=previous_ids,
        )
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

    catalog = OpenRouterCatalog()
    session_usage = SessionUsage()

    messages = []
    model = available_models[0]
    ui = ChatUI(model=model, available_models=available_models)
    ui.print_welcome()

    async def load_catalog() -> None:
        ui.set_model_catalog(await catalog.models())

    # Fetched in the background so the first prompt is not delayed; the
    # in-loop refresh below retries (rate-limited by the catalog's cooldown)
    # if this initial attempt failed.
    catalog_task = asyncio.create_task(load_catalog())
    vision_task = asyncio.create_task(asyncio.to_thread(preload_vision_model))

    try:
        while True:
            user_content = await ui.get_input()

            if not user_content.strip():
                continue

            if catalog_task.done() and not ui.model_catalog:
                ui.set_model_catalog(await catalog.models())

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
                catalog=catalog,
                session_usage=session_usage,
            )
    finally:
        catalog_task.cancel()
        vision_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
