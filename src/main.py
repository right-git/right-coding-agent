import asyncio
import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.config.logging import logger
from src.config.settings import settings
from src.llm.history import finalize_turn_history
from src.llm.providers import OpenRouterCatalog
from src.llm.types import LLMProvider
from src.llm.statistics import SessionUsage, turn_usage_from_messages
from src.llm.utils import (
    collect_message_ids,
    is_empty_final_response,
    trim_incomplete_tool_calls,
)
from src.ui import ChatUI

# The startup model comes from .env (LLM_DEFAULT_MODEL); /model can switch
# to anything else at runtime.
available_models = [settings.llm_default_model]

EMPTY_RESPONSE_NUDGE = (
    "Your last message was empty. Continue: finish the remaining steps of "
    "the task, or state what you found and what is still missing."
)


def preload_vision_model(ui: ChatUI | None = None) -> None:
    """Load the LocateAnything vision model so the first screen query is fast.

    Runs in a worker thread at startup; a failure only costs the warm start,
    the locator will retry lazily on first use. Progress is reported through
    `ui.set_model_status`, shown live at the right of the prompt.
    """
    set_status = getattr(ui, "set_model_status", None) or (lambda *_: None)
    try:
        from src.llm.tools import warm_up_computer

        logger.info("Preloading the vision locator model")
        set_status("vision", "loading")
        warm_up_computer()
        set_status("vision", "ready")
        logger.info("Vision locator model is ready")
    except Exception:
        set_status("vision", "failed")
        logger.exception("Vision model preload failed")


async def report_usage(
    *,
    ui: ChatUI,
    catalog: OpenRouterCatalog | None,
    session_usage: SessionUsage | None,
    model: str,
    response_messages,
    previous_ids: frozenset[str],
    duration: float | None = None,
) -> None:
    """Print the context/token/cost/time footer for one finished turn.

    Reporting must never break a turn that already succeeded, so every
    failure here is logged and swallowed.
    """
    if catalog is None or session_usage is None:
        return
    try:
        turn = turn_usage_from_messages(response_messages, previous_ids)
        model_info = await catalog.get(model)
        cost = (
            model_info.cost_of(turn.input_tokens, turn.output_tokens) if model_info is not None and turn.calls else None
        )
        session_usage.add(turn, cost, duration or 0.0)
        ui.print_usage(turn, model_info, cost, session_usage, duration)
        logger.info(
            "Turn usage model [{}] input [{}] output [{}] context [{}] " "cost [{}] duration [{}] session_tokens [{}]",
            model,
            turn.input_tokens,
            turn.output_tokens,
            turn.context_tokens,
            cost,
            duration,
            session_usage.total_tokens,
        )
    except Exception:
        logger.exception("Failed to report token usage for model [{}]", model)


def turn_callbacks(ui: ChatUI, stream) -> tuple:
    """The turn's (on_message, on_token) pair; tees tokens into TTS in voice mode."""
    on_message = getattr(stream, "on_message", None)
    on_token = getattr(stream, "on_token", None)
    voice = getattr(ui, "voice", None)
    if voice is not None and voice.speak_replies:
        on_token = voice.wrap_on_token(on_token)
    return on_message, on_token


async def process_user_turn(
    *,
    agents,
    ui: ChatUI,
    messages: list[HumanMessage | AIMessage | ToolMessage],
    model: str,
    user_content: str | list[dict],
    catalog: OpenRouterCatalog | None = None,
    session_usage: SessionUsage | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
) -> list[HumanMessage | AIMessage | ToolMessage]:
    started = time.perf_counter()
    voice_mode = getattr(ui, "voice_active", False)
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

    printed_ids: set[str] = set()

    try:
        with ui.turn_stream() as stream:
            on_message, on_token = turn_callbacks(ui, stream)
            response = await agents.right_coding_agent(
                messages=working_messages,
                model=model,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                voice_mode=voice_mode,
                on_message=on_message,
                on_token=on_token,
            )
        if stream is not None:
            printed_ids |= stream.printed_ids
        raw_messages = response["messages"]

        if is_empty_final_response(raw_messages):
            # Seen with Gemini via OpenRouter: a "successful" completion with
            # no text, no tool calls, zero usage. Drop it and nudge once.
            logger.warning(
                "Model returned an empty final response model [{}]; " "nudging once to continue",
                model,
            )
            retry_messages = [
                *raw_messages[:-1],
                HumanMessage(EMPTY_RESPONSE_NUDGE),
            ]
            with ui.turn_stream() as stream:
                on_message, on_token = turn_callbacks(ui, stream)
                response = await agents.right_coding_agent(
                    messages=retry_messages,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    voice_mode=voice_mode,
                    on_message=on_message,
                    on_token=on_token,
                )
            if stream is not None:
                printed_ids |= stream.printed_ids
            raw_messages = response["messages"]
            if is_empty_final_response(raw_messages):
                logger.warning(
                    "Model returned an empty final response twice model [{}]",
                    model,
                )
                ui.print_warning("model ended the turn with an empty response twice")

        trimmed_messages = trim_incomplete_tool_calls(raw_messages)
        new_messages = trimmed_messages[shown_count:]
        visible_output = ui.has_visible_output(new_messages)

        if len(raw_messages) != len(trimmed_messages):
            logger.warning(
                "Trimmed incomplete tool-call tail model [{}] raw_messages [{}] " "trimmed_messages [{}]",
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
            ui.print_warning("assistant completed the turn but produced no visible output")
            await report_usage(
                ui=ui,
                catalog=catalog,
                session_usage=session_usage,
                model=model,
                response_messages=raw_messages,
                previous_ids=previous_ids,
                duration=time.perf_counter() - started,
            )
            ui.finish_voice_turn()
            ui.notify_done()
            return finalize_turn_history(trimmed_messages)

        if printed_ids:
            ui.print_response(new_messages, skip_ids=printed_ids)
        else:
            ui.print_response(new_messages)
        await report_usage(
            ui=ui,
            catalog=catalog,
            session_usage=session_usage,
            model=model,
            response_messages=raw_messages,
            previous_ids=previous_ids,
            duration=time.perf_counter() - started,
        )
        ui.finish_voice_turn()
        ui.notify_done()
        return finalize_turn_history(trimmed_messages)
    except Exception as e:
        logger.exception(
            "User turn failed model [{}] message_chars [{}]",
            model,
            len(user_content),
        )
        ui.finish_voice_turn()
        ui.print_error(e)
        return base_messages


async def main():
    try:
        from src.llm.agents import Agents
    except Exception:
        logger.exception("Failed to import agent implementation")
        raise

    agents = Agents(
        [
            LLMProvider(
                provider_name="openai",
                api_key=settings.llm_api_key,
                api_base=settings.llm_api_base,
            )
        ]
    )

    catalog = OpenRouterCatalog()
    session_usage = SessionUsage()

    messages = []
    model = available_models[0]
    ui = ChatUI(model=model, available_models=available_models)
    try:
        # Push-to-talk is always on: the hotkey works from the first prompt,
        # /voice only toggles whether replies are spoken.
        ui.start_voice_input()
    except Exception:
        logger.exception("Push-to-talk startup failed")
        ui.print_warning("push-to-talk unavailable (see logs.log)")
    ui.print_welcome()

    async def load_catalog() -> None:
        ui.set_model_catalog(await catalog.models())

    # Fetched in the background so the first prompt is not delayed; the
    # in-loop refresh below retries (rate-limited by the catalog's cooldown)
    # if this initial attempt failed.
    catalog_task = asyncio.create_task(load_catalog())
    vision_task = asyncio.create_task(asyncio.to_thread(preload_vision_model, ui))

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
                user_content=ui.take_user_content(user_content),
                catalog=catalog,
                session_usage=session_usage,
                reasoning_effort=ui.reasoning_effort,
                temperature=ui.temperature,
            )
    finally:
        catalog_task.cancel()
        vision_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
