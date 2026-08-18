import asyncio
import os
import signal
import threading
import time
from pathlib import Path

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
from src.ui.commands import McpAction, SkillAction, run_mcp_action
from src.ui.interrupt import InterruptPolicy, TurnCancelled

# The startup model comes from .env (LLM_DEFAULT_MODEL); /model can switch
# to anything else at runtime.
available_models = [settings.llm_default_model]

EMPTY_RESPONSE_NUDGE = (
    "Your last message was empty. Continue: finish the remaining steps of "
    "the task, or state what you found and what is still missing."
)


def _status_word(state) -> str:
    """Map an MCP `ServerState` onto `ChatUI.set_model_status`'s vocabulary.

    `set_model_status` only renders three words ("loading" / "ready" /
    "failed" — see `ChatUI._model_status_text`); anything else is stored but
    stays invisible in the status line, which is the right outcome for a
    plain disconnect (stop, or the moment between teardown and reconnect) —
    it is not a failure and should not flash a red mark. `ServerState` is a
    `str` subclass, so comparing against literals is exact.
    """
    if state == "connecting":
        return "loading"
    if state == "connected":
        return "ready"
    if state in ("failed", "needs auth"):
        return "failed"
    return "disconnected"


def preload_vision_model(ui: ChatUI | None = None) -> None:
    """Load the LocateAnything vision model so the first screen query is fast.

    Runs in a worker thread at startup; a failure only costs the warm start,
    the locator will retry lazily on first use. Progress is reported through
    `ui.set_model_status`, shown live at the right of the prompt. With
    ENABLE_VISION_MODEL off (the default) this is a no-op — the locator tools
    are not registered either, so nothing can load the model later.
    """
    if not settings.enable_vision_model:
        logger.info("Vision locator disabled (ENABLE_VISION_MODEL is off); skipping preload")
        return
    set_status = getattr(ui, "set_model_status", None) or (lambda *_: None)
    try:
        from src.llm.tools import warm_up_computer
        from src.utils.downloads import reporting_progress

        logger.info("Preloading the vision locator model")
        set_status("vision", "loading")
        with reporting_progress(lambda detail: set_status("vision", "loading", detail)):
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
    streamed_messages: list | None = None,
    cache_writes_expected: bool = False,
) -> None:
    """Print the context/token/cost/time footer for one finished turn.

    `streamed_messages` is the turn's live message stream — the complete
    record even when mid-turn summarization dropped early messages from the
    final state (accounting once missed 21 of 28 calls without it); the two
    lists are concatenated and deduplicated by id in
    `turn_usage_from_messages`. `cache_writes_expected` says the client
    requested prompt caching for this model, so fresh input is billed at the
    cache-write price and `saved` is the net effect vs no caching — negative
    when a turn wrote to the cache but read nothing back. Reporting must
    never break a turn that already succeeded, so every failure here is
    logged and swallowed.
    """
    if catalog is None or session_usage is None:
        return
    try:
        turn = turn_usage_from_messages([*(streamed_messages or []), *response_messages], previous_ids)
        model_info = await catalog.get(model)
        cost = saved = None
        if model_info is not None and turn.calls:
            cost = model_info.cost_of(
                turn.input_tokens,
                turn.output_tokens,
                turn.cached_input_tokens,
                assume_cache_writes=cache_writes_expected,
            )
            baseline = model_info.cost_of(turn.input_tokens, turn.output_tokens)
            if cost is not None and baseline is not None:
                saved = baseline - cost
        session_usage.add(turn, cost, duration or 0.0, saved=saved)
        ui.print_usage(turn, model_info, cost, session_usage, duration, saved=saved)
        logger.info(
            "Turn usage model [{}] input [{}] output [{}] cached [{}] context [{}] "
            "cost [{}] saved [{}] duration [{}] session_tokens [{}]",
            model,
            turn.input_tokens,
            turn.output_tokens,
            turn.cached_input_tokens,
            turn.context_tokens,
            cost,
            saved,
            duration,
            session_usage.total_tokens,
        )
    except Exception:
        logger.exception("Failed to report token usage for model [{}]", model)


def make_sigint_handler(policy: InterruptPolicy, ui, current_turn: dict, *, force_exit=os._exit, clock=time.monotonic):
    """Ctrl+C during a turn: cancel it; a double press force-quits the process.

    Cancelling a turn stuck inside `asyncio.to_thread` only takes effect when
    the thread ends — the executor job itself cannot be interrupted — so the
    double press is the guaranteed way out: process death frees the
    microphone and the GPU no matter what any thread is doing. Installed via
    `loop.add_signal_handler` (Unix; at the prompt prompt_toolkit's raw mode
    handles Ctrl+C itself and returns /quit).
    """

    def on_sigint() -> None:
        task = current_turn.get("task")
        if task is not None and not task.done():
            if getattr(task, "cancelling", lambda: 0)():
                # The first Ctrl+C was ignored (a tool call stuck in an
                # executor thread) — no time window here, the next press
                # must always get the user out.
                logger.warning("Force quit: the turn ignored the first Ctrl+C")
                force_exit(130)
                return
            ui.print_warning("interrupting the turn — press Ctrl+C again to force quit")
            task.cancel()
            return
        if policy.press(now=clock()) == "force":
            logger.warning("Force quit on double Ctrl+C")
            force_exit(130)
        else:
            ui.print_warning("press Ctrl+C again to force quit")

    return on_sigint


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
    try:
        cache_writes_expected = bool(agents.wants_prompt_cache(agents.get_default_provider(), model))
    except Exception:
        cache_writes_expected = False
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
    streamed_messages: list = []

    def collecting(handler):
        """Tee every streamed message into our own list: summarization can
        drop early messages from the final state mid-turn, and usage
        accounting must still see them."""

        def on_collected(message):
            streamed_messages.append(message)
            if handler is not None:
                handler(message)

        return on_collected

    try:
        with ui.turn_stream() as stream:
            on_message, on_token = turn_callbacks(ui, stream)
            on_message = collecting(on_message)
            response = await ui.run_cancellable(
                agents.right_coding_agent(
                    messages=working_messages,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    voice_mode=voice_mode,
                    on_message=on_message,
                    on_token=on_token,
                )
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
                on_message = collecting(on_message)
                response = await ui.run_cancellable(
                    agents.right_coding_agent(
                        messages=retry_messages,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        temperature=temperature,
                        voice_mode=voice_mode,
                        on_message=on_message,
                        on_token=on_token,
                    )
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
                streamed_messages=streamed_messages,
                cache_writes_expected=cache_writes_expected,
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
            streamed_messages=streamed_messages,
            cache_writes_expected=cache_writes_expected,
        )
        ui.finish_voice_turn()
        ui.notify_done()
        return finalize_turn_history(trimmed_messages)
    except TurnCancelled:
        logger.info("Turn cancelled by user model [{}]", model)
        ui.cancel_voice_turn()
        ui.print_warning("request cancelled (Esc)")
        return base_messages
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
    if settings.enable_voice_model:
        try:
            # Push-to-talk is on whenever the voice models are enabled: the
            # hotkey works from the first prompt, /voice only toggles whether
            # replies are spoken.
            ui.start_voice_input()
        except PermissionError as error:
            logger.warning("Push-to-talk permission missing: {}", error)
            ui.print_warning(str(error))
        except Exception:
            logger.exception("Push-to-talk startup failed")
            ui.print_warning("push-to-talk unavailable (see logs.log)")
    else:
        logger.info("Voice models disabled (ENABLE_VOICE_MODEL is off); push-to-talk not started")
    try:
        # The breathing screen border while the agent drives the desktop:
        # every screen tool pings the status overlay.
        from src.llm.tools.computer import set_activity_listener
        from src.ui.overlay import get_status_overlay

        set_activity_listener(lambda: get_status_overlay().ping_computer())
    except Exception:
        logger.exception("Status overlay wiring failed")
    ui.print_welcome()

    async def load_catalog() -> None:
        ui.set_model_catalog(await catalog.models())

    # Fetched in the background so the first prompt is not delayed; the
    # in-loop refresh below retries (rate-limited by the catalog's cooldown)
    # if this initial attempt failed.
    catalog_task = asyncio.create_task(load_catalog())
    # A daemon thread, NOT asyncio.to_thread: the default executor's workers
    # are non-daemon, and /quit during a first-run model download would hang
    # the interpreter in concurrent.futures' atexit join.
    threading.Thread(target=preload_vision_model, args=(ui,), name="vision-preload", daemon=True).start()

    from src.llm.tools.mcp.manager import get_mcp_manager

    mcp_manager = get_mcp_manager()
    mcp_manager.on_status = lambda name, state: ui.set_model_status(f"mcp:{name}", _status_word(state))
    mcp_task = asyncio.create_task(mcp_manager.start())

    from src.llm.tools.skills.store import get_skill_store, skills_startup_report, start_skill_store

    try:
        skill_store = start_skill_store()
        notice = skills_startup_report(skill_store, auto_import=settings.skills_auto_import, repo_root=Path.cwd())
        if notice:
            ui.console.print(f"  {notice}", style="dim", markup=False, highlight=False)
    except Exception:
        logger.exception("Skill store startup failed")

    current_turn: dict = {"task": None}
    try:
        asyncio.get_running_loop().add_signal_handler(
            signal.SIGINT, make_sigint_handler(InterruptPolicy(), ui, current_turn)
        )
    except (NotImplementedError, RuntimeError):
        pass  # Windows event loops have no add_signal_handler; Esc covers turns there

    try:
        while True:
            user_content = await ui.get_input()

            if not user_content.strip():
                continue

            store = get_skill_store()
            if store is not None:
                try:
                    store.refresh()
                except Exception:
                    logger.exception("Skill refresh failed")

            if catalog_task.done() and not ui.model_catalog:
                ui.set_model_catalog(await catalog.models())

            if user_content.startswith("/"):
                result = ui.handle_command(user_content)
                if result == "clear":
                    messages = []
                model = ui.model
                if isinstance(result, SkillAction):
                    user_content = result.text  # fall through into the turn below
                elif isinstance(result, McpAction):
                    prompt_text = await run_mcp_action(result, mcp_manager, ui.console)
                    if prompt_text is None:
                        continue
                    user_content = prompt_text  # fall through into the turn below
                else:
                    continue

            turn = asyncio.create_task(
                process_user_turn(
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
            )
            current_turn["task"] = turn
            try:
                messages = await turn
            except asyncio.CancelledError:
                # Ctrl+C: keep the pre-turn history, like an Esc cancel.
                ui.cancel_voice_turn()
                ui.print_warning("turn interrupted (Ctrl+C)")
            finally:
                current_turn["task"] = None
    finally:
        catalog_task.cancel()
        mcp_task.cancel()
        try:
            await mcp_manager.stop()
        except Exception:
            logger.exception("MCP manager shutdown failed")


def cli_main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        from src.llm.tools.mcp.cli import run_mcp_cli

        raise SystemExit(run_mcp_cli(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "skills":
        from src.llm.tools.skills.cli import run_skills_cli

        raise SystemExit(run_skills_cli(sys.argv[2:]))
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
