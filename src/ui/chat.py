import asyncio
import os
import time
from contextlib import contextmanager

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from src.ui.clipboard import encode_image, grab_clipboard_image
from src.ui.commands import CommandHandler
from src.ui.completer import CommandCompleter
from src.ui.sound import play_done_sound
from src.ui.stream import TurnStream
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from src.llm.middlewares import scrub_text
from src.llm.providers import ModelInfo
from src.llm.types import TurnUsage
from src.llm.statistics import SessionUsage
from src.llm.utils import format_duration, format_money

theme = Theme(
    {
        "user.prompt": "bold cyan",
        "assistant.name": "bold magenta",
        "tool.name": "bold yellow",
        "info": "dim",
        "success": "bold green",
        "error": "bold red",
    }
)


class ChatUI:
    def __init__(self, model: str, available_models: list[str] | None = None):
        self.console = Console(theme=theme)
        self.model = model
        self.available_models = available_models or []
        self.model_catalog: dict[str, ModelInfo] = {}
        self.prompt_session: PromptSession | None = None
        self.reasoning_effort: str | None = None
        self.temperature: float | None = None
        self.pending_images: list[dict] = []
        self.sound_enabled = True
        self.voice = None  # VoiceController once /voice enables it
        self.model_status: dict[str, tuple[str, float, str | None]] = {}  # name → (state, since, detail)
        self._type_ahead = ""  # keys typed during a turn, returned to the next prompt
        self.commands = CommandHandler(self)

    def set_model_catalog(self, catalog: dict[str, ModelInfo] | None) -> None:
        """Install OpenRouter metadata used by /models, /model, and the usage footer."""
        self.model_catalog = dict(catalog or {})

    def _get_prompt_session(self) -> PromptSession:
        if self.prompt_session is None:
            self.prompt_session = PromptSession(
                history=InMemoryHistory(),
                completer=CommandCompleter(self),
                complete_while_typing=True,
                key_bindings=self._build_key_bindings(),
                rprompt=self._rprompt_status,
                # Auto-redraw keeps the loading/recording stopwatches ticking
                # without any thread having to invalidate the app.
                refresh_interval=0.5,
            )
        return self.prompt_session

    READY_LINGER_SECONDS = 5.0

    def set_model_status(self, name: str, state: str, detail: str | None = None) -> None:
        """Report a model's loading state ("loading" / "ready" / "failed").

        Called from loader threads; rendered live at the right of the prompt.
        `detail` carries short download progress ("↓ 1.2/6.4 GB 19%"); updates
        within the same state keep the original stopwatch start.
        """
        previous = self.model_status.get(name)
        since = previous[1] if previous is not None and previous[0] == state else time.monotonic()
        self.model_status[name] = (state, since, detail)

    def _model_status_text(self) -> str:
        parts = []
        for name, (state, since, detail) in list(self.model_status.items()):
            elapsed = time.monotonic() - since
            if state == "loading":
                progress = f" {detail}" if detail else ""
                parts.append(f"⏳ {name}{progress} {elapsed:.0f}s")
            elif state == "failed":
                parts.append(f"✗ {name}")
            elif state == "ready" and elapsed < self.READY_LINGER_SECONDS:
                parts.append(f"✓ {name}")
        return " · ".join(parts)

    def _rprompt_status(self) -> str:
        """Right-side prompt status: voice recording/speaking + model loading."""
        voice = self.voice.status() if self.voice is not None else ""
        models = self._model_status_text()
        return " · ".join(part for part in (voice, models) if part)

    def _voice_welcome_line(self) -> str:
        if self.voice is not None:
            key = self.voice.key_spec
        else:
            try:
                from src.config.settings import settings

                if not settings.enable_voice_model:
                    return "voice off — set ENABLE_VOICE_MODEL=1 in .env for push-to-talk and spoken replies"
                key = settings.voice_ptt_key
            except Exception:
                key = "alt_r"
        try:
            from src.voice.hotkey import describe_hotkey

            key = describe_hotkey(key)
        except Exception:
            pass
        replies = "on" if self.voice_active else "off"
        return f"push-to-talk {key} · spoken replies {replies} (/voice)"

    @property
    def voice_active(self) -> bool:
        """True when the agent answers aloud (drives the TTS prompt suffix)."""
        return self.voice is not None and self.voice.speak_replies

    def start_voice_input(self) -> None:
        """Start always-on push-to-talk; refused while ENABLE_VOICE_MODEL is off."""
        from src.config.settings import settings

        if not settings.enable_voice_model:
            raise RuntimeError("voice is disabled — set ENABLE_VOICE_MODEL=1 in .env and restart")
        if self.voice is None:
            from src.ui.voice import VoiceController

            self.voice = VoiceController(self)
        self.voice.start_input()

    def set_voice_replies(self, on: bool) -> None:
        """`/voice on|off`: whether replies are spoken; push-to-talk stays on."""
        if self.voice is None:
            if not on:
                return  # nothing to silence, and no reason to start the models
            self.start_voice_input()  # retry input too if startup failed
        self.voice.set_speaking(on)

    def finish_voice_turn(self) -> None:
        """Flush the spoken tail of a finished turn; safe to call anytime."""
        if self.voice is not None:
            self.voice.finish_turn()

    def _build_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("c-v")
        def paste(event) -> None:
            # Terminals that paste on Ctrl+V themselves never deliver the key
            # here — /paste is the reliable fallback. When we do get it: an
            # image in the clipboard beats text.
            if self.attach_clipboard_image():
                event.current_buffer.insert_text(f"[image {len(self.pending_images)}] ")
                return
            import pyperclip

            event.current_buffer.insert_text(pyperclip.paste() or "")

        return bindings

    def attach_clipboard_image(self) -> bool:
        """Queue the clipboard's image for the next message; False when none."""
        image = grab_clipboard_image()
        if image is None:
            return False
        self.pending_images.append(encode_image(image))
        return True

    def take_user_content(self, text: str) -> str | list[dict]:
        """The next message's content: plain text, or text plus queued images."""
        if not self.pending_images:
            return text
        blocks: list[dict] = []
        if text.strip():
            blocks.append({"type": "text", "text": text})
        for image in self.pending_images:
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image['mime_type']};base64,{image['base64_data']}"},
                }
            )
        self.pending_images.clear()
        return blocks

    def settings_line(self) -> str:
        """`effort high · temperature default (1.0)` for the session settings.

        "default" means the parameter is not sent and the provider decides;
        when the OpenRouter catalog publishes the model's own default, it is
        shown in parentheses.
        """
        info = self.model_catalog.get(self.model)
        effort = self.reasoning_effort or "default"
        if self.temperature is not None:
            temperature = f"{self.temperature:g}"
        elif info is not None and info.default_temperature is not None:
            temperature = f"default ({info.default_temperature:g})"
        else:
            temperature = "default"
        return f"effort {effort} · temperature {temperature}"

    def print_welcome(self):
        # No screen clearing here: the banner joins the normal output flow,
        # so the terminal never jumps and nothing scrolls out of reach when
        # the input gets focus. /clear wipes the screen itself before calling.
        info = Table.grid(padding=(0, 2))
        info.add_column(style="info", justify="right", no_wrap=True)
        info.add_column()
        info.add_row("model", self.model)
        info.add_row("settings", self.settings_line())
        info.add_row("cwd", os.getcwd())
        info.add_row("logs", "logs.log")
        info.add_row("vision", "nvidia/LocateAnything-3B · progress shown right of the prompt")
        info.add_row("voice", self._voice_welcome_line())
        body = Group(
            Text("✻  Chattler.AI Open Source", style="bold magenta"),
            Text(""),
            info,
            Text(""),
            Text("/help for commands · /quit to exit", style="info"),
        )
        self.console.print()
        self.console.print(
            Panel(
                body,
                border_style="magenta",
                box=box.ROUNDED,
                padding=(0, 2),
                expand=False,
            )
        )
        self.console.print()

    async def get_input(self) -> str:
        if self.voice is not None:
            pending = self.voice.take_pending_text()
            if pending:
                # A transcript that finished while no prompt was up; echo it
                # the way typed input would look and use it as the message.
                self.console.print(f"> {pending}", style="user.prompt", markup=False, highlight=False)
                return pending
        default, self._type_ahead = self._type_ahead, ""
        try:
            return await self._get_prompt_session().prompt_async("> ", default=default)
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return "/quit"

    CANCEL_GRACE_SECONDS = 1.5

    async def run_cancellable(self, coroutine, watcher=None):
        """Await a turn while watching the console for Esc; Esc cancels it.

        Raises `TurnCancelled` on Esc. Keys the watcher consumed come back as
        type-ahead in the next prompt. Without a console watcher (redirected
        stdin) the turn simply runs to completion. A cancel can only take
        effect when the turn's current tool call returns — a call stuck in an
        executor thread cannot be aborted — so if it takes longer than the
        grace period, the wait is explained instead of looking dead.
        """
        from src.ui.interrupt import EscapeWatcher, TurnCancelled

        watcher = watcher or EscapeWatcher.create()
        if watcher is None:
            return await coroutine
        task = asyncio.ensure_future(coroutine)
        try:
            with watcher:
                while True:
                    done, _ = await asyncio.wait({task}, timeout=0.1)
                    if done:
                        return task.result()
                    if watcher.pressed.is_set():
                        break
        finally:
            self._type_ahead += watcher.typed_text
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=self.CANCEL_GRACE_SECONDS)
        if not done:
            self.print_warning("cancelling — waiting for the running tool call to finish (Ctrl+C twice force-quits)")
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise TurnCancelled()

    def cancel_voice_turn(self) -> None:
        """Drop any buffered/playing speech of a cancelled turn."""
        if self.voice is not None:
            self.voice.cancel_turn()

    def handle_command(self, text: str) -> str | None:
        """Dispatch a slash command; returns "clear" when history must reset."""
        return self.commands.handle(text)

    def _format_tool_call(self, tc: dict) -> tuple[str, str]:
        """Return (label, detail) for a tool call."""
        name = tc["name"]
        args = tc["args"]

        if name == "search_tools":
            return f"Search tools · {args.get('query', '')}", ""
        if name == "get_tool":
            requested = args.get("names") or args.get("name") or ""
            if isinstance(requested, list):
                requested = ", ".join(str(item) for item in requested)
            return f"Read tool docs · {requested}", ""
        if name == "run_tools":
            return "Run tools script", args.get("code", "")

        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return name, args_str

    MAX_DETAIL_LINES = 6

    def _print_tool_call(self, tc: dict) -> None:
        label, detail = self._format_tool_call(tc)
        self.console.print(f"  ● {label}", style="tool.name", markup=False)
        if not detail:
            return
        lines = detail.splitlines()
        for line in lines[: self.MAX_DETAIL_LINES]:
            self.console.print(f"  │ {line}", style="info", markup=False, highlight=False)
        if len(lines) > self.MAX_DETAIL_LINES:
            self.console.print(
                f"  │ … +{len(lines) - self.MAX_DETAIL_LINES} more lines",
                style="info",
                markup=False,
            )

    def _print_tool_result(self, msg: ToolMessage, duration: float | None = None) -> None:
        content_str = str(msg.content)
        is_error = (
            msg.status == "error" or content_str.startswith("Error") or content_str.startswith("Tool call failed")
        )
        preview = scrub_text(" ".join(content_str.split()), 200)
        suffix = f" · {format_duration(duration)}" if duration is not None else ""
        self.console.print(
            f"  ⎿ {preview}{suffix}",
            style="error" if is_error else "info",
            markup=False,
            highlight=False,
        )

    def _render_ai_content(self, content: str | list[dict] | None) -> str:
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return ""

        text_parts: list[str] = []
        reasoning_parts: list[str] = []

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)
                continue

            if block_type == "reasoning":
                for summary in block.get("summary") or []:
                    if not isinstance(summary, dict):
                        continue
                    text = summary.get("text")
                    if isinstance(text, str) and text.strip():
                        reasoning_parts.append(text)

        if text_parts:
            return "\n\n".join(text_parts)

        if reasoning_parts:
            return "\n\n".join(reasoning_parts)

        return ""

    def has_visible_output(
        self,
        messages: list[HumanMessage | AIMessage | ToolMessage],
    ) -> bool:
        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    return True
                if self._render_ai_content(msg.content):
                    return True
            elif isinstance(msg, ToolMessage):
                if str(msg.content).strip():
                    return True
        return False

    def print_response(
        self,
        messages: list[HumanMessage | AIMessage | ToolMessage],
        skip_ids: set[str] | frozenset[str] = frozenset(),
    ):
        for msg in messages:
            if getattr(msg, "id", None) in skip_ids:
                continue  # already shown live by the turn stream
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        self._print_tool_call(tc)
                else:
                    rendered_content = self._render_ai_content(msg.content)
                    if not rendered_content:
                        continue
                    self.console.print()
                    md = Markdown(rendered_content)
                    self.console.print(
                        Panel(
                            md,
                            border_style="magenta",
                            padding=(0, 1),
                        )
                    )
                    self.console.print()

            elif isinstance(msg, ToolMessage):
                self._print_tool_result(msg)

    async def get_tool_approval(self, tool_calls: list[dict]) -> dict:
        """Show pending tool calls and ask user to approve, edit, or reject.

        Returns an HITLResponse dict: {"decisions": [{"type": "approve"}, ...]}
        with one decision per tool call.
        """
        decisions = []
        for tc in tool_calls:
            label, detail = self._format_tool_call(tc)
            self.console.print(f"  ? {label}", style="tool.name", markup=False)
            for line in detail.splitlines()[: self.MAX_DETAIL_LINES]:
                self.console.print(f"  │ {line}", style="info", markup=False, highlight=False)
            self.console.print("    [Y] approve  [e] edit  [n] reject", style="info")
            try:
                choice = (await self._get_prompt_session().prompt_async("  > ")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                decisions.append({"type": "reject"})
                continue
            if choice in ("n", "no", "reject"):
                decisions.append({"type": "reject"})
            elif choice in ("e", "edit"):
                decisions.append(
                    {
                        "type": "edit",
                        "edited_action": {"name": tc["name"], "args": tc["args"]},
                    }
                )
            else:
                decisions.append({"type": "approve"})
        self.console.print()
        return {"decisions": decisions}

    @contextmanager
    def turn_stream(self):
        """Live view of one running turn: stopwatch, tool events, streamed text.

        Yields a `TurnStream` whose `on_message`/`on_token` callbacks the LLM
        layer drives; whatever it printed live is recorded in `printed_ids`
        so `print_response` can skip it afterwards.
        """
        stream = TurnStream(self)
        with Live(stream, console=self.console, refresh_per_second=10, transient=True):
            yield stream

    def notify_done(self) -> None:
        """Play the completion sound, when enabled; in voice mode speech is the signal."""
        if self.voice_active:
            return
        if self.sound_enabled:
            play_done_sound()

    def print_error(self, error: Exception):
        self.console.print(f"  error: {error}", style="error")

    def print_warning(self, message: str):
        self.console.print(f"  warning: {message}", style="error")

    def print_goodbye(self):
        self.console.print("\n  goodbye!\n", style="info")

    @staticmethod
    def _context_bar(used: int, limit: int, width: int = 20) -> str:
        """A colored fill bar for context usage, green → yellow → red."""
        ratio = min(1.0, used / limit) if limit > 0 else 0.0
        filled = round(width * ratio)
        if used > 0:
            filled = max(filled, 1)
        filled = min(filled, width)
        if ratio < 0.7:
            color = "green"
        elif ratio < 0.9:
            color = "yellow"
        else:
            color = "red"
        return f"[{color}]{'█' * filled}[/]{'░' * (width - filled)}"

    def print_usage(
        self,
        turn: TurnUsage,
        model_info: ModelInfo | None,
        cost: float | None,
        session: SessionUsage,
        duration: float | None = None,
        saved: float | None = None,
    ) -> None:
        """One dim footer line: context fill, turn tokens and cost, cache
        savings (`saved` = dollars the prompt-cache reads knocked off the
        full input price), session totals."""
        if turn.calls == 0:
            self.console.print("  usage: provider reported no token counts", style="info")
            return

        limit = model_info.context_length if model_info else None
        if limit:
            percent = 100 * turn.context_tokens / limit
            bar = self._context_bar(turn.context_tokens, limit)
            context_part = f"ctx {bar} {turn.context_tokens:,}/{limit:,} ({percent:.1f}%)"
        else:
            context_part = f"ctx {turn.context_tokens:,} (limit unknown)"

        turn_part = f"turn {turn.input_tokens:,} in + {turn.output_tokens:,} out"
        turn_part += f" ({format_money(cost)})" if cost is not None else " (price unknown)"

        parts = [context_part, turn_part]

        if turn.cached_input_tokens or (saved is not None and saved < 0):
            share = 100 * turn.cached_input_tokens / turn.input_tokens if turn.input_tokens else 0.0
            cache_part = f"cache {turn.cached_input_tokens:,} read ({share:.0f}% of input)"
            if saved and saved > 0:
                cache_part += f", saved {format_money(saved)}"
            elif saved and saved < 0:
                # The turn wrote to the cache but read nothing back — the
                # write premium made caching a net cost this time.
                cache_part += f", writes cost {format_money(-saved)} extra"
            parts.append(cache_part)

        if duration is not None and duration > 0:
            parts.append(f"took {format_duration(duration)}")

        if turn.tool_calls or turn.script_tool_calls:
            tools_part = f"tools {turn.tool_calls}"
            if turn.script_tool_calls:
                tools_part += f" (+{turn.script_tool_calls} in scripts)"
            parts.append(tools_part)

        session_part = f"session {session.total_tokens:,} tokens"
        approx = "≈" if session.unpriced_turns else ""
        details = f"{approx}{format_money(session.cost)}"
        if session.duration > 0:
            details += f", {format_duration(session.duration)}"
        if session.saved > 0:
            details += f", saved {format_money(session.saved)}"
        elif session.saved < 0:
            details += f", cache overhead {format_money(-session.saved)}"
        session_part += f" ({details})"
        parts.append(session_part)

        self.console.print("  " + " · ".join(parts), style="info")
