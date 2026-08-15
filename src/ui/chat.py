import os
import sys
from contextlib import contextmanager

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from src.ui.stream import TurnStream
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from src.config.logging import app_logging
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

    def set_model_catalog(self, catalog: dict[str, ModelInfo] | None) -> None:
        """Install OpenRouter metadata used by /models, /model, and the usage footer."""
        self.model_catalog = dict(catalog or {})

    def _get_prompt_session(self) -> PromptSession:
        if self.prompt_session is None:
            self.prompt_session = PromptSession(history=InMemoryHistory())
        return self.prompt_session

    def print_welcome(self):
        self.console.clear()
        info = Table.grid(padding=(0, 2))
        info.add_column(style="info", justify="right", no_wrap=True)
        info.add_column()
        info.add_row("model", self.model)
        info.add_row("cwd", os.getcwd())
        info.add_row("logs", "logs.log")
        info.add_row("vision", "nvidia/LocateAnything-3B · loads in background")
        info.add_row("", Text("/help for commands · /quit to exit", style="info"))
        self.console.print()
        self.console.print(
            Panel(
                info,
                title="[bold magenta]✻ Right Code[/]",
                title_align="left",
                border_style="magenta",
                box=box.ROUNDED,
                padding=(0, 2),
                expand=False,
            )
        )
        self.console.print()

    async def get_input(self) -> str:
        try:
            return await self._get_prompt_session().prompt_async("> ")
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return "/quit"

    def handle_command(self, text: str) -> str | None:
        stripped = text.strip()
        cmd = stripped.lower()

        if cmd in ("/quit", "/exit", "/q"):
            self.print_goodbye()
            sys.exit(0)

        if cmd == "/help":
            self._print_help()
            return None

        if cmd == "/models":
            self._print_models()
            return None

        if cmd.startswith("/model "):
            return self._switch_model(text.strip()[7:].strip())

        if cmd in ("/log-level", "/loglevel"):
            self.console.print(
                f"  current log level: {app_logging.get_level()}",
                style="info",
            )
            return None

        if cmd.startswith("/log-level ") or cmd.startswith("/loglevel "):
            _, level_name = stripped.split(maxsplit=1)
            return self._switch_log_level(level_name)

        if cmd == "/clear":
            self.console.clear()
            self.print_welcome()
            return "clear"

        return None

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

    def print_error(self, error: Exception):
        self.console.print(f"  error: {error}", style="error")

    def print_warning(self, message: str):
        self.console.print(f"  warning: {message}", style="error")

    def print_goodbye(self):
        self.console.print("\n  goodbye!\n", style="info")

    def _print_help(self):
        commands = [
            ("/help", "show this help"),
            ("/models", "list available models"),
            ("/model <name>", "switch model"),
            ("/log-level", "show current log level"),
            ("/log-level <name>", "change log level"),
            ("/clear", "clear screen"),
            ("/quit", "exit"),
        ]
        self.console.print()
        for cmd, desc in commands:
            self.console.print(f"  {cmd:<20} {desc}", style="info")
        self.console.print()

    def _model_summary(self, model_id: str) -> str | None:
        """`ctx 1,048,576 · $0.075/M in, $0.30/M out` for a known model."""
        info = self.model_catalog.get(model_id)
        if info is None:
            return None
        parts = []
        if info.context_length:
            parts.append(f"ctx {info.context_length:,}")
        if info.prompt_price is not None and info.completion_price is not None:
            parts.append(f"${info.prompt_price * 1e6:.3g}/M in, " f"${info.completion_price * 1e6:.3g}/M out")
        return " · ".join(parts) or None

    def _model_line(self, model_id: str) -> str:
        summary = self._model_summary(model_id)
        return f"{model_id}  ({summary})" if summary else model_id

    def _print_models(self):
        self.console.print()
        listed = list(self.available_models)
        if self.model not in listed:
            listed.append(self.model)
        for m in listed:
            marker = "●" if m == self.model else "○"
            style = "success" if m == self.model else "info"
            self.console.print(f"  {marker} {self._model_line(m)}", style=style)
        hint = "/model <id> also accepts any OpenRouter model id"
        if not self.model_catalog:
            hint += " (OpenRouter catalog not loaded)"
        self.console.print(f"  {hint}", style="info")
        self.console.print()

    def _apply_model(self, model_name: str) -> None:
        self.model = model_name
        self.console.print(f"  switched to {self._model_line(model_name)}", style="success")

    def _switch_model(self, model_name: str) -> str | None:
        if model_name in self.available_models or model_name in self.model_catalog:
            self._apply_model(model_name)
            return None

        # try partial match over the curated list and the OpenRouter catalog
        needle = model_name.lower()
        candidates = sorted(
            {m for m in self.available_models if needle in m.lower()}
            | {m for m in self.model_catalog if needle in m.lower()}
        )
        if len(candidates) == 1:
            self._apply_model(candidates[0])
            return None
        if candidates:
            self.console.print(f"  {len(candidates)} models match {model_name!r}:", style="info")
            for candidate in candidates[:8]:
                self.console.print(f"    {self._model_line(candidate)}", style="info")
            if len(candidates) > 8:
                self.console.print(f"    … and {len(candidates) - 8} more", style="info")
            return None

        if not self.model_catalog:
            self.model = model_name
            self.console.print(
                f"  switched to {model_name} " "(not verified — OpenRouter catalog unavailable)",
                style="success",
            )
            return None

        self.console.print(f"  unknown model: {model_name}", style="error")
        return None

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
    ) -> None:
        """One dim footer line: context fill, turn tokens and cost, session totals."""
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
        session_part += f" ({details})"
        parts.append(session_part)

        self.console.print("  " + " · ".join(parts), style="info")

    def _switch_log_level(self, level_name: str) -> str | None:
        try:
            new_level = app_logging.set_level(level_name)
        except ValueError as exc:
            self.console.print(f"  {exc}", style="error")
            return None

        self.console.print(
            f"  log level set to {new_level}",
            style="success",
        )
        return None
