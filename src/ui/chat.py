import sys
from contextlib import contextmanager

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.theme import Theme
from src.config.logging import app_logging

theme = Theme({
    "user.prompt": "bold cyan",
    "assistant.name": "bold magenta",
    "tool.name": "bold yellow",
    "info": "dim",
    "success": "bold green",
    "error": "bold red",
})


class ChatUI:
    def __init__(self, model: str, available_models: list[str] | None = None):
        self.console = Console(theme=theme)
        self.model = model
        self.available_models = available_models or []
        self.prompt_session: PromptSession | None = None

    def _get_prompt_session(self) -> PromptSession:
        if self.prompt_session is None:
            self.prompt_session = PromptSession(history=InMemoryHistory())
        return self.prompt_session

    def print_welcome(self):
        self.console.clear()
        self.console.print()
        title = Text("  Right Code  ", style="bold white on magenta")
        self.console.print(title)
        self.console.print(
            f"  model: {self.model}", style="info"
        )
        self.console.print(
            "  type /help for commands, /quit to exit", style="info"
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

        if name == "ls":
            return f"List {args.get('path', '.')}", ""
        if name == "read_file":
            return f"Read {args.get('file_path', '')}", ""
        if name == "write_file":
            return f"Write {args.get('file_path', '')}", ""
        if name == "edit_file":
            return f"Edit {args.get('file_path', '')}", ""
        if name == "execute":
            return "Run", args.get("command", "")
        if name == "web_search":
            return "Fetch", args.get("url", "")

        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return name, args_str

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

    def print_response(self, messages: list[HumanMessage | AIMessage | ToolMessage]):
        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        label, detail = self._format_tool_call(tc)
                        self.console.print(
                            f"  ● {label}",
                            style="tool.name",
                        )
                        if detail:
                            self.console.print(
                                f"    {detail}", style="info"
                            )
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
                content_str = str(msg.content)
                is_error = (
                    msg.status == "error"
                    or content_str.startswith("Error")
                    or content_str.startswith("Tool call failed")
                )
                marker = "✗" if is_error else "✓"
                style = "error" if is_error else "success"
                content_preview = content_str[:120]
                if len(content_str) > 120:
                    content_preview += "…"
                self.console.print(
                    f"  {marker} {content_preview}", style=style
                )

    async def get_tool_approval(self, tool_calls: list[dict]) -> dict:
        """Show pending tool calls and ask user to approve, edit, or reject.

        Returns an HITLResponse dict: {"decisions": [{"type": "approve"}, ...]}
        with one decision per tool call.
        """
        decisions = []
        for tc in tool_calls:
            label, detail = self._format_tool_call(tc)
            self.console.print(f"  ? {label}", style="tool.name")
            if detail:
                self.console.print(f"    {detail}", style="info")
            self.console.print(
                "    [Y] approve  [e] edit  [n] reject", style="info"
            )
            try:
                choice = (
                    await self._get_prompt_session().prompt_async("  > ")
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                decisions.append({"type": "reject"})
                continue
            if choice in ("n", "no", "reject"):
                decisions.append({"type": "reject"})
            elif choice in ("e", "edit"):
                decisions.append({"type": "edit", "edited_action": {"name": tc["name"], "args": tc["args"]}})
            else:
                decisions.append({"type": "approve"})
        self.console.print()
        return {"decisions": decisions}

    @contextmanager
    def loading(self, message: str = "thinking"):
        """Show a spinner animation while awaiting a response."""
        spinner = Spinner("dots", text=Text(f"  {message}...", style="info"))
        with Live(spinner, console=self.console, refresh_per_second=10, transient=True):
            yield

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

    def _print_models(self):
        self.console.print()
        for m in self.available_models:
            marker = "●" if m == self.model else "○"
            style = "success" if m == self.model else "info"
            self.console.print(f"  {marker} {m}", style=style)
        self.console.print()

    def _switch_model(self, model_name: str) -> str | None:
        if model_name in self.available_models:
            self.model = model_name
            self.console.print(
                f"  switched to {model_name}", style="success"
            )
            return None

        # try partial match
        matches = [m for m in self.available_models if model_name in m]
        if len(matches) == 1:
            self.model = matches[0]
            self.console.print(
                f"  switched to {matches[0]}", style="success"
            )
            return None

        self.console.print(
            f"  unknown model: {model_name}", style="error"
        )
        return None

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
