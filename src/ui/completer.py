"""Inline completion for slash commands and their arguments.

Wired into the prompt session with `complete_while_typing`, so suggestions
appear as the user types: command names after `/`, model ids for `/model`
and `/models` (tool-capable catalog entries plus the curated list, `:batch`
variants excluded — the same pool `/model` accepts), effort levels for
`/effort` and the `/model <id> <effort>` suffix, log levels for
`/log-level`, and registry tool names (built-in and MCP) for `/tool`.
"""

from prompt_toolkit.completion import Completer, Completion

from src.config.logging import app_logging
from src.ui.commands import EFFORT_LEVELS

COMMANDS: dict[str, str] = {
    "/help": "show help",
    "/models": "list or search models",
    "/model": "switch model (and effort)",
    "/effort": "reasoning effort",
    "/temperature": "sampling temperature",
    "/paste": "attach a clipboard image",
    "/copy": "copy the last answer / a code block",
    "/sound": "toggle the completion sound",
    "/voice": "voice mode (push-to-talk + TTS)",
    "/check": "check & request macOS permissions",
    "/log-level": "show or change log level",
    "/mcp": "MCP servers: status / reconnect / login / logout",
    "/tool": "pin a tool for the next message",
    "/clear": "clear screen and history",
    "/quit": "exit",
}
MAX_MODEL_COMPLETIONS = 20


class CommandCompleter(Completer):
    def __init__(self, ui) -> None:
        self.ui = ui

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        if " " not in text:
            needle = text.lower()
            for command, description in COMMANDS.items():
                if command.startswith(needle):
                    yield Completion(command, start_position=-len(text), display_meta=description)
            for command, description in self._mcp_prompt_commands():
                if command.startswith(needle):
                    yield Completion(command, start_position=-len(text), display_meta=description)
            return

        command, _, argument = text.partition(" ")
        yield from self._argument_completions(command.lower(), argument)

    def _argument_completions(self, command: str, argument: str):
        word = argument.rsplit(" ", 1)[-1]

        if command == "/models" or (command == "/model" and " " not in argument.strip()):
            yield from self._model_completions(word)
            return
        if command == "/model":  # second word: the optional effort suffix
            yield from self._option_completions(word, (*EFFORT_LEVELS, "none"))
            return
        if command == "/effort":
            yield from self._option_completions(word, (*EFFORT_LEVELS, "none"))
            return
        if command in ("/temperature", "/temp"):
            yield from self._option_completions(word, ("none",))
            return
        if command in ("/sound", "/voice"):
            yield from self._option_completions(word, ("on", "off"))
            return
        if command == "/copy":
            yield from self._option_completions(word, ("code",))
            return
        if command in ("/log-level", "/loglevel"):
            levels = tuple(sorted(level.lower() for level in app_logging.VALID_LEVELS))
            yield from self._option_completions(word, levels)
            return
        if command == "/mcp":
            first, _, rest = argument.partition(" ")
            if not rest and " " not in argument:
                yield from self._option_completions(word, ("reconnect", "login", "logout"))
                return
            yield from self._option_completions(word, tuple(self._mcp_server_names()))
            return
        if command == "/tool":
            yield from self._tool_name_completions(word)
            return

    def _model_completions(self, word: str):
        matched, _ = self.ui.commands.catalog_matches(word)
        curated = [m for m in self.ui.available_models if word.lower() in m.lower()]
        offered: list[str] = []
        for model_id in [*curated, *matched]:
            if model_id not in offered:
                offered.append(model_id)
        for model_id in offered[:MAX_MODEL_COMPLETIONS]:
            yield Completion(
                model_id,
                start_position=-len(word),
                display_meta=self.ui.commands.model_summary(model_id) or "",
            )

    @staticmethod
    def _option_completions(word: str, options: tuple[str, ...]):
        needle = word.lower()
        for option in options:
            if option.startswith(needle):
                yield Completion(option, start_position=-len(word))

    @staticmethod
    def _mcp_prompt_commands() -> list[tuple[str, str]]:
        try:
            from src.llm.tools.mcp.manager import get_mcp_manager

            return get_mcp_manager().prompt_commands()
        except Exception:
            return []

    @staticmethod
    def _mcp_server_names() -> list[str]:
        try:
            from src.llm.tools.mcp.manager import get_mcp_manager

            return [status.name for status in get_mcp_manager().statuses()]
        except Exception:
            return []

    @staticmethod
    def _tool_name_completions(word: str):
        try:
            from src.llm.tools import get_registry

            registry = get_registry()
        except Exception:
            return
        needle = word.lower()
        for tool_obj in registry.all_tools():
            if needle in tool_obj.name.lower():
                yield Completion(tool_obj.name, start_position=-len(word))
