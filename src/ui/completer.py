"""Inline completion for slash commands and their arguments.

Wired into the prompt session with `complete_while_typing`, so suggestions
appear as the user types: command names after `/`, model ids for `/model`
and `/models` (tool-capable catalog entries plus the curated list, `:batch`
variants excluded — the same pool `/model` accepts), effort levels for
`/effort` and the `/model <id> <effort>` suffix, and log levels for
`/log-level`.
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
    "/sound": "toggle the completion sound",
    "/log-level": "show or change log level",
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
        if command == "/sound":
            yield from self._option_completions(word, ("on", "off"))
            return
        if command in ("/log-level", "/loglevel"):
            levels = tuple(sorted(level.lower() for level in app_logging.VALID_LEVELS))
            yield from self._option_completions(word, levels)

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
