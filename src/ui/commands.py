"""Slash commands of the chat UI.

Every `/command` the user can type lives here; `ChatUI.handle_command`
delegates to `CommandHandler.handle`, which returns `"clear"` when the main
loop must drop the conversation history and `None` otherwise. The handler
reads and mutates the session state the `ChatUI` instance owns (current
model, reasoning effort, temperature, model catalog).

Model selection is capability-aware: search and switching consult the
OpenRouter catalog's `supported_parameters`, so a model that positively
cannot call tools is never offered to an agent whose whole workflow is tool
calls. Unknown models (catalog empty or entry without a capability list)
are allowed — the catalog's absence must not lock the user out.
"""

import sys

from src.config.logging import app_logging

EFFORT_LEVELS = ("minimal", "low", "medium", "high")
CLEAR_WORDS = ("none", "off", "default")
MAX_LISTED_MATCHES = 8
MAX_SEARCH_RESULTS = 15


class CommandHandler:
    def __init__(self, ui) -> None:
        self.ui = ui

    @property
    def console(self):
        return self.ui.console

    def handle(self, text: str) -> str | None:
        stripped = text.strip()
        command, _, argument = stripped.partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command in ("/quit", "/exit", "/q"):
            self.ui.print_goodbye()
            sys.exit(0)
        if command == "/help":
            return self._print_help()
        if command == "/models":
            return self._print_models(argument)
        if command == "/model":
            return self._switch_model(argument) if argument else self._print_models("")
        if command == "/effort":
            return self._switch_effort(argument)
        if command in ("/temperature", "/temp"):
            return self._switch_temperature(argument)
        if command == "/paste":
            return self._paste_image()
        if command == "/sound":
            return self._toggle_sound(argument)
        if command in ("/log-level", "/loglevel"):
            return self._switch_log_level(argument) if argument else self._print_log_level()
        if command == "/clear":
            self.ui.pending_images.clear()
            self.console.clear()
            self.ui.print_welcome()
            return "clear"

        self.console.print(f"  unknown command: {command} — try /help", style="error")
        return None

    # ------------------------------------------------------------------ help

    def _print_help(self) -> None:
        commands = [
            ("/help", "show this help"),
            ("/models [query]", "list models, or search the OpenRouter catalog"),
            ("/model <name> [effort]", "switch model (and optionally reasoning effort)"),
            ("/effort [level]", f"reasoning effort: {', '.join(EFFORT_LEVELS)}, or none"),
            ("/temperature [value]", "sampling temperature 0..2, or none"),
            ("/paste", "attach an image from the clipboard (also Ctrl+V in many terminals)"),
            ("/sound [on|off]", "toggle the completion sound"),
            ("/log-level [name]", "show or change the log level"),
            ("/clear", "clear screen and history"),
            ("/quit", "exit"),
        ]
        self.console.print()
        for command, description in commands:
            self.console.print(f"  {command:<24} {description}", style="info")
        self.console.print()

    # ---------------------------------------------------------------- models

    def model_summary(self, model_id: str) -> str | None:
        """`ctx 1,048,576 · $0.075/M in, $0.30/M out · reasoning` for a known model."""
        info = self.ui.model_catalog.get(model_id)
        if info is None:
            return None
        parts = []
        if info.context_length:
            parts.append(f"ctx {info.context_length:,}")
        if info.prompt_price is not None and info.completion_price is not None:
            parts.append(f"${info.prompt_price * 1e6:.3g}/M in, " f"${info.completion_price * 1e6:.3g}/M out")
        if info.supports_reasoning:
            parts.append("reasoning")
        return " · ".join(parts) or None

    def _model_line(self, model_id: str) -> str:
        summary = self.model_summary(model_id)
        return f"{model_id}  ({summary})" if summary else model_id

    def _print_models(self, query: str) -> None:
        if query:
            return self._search_models(query)
        self.console.print()
        listed = list(self.ui.available_models)
        if self.ui.model not in listed:
            listed.append(self.ui.model)
        for model_id in listed:
            marker = "●" if model_id == self.ui.model else "○"
            style = "success" if model_id == self.ui.model else "info"
            self.console.print(f"  {marker} {self._model_line(model_id)}", style=style)
        self.console.print(f"  {self.ui.settings_line()}", style="info")
        hint = "/models <query> searches the OpenRouter catalog (tool-capable models only)"
        if not self.ui.model_catalog:
            hint += " (catalog not loaded)"
        self.console.print(f"  {hint}", style="info")
        self.console.print()

    def catalog_matches(self, needle: str) -> tuple[list[str], int]:
        """Tool-capable catalog ids matching `needle`, plus how many were hidden.

        Matches against the id and the human-readable name; models the
        catalog positively marks as unable to call tools are dropped and
        counted, so the user learns why a model is not offered.
        """
        needle = needle.lower()
        matched: list[str] = []
        hidden = 0
        for model_id, info in self.ui.model_catalog.items():
            if model_id.endswith(":batch"):
                continue  # batch endpoints are not for interactive chat
            if needle not in model_id.lower() and needle not in info.name.lower():
                continue
            if info.lacks_tools:
                hidden += 1
                continue
            matched.append(model_id)
        return sorted(matched), hidden

    def _search_models(self, query: str) -> None:
        if not self.ui.model_catalog:
            self.console.print("  OpenRouter catalog not loaded — cannot search", style="error")
            return None
        matched, hidden = self.catalog_matches(query)
        self.console.print()
        if not matched:
            self.console.print(f"  nothing matches {query!r}", style="info")
        for model_id in matched[:MAX_SEARCH_RESULTS]:
            self.console.print(f"  {self._model_line(model_id)}", style="info")
        if len(matched) > MAX_SEARCH_RESULTS:
            self.console.print(f"  … and {len(matched) - MAX_SEARCH_RESULTS} more", style="info")
        if hidden:
            self.console.print(f"  ({hidden} match(es) hidden: no tool-call support)", style="info")
        self.console.print()

    # ----------------------------------------------------------- model switch

    def _split_effort_suffix(self, argument: str) -> tuple[str, str | None]:
        """`gpt-5.1 high` -> ("gpt-5.1", "high"); no suffix -> (argument, None)."""
        head, _, tail = argument.rpartition(" ")
        if head and (tail.lower() in EFFORT_LEVELS or tail.lower() in CLEAR_WORDS):
            return head.strip(), tail.lower()
        return argument, None

    def _switch_model(self, argument: str) -> None:
        model_name, effort = self._split_effort_suffix(argument)

        if model_name in self.ui.available_models or model_name in self.ui.model_catalog:
            return self._apply_model(model_name, effort)

        # Partial match over the curated list and the tool-capable catalog.
        needle = model_name.lower()
        curated = {m for m in self.ui.available_models if needle in m.lower()}
        from_catalog, hidden = self.catalog_matches(model_name)
        candidates = sorted(curated | set(from_catalog))

        if len(candidates) == 1:
            return self._apply_model(candidates[0], effort)
        if candidates:
            self.console.print(f"  {len(candidates)} models match {model_name!r}:", style="info")
            for candidate in candidates[:MAX_LISTED_MATCHES]:
                self.console.print(f"    {self._model_line(candidate)}", style="info")
            if len(candidates) > MAX_LISTED_MATCHES:
                self.console.print(f"    … and {len(candidates) - MAX_LISTED_MATCHES} more", style="info")
            return None
        if hidden:
            self.console.print(
                f"  {hidden} model(s) match {model_name!r} but cannot call tools — this agent needs tool support",
                style="error",
            )
            return None

        if not self.ui.model_catalog:
            self.ui.model = model_name
            self._apply_effort(effort)
            self.console.print(
                f"  switched to {model_name} " "(not verified — OpenRouter catalog unavailable)",
                style="success",
            )
            return None

        self.console.print(f"  unknown model: {model_name}", style="error")
        return None

    def _apply_model(self, model_name: str, effort: str | None = None) -> None:
        info = self.ui.model_catalog.get(model_name)
        if info is not None and info.lacks_tools and model_name not in self.ui.available_models:
            self.console.print(
                f"  {model_name} cannot call tools — this agent needs tool support",
                style="error",
            )
            return None
        self.ui.model = model_name
        self._apply_effort(effort)
        self.console.print(f"  switched to {self._model_line(model_name)}", style="success")
        self.console.print(f"  {self.ui.settings_line()}", style="info")
        return None

    # ------------------------------------------------------- effort and temp

    def _apply_effort(self, effort: str | None) -> None:
        if effort is None:
            return
        self.ui.reasoning_effort = None if effort in CLEAR_WORDS else effort

    def _switch_effort(self, argument: str) -> None:
        if not argument:
            self.console.print(
                f"  reasoning effort: {self.ui.reasoning_effort or 'default'} "
                f"(choose {', '.join(EFFORT_LEVELS)}, or none)",
                style="info",
            )
            if self.ui.reasoning_effort is None:
                self.console.print(
                    "  default sends no effort parameter — the provider decides "
                    "(OpenAI-style reasoning models default to medium)",
                    style="info",
                )
            return None

        value = argument.lower()
        if value in CLEAR_WORDS:
            self.ui.reasoning_effort = None
            self.console.print("  reasoning effort reset to default", style="success")
            return None
        if value not in EFFORT_LEVELS:
            self.console.print(
                f"  invalid effort: {argument} (choose {', '.join(EFFORT_LEVELS)}, or none)", style="error"
            )
            return None

        info = self.ui.model_catalog.get(self.ui.model)
        if info is not None and info.lacks_reasoning:
            self.console.print(f"  {self.ui.model} does not support reasoning effort", style="error")
            return None
        self.ui.reasoning_effort = value
        self.console.print(f"  reasoning effort set to {value}", style="success")
        return None

    def _switch_temperature(self, argument: str) -> None:
        if not argument:
            current = "default" if self.ui.temperature is None else f"{self.ui.temperature:g}"
            self.console.print(f"  temperature: {current} (choose 0..2, or none)", style="info")
            if self.ui.temperature is None:
                info = self.ui.model_catalog.get(self.ui.model)
                if info is not None and info.default_temperature is not None:
                    self.console.print(
                        f"  default sends no temperature — {self.ui.model} uses {info.default_temperature:g}",
                        style="info",
                    )
                else:
                    self.console.print(
                        "  default sends no temperature — the provider decides (typically 1.0)",
                        style="info",
                    )
            return None

        if argument.lower() in CLEAR_WORDS:
            self.ui.temperature = None
            self.console.print("  temperature reset to default", style="success")
            return None
        try:
            value = float(argument)
        except ValueError:
            self.console.print(f"  invalid temperature: {argument} (choose 0..2, or none)", style="error")
            return None
        if not 0 <= value <= 2:
            self.console.print(f"  temperature must be between 0 and 2, got {value:g}", style="error")
            return None

        info = self.ui.model_catalog.get(self.ui.model)
        if info is not None and info.lacks_temperature:
            self.console.print(f"  {self.ui.model} does not support temperature", style="error")
            return None
        self.ui.temperature = value
        self.console.print(f"  temperature set to {value:g}", style="success")
        return None

    # --------------------------------------------------------------- images

    def _paste_image(self) -> None:
        if not self.ui.attach_clipboard_image():
            self.console.print("  no image in the clipboard", style="error")
            return None
        image = self.ui.pending_images[-1]
        self.console.print(
            f"  image attached ({image['width']}×{image['height']}) — "
            f"{len(self.ui.pending_images)} image(s) will go with your next message",
            style="success",
        )
        return None

    def _toggle_sound(self, argument: str) -> None:
        value = argument.lower()
        if value in ("on", "off"):
            self.ui.sound_enabled = value == "on"
        elif value:
            self.console.print(f"  invalid value: {argument} (use /sound, /sound on, or /sound off)", style="error")
            return None
        else:
            self.ui.sound_enabled = not self.ui.sound_enabled
        state = "on" if self.ui.sound_enabled else "off"
        self.console.print(f"  completion sound {state}", style="success")
        return None

    # -------------------------------------------------------------- logging

    def _print_log_level(self) -> None:
        self.console.print(f"  current log level: {app_logging.get_level()}", style="info")
        return None

    def _switch_log_level(self, level_name: str) -> None:
        try:
            new_level = app_logging.set_level(level_name)
        except ValueError as exc:
            self.console.print(f"  {exc}", style="error")
            return None

        self.console.print(f"  log level set to {new_level}", style="success")
        return None
