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

import re
import sys
from dataclasses import dataclass

from src.config.logging import app_logging

EFFORT_LEVELS = ("minimal", "low", "medium", "high")
CODE_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)
CLEAR_WORDS = ("none", "off", "default")
MAX_LISTED_MATCHES = 8
MAX_SEARCH_RESULTS = 15
MCP_SUBCOMMANDS = ("reconnect", "login", "logout")
TOOL_DIRECTIVE_HEADER = (
    "[Tool directive: use the tool(s) below for this request — the user "
    "picked them explicitly. Contracts follow; no need for search_tools/get_tool.]"
)
# Literal style specs, not the app theme's "info"/"success"/"error" aliases:
# `run_mcp_action` is called with a plain console in tests (and could be
# handed any console), and a literal spec renders identically wherever the
# theme happens to define those names the same way ("error" -> "bold red").
MCP_INFO = "dim"
MCP_SUCCESS = "bold green"
MCP_ERROR = "bold red"


@dataclass(frozen=True)
class McpAction:
    """An `/mcp`-family outcome the main loop must act on outside the UI.

    `kind` is "reconnect" | "login" | "logout" | "prompt"; `argument` is the
    server name for the first three, or the full `/mcp__srv__prompt arg...`
    text for "prompt". `CommandHandler.handle` returns this instead of acting
    directly because reconnecting and fetching a prompt are async and touch
    the MCP manager, which the synchronous command handler does not own.
    """

    kind: str
    argument: str


def _map_prompt_arguments(words: list[str], arguments: list | None) -> dict[str, str]:
    """Whitespace-split words onto prompt arguments, positionally.

    Extra words beyond the argument count join into the last argument; with
    no declared arguments (or no words), the mapping is empty. The MCP SDK's
    `Prompt.arguments` defaults to `None` (not `[]`) for a no-argument
    prompt, so `None` is treated the same as an empty list here.
    """
    names = [argument.name for argument in (arguments or [])]
    mapping: dict[str, str] = {}
    for index, name in enumerate(names):
        if index >= len(words):
            break
        if index == len(names) - 1:
            mapping[name] = " ".join(words[index:])
        else:
            mapping[name] = words[index]
    return mapping


async def run_mcp_action(action: McpAction, manager, console) -> str | None:
    """Execute one `McpAction`; returns text to send as a user turn, or None.

    `reconnect`/`login`/`logout` never return text — they print their result
    and the main loop just continues. `prompt` fetches the rendered prompt
    text and returns it so the caller can feed it into a turn. Every failure
    is caught and printed rather than raised — a bad MCP server must not take
    down the REPL. `prompt_commands`/`find_prompt` can name prompts of a
    currently-disconnected server, so the prompt branch handles the resulting
    `ConnectionError` with a clear reconnect hint instead of a raw traceback.
    """
    try:
        if action.kind == "reconnect":
            status = await manager.reconnect(action.argument)
            style = MCP_SUCCESS if status.state.value == "connected" else MCP_ERROR
            console.print(f"  {action.argument}: {status.state.value}", style=style)
            return None

        if action.kind in ("login", "logout"):
            if action.kind == "login":
                # The browser flow blocks the REPL until consent comes back,
                # so say so before disappearing into it.
                console.print(f"  {action.argument}: opening a browser to authorize…", style=MCP_INFO)
                status = await manager.login(action.argument)
            else:
                status = await manager.logout(action.argument)
            style = MCP_SUCCESS if status.state.value == "connected" else MCP_ERROR
            detail = f" — {status.error}" if status.error else ""
            console.print(
                f"  {action.argument}: {status.state.value}{detail}",
                style=style,
                markup=False,
                highlight=False,
            )
            return None

        if action.kind == "prompt":
            command, _, rest = action.argument.partition(" ")
            found = manager.find_prompt(command)
            if found is None:
                console.print(f"  unknown MCP prompt command: {command}", style=MCP_ERROR)
                return None
            server, prompt = found
            arguments = _map_prompt_arguments(rest.split(), prompt.arguments)
            try:
                return await manager.get_prompt(server, prompt.name, arguments)
            except ConnectionError:
                console.print(
                    f"  server '{server}' is not connected — /mcp reconnect {server}",
                    style=MCP_ERROR,
                )
                return None

        console.print(f"  unknown MCP action: {action.kind}", style=MCP_ERROR)
        return None
    except Exception as error:
        console.print(f"  {error}", style=MCP_ERROR, markup=False, highlight=False)
        return None


class CommandHandler:
    def __init__(self, ui) -> None:
        self.ui = ui

    @property
    def console(self):
        return self.ui.console

    def handle(self, text: str) -> str | McpAction | None:
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
        if command == "/copy":
            return self._copy(argument)
        if command == "/sound":
            return self._toggle_sound(argument)
        if command == "/voice":
            return self._toggle_voice(argument)
        if command == "/check":
            return self._run_check()
        if command in ("/log-level", "/loglevel"):
            return self._switch_log_level(argument) if argument else self._print_log_level()
        if command == "/clear":
            self.ui.pending_images.clear()
            self.console.clear()
            self.ui.print_welcome()
            return "clear"
        if command == "/mcp":
            return self._mcp(argument)
        if command.startswith("/mcp__"):
            return self._mcp_prompt(stripped)
        if command == "/tool":
            return self._tool_pin(argument)

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
            ("/copy [code [n]]", "copy the last answer — or its n-th code block — to the clipboard"),
            ("/sound [on|off]", "toggle the completion sound"),
            ("/voice [on|off]", "spoken replies on/off (needs ENABLE_VOICE_MODEL=1)"),
            ("/check", "check macOS permissions and raise the missing consent dialogs"),
            ("/log-level [name]", "show or change the log level"),
            ("/mcp", "MCP servers: status, reconnect <name>, login/logout <name>"),
            ("/tool <name>", "pin a tool for the next message (its contract goes along)"),
            ("/clear", "clear screen and history"),
            ("/quit", "exit"),
        ]
        self.console.print()
        for command, description in commands:
            self.console.print(f"  {command:<24} {description}", style="info")
        self.console.print()

    # ----------------------------------------------------------- permissions

    def _run_check(self, platform: str | None = None) -> None:
        """`/check`: probe every OS permission the app needs and prompt for the missing ones."""
        from src.utils import permissions

        if (platform or sys.platform) != "darwin":
            self.console.print("  no OS permission setup needed on this platform", style="info")
            return None
        self.console.print("  checking macOS permissions — grant any dialogs that appear:", style="info")
        for status in permissions.check_permissions():
            if status.granted is True:
                icon, style = "✓", "success"
            elif status.granted is False:
                icon, style = "✗", "error"
            else:
                icon, style = "?", "info"
            line = f"  {icon} {status.name} — {status.purpose}"
            if status.granted is not True:
                line += f" · System Settings → {status.settings_pane}"
            self.console.print(line, style=style, markup=False, highlight=False)
        self.console.print(
            "  ? means the state cannot be read here — if the feature misbehaves, "
            "enable it in the pane above. After granting anything new, restart the app.",
            style="info",
        )
        return None

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

    # -------------------------------------------------------------- copying

    def _set_clipboard(self, text: str) -> bool:
        try:
            import pyperclip

            pyperclip.copy(text)
            return True
        except Exception as error:
            self.console.print(f"  clipboard unavailable: {error}", style="error")
            return False

    def _copy(self, argument: str) -> None:
        """`/copy`: the whole last answer; `/copy code [n]`: one of its code blocks."""
        answer = self.ui.last_answer
        if not answer:
            self.console.print("  nothing to copy yet — no answer in this session", style="error")
            return None
        kind, _, index_text = argument.partition(" ")
        if kind.lower() not in ("", "code"):
            self.console.print(
                "  usage: /copy — the whole last answer; /copy code [n] — its n-th code block",
                style="error",
            )
            return None
        if kind.lower() != "code":
            if self._set_clipboard(answer):
                self.console.print(f"  copied the last answer ({len(answer)} chars)", style="success")
            return None

        blocks = [block.strip("\n") for block in CODE_FENCE_RE.findall(answer) if block.strip()]
        if not blocks:
            self.console.print("  the last answer has no fenced code blocks", style="error")
            return None
        if index_text.strip():
            try:
                index = int(index_text)
            except ValueError:
                self.console.print(f"  invalid block number: {index_text}", style="error")
                return None
            if not 1 <= index <= len(blocks):
                self.console.print(
                    f"  block number out of range — the answer has {len(blocks)} code block(s)",
                    style="error",
                )
                return None
        elif len(blocks) == 1:
            index = 1
        else:
            self.console.print(f"  {len(blocks)} code blocks — pick one with /copy code <n>:", style="info")
            for number, block in enumerate(blocks, 1):
                first_line = block.strip().splitlines()[0]
                self.console.print(
                    f"    {number}. {first_line[:60]} ({len(block.splitlines())} lines)",
                    style="info",
                    markup=False,
                    highlight=False,
                )
            return None
        block = blocks[index - 1]
        if self._set_clipboard(block):
            self.console.print(f"  copied code block {index} ({len(block.splitlines())} lines)", style="success")
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

    def _toggle_voice(self, argument: str) -> None:
        value = argument.lower()
        if value not in ("", "on", "off"):
            self.console.print(f"  invalid value: {argument} (use /voice, /voice on, or /voice off)", style="error")
            return None
        turn_on = value == "on" or (not value and not self.ui.voice_active)
        try:
            self.ui.set_voice_replies(turn_on)
            if turn_on:
                self.console.print(
                    "  spoken replies on — the agent answers with voice " "(the TTS model warms up in the background)",
                    style="success",
                )
            else:
                self.console.print("  spoken replies off — push-to-talk stays active", style="success")
        except Exception as error:
            self.console.print(f"  voice mode failed: {error}", style="error")
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

    # -------------------------------------------------------------------- mcp

    _MCP_STATE_GLYPHS = {
        "connected": "●",
        "connecting": "○",
        "failed": "✗",
        "needs auth": "✗",
        "disconnected": "○",
    }
    _MCP_STATE_STYLES = {
        "connected": MCP_SUCCESS,
        "failed": MCP_ERROR,
        "needs auth": MCP_ERROR,
    }
    _MCP_HINT = (
        "  /mcp reconnect <name>, /mcp login <name>; " "manage servers with: uv run python -m src.main mcp add ..."
    )

    def _mcp(self, argument: str) -> McpAction | None:
        """`/mcp`: status table; `/mcp reconnect|login|logout <name>`: an action."""
        from src.llm.tools.mcp.manager import get_mcp_manager

        manager = get_mcp_manager()
        parts = argument.split(None, 1)
        if not parts:
            return self._render_mcp_status(manager)

        sub = parts[0].lower()
        if sub not in MCP_SUBCOMMANDS:
            self.console.print(
                f"  unknown /mcp subcommand: {sub} — try /mcp, /mcp reconnect <name>, /mcp login|logout <name>",
                style=MCP_ERROR,
            )
            return None

        name = parts[1].strip().split()[0] if len(parts) > 1 and parts[1].strip() else ""
        if not name:
            self.console.print(f"  usage: /mcp {sub} <name>", style=MCP_ERROR)
            return None

        known = {status.name for status in manager.statuses()}
        if name not in known:
            self.console.print(f"  unknown MCP server: {name}", style=MCP_ERROR)
            return None
        return McpAction(sub, name)

    def _render_mcp_status(self, manager) -> None:
        statuses = manager.statuses()
        self.console.print()
        if not statuses:
            self.console.print("  no MCP servers configured", style=MCP_INFO)
            self.console.print(
                "  manage servers with: uv run python -m src.main mcp add ...",
                style=MCP_INFO,
            )
            self.console.print()
            return None

        for status in statuses:
            state = status.state.value
            glyph = self._MCP_STATE_GLYPHS.get(state, "○")
            style = self._MCP_STATE_STYLES.get(state, MCP_INFO)
            line = f"  {glyph} {status.name}  {status.transport}/{status.scope}  {status.tool_count} tools  {state}"
            if status.error:
                line += f"  — {status.error}"
            self.console.print(line, style=style, markup=False, highlight=False)

        prompts = manager.prompt_commands()
        if prompts:
            self.console.print()
            for command, description in prompts:
                self.console.print(f"  {command:<28} {description}", style=MCP_INFO, markup=False, highlight=False)

        self.console.print()
        self.console.print(self._MCP_HINT, style=MCP_INFO)
        self.console.print()
        return None

    def _mcp_prompt(self, text: str) -> McpAction | None:
        """`/mcp__server__prompt [args...]`: an MCP prompt command."""
        from src.llm.tools.mcp.manager import get_mcp_manager

        manager = get_mcp_manager()
        command = text.split(None, 1)[0]
        if manager.find_prompt(command) is None:
            self.console.print(f"  unknown MCP prompt command: {command}", style=MCP_ERROR)
            prompts = manager.prompt_commands()
            if prompts:
                self.console.print("  available:", style=MCP_INFO)
                for cmd, description in prompts:
                    self.console.print(f"    {cmd:<28} {description}", style=MCP_INFO, markup=False, highlight=False)
            else:
                self.console.print("  no MCP prompt commands are available", style=MCP_INFO)
            return None
        return McpAction("prompt", text)

    # ------------------------------------------------------------- tool pin

    def _tool_pin(self, argument: str) -> None:
        """`/tool [name|none]`: pin a registry tool's full contract onto the
        next outgoing message (`ChatUI._apply_tool_pins` does the appending
        and consumes the pin), so the model can skip search_tools/get_tool
        discovery for it. With no argument, shows what is currently pinned;
        `none`/`off`/`default` clears the pins. An unknown name is matched by
        substring — a single match pins it, several are listed as candidates
        rather than guessed. Tool names may come from an MCP server, so
        dynamic text here is printed with markup disabled."""
        from src.llm.tools import get_registry

        registry = get_registry()
        if not argument:
            pinned = ", ".join(self.ui.pinned_tools) or "nothing"
            self.console.print(
                f"  pinned for the next message: {pinned} (/tool <name>, /tool none)",
                style="info",
                markup=False,
                highlight=False,
            )
            return None
        if argument.lower() in CLEAR_WORDS:
            self.ui.pinned_tools.clear()
            self.console.print("  tool pins cleared", style="success")
            return None
        name = argument.strip()
        if registry.get(name) is None:
            matches = [t.name for t in registry.all_tools() if name.lower() in t.name.lower()]
            if len(matches) == 1:
                name = matches[0]
            else:
                listed = ", ".join(matches[:MAX_LISTED_MATCHES]) or "no similar names"
                self.console.print(
                    f"  unknown tool: {name} — {listed}",
                    style="error",
                    markup=False,
                    highlight=False,
                )
                return None
        if name not in self.ui.pinned_tools:
            self.ui.pinned_tools.append(name)
        self.console.print(
            f"  pinned {name} — its contract goes with your next message",
            style="success",
            markup=False,
            highlight=False,
        )
        return None
