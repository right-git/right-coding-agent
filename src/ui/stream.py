"""Live progress of one agent turn: stopwatch, tool events, streamed text.

`ChatUI.turn_stream()` opens one rich Live region pinned to the bottom of
the terminal and yields a `TurnStream`. The LLM layer drives it with two
callbacks while the turn runs: `on_token` accumulates the model's streamed
text and shows it in full while it is written (only the terminal's height
crops it), `on_reasoning` fills that same space, dimmed, while the model is
still thinking, `on_message` prints tool calls and tool results the moment
they happen — plus a dim "thought for Ns" line before each action, so the wait is
always attributed. The first streamed token ends the "thinking" phase and
flips the label to "responding": tokens are answer text by construction,
never reasoning. Regular console prints
surface above the live region, so finished lines scroll away naturally
while the ticker stays pinned.
"""

import time

from langchain_core.messages import AIMessage, ToolMessage
from rich.text import Text

from src.config.logging import logger
from src.llm.utils import format_duration

TICKER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
TEXT_INDENT = "  "
TEXT_MIN_WIDTH = 20


class TurnTicker:
    """A labeled stopwatch; the display re-reads it on every refresh."""

    def __init__(self, label: str = "thinking") -> None:
        self.label = label
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def reset(self, label: str) -> None:
        self.label = label
        self.started = time.monotonic()


class TurnStream:
    """Receives turn events from the LLM layer and renders them live.

    Also the Live renderable itself: rich re-renders `__rich_console__`
    ~10 times per second, which is what animates the frames and the clock.
    `printed_ids` collects the ids of messages already shown live, so the
    final `print_response` can skip them instead of printing twice.
    """

    def __init__(self, ui) -> None:
        self._ui = ui
        self.ticker = TurnTicker()
        self.printed_ids: set[str] = set()
        self._seen: set[str] = set()
        self._text = ""
        self._reasoning = ""

    def __rich_console__(self, console, options):
        # The answer is shown as it streams, wrapped and undimmed — it is the
        # result, not a peek at one. Until it starts, the model's reasoning
        # fills the same space, dimmed, so a long "thinking" is never a blank
        # wait. Only the terminal's own height limits either: the newest lines
        # win, so what is being written is always on screen.
        body, style = (self._text, None) if self._text else (self._reasoning, "info")
        if body:
            height = getattr(options, "max_height", None) or console.size.height
            budget = max(1, height - 2)  # leave room for the ticker line
            width = max(TEXT_MIN_WIDTH, options.max_width - len(TEXT_INDENT))
            wrapped = Text(body.strip(), style=style or "").wrap(console, width, overflow="fold")
            for line in wrapped[-budget:]:
                yield Text(TEXT_INDENT, style=style or "").append_text(line)
        frame = TICKER_FRAMES[int(time.monotonic() * 10) % len(TICKER_FRAMES)]
        yield Text(
            f"  {frame} {self.ticker.label}… {format_duration(self.ticker.elapsed)}",
            style="info",
        )

    # Both callbacks swallow their own failures: rendering progress must
    # never break a turn that the model is still executing.

    def on_token(self, piece: str) -> None:
        try:
            if not piece:
                return
            # Only answer text reaches this callback — reasoning arrives as
            # `reasoning` content blocks, which `AIMessageChunk.text` drops.
            # So the first token means thinking is over and the model is
            # writing; without this the ticker kept saying "thinking" while
            # the answer itself scrolled past, reading as reasoning.
            self._announce_thought()
            if self.ticker.label == "thinking":
                self.ticker.reset("responding")
            self._reasoning = ""  # the answer takes over the live view
            self._text += piece
        except Exception:
            logger.exception("Failed to buffer streamed text")

    def on_reasoning(self, piece: str) -> None:
        """Buffer the model's reasoning; unlike answer text it does not end
        the thinking phase — it is what the thinking *is*."""
        try:
            if piece:
                self._reasoning += piece
        except Exception:
            logger.exception("Failed to buffer streamed reasoning")

    def on_message(self, message) -> None:
        try:
            self._handle(message)
        except Exception:
            logger.exception("Failed to render a streamed message")

    def _handle(self, message) -> None:
        identifier = getattr(message, "id", None)
        if identifier and identifier in self._seen:
            return
        if identifier:
            self._seen.add(identifier)

        if isinstance(message, ToolMessage):
            took = self.ticker.elapsed if self.ticker.label == "running tools" else None
            self._ui._print_tool_result(message, duration=took)
            if identifier:
                self.printed_ids.add(identifier)
            self.ticker.reset("thinking")
            return

        if not isinstance(message, AIMessage):
            return

        if message.tool_calls:
            self._announce_thought()
            for tool_call in message.tool_calls:
                self._ui._print_tool_call(tool_call)
            if identifier:
                self.printed_ids.add(identifier)
            self._text = ""
            self._reasoning = ""
            self.ticker.reset("running tools")
        else:
            # The final answer: announce the thinking time (a no-op once the
            # streamed text already ended the thinking phase) and let the turn
            # loop print the text itself once the response is complete.
            self._announce_thought()
            self.ticker.reset("finishing")

    def _announce_thought(self) -> None:
        if self.ticker.label != "thinking":
            return
        self._ui.console.print(f"  ✻ thought for {format_duration(self.ticker.elapsed)}", style="info")
