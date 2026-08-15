import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .service import ComputerUse
from .types import TargetMode

CommandKind = Literal["empty", "exit", "mode", "query", "mark", "click"]

EXIT_COMMANDS = frozenset({"exit", "quit"})
MODE_USAGE = "Use :mode first or :mode all"
MARK_USAGE = "Use :mark <описание> | <подсказка>"
CLICK_USAGE = "Use :click <описание>"
HELP_TEXT = "Commands: :mode first, :mode all, " ":mark <описание> | <подсказка>, :click <описание>, exit"


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    argument: str = ""
    note: str = ""
    mode: TargetMode | None = None


def parse_command(text: str) -> Command:
    """Turn one line of user input into a command. Raises ValueError on misuse."""
    stripped = text.strip()
    if not stripped:
        return Command(kind="empty")
    if stripped.casefold() in EXIT_COMMANDS:
        return Command(kind="exit")

    keyword = stripped.split(maxsplit=1)[0].casefold()
    remainder = stripped[len(keyword) :].strip()

    if keyword == ":mode":
        if remainder.casefold() in ("first", "all"):
            return Command(kind="mode", mode=remainder.casefold())  # type: ignore[arg-type]
        raise ValueError(MODE_USAGE)

    if keyword == ":mark":
        description, separator, note = remainder.partition("|")
        if not description.strip():
            raise ValueError(MARK_USAGE)
        return Command(
            kind="mark",
            argument=description.strip(),
            note=note.strip() if separator else "",
        )

    if keyword == ":click":
        if not remainder:
            raise ValueError(CLICK_USAGE)
        return Command(kind="click", argument=remainder)

    return Command(kind="query", argument=stripped)


def run_interactive_loop(
    computer: ComputerUse,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    pause_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Prompt for descriptions, then locate, mark, or click what was asked for."""
    mode: TargetMode = "first"
    output_fn(HELP_TEXT)

    while True:
        try:
            raw_input = input_fn(f"[{mode}] Что найти? ")
        except (EOFError, KeyboardInterrupt):
            output_fn("Stopped.")
            return

        try:
            command = parse_command(raw_input)
        except ValueError as error:
            output_fn(str(error))
            continue

        if command.kind == "empty":
            continue
        if command.kind == "exit":
            return
        if command.kind == "mode":
            mode = command.mode or mode
            output_fn(f"Mode: {mode}")
            continue

        try:
            _run_command(computer, command, mode, output_fn, pause_fn)
        except KeyboardInterrupt:
            output_fn("Stopped.")
            return
        except Exception as error:
            output_fn(f"Query failed: {error}")


def _run_command(
    computer: ComputerUse,
    command: Command,
    mode: TargetMode,
    output_fn: Callable[[str], None],
    pause_fn: Callable[[float], None],
) -> None:
    if command.kind == "mark":
        detection = computer.mark_object(command.argument, command.note)
        if detection is None:
            output_fn("Nothing found; pointer was not moved.")
            return
        output_fn(f"Marked {detection.label} at {computer.center_of(detection)}")
        return

    if command.kind == "click":
        detection = computer.click_object(command.argument)
        if detection is None:
            output_fn("Nothing found; pointer was not moved.")
            return
        output_fn(f"Clicked {detection.label} at {computer.center_of(detection)}")
        return

    detections = computer.locate_object(command.argument)
    computer.save_annotated(detections=detections)

    if not detections:
        output_fn("Nothing found; pointer was not moved.")
        return

    output_fn(computer.describe(detections))

    targets = detections[:1] if mode == "first" else detections
    for index, detection in enumerate(targets):
        computer.move_mouse(*computer.center_of(detection))
        if index < len(targets) - 1:
            pause_fn(0.35)
