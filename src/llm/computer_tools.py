"""LangChain tools that let the agent see and drive the user's desktop."""

import asyncio

from langchain_core.tools import tool

from src.tools.computer_use import ComputerUse


_computer: ComputerUse | None = None


def get_computer() -> ComputerUse:
    """The process-wide desktop session, created on first use."""
    global _computer
    if _computer is None:
        _computer = ComputerUse()
    return _computer


def set_computer(computer: ComputerUse | None) -> None:
    """Replace the shared session (used by tests and by headless runs)."""
    global _computer
    _computer = computer


@tool(parse_docstring=True, return_direct=False)
async def screen_locate(description: str) -> str:
    """Find where something is on the user's screen right now.

    Call this whenever the user asks about something visible on their screen —
    a button, a panel, an error message — and you need its position before
    answering or acting. Takes a fresh screenshot on every call.

    Args:
        description: What to look for, in plain language, for example
            "the render button in the export panel".

    Returns:
        Every match with its bounding box and clickable center, or a note that
        nothing matched.
    """
    try:
        computer = get_computer()
        detections = await asyncio.to_thread(computer.locate_object, description)
        return computer.describe(detections)
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def screen_mark(description: str, note: str, title: str = "") -> str:
    """Show the user where an element is, without clicking it.

    Call this when the user cannot find something ("where is the render
    button?", "I don't see the export option"). It moves the mouse onto the
    element, outlines it, and shows a tooltip next to the cursor with your
    explanation. Prefer this over clicking when the user asked *where*
    something is rather than asking you to do it.

    Args:
        description: The element to point at, in plain language. This is the
            search query, so describe the element and where it sits.
        note: Short explanation shown to the user: what the element is and what
            happens when it is used.
        title: Heading shown above the note, a few words in the user's
            language, such as "Кнопка Render". Always set this — without it the
            heading falls back to the whole search query, which reads badly.

    Returns:
        Confirmation with the marked label and screen coordinates.
    """
    try:
        computer = get_computer()
        detection = await asyncio.to_thread(
            lambda: computer.mark_object(description, note, title=title or None)
        )
        if detection is None:
            return f"Nothing on screen matched: {description}"
        return (
            f"Marked '{detection.label}' at {computer.center_of(detection)} "
            "and showed the note to the user."
        )
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def screen_click(description: str, double: bool = False) -> str:
    """Click an element described in plain language.

    Call this only when the user asked you to *do* something on their machine.
    If they only asked where something is, use screen_mark instead.

    Args:
        description: The element to click, in plain language.
        double: Whether to double-click instead of single-click.

    Returns:
        Confirmation with the clicked label and coordinates, or a note that
        nothing matched.
    """
    try:
        computer = get_computer()
        detection = await asyncio.to_thread(
            lambda: computer.click_object(description, count=2 if double else 1)
        )
        if detection is None:
            return f"Nothing on screen matched: {description}"
        return f"Clicked '{detection.label}' at {computer.center_of(detection)}."
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def screen_type(text: str) -> str:
    """Type text into whatever currently has keyboard focus.

    Call this after clicking a field, when the user asked you to enter
    something on their machine.

    Args:
        text: The text to type. Newlines are sent as Enter.

    Returns:
        Confirmation of how much text was typed.
    """
    try:
        await asyncio.to_thread(get_computer().type_text, text)
        return f"Typed {len(text)} characters."
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def screen_key(combination: str) -> str:
    """Press a keyboard shortcut on the user's machine.

    Call this for shortcuts and navigation keys instead of hunting for a menu
    item with the mouse.

    Args:
        combination: One or more combinations separated by spaces, such as
            "ctrl+s", "alt+f4", or "ctrl+a delete". Letters, digits, and named
            keys (enter, tab, esc, delete, arrows, f1-f24) are supported.

    Returns:
        Confirmation of the pressed keys.
    """
    try:
        await asyncio.to_thread(get_computer().key, combination)
        return f"Pressed {combination}."
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def screen_scroll(direction: str, amount: int = 3) -> str:
    """Scroll the window under the mouse pointer.

    Call this when the element you need is likely off-screen, then locate it
    again afterwards.

    Args:
        direction: One of "up", "down", "left", "right".
        amount: Number of wheel notches, three by default.

    Returns:
        Confirmation of the scroll.
    """
    try:
        await asyncio.to_thread(get_computer().scroll, direction, amount)
        return f"Scrolled {direction} by {amount}."
    except Exception as error:
        return f"Tool call failed, error: {error}"


COMPUTER_TOOLS = [
    screen_locate,
    screen_mark,
    screen_click,
    screen_type,
    screen_key,
    screen_scroll,
]
