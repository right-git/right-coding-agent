"""LangChain tools that let the agent see and drive the user's desktop."""

import asyncio

from langchain_core.tools import tool

from .service import ComputerUse

from ..meta.attachments import attach_image

_computer: ComputerUse | None = None
_activity_listener = None


def set_activity_listener(listener) -> None:
    """UI hook fired whenever a screen tool starts working with the desktop.

    The chat UI wires this to the status overlay's border ("the AI is driving
    your computer — hands off"); tests and library use leave it unset.
    """
    global _activity_listener
    _activity_listener = listener


def _ping_activity() -> None:
    if _activity_listener is None:
        return
    try:
        _activity_listener()
    except Exception:
        pass  # indication must never break a tool call


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


def warm_up_computer() -> None:
    """Load the vision locator now instead of on the first screen query."""
    locator = get_computer().locator
    load = getattr(locator, "load", None)
    if callable(load):
        load()


def _resolve_match(computer, description, detections, match, *, action):
    """Pick exactly one detection, or explain why nothing will be done.

    A vague description often matches several elements (every screen has
    many "input fields"); acting on the first one silently clicks the wrong
    thing. Callers act only on a unique match or an explicit `match` pick.
    """
    if not detections:
        return None, f"Nothing on screen matched: {description}"
    if match:
        if not 1 <= match <= len(detections):
            return None, (
                f"match={match} is out of range — {len(detections)} "
                f"element(s) matched {description!r}:\n"
                f"{computer.describe(detections)}"
            )
        return detections[match - 1], ""
    if len(detections) > 1:
        return None, (
            f"Did not {action}: {len(detections)} elements matched "
            f"{description!r}. Refine the description — say where the "
            f"element sits and what is around it — or call again with "
            f"match=<number> to pick one of:\n"
            f"{computer.describe(detections)}"
        )
    return detections[0], ""


@tool(parse_docstring=True, return_direct=False)
async def screen_locate(
    description: str,
    return_screen: bool = False,
    mark: bool = False,
    note: str = "",
    title: str = "",
    match: int = 0,
    region: str = "",
) -> str:
    """Find where something is on the user's screen — and optionally show it to them.

    Call this whenever the user asks about something visible on their screen —
    a button, a panel, an error message. Takes a fresh screenshot on every
    call. When the user cannot find something ("where is the render button?"),
    set mark=True with a note — the SAME call then outlines the element on
    their screen and shows a tooltip, so never call twice for find-then-show.

    Args:
        description: What to look for, in plain language. Be concrete — name
            the element type, its text or icon, and WHERE it sits ("the URL
            address bar at the very top of the browser window", not "the
            input field" — a screen usually has several inputs and a vague
            query matches all of them).
        return_screen: Also capture the screen with every match outlined and
            attach it to the conversation, so you can see the layout instead
            of only coordinates.
        mark: Also point the element out to the user on their screen — move
            the mouse onto it, outline it, and show a tooltip. Requires the
            description to match exactly one element - with several matches
            nothing is marked and the candidates are listed instead.
        note: Tooltip text shown to the user when marking - what the element
            is and what happens when it is used. Giving a note implies mark.
        title: Tooltip heading when marking, a few words in the user's
            language, such as "Кнопка Render". Always set it when marking -
            without it the heading falls back to the detected label.
        match: 1-based candidate number to mark when a previous call listed
            several matches for the same description. Leave 0 when the
            description should match uniquely.
        region: Limit the search to part of the screen - MUCH faster and more
            accurate, use it whenever you know the neighbourhood. Accepts the
            same grid names results are reported with ("top-right", "center",
            "bottom-left"), edge strips for menu bars, docks, and side panels
            ("top-bar", "bottom-bar", "left-bar", "right-bar"), or exact
            pixel bounds "l,t,r,b" from an earlier result.

    Returns:
        Every match with its bounding box, clickable center, and coarse
        position (top-left … bottom-right), or a note that nothing matched.
        With mark, confirmation that the element was pointed out — or the
        candidate list when the description was ambiguous. With return_screen
        the annotated screenshot is attached as an image you can see.
    """
    try:
        _ping_activity()
        computer = get_computer()
        detections = await asyncio.to_thread(lambda: computer.locate_object(description, region=region or None))
        text = computer.describe(detections)
        if mark or note.strip():
            target, report = _resolve_match(computer, description, detections, match, action="mark")
            if target is None:
                return report
            anchor = await asyncio.to_thread(
                lambda: computer.mark_box(target.box, title or target.label or description, note)
            )
            text += f"\nMarked '{target.label}' at {anchor} and showed the note to the user."
        if return_screen:
            encoded = await asyncio.to_thread(computer.annotated_base64)
            if attach_image(encoded, "image/jpeg", label=f"screen_locate: {description}"):
                text += "\n(annotated screenshot attached as an image)"
            else:
                text += f"\nAnnotated screenshot, base64 JPEG: {encoded}"
        return text
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def screen_screenshot(return_base64: bool = False, max_side: int = 1280) -> str:
    """Capture the user's screen so you can look at what is on it.

    The captured image is attached to the conversation and shown to you as a
    picture. Use it to understand layout, read anything the locator cannot
    describe, or check the result of an action.

    Args:
        return_base64: Also include the raw base64 JPEG in the text result,
            for passing to another tool. You cannot see an image from base64
            text — the attached picture is what you look at.
        max_side: Longest side of the captured image in pixels; smaller is
            cheaper, larger is sharper.

    Returns:
        Confirmation text; the screenshot itself arrives as an attached
        image. With return_base64 the raw base64 JPEG is appended.
    """
    try:
        _ping_activity()
        computer = get_computer()
        encoded = await asyncio.to_thread(lambda: computer.screenshot_base64(max_side=max_side))
        attached = attach_image(encoded, "image/jpeg", label="screenshot")
        parts = ["Captured the screen."]
        if attached:
            parts.append("The screenshot is attached as an image.")
        if return_base64 or not attached:
            parts.append(f"base64 JPEG: {encoded}")
        return " ".join(parts)
    except Exception as error:
        return f"Tool call failed, error: {error}"


@tool(parse_docstring=True, return_direct=False)
async def screen_click(description: str, double: bool = False, match: int = 0, region: str = "") -> str:
    """Click an element described in plain language.

    Call this only when the user asked you to *do* something on their machine.
    If they only asked where something is, use screen_locate with mark=True
    instead. When several elements match the description, nothing is clicked —
    the candidates are listed so you can refine the description or pick one.

    Args:
        description: The element to click. Be concrete enough to match
            exactly one element — its type, its text or icon, and where it
            sits ("the URL address bar at the very top of the browser
            window", not "the input field", which matches every input on
            the screen).
        double: Whether to double-click instead of single-click.
        match: 1-based number of the candidate to click when a previous call
            listed several matches for the same description. Leave 0 when
            the description should match uniquely.
        region: Limit the search to part of the screen - MUCH faster and more
            accurate, use it whenever you know the neighbourhood. Accepts
            grid names ("top-right", "center"), edge strips ("top-bar",
            "bottom-bar", "left-bar", "right-bar"), or pixel bounds
            "l,t,r,b" from an earlier locate result.

    Returns:
        Confirmation with the clicked label and coordinates; the candidate
        list when the description was ambiguous; or a note that nothing
        matched.
    """
    try:
        _ping_activity()
        computer = get_computer()
        detections = await asyncio.to_thread(lambda: computer.locate_object(description, region=region or None))
        target, report = _resolve_match(computer, description, detections, match, action="click")
        if target is None:
            return report
        x, y = computer.center_of(target)
        await asyncio.to_thread(lambda: computer.double_click(x, y) if double else computer.left_click(x, y))
        return f"Clicked '{target.label}' at {(x, y)}."
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
        _ping_activity()
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
        _ping_activity()
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
        _ping_activity()
        await asyncio.to_thread(get_computer().scroll, direction, amount)
        return f"Scrolled {direction} by {amount}."
    except Exception as error:
        return f"Tool call failed, error: {error}"


COMPUTER_TOOLS = [
    screen_locate,
    screen_screenshot,
    screen_click,
    screen_type,
    screen_key,
    screen_scroll,
]
