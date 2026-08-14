import ctypes
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from ctypes import wintypes
from dataclasses import dataclass
from typing import Literal, Protocol

from .types import MouseButton, Point, ScrollDirection


INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

WHEEL_DELTA = 120

CLICK_FLAGS: dict[str, tuple[int, int]] = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}

VIRTUAL_KEYS: dict[str, int] = {
    "ctrl": 0x11,
    "control": 0x11,
    "shift": 0x10,
    "alt": 0x12,
    "option": 0x12,
    "win": 0x5B,
    "super": 0x5B,
    "meta": 0x5B,
    "cmd": 0x5B,
    "command": 0x5B,
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "backspace": 0x08,
    "delete": 0x2E,
    "del": 0x2E,
    "insert": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "pgup": 0x21,
    "pgdn": 0x22,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "capslock": 0x14,
    "printscreen": 0x2C,
    "pause": 0x13,
    "numlock": 0x90,
    "menu": 0x5D,
}
VIRTUAL_KEYS.update({f"f{number}": 0x6F + number for number in range(1, 25)})

# Keys that live on the extended half of the keyboard and need the extended flag.
EXTENDED_KEY_CODES = frozenset(
    {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2C, 0x2D, 0x2E, 0x5B, 0x5C, 0x90}
)
MODIFIER_KEY_CODES = frozenset({0x10, 0x11, 0x12, 0x5B, 0x5C})

VK_RETURN = 0x0D
VK_TAB = 0x09

TEXT_KEY_OVERRIDES: dict[str, int] = {
    "\n": VK_RETURN,
    "\r": VK_RETURN,
    "\t": VK_TAB,
}


@dataclass(frozen=True)
class KeyEvent:
    code: int
    pressed: bool
    extended: bool = False


@dataclass(frozen=True)
class TextAction:
    kind: Literal["key", "unicode"]
    code: int


def resolve_key(name: str) -> int:
    """Map a key name (`ctrl`, `enter`, `f5`, `a`, `7`) to a virtual-key code."""
    normalized = name.strip().casefold()
    if not normalized:
        raise ValueError("key name must not be empty")

    known_code = VIRTUAL_KEYS.get(normalized)
    if known_code is not None:
        return known_code

    if len(normalized) == 1 and normalized.isascii() and normalized.isalnum():
        return ord(normalized.upper())

    raise ValueError(
        f"Unsupported key name: {name!r}. "
        "Use a named key, a letter, a digit, or type_text() for punctuation."
    )


def key_combination_events(combination: str) -> list[KeyEvent]:
    """Build press/release events for one combination such as `ctrl+shift+s`."""
    names = [part for part in combination.strip().split("+") if part]
    if not names:
        raise ValueError("key combination must not be empty")

    codes = [resolve_key(name) for name in names]
    for code in codes[:-1]:
        if code not in MODIFIER_KEY_CODES:
            raise ValueError(
                f"Only modifier keys may precede the final key in {combination!r}"
            )

    events = [
        KeyEvent(code=code, pressed=True, extended=code in EXTENDED_KEY_CODES)
        for code in codes
    ]
    events.extend(
        KeyEvent(code=code, pressed=False, extended=code in EXTENDED_KEY_CODES)
        for code in reversed(codes)
    )
    return events


def key_events(combinations: str) -> list[KeyEvent]:
    """Build events for whitespace-separated combinations (`ctrl+a delete`)."""
    parts = combinations.split()
    if not parts:
        raise ValueError("key sequence must not be empty")

    events: list[KeyEvent] = []
    for combination in parts:
        events.extend(key_combination_events(combination))
    return events


def text_actions(text: str) -> list[TextAction]:
    """Split text into virtual-key presses and UTF-16 code units to inject."""
    actions: list[TextAction] = []
    for position, character in enumerate(text):
        override = TEXT_KEY_OVERRIDES.get(character)
        if override is not None:
            actions.append(TextAction(kind="key", code=override))
            continue
        try:
            encoded = character.encode("utf-16-le")
        except UnicodeEncodeError as error:
            raise ValueError(
                f"text is not valid Unicode at position {position}: "
                f"U+{ord(character):04X} is an unpaired surrogate. This "
                "usually means the text was decoded with the wrong codec — "
                "read it as UTF-8 before typing it."
            ) from error
        for index in range(0, len(encoded), 2):
            actions.append(
                TextAction(
                    kind="unicode",
                    code=int.from_bytes(encoded[index : index + 2], "little"),
                )
            )
    return actions


def scroll_command(direction: ScrollDirection, amount: int) -> tuple[int, int]:
    """Wheel flags and signed wheel delta for a scroll in `direction`."""
    if amount < 0:
        raise ValueError("scroll amount must not be negative")

    if direction == "up":
        return MOUSEEVENTF_WHEEL, amount * WHEEL_DELTA
    if direction == "down":
        return MOUSEEVENTF_WHEEL, -amount * WHEEL_DELTA
    if direction == "right":
        return MOUSEEVENTF_HWHEEL, amount * WHEEL_DELTA
    if direction == "left":
        return MOUSEEVENTF_HWHEEL, -amount * WHEEL_DELTA
    raise ValueError(f"Unsupported scroll direction: {direction}")


def interpolate(start: Point, end: Point, steps: int) -> list[Point]:
    """Intermediate points for a drag, `end` included, `start` excluded."""
    if steps < 1:
        raise ValueError("steps must be positive")

    start_x, start_y = start
    end_x, end_y = end
    return [
        (
            round(start_x + (end_x - start_x) * step / steps),
            round(start_y + (end_y - start_y) * step / steps),
        )
        for step in range(1, steps + 1)
    ]


class InputDispatcher(Protocol):
    def mouse(self, flags: int, data: int = 0) -> None: ...

    def key(self, code: int, *, pressed: bool, extended: bool = False) -> None: ...

    def unicode(self, code_unit: int, *, pressed: bool) -> None: ...


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("payload", _INPUTUNION)]


class Win32InputDispatcher:
    """Sends synthesized mouse and keyboard events through `SendInput`."""

    def __init__(self, *, platform: str = sys.platform) -> None:
        if platform != "win32":
            raise RuntimeError("Win32InputDispatcher requires Windows")
        self._send_input = ctypes.windll.user32.SendInput
        self._send_input.argtypes = [wintypes.UINT, ctypes.c_void_p, ctypes.c_int]
        self._send_input.restype = wintypes.UINT

    def _send(self, entry: _INPUT) -> None:
        sent = self._send_input(1, ctypes.byref(entry), ctypes.sizeof(_INPUT))
        if sent != 1:
            raise OSError("SendInput failed")

    def mouse(self, flags: int, data: int = 0) -> None:
        entry = _INPUT(type=INPUT_MOUSE)
        entry.payload.mi = _MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=data & 0xFFFFFFFF,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        self._send(entry)

    def key(self, code: int, *, pressed: bool, extended: bool = False) -> None:
        flags = 0
        if extended:
            flags |= KEYEVENTF_EXTENDEDKEY
        if not pressed:
            flags |= KEYEVENTF_KEYUP
        entry = _INPUT(type=INPUT_KEYBOARD)
        entry.payload.ki = _KEYBDINPUT(
            wVk=code, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0
        )
        self._send(entry)

    def unicode(self, code_unit: int, *, pressed: bool) -> None:
        flags = KEYEVENTF_UNICODE
        if not pressed:
            flags |= KEYEVENTF_KEYUP
        entry = _INPUT(type=INPUT_KEYBOARD)
        entry.payload.ki = _KEYBDINPUT(
            wVk=0, wScan=code_unit, dwFlags=flags, time=0, dwExtraInfo=0
        )
        self._send(entry)


def _default_set_cursor_position() -> Callable[[int, int], int | bool]:
    if sys.platform != "win32":
        raise RuntimeError("pointer movement requires Windows")
    setter = ctypes.windll.user32.SetCursorPos
    setter.argtypes = [ctypes.c_int, ctypes.c_int]
    setter.restype = ctypes.c_int
    return setter


def _default_get_cursor_position() -> Callable[[], Point]:
    if sys.platform != "win32":
        raise RuntimeError("pointer position requires Windows")
    getter = ctypes.windll.user32.GetCursorPos

    def read_position() -> Point:
        position = wintypes.POINT()
        if not getter(ctypes.byref(position)):
            raise OSError("GetCursorPos failed")
        return (position.x, position.y)

    return read_position


def move_pointer(
    x: int,
    y: int,
    *,
    set_cursor_position: Callable[[int, int], int | bool] | None = None,
) -> None:
    if set_cursor_position is None:
        set_cursor_position = _default_set_cursor_position()

    if not set_cursor_position(int(x), int(y)):
        raise OSError("SetCursorPos failed")


class Pointer:
    """Mouse and keyboard control, with every OS call behind an injected seam."""

    def __init__(
        self,
        *,
        dispatcher: InputDispatcher | None = None,
        set_cursor_position: Callable[[int, int], int | bool] | None = None,
        get_cursor_position: Callable[[], Point] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        click_interval: float = 0.02,
        drag_steps: int = 12,
        drag_interval: float = 0.01,
    ) -> None:
        self._dispatcher = dispatcher
        self._set_cursor_position = set_cursor_position
        self._get_cursor_position = get_cursor_position
        self._sleep = sleep
        self._click_interval = click_interval
        self._drag_steps = drag_steps
        self._drag_interval = drag_interval

    @property
    def dispatcher(self) -> InputDispatcher:
        if self._dispatcher is None:
            self._dispatcher = Win32InputDispatcher()
        return self._dispatcher

    def move(self, x: int, y: int) -> None:
        if self._set_cursor_position is None:
            self._set_cursor_position = _default_set_cursor_position()
        move_pointer(x, y, set_cursor_position=self._set_cursor_position)

    def position(self) -> Point:
        if self._get_cursor_position is None:
            self._get_cursor_position = _default_get_cursor_position()
        return self._get_cursor_position()

    def _press_modifiers(self, modifiers: Sequence[str]) -> list[int]:
        codes = [resolve_key(modifier) for modifier in modifiers]
        for code in codes:
            if code not in MODIFIER_KEY_CODES:
                raise ValueError(f"{modifiers!r} contains a non-modifier key")
        for code in codes:
            self.dispatcher.key(
                code, pressed=True, extended=code in EXTENDED_KEY_CODES
            )
        return codes

    def _release_modifiers(self, codes: Iterable[int]) -> None:
        for code in reversed(list(codes)):
            self.dispatcher.key(
                code, pressed=False, extended=code in EXTENDED_KEY_CODES
            )

    def click(
        self,
        button: MouseButton = "left",
        count: int = 1,
        modifiers: Sequence[str] = (),
    ) -> None:
        if count < 1:
            raise ValueError("click count must be positive")
        down_flag, up_flag = self._click_flags(button)

        modifier_codes = self._press_modifiers(modifiers)
        try:
            for index in range(count):
                self.dispatcher.mouse(down_flag)
                self.dispatcher.mouse(up_flag)
                if index < count - 1:
                    self._sleep(self._click_interval)
        finally:
            self._release_modifiers(modifier_codes)

    def mouse_down(self, button: MouseButton = "left") -> None:
        self.dispatcher.mouse(self._click_flags(button)[0])

    def mouse_up(self, button: MouseButton = "left") -> None:
        self.dispatcher.mouse(self._click_flags(button)[1])

    def drag(
        self,
        start: Point,
        end: Point,
        *,
        button: MouseButton = "left",
        steps: int | None = None,
    ) -> None:
        self.move(*start)
        self.mouse_down(button)
        try:
            for point in interpolate(start, end, steps or self._drag_steps):
                self.move(*point)
                self._sleep(self._drag_interval)
        finally:
            self.mouse_up(button)

    def scroll(
        self,
        direction: ScrollDirection,
        amount: int = 3,
        modifiers: Sequence[str] = (),
    ) -> None:
        flags, data = scroll_command(direction, amount)
        modifier_codes = self._press_modifiers(modifiers)
        try:
            self.dispatcher.mouse(flags, data)
        finally:
            self._release_modifiers(modifier_codes)

    def type_text(self, text: str) -> None:
        for action in text_actions(text):
            if action.kind == "key":
                self.dispatcher.key(action.code, pressed=True)
                self.dispatcher.key(action.code, pressed=False)
            else:
                self.dispatcher.unicode(action.code, pressed=True)
                self.dispatcher.unicode(action.code, pressed=False)

    def key(self, combination: str) -> None:
        for event in key_events(combination):
            self.dispatcher.key(
                event.code, pressed=event.pressed, extended=event.extended
            )

    def hold_key(self, combination: str, duration: float) -> None:
        if duration < 0:
            raise ValueError("duration must not be negative")
        events = key_combination_events(combination)
        for event in (event for event in events if event.pressed):
            self.dispatcher.key(event.code, pressed=True, extended=event.extended)
        try:
            self._sleep(duration)
        finally:
            for event in (event for event in events if not event.pressed):
                self.dispatcher.key(
                    event.code, pressed=False, extended=event.extended
                )

    @staticmethod
    def _click_flags(button: MouseButton) -> tuple[int, int]:
        flags = CLICK_FLAGS.get(button)
        if flags is None:
            raise ValueError(f"Unsupported mouse button: {button}")
        return flags
