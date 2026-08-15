"""Mouse and keyboard control via pynput, for macOS and Linux.

Implements the same `types.PointerBackend` contract as the native Windows
pointer, including the key-combination language of `screen_key`: `ctrl+c`,
`cmd+shift+s`, whitespace-separated sequences (`ctrl+a delete`), letters,
digits, and named keys. On macOS the process needs the Accessibility
permission; on Linux a running X11 session (or uinput) is expected.
"""

import time
from collections.abc import Callable, Sequence

from ...types import MouseButton, Point, ScrollDirection

MODIFIER_NAMES = frozenset({"ctrl", "control", "shift", "alt", "option", "cmd", "command", "win", "super", "meta"})

# key name (as the model writes it) -> pynput `Key` attribute name
NAMED_KEYS: dict[str, str] = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "shift": "shift",
    "alt": "alt",
    "option": "alt",
    "cmd": "cmd",
    "command": "cmd",
    "win": "cmd",
    "super": "cmd",
    "meta": "cmd",
    "enter": "enter",
    "return": "enter",
    "tab": "tab",
    "esc": "esc",
    "escape": "esc",
    "space": "space",
    "backspace": "backspace",
    "delete": "delete",
    "del": "delete",
    "insert": "insert",
    "home": "home",
    "end": "end",
    "pageup": "page_up",
    "pgup": "page_up",
    "pagedown": "page_down",
    "pgdn": "page_down",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "capslock": "caps_lock",
    "printscreen": "print_screen",
    "pause": "pause",
    "numlock": "num_lock",
    "menu": "menu",
}
NAMED_KEYS.update({f"f{number}": f"f{number}" for number in range(1, 21)})

SCROLL_VECTORS: dict[str, tuple[int, int]] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


def split_combination(combination: str) -> list[str]:
    """Key names of one `ctrl+shift+s`-style combination, validated."""
    names = [part for part in combination.strip().split("+") if part]
    if not names:
        raise ValueError("key combination must not be empty")
    for name in names[:-1]:
        if name.strip().casefold() not in MODIFIER_NAMES:
            raise ValueError(f"Only modifier keys may precede the final key in {combination!r}")
    return names


def _interpolate(start: Point, end: Point, steps: int) -> list[Point]:
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


class PortablePointer:
    """Mouse and keyboard control through pynput controllers."""

    def __init__(
        self,
        *,
        sleep: Callable[[float], None] = time.sleep,
        click_interval: float = 0.02,
        drag_steps: int = 12,
        drag_interval: float = 0.01,
    ) -> None:
        from pynput.keyboard import Controller as KeyboardController
        from pynput.keyboard import Key
        from pynput.mouse import Button
        from pynput.mouse import Controller as MouseController

        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        self._buttons = {"left": Button.left, "right": Button.right, "middle": Button.middle}
        self._key_type = Key
        self._sleep = sleep
        self._click_interval = click_interval
        self._drag_steps = drag_steps
        self._drag_interval = drag_interval

    # ------------------------------------------------------------------ keys

    def resolve_key(self, name: str):
        """Map a key name (`ctrl`, `enter`, `f5`, `a`, `7`) to a pynput key."""
        normalized = name.strip().casefold()
        if not normalized:
            raise ValueError("key name must not be empty")

        attribute = NAMED_KEYS.get(normalized)
        if attribute is not None:
            key = getattr(self._key_type, attribute, None)
            if key is None:
                raise ValueError(f"Key {name!r} is not available on this platform")
            return key

        if len(normalized) == 1 and normalized.isascii() and normalized.isalnum():
            return normalized

        raise ValueError(
            f"Unsupported key name: {name!r}. Use a named key, a letter, a digit, or type_text() for punctuation."
        )

    def _press_combination(self, combination: str) -> None:
        keys = [self.resolve_key(name) for name in split_combination(combination)]
        pressed = []
        try:
            for key in keys[:-1]:
                self._keyboard.press(key)
                pressed.append(key)
            self._keyboard.press(keys[-1])
            self._keyboard.release(keys[-1])
        finally:
            for key in reversed(pressed):
                self._keyboard.release(key)

    def _press_modifiers(self, modifiers: Sequence[str]) -> list:
        for name in modifiers:
            if name.strip().casefold() not in MODIFIER_NAMES:
                raise ValueError(f"{modifiers!r} contains a non-modifier key")
        keys = [self.resolve_key(name) for name in modifiers]
        for key in keys:
            self._keyboard.press(key)
        return keys

    def _release_modifiers(self, keys) -> None:
        for key in reversed(list(keys)):
            self._keyboard.release(key)

    def type_text(self, text: str) -> None:
        self._keyboard.type(text)

    def key(self, combinations: str) -> None:
        parts = combinations.split()
        if not parts:
            raise ValueError("key sequence must not be empty")
        for combination in parts:
            self._press_combination(combination)

    def hold_key(self, combination: str, duration: float) -> None:
        if duration < 0:
            raise ValueError("duration must not be negative")
        keys = [self.resolve_key(name) for name in split_combination(combination)]
        for key in keys:
            self._keyboard.press(key)
        try:
            self._sleep(duration)
        finally:
            for key in reversed(keys):
                self._keyboard.release(key)

    # ----------------------------------------------------------------- mouse

    def _button(self, button: MouseButton):
        resolved = self._buttons.get(button)
        if resolved is None:
            raise ValueError(f"Unsupported mouse button: {button}")
        return resolved

    def move(self, x: int, y: int) -> None:
        self._mouse.position = (int(x), int(y))

    def position(self) -> Point:
        x, y = self._mouse.position
        return (int(x), int(y))

    def click(
        self,
        button: MouseButton = "left",
        count: int = 1,
        modifiers: Sequence[str] = (),
    ) -> None:
        if count < 1:
            raise ValueError("click count must be positive")
        resolved = self._button(button)

        modifier_keys = self._press_modifiers(modifiers)
        try:
            for index in range(count):
                self._mouse.click(resolved)
                if index < count - 1:
                    self._sleep(self._click_interval)
        finally:
            self._release_modifiers(modifier_keys)

    def mouse_down(self, button: MouseButton = "left") -> None:
        self._mouse.press(self._button(button))

    def mouse_up(self, button: MouseButton = "left") -> None:
        self._mouse.release(self._button(button))

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
            for point in _interpolate(start, end, steps or self._drag_steps):
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
        if amount < 0:
            raise ValueError("scroll amount must not be negative")
        vector = SCROLL_VECTORS.get(direction)
        if vector is None:
            raise ValueError(f"Unsupported scroll direction: {direction}")

        modifier_keys = self._press_modifiers(modifiers)
        try:
            self._mouse.scroll(vector[0] * amount, vector[1] * amount)
        finally:
            self._release_modifiers(modifier_keys)
