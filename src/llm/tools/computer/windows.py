"""Finding and focusing top-level windows.

Typing without checking which window is focused is how automation ends up
entering text into the wrong application — including the one driving it. Every
`type_text` in a sequence should be preceded by a known-good focus.
"""

import ctypes
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass

from .types import Box

SW_RESTORE = 9
SW_SHOW = 5


@dataclass(frozen=True)
class Window:
    handle: int
    title: str
    rect: Box

    @property
    def size(self) -> tuple[int, int]:
        return (self.rect[2] - self.rect[0], self.rect[3] - self.rect[1])


def _user32():
    if sys.platform != "win32":
        raise RuntimeError("window control requires Windows")
    return ctypes.windll.user32


def _title_of(user32, handle: int) -> str:
    length = user32.GetWindowTextLengthW(handle)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value


def _rect_of(user32, handle: int) -> Box:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        raise OSError("GetWindowRect failed")
    return (rect.left, rect.top, rect.right, rect.bottom)


def list_windows(*, visible_only: bool = True) -> list[Window]:
    """Every top-level window, most recently active first."""
    user32 = _user32()
    windows: list[Window] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def collect(handle, _lparam):
        if visible_only and not user32.IsWindowVisible(handle):
            return True
        title = _title_of(user32, handle)
        if title:
            windows.append(Window(int(handle), title, _rect_of(user32, handle)))
        return True

    user32.EnumWindows(collect, 0)
    return windows


def foreground_window() -> Window | None:
    """The window that currently receives keyboard input."""
    user32 = _user32()
    handle = user32.GetForegroundWindow()
    if not handle:
        return None
    return Window(int(handle), _title_of(user32, handle), _rect_of(user32, handle))


def find_window(title_contains: str) -> Window | None:
    """First visible window whose title contains `title_contains`."""
    needle = title_contains.casefold()
    for window in list_windows():
        if needle in window.title.casefold():
            return window
    return None


def focus_window(
    title_contains: str,
    *,
    attempts: int = 3,
    settle: float = 0.4,
    sleep=time.sleep,
) -> Window:
    """Bring a window to the foreground and confirm it got there.

    Windows refuses `SetForegroundWindow` from a process that does not own the
    current foreground window, and it fails silently, so the result is verified
    rather than assumed.
    """
    user32 = _user32()
    window = find_window(title_contains)
    if window is None:
        raise LookupError(f"no visible window matching {title_contains!r}")

    for _ in range(attempts):
        user32.ShowWindow(window.handle, SW_RESTORE)
        user32.SetForegroundWindow(window.handle)
        sleep(settle)
        current = foreground_window()
        if current is not None and current.handle == window.handle:
            return current
        # Nudge the foreground lock: a synthetic Alt tap makes Windows treat
        # this process as having input, which lets the next call through.
        user32.keybd_event(0x12, 0, 0, 0)
        user32.keybd_event(0x12, 0, 2, 0)

    raise OSError(
        f"could not focus {title_contains!r}; foreground is "
        f"{(foreground_window() or Window(0, '<none>', (0, 0, 0, 0))).title!r}"
    )
