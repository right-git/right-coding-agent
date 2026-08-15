"""Per-OS backends behind one lazy factory.

`windows/` is the native implementation (SendInput, Win32 clipboard, window
focus); `portable/` covers macOS and Linux on top of cross-platform libraries
(pynput for input, mss for capture, pyperclip for the clipboard). Every
import here is lazy on purpose: importing the computer package must never
touch an OS API that does not exist on the current platform (the Win32
modules, for one, only import on Windows).
"""

import sys

IS_WINDOWS = sys.platform == "win32"


def default_screen():
    """The screen backend for this OS."""
    if IS_WINDOWS:
        from .windows.screen import PrimaryScreen

        return PrimaryScreen()
    from .portable.screen import MssScreen

    return MssScreen()


def default_pointer():
    """The mouse/keyboard backend for this OS."""
    if IS_WINDOWS:
        from .windows.pointer import Pointer

        return Pointer()
    from .portable.pointer import PortablePointer

    return PortablePointer()


def default_clipboard():
    """The clipboard backend for this OS."""
    if IS_WINDOWS:
        from .windows.clipboard import Win32Clipboard

        return Win32Clipboard()
    from .portable.clipboard import PyperclipClipboard

    return PyperclipClipboard()


def enable_dpi_awareness() -> None:
    """Windows per-monitor DPI awareness; a no-op on other platforms."""
    if IS_WINDOWS:
        from .windows.screen import enable_dpi_awareness as enable

        enable()


def foreground_window():
    """The window currently receiving keyboard input, or None where unknown."""
    if IS_WINDOWS:
        from .windows.focus import foreground_window as query

        return query()
    from .portable.focus import foreground_window as query

    return query()


def focus_window(title_contains: str):
    """Bring a window to the foreground; raises where unsupported."""
    if IS_WINDOWS:
        from .windows.focus import focus_window as focus

        return focus(title_contains)
    from .portable.focus import focus_window as focus

    return focus(title_contains)
