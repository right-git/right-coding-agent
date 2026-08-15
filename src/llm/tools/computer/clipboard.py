"""Reading the Windows clipboard.

Copying a link and reading it back is exact, where transcribing it from a
screenshot is a guess — an eleven-character video id has no redundancy, so a
single misread letter yields a working-looking but wrong URL.
"""

import ctypes
import sys
from ctypes import wintypes
from typing import Protocol

CF_UNICODETEXT = 13


GMEM_MOVEABLE = 0x0002


class ClipboardBackend(Protocol):
    def read_text(self) -> str: ...

    def write_text(self, text: str) -> None: ...


class Win32Clipboard:
    """Clipboard backed by the Win32 API."""

    def __init__(self, *, platform: str = sys.platform) -> None:
        if platform != "win32":
            raise RuntimeError("Win32Clipboard requires Windows")
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        self._user32.OpenClipboard.argtypes = [wintypes.HWND]
        self._user32.OpenClipboard.restype = wintypes.BOOL
        self._user32.GetClipboardData.argtypes = [wintypes.UINT]
        self._user32.GetClipboardData.restype = wintypes.HANDLE
        self._user32.CloseClipboard.argtypes = []
        self._user32.CloseClipboard.restype = wintypes.BOOL
        self._kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        self._kernel32.GlobalLock.restype = wintypes.LPVOID
        self._kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        self._kernel32.GlobalUnlock.restype = wintypes.BOOL
        self._kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        self._kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        self._user32.EmptyClipboard.argtypes = []
        self._user32.EmptyClipboard.restype = wintypes.BOOL
        self._user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        self._user32.SetClipboardData.restype = wintypes.HANDLE

    def read_text(self) -> str:
        """Current clipboard text, or an empty string when it holds none."""
        if not self._user32.OpenClipboard(None):
            raise OSError("OpenClipboard failed")
        try:
            handle = self._user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return ""
            pointer = self._kernel32.GlobalLock(handle)
            if not pointer:
                raise OSError("GlobalLock failed")
            try:
                return ctypes.c_wchar_p(pointer).value or ""
            finally:
                self._kernel32.GlobalUnlock(handle)
        finally:
            self._user32.CloseClipboard()

    def write_text(self, text: str) -> None:
        """Replace the clipboard contents with `text`.

        Pasting beats typing for anything long or multi-line: SendInput emits
        two events per character, and a newline typed into a chat box sends the
        message instead of breaking the line.
        """
        buffer = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(buffer)
        handle = self._kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            raise OSError("GlobalAlloc failed")

        pointer = self._kernel32.GlobalLock(handle)
        if not pointer:
            raise OSError("GlobalLock failed")
        try:
            ctypes.memmove(pointer, ctypes.byref(buffer), size)
        finally:
            self._kernel32.GlobalUnlock(handle)

        if not self._user32.OpenClipboard(None):
            raise OSError("OpenClipboard failed")
        try:
            self._user32.EmptyClipboard()
            if not self._user32.SetClipboardData(CF_UNICODETEXT, handle):
                raise OSError("SetClipboardData failed")
        finally:
            self._user32.CloseClipboard()


class MemoryClipboard:
    """Clipboard double for dry runs and tests."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.reads = 0
        self.writes: list[str] = []

    def read_text(self) -> str:
        self.reads += 1
        return self.text

    def write_text(self, text: str) -> None:
        self.text = text
        self.writes.append(text)
