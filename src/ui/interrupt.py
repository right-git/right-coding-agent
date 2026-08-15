"""Cancel a running turn from the keyboard (Esc).

While the model works, the prompt is closed and prompt_toolkit key bindings
cannot fire, so a watcher thread reads console keys raw. Esc requests
cancellation; every printable key the watcher inevitably consumes is stashed
as `typed_text` and handed back to the next prompt as type-ahead, so typing
the next message while a turn runs still works. Raw console reading is
Windows-only for now (msvcrt); elsewhere `EscapeWatcher.create()` returns
None and turns simply run to completion.
"""

import sys
import threading


class TurnCancelled(Exception):
    """The user pressed Esc while the turn was running."""


def _windows_reader():
    import msvcrt

    def read() -> str:
        chars: list[str] = []
        while msvcrt.kbhit():
            char = msvcrt.getwch()
            if char in ("\x00", "\xe0"):  # arrows/F-keys arrive as a two-part code
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            chars.append(char)
        return "".join(chars)

    return read


def default_reader():
    """A no-argument callable draining pending console keys, or None."""
    if sys.platform != "win32":
        return None
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return None
        import msvcrt  # noqa: F401
    except Exception:
        return None
    return _windows_reader()


class EscapeWatcher:
    """Background key watcher for one running turn."""

    POLL_SECONDS = 0.05

    def __init__(self, read_keys):
        self.pressed = threading.Event()
        self._typed: list[str] = []
        self._read_keys = read_keys
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def create(cls) -> "EscapeWatcher | None":
        reader = default_reader()
        return cls(reader) if reader is not None else None

    @property
    def typed_text(self) -> str:
        return "".join(self._typed)

    def handle(self, chars: str) -> None:
        for char in chars:
            if char == "\x1b":
                self.pressed.set()
            elif char == "\x08":  # backspace edits the type-ahead stash
                if self._typed:
                    self._typed.pop()
            elif char >= " ":
                self._typed.append(char)

    def _run(self) -> None:
        while not self._stop.wait(self.POLL_SECONDS):
            try:
                self.handle(self._read_keys())
            except Exception:
                return  # a broken console must not loop hot; cancel just stops working

    def __enter__(self) -> "EscapeWatcher":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
