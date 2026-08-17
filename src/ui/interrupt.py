"""Cancel a running turn from the keyboard (Esc, Ctrl+C).

While the model works, the prompt is closed and prompt_toolkit key bindings
cannot fire, so a watcher thread reads console keys raw. Esc requests
cancellation; every printable key the watcher inevitably consumes is stashed
as `typed_text` and handed back to the next prompt as type-ahead, so typing
the next message while a turn runs still works. Raw console reading is
Windows-only for now (msvcrt); elsewhere `EscapeWatcher.create()` returns
None and turns simply run to completion.

`InterruptPolicy` backs the Unix SIGINT handler: the first Ctrl+C cancels
the turn, a second within the window force-quits. The distinction matters
because a turn blocked inside `asyncio.to_thread` (MPS inference, whisper
finalize) cannot actually be cancelled until the thread ends — asyncio only
defers the cancellation — so a stuck turn made Ctrl+C look dead and pushed
users toward Ctrl+Z, which suspends the process with the microphone still
attached.
"""

import sys
import threading


class TurnCancelled(Exception):
    """The user pressed Esc while the turn was running."""


class InterruptPolicy:
    """First Ctrl+C asks to cancel; a repeat within `window` seconds forces quit."""

    def __init__(self, window: float = 2.0):
        self.window = window
        self._last = float("-inf")

    def press(self, now: float) -> str:
        """ "cancel" for a lone press, "force" for a double press."""
        action = "force" if now - self._last <= self.window else "cancel"
        self._last = now
        return action


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


class _UnixKeyReader:
    """cbreak-mode stdin reader for the turn window; restores the terminal.

    `start()` (called from the watcher's __enter__) flips the tty to cbreak —
    keys arrive without Enter, Ctrl+C still raises SIGINT (ISIG stays on) —
    and `close()` restores the saved state, so the next prompt_toolkit prompt
    finds the terminal exactly as it was.
    """

    def __init__(self, fd: int | None = None):
        self._fd = sys.stdin.fileno() if fd is None else fd
        self._saved = None

    def start(self) -> None:
        import termios
        import tty

        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def __call__(self) -> str:
        import os
        import select

        chunks: list[str] = []
        while True:
            ready, _, _ = select.select([self._fd], [], [], 0)
            if not ready:
                break
            data = os.read(self._fd, 64)
            if not data:
                break
            chunks.append(data.decode(errors="ignore"))
        return "".join(chunks)

    def close(self) -> None:
        if self._saved is None:
            return
        import termios

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        self._saved = None


def default_reader():
    """A key reader for this console (callable, optional start/close), or None."""
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return None
        if sys.platform == "win32":
            import msvcrt  # noqa: F401

            return _windows_reader()
        return _UnixKeyReader()
    except Exception:
        return None


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
        index = 0
        while index < len(chars):
            char = chars[index]
            if char == "\x1b":
                following = chars[index + 1] if index + 1 < len(chars) else ""
                if following in ("[", "O"):
                    # An arrow/function key arrives as "\x1b[A"-style CSI/SS3 —
                    # skip the whole sequence, it is not a cancel request.
                    index += 2
                    while index < len(chars) and not chars[index].isalpha() and chars[index] != "~":
                        index += 1
                    index += 1
                    continue
                self.pressed.set()
            elif char in ("\x08", "\x7f"):  # backspace/DEL edits the type-ahead stash
                if self._typed:
                    self._typed.pop()
            elif char >= " ":
                self._typed.append(char)
            index += 1

    def _run(self) -> None:
        while not self._stop.wait(self.POLL_SECONDS):
            try:
                self.handle(self._read_keys())
            except Exception:
                return  # a broken console must not loop hot; cancel just stops working

    def __enter__(self) -> "EscapeWatcher":
        start = getattr(self._read_keys, "start", None)
        if start is not None:
            try:
                start()
            except Exception:
                return self  # no cbreak — Esc simply stays inert this turn
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        close = getattr(self._read_keys, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass  # restoring a gone terminal must not break the turn
