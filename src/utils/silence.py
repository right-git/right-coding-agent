"""Mute noisy third-party output from a single thread.

Loading the vision model (transformers plus its remote code) prints
deprecation warnings, fallback notices, and progress bars straight to the
terminal. That load runs in a background thread, so the noise lands on top
of the interactive prompt. Redirecting `sys.stdout` globally would also eat
the UI's output — instead the streams are replaced once with proxies that
route per thread: a thread inside `silenced()` writes to nowhere, every
other thread writes through untouched.
"""

import os
import sys
import threading
import warnings
from contextlib import contextmanager

_local = threading.local()
_lock = threading.Lock()

_fd_lock = threading.Lock()
_fd_depth = 0
_fd_saved: int | None = None


class _ThreadRoutedStream:
    """Forwards to the wrapped stream unless the writing thread is silenced."""

    def __init__(self, original):
        self._original = original

    def write(self, text: str) -> int:
        if getattr(_local, "silenced", False):
            return len(text)
        return self._original.write(text)

    def writelines(self, lines) -> None:
        if getattr(_local, "silenced", False):
            return
        self._original.writelines(lines)

    def flush(self) -> None:
        if getattr(_local, "silenced", False):
            return
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install() -> None:
    # Checked by type, not by flag: stdout may be swapped by test harnesses
    # (redirect_stdout), and a stale flag would leave the new stream unwrapped.
    with _lock:
        if not isinstance(sys.stdout, _ThreadRoutedStream):
            sys.stdout = _ThreadRoutedStream(sys.stdout)
        if not isinstance(sys.stderr, _ThreadRoutedStream):
            sys.stderr = _ThreadRoutedStream(sys.stderr)


@contextmanager
def silenced():
    """Drop stdout/stderr writes and warnings from the current thread."""
    _install()
    _local.silenced = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        _local.silenced = False


@contextmanager
def suppress_native_stderr():
    """Drop C-level stderr while a native library loads (process-wide).

    `silenced()` only intercepts Python's `sys.stderr`; the Objective-C
    runtime's duplicate-class warnings (av and cv2 both bundle libavdevice)
    are written straight to fd 2 and land on top of the prompt. This dups
    /dev/null over fd 2 for the duration — process-global by nature, so keep
    the window to the model-construction call itself, never around a long
    download. Nesting/overlap from concurrent loader threads is refcounted;
    the fd is restored at the outermost exit.
    """
    global _fd_depth, _fd_saved
    with _fd_lock:
        if _fd_depth == 0:
            _fd_saved = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        _fd_depth += 1
    try:
        yield
    finally:
        with _fd_lock:
            _fd_depth -= 1
            if _fd_depth == 0 and _fd_saved is not None:
                os.dup2(_fd_saved, 2)
                os.close(_fd_saved)
                _fd_saved = None
