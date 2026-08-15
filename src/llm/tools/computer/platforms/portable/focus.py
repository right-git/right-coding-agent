"""Window focus on macOS (AppleScript) and Linux (wmctrl/xdotool).

Both are driven through small CLI calls so no extra Python dependency is
needed: macOS ships `osascript`, Linux needs `wmctrl` (and `xdotool` for
reading the active window) from the distribution's package manager. Every
function takes a `runner` seam so tests never touch a real desktop.
"""

import shutil
import subprocess
import sys
from dataclasses import dataclass

RUN_TIMEOUT = 10.0

# Enumerates visible processes and raises the first window whose title
# contains the needle (AppleScript `contains` is case-insensitive).
_DARWIN_FOCUS_SCRIPT = """
on run argv
    set needle to item 1 of argv
    tell application "System Events"
        repeat with proc in (application processes whose visible is true)
            repeat with win in windows of proc
                if name of win contains needle then
                    set frontmost of proc to true
                    perform action "AXRaise" of win
                    return name of win
                end if
            end repeat
        end repeat
    end tell
    return ""
end run
"""

_DARWIN_FOREGROUND_SCRIPT = """
tell application "System Events"
    set proc to first application process whose frontmost is true
    try
        return name of front window of proc
    on error
        return name of proc
    end try
end tell
"""


@dataclass(frozen=True)
class FocusedWindow:
    """The slice of window metadata available on every OS."""

    title: str


def _run(command: list[str], runner=subprocess.run):
    return runner(command, capture_output=True, text=True, timeout=RUN_TIMEOUT)


def foreground_window(*, platform: str = sys.platform, runner=subprocess.run):
    """The window receiving keyboard input, or None where it cannot be read."""
    try:
        if platform == "darwin":
            result = _run(["osascript", "-e", _DARWIN_FOREGROUND_SCRIPT], runner)
            title = (result.stdout or "").strip()
            return FocusedWindow(title) if result.returncode == 0 and title else None
        if platform.startswith("linux") and shutil.which("xdotool"):
            result = _run(["xdotool", "getactivewindow", "getwindowname"], runner)
            title = (result.stdout or "").strip()
            return FocusedWindow(title) if result.returncode == 0 and title else None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def focus_window(title_contains: str, *, platform: str = sys.platform, runner=subprocess.run) -> FocusedWindow:
    """Bring the first window whose title contains `title_contains` to the front."""
    if platform == "darwin":
        return _focus_darwin(title_contains, runner)
    if platform.startswith("linux"):
        return _focus_linux(title_contains, runner)
    raise RuntimeError(
        "focus_window is not supported on this platform — bring the "
        f"target window ({title_contains!r}) to the foreground manually"
    )


def _focus_darwin(title_contains: str, runner=subprocess.run) -> FocusedWindow:
    result = _run(["osascript", "-e", _DARWIN_FOCUS_SCRIPT, title_contains], runner)
    if result.returncode != 0:
        raise OSError(
            f"osascript failed while focusing {title_contains!r}: "
            f"{(result.stderr or '').strip() or 'is the Accessibility permission granted?'}"
        )
    title = (result.stdout or "").strip()
    if not title:
        raise LookupError(f"no visible window matching {title_contains!r}")
    return FocusedWindow(title)


def _focus_linux(title_contains: str, runner=subprocess.run) -> FocusedWindow:
    if not shutil.which("wmctrl"):
        raise RuntimeError("focus_window needs wmctrl on Linux — install it (e.g. `sudo apt install wmctrl`)")
    # `wmctrl -a` raises the first window whose title contains the string,
    # matched case-insensitively, switching desktops if needed.
    result = _run(["wmctrl", "-a", title_contains], runner)
    if result.returncode != 0:
        raise LookupError(f"no visible window matching {title_contains!r}")
    current = foreground_window(platform="linux", runner=runner)
    return current or FocusedWindow(title_contains)
