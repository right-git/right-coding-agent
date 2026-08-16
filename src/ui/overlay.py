"""Always-on-top status indication drawn over the desktop.

Two independent layers:

- the **voice pill** at the bottom of the screen — animated level bars while
  recording ("listening"), pulsing dots from send until the reply lands
  ("syncing");
- the **computer border** — a breathing accent frame around the screen while
  the agent drives the desktop: a "hands off, the AI is working" signal,
  refreshed by every screen tool call and fading out a few seconds after the
  last one.

Backends: on Windows/Linux one Tk loop THREAD owns a fullscreen transparent
click-through window. On macOS AppKit only allows window creation on the main
thread (a Tk root on any other thread aborts the whole process — shipped once
as SIGABRT in TkMacOSXMakeRealWindowExist), so the overlay runs as a CHILD
PROCESS: `python src/ui/overlay.py` puts Tk on the child's main thread and
reads JSON commands from stdin, and it draws only the voice pill in a small
bottom-center window (no fullscreen border — a small window cannot block the
desktop even if click-through fails). This module must therefore stay
importable as a plain script: stdlib + loguru only, no `src.*` imports at
module level.

`set_voice`/`ping_computer` only queue/send commands and never block. Every
failure is swallowed and logged — status indication must never break a turn.
`_apply`/`computer_active`, the wire format, and the layout helpers are pure
and unit-tested without Tk.
"""

import atexit
import gc
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

from loguru import logger


def tk_thread_supported(platform: str | None = None) -> bool:
    """Whether a Tk window may live on a background thread here (not on macOS).

    Kept in sync with the canonical copy in `src/llm/tools/computer/overlay.py`;
    duplicated so this file stays runnable as a child script without `src.*`.
    """
    return (platform or sys.platform) != "darwin"


TRANSPARENT_KEY = "#0B1F0B"
PILL_BACKGROUND = "#11161C"
LISTENING_COLOR = "#FF3B30"
SYNCING_COLOR = "#3B82F6"
BORDER_COLOR = "#FF3B30"
BORDER_DIM = "#57120D"
FRAME_MS = 66  # ~15 fps
COMPUTER_LINGER = 3.0
PILL_WIDTH, PILL_HEIGHT, PILL_MARGIN = 240, 56, 96


def blend(color_a: str, color_b: str, t: float) -> str:
    """Linear mix of two `#rrggbb` colors, t in [0, 1]."""
    t = min(1.0, max(0.0, t))
    parts = []
    for index in (1, 3, 5):
        a = int(color_a[index : index + 2], 16)
        b = int(color_b[index : index + 2], 16)
        parts.append(round(a + (b - a) * t))
    return "#{:02x}{:02x}{:02x}".format(*parts)


COMMANDS = ("voice", "computer", "stop")


def encode_command(command: str, payload=None) -> str:
    """One overlay command as a single JSON line (parent → child stdin)."""
    return json.dumps({"command": command, "payload": payload})


def decode_command(line: str) -> tuple | None:
    """Parse a wire line back to (command, payload); None for anything invalid."""
    try:
        data = json.loads(line)
        command = data["command"]
    except Exception:
        return None
    if command not in COMMANDS:
        return None
    return command, data.get("payload")


def pill_top(height: int) -> int:
    """Top of the pill: `PILL_MARGIN` above the bottom, centered in small windows."""
    top = height - PILL_MARGIN - PILL_HEIGHT
    return top if top >= 0 else (height - PILL_HEIGHT) // 2


def _ensure_tcl_env(base_prefix: str | None = None) -> None:
    """Point Tcl/Tk at the base interpreter's bundled runtime when unset.

    uv-managed pythons look for init.tcl relative to the VENV prefix and miss
    the base installation's `lib/tcl8.6`, failing with "Can't find a usable
    init.tcl" depending on the environment. Never overrides an explicit value.
    """
    base = Path(base_prefix or sys.base_prefix) / "lib"
    for variable, name in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
        path = base / name
        if variable not in os.environ and path.is_dir():
            os.environ[variable] = str(path)


class StatusOverlay:
    """Queue/pipe-driven overlay; safe to call from any thread."""

    def __init__(self, screen_size=None, child_factory=None, pill_only: bool = False):
        self.screen_size = screen_size
        self.pill_only = pill_only  # child mode: small pill window, no border
        self.voice_state: str | None = None  # None | "listening" | "syncing"
        self.computer_until = 0.0  # monotonic deadline of the border layer
        self._commands: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._child = None  # subprocess.Popen-shaped, macOS backend
        self._child_factory = child_factory
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()  # one stdin writer at a time
        self._failed = False

    # ------------------------------------------------------------ public API

    def set_voice(self, state: str | None) -> None:
        if state is None and not self._is_active():
            return  # already hidden — don't create a window just to show nothing
        self._dispatch("voice", state)

    def ping_computer(self, linger: float = COMPUTER_LINGER) -> None:
        if not tk_thread_supported():
            return  # the macOS child draws only the voice pill — no border to ping
        self._dispatch("computer", linger)

    def prewarm(self) -> None:
        """Start the backend early so the first real command shows instantly.

        Matters on macOS: booting the child process (python + Tk) takes about
        a second, too slow for a pill that should appear on the first press.
        """
        self._start()

    def close(self) -> None:
        with self._lock:
            thread = self._thread
            child = self._child
            self._thread = None
            self._child = None
        if child is not None and child.poll() is None:
            try:
                child.stdin.write(encode_command("stop") + "\n")
                child.stdin.flush()
                child.stdin.close()
            except Exception:
                pass
            try:
                child.wait(timeout=2.0)
            except Exception:
                child.terminate()
        if thread is None or not thread.is_alive():
            return
        self._commands.put(("stop", None))
        thread.join(timeout=5.0)

    # ------------------------------------------------------------ pure state

    def _apply(self, command: str, payload, *, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if command == "voice":
            self.voice_state = payload
        elif command == "computer":
            self.computer_until = max(self.computer_until, now + payload)

    def computer_active(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return now < self.computer_until

    # -------------------------------------------------------------- Tk loop

    def _is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _is_active(self) -> bool:
        child = self._child
        return self._is_running() or (child is not None and child.poll() is None)

    def _dispatch(self, command: str, payload) -> None:
        if not self._start():
            return
        if self._child is not None:
            self._send_child(command, payload)
        else:
            self._commands.put((command, payload))

    def _start(self) -> bool:
        if self._failed:
            return False
        if tk_thread_supported():
            return self._start_thread()
        return self._start_child()

    def _start_thread(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._thread = threading.Thread(target=self._serve, name="status-overlay", daemon=True)
            self._thread.start()
            atexit.register(self.close)
            return True

    def _start_child(self) -> bool:
        with self._lock:
            if self._child is not None and self._child.poll() is None:
                return True
            try:
                self._child = (self._child_factory or self._spawn_child)()
            except Exception:
                logger.exception("Status overlay child failed to start")
                self._failed = True
                return False
            atexit.register(self.close)
            return True

    def _spawn_child(self):
        import subprocess

        # The child's stderr goes to the native-noise log, so a Tk failure
        # there stays diagnosable (see src/utils/silence.py).
        from src.utils.silence import NATIVE_STDERR_LOG

        try:
            stderr = open(NATIVE_STDERR_LOG, "ab")
        except OSError:
            stderr = subprocess.DEVNULL
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            text=True,
        )
        if stderr is not subprocess.DEVNULL:
            stderr.close()  # the child holds its own copy of the fd
        logger.info("Status overlay child started (pid {})", child.pid)
        return child

    def _send_child(self, command: str, payload) -> None:
        child = self._child
        try:
            with self._send_lock:
                child.stdin.write(encode_command(command, payload) + "\n")
                child.stdin.flush()
        except Exception:
            logger.warning("Status overlay child is gone; disabling the overlay")
            self._failed = True

    def _serve(self) -> None:
        try:
            import tkinter
        except ImportError:
            self._failed = True
            return
        try:
            root = tkinter.Tk()
        except Exception:
            logger.exception("Status overlay could not create a window")
            self._failed = True
            return
        try:
            self._run_loop(root, tkinter)
        except Exception:
            logger.exception("Status overlay crashed")
        finally:
            try:
                root.destroy()
            except Exception:
                pass
            del root
            gc.collect()

    def _window_geometry(self, root) -> tuple[int, int]:
        """Window size; pill mode uses a small bottom-center window, not fullscreen."""
        screen_width, screen_height = self.screen_size or (root.winfo_screenwidth(), root.winfo_screenheight())
        if not self.pill_only:
            root.geometry(f"{screen_width}x{screen_height}+0+0")
            return screen_width, screen_height
        width, height = PILL_WIDTH + 32, PILL_HEIGHT + 32
        x = (screen_width - width) // 2
        y = screen_height - height - PILL_MARGIN
        root.geometry(f"{width}x{height}+{x}+{y}")
        return width, height

    @staticmethod
    def _pick_background(root, tkinter) -> str:
        """Best transparency the platform offers, most transparent first."""
        try:
            root.attributes("-transparentcolor", TRANSPARENT_KEY)  # Windows: color key
            return TRANSPARENT_KEY
        except tkinter.TclError:
            pass
        if sys.platform == "darwin":
            try:
                root.attributes("-transparent", True)  # macOS: true per-pixel alpha
                root.config(bg="systemTransparent")
                return "systemTransparent"
            except tkinter.TclError:
                pass
        root.attributes("-alpha", 0.85)
        return PILL_BACKGROUND

    def _enable_click_through(self, root) -> None:
        if sys.platform == "win32":
            from src.llm.tools.computer.overlay import enable_click_through

            try:
                enable_click_through(int(root.winfo_id()))
            except (OSError, AttributeError, ValueError):
                pass
        elif sys.platform == "darwin":
            _macos_click_through()

    def _run_loop(self, root, tkinter) -> None:
        if sys.platform == "darwin":
            _macos_hide_from_dock()
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        width, height = self._window_geometry(root)
        background = self._pick_background(root, tkinter)
        canvas = tkinter.Canvas(root, width=width, height=height, bg=background, highlightthickness=0, borderwidth=0)
        canvas.pack(fill="both", expand=True)

        visible = [False]

        def tick() -> None:
            try:
                while True:
                    command, payload = self._commands.get_nowait()
                    if command == "stop":
                        root.quit()
                        return
                    self._apply(command, payload)
            except queue.Empty:
                pass

            now = time.monotonic()
            active = self.voice_state is not None or self.computer_active(now)
            if active:
                canvas.delete("all")
                self._draw(canvas, width, height, now)
                if not visible[0]:
                    root.deiconify()
                    root.attributes("-topmost", True)
                    root.update_idletasks()
                    self._enable_click_through(root)
                    visible[0] = True
            elif visible[0]:
                canvas.delete("all")
                root.withdraw()
                visible[0] = False
            root.after(FRAME_MS, tick)

        root.after(0, tick)
        root.mainloop()

    # -------------------------------------------------------------- drawing

    def _draw(self, canvas, width: int, height: int, now: float) -> None:
        if self.computer_active(now) and not self.pill_only:
            self._draw_border(canvas, width, height, now)
        if self.voice_state is not None:
            self._draw_pill(canvas, width, height, now)

    @staticmethod
    def _draw_border(canvas, width: int, height: int, now: float) -> None:
        pulse = 0.5 + 0.5 * math.sin(now * 3.5)  # дыхание рамки
        for inset, weight in ((3, 6), (11, 2)):
            color = blend(BORDER_DIM, BORDER_COLOR, pulse if weight == 6 else pulse * 0.6)
            canvas.create_rectangle(inset, inset, width - inset, height - inset, outline=color, width=weight)

    def _draw_pill(self, canvas, width: int, height: int, now: float) -> None:
        accent = LISTENING_COLOR if self.voice_state == "listening" else SYNCING_COLOR
        left = (width - PILL_WIDTH) // 2
        top = pill_top(height)
        right, bottom = left + PILL_WIDTH, top + PILL_HEIGHT
        radius = PILL_HEIGHT // 2
        # Пилюля из двух кругов и прямоугольника — у Tk нет скруглённых углов.
        for shape in (
            (left, top, left + 2 * radius, bottom),
            (right - 2 * radius, top, right, bottom),
        ):
            canvas.create_oval(*shape, fill=PILL_BACKGROUND, outline=accent, width=2)
        canvas.create_rectangle(left + radius, top, right - radius, bottom, fill=PILL_BACKGROUND, outline="")
        for y in (top, bottom):
            canvas.create_line(left + radius, y, right - radius, y, fill=accent, width=2)

        center_y = (top + bottom) // 2
        if self.voice_state == "listening":
            bars = 9
            spacing = 14
            start_x = (width - (bars - 1) * spacing) // 2
            for index in range(bars):
                wave = abs(math.sin(now * 6.0 + index * 0.9))
                half = 4 + wave * 14
                x = start_x + index * spacing
                canvas.create_line(x, center_y - half, x, center_y + half, fill=accent, width=4, capstyle="round")
        else:  # syncing: три пульсирующие точки
            for index in range(3):
                wave = 0.5 + 0.5 * math.sin(now * 5.0 - index * 0.9)
                radius_dot = 4 + wave * 5
                x = width // 2 + (index - 1) * 26
                canvas.create_oval(
                    x - radius_dot,
                    center_y - radius_dot,
                    x + radius_dot,
                    center_y + radius_dot,
                    fill=blend(PILL_BACKGROUND, accent, 0.4 + 0.6 * wave),
                    outline="",
                )


def _macos_hide_from_dock() -> None:
    """No Dock icon, no menu bar, no focus stealing for the overlay child."""
    try:
        import AppKit

        AppKit.NSApp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    except Exception:
        logger.debug("Could not set the macOS activation policy")


def _macos_click_through() -> None:
    """Let clicks pass through the pill window; show it on every Space."""
    try:
        import AppKit

        can_join_all_spaces = 1 << 0
        fullscreen_auxiliary = 1 << 8
        for window in AppKit.NSApp.windows():
            window.setIgnoresMouseEvents_(True)
            window.setCollectionBehavior_(window.collectionBehavior() | can_join_all_spaces | fullscreen_auxiliary)
    except Exception:
        logger.debug("Could not enable macOS click-through for the overlay")


# ------------------------------------------------------------- child process


def _read_commands(stream, commands: queue.Queue) -> None:
    """Feed decoded stdin lines into the Tk loop's queue; EOF means stop.

    The parent closing the pipe (normal exit or crash) is the child's signal
    to shut down — no orphaned overlay processes.
    """
    for line in stream:
        decoded = decode_command(line)
        if decoded is None:
            continue
        commands.put(decoded)
        if decoded[0] == "stop":
            return
    commands.put(("stop", None))


def run_overlay_child() -> None:
    """Entry point of the overlay child: Tk on THIS process's main thread."""
    _ensure_tcl_env()
    overlay = StatusOverlay(pill_only=True)
    threading.Thread(target=_read_commands, args=(sys.stdin, overlay._commands), daemon=True).start()
    overlay._serve()


_overlay: StatusOverlay | None = None


def get_status_overlay() -> StatusOverlay:
    """The process-wide status overlay, created on first use."""
    global _overlay
    if _overlay is None:
        _overlay = StatusOverlay()
    return _overlay


def set_status_overlay(overlay: StatusOverlay | None) -> None:
    """Test/headless seam, mirroring `set_computer`."""
    global _overlay
    _overlay = overlay


if __name__ == "__main__":
    run_overlay_child()
