"""Always-on-top status indication drawn over the desktop.

Independent layers:

- the **voice indicator** — animated level bars while recording
  ("listening"), pulsing dots from send until the reply lands ("syncing").
  On Windows/Linux it is a small pill at the bottom of the screen; on macOS
  it is a **notch island**: an opaque black rounded-bottom slab that pops
  out of the physical notch at the top center (width from
  NSScreen.safeAreaInsets/auxiliary areas, NOTCH_FALLBACK_WIDTH without a
  notch) and retracts when idle, with ease-out animation both ways;
- the **computer border** — a breathing accent frame around the screen while
  the agent drives the desktop, refreshed by every screen tool call;
- **element markers** — screen_locate(mark=True) highlights with tooltips.

Backends: on Windows/Linux one Tk loop THREAD owns a fullscreen transparent
click-through window. On macOS AppKit only allows window creation on the main
thread (a Tk root on any other thread aborts the whole process — shipped once
as SIGABRT in TkMacOSXMakeRealWindowExist), so the overlay runs as a CHILD
PROCESS: `python src/ui/overlay.py` puts Tk on the child's main thread and
reads JSON commands from stdin; without AppKit (no click-through guarantee)
it falls back to a small bottom pill window that cannot block the desktop.
This module must therefore stay importable as a plain script: stdlib +
loguru only, no `src.*` imports at module level.

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
import textwrap
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
PILL_WIDTH, PILL_HEIGHT, PILL_MARGIN = 150, 36, 48
OVERLAY_ALPHA = 0.75  # macOS pill fallback: translucent, unobtrusive
ISLAND_ALPHA = 1.0  # the island must read as notch-black glass, not a tint

# Dynamic-island mode (macOS): the indicator pops out of the notch at the top
# center instead of a bottom pill. Content sits BELOW the physical notch —
# anything drawn behind it is invisible.
ISLAND_HEIGHT = 64
ISLAND_EXTRA = 90  # how much wider than the notch when fully out
ISLAND_ANIM_SECONDS = 0.25
ISLAND_RADIUS = 18
ISLAND_BACKGROUND = "#0A0A0A"  # the notch is pure black — blend into it
NOTCH_FALLBACK_WIDTH = 200  # no notch detected: float a tab of this width
NOTCH_BOTTOM = 38  # menu-bar/notch depth on notched Macs, in points


def ease_out_cubic(t: float) -> float:
    """Fast start, soft landing — the pop-out feel; input clamped to [0, 1]."""
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def island_geometry(screen_width: int, notch_width: int, progress: float) -> tuple[int, int, int]:
    """(left, right, height) of the island at `progress` — collapsed it IS the notch."""
    width = notch_width + round(ISLAND_EXTRA * progress)
    left = (screen_width - width) // 2
    return left, left + width, round(ISLAND_HEIGHT * progress)


def blend(color_a: str, color_b: str, t: float) -> str:
    """Linear mix of two `#rrggbb` colors, t in [0, 1]."""
    t = min(1.0, max(0.0, t))
    parts = []
    for index in (1, 3, 5):
        a = int(color_a[index : index + 2], 16)
        b = int(color_b[index : index + 2], 16)
        parts.append(round(a + (b - a) * t))
    return "#{:02x}{:02x}{:02x}".format(*parts)


COMMANDS = ("voice", "computer", "marker", "stop")
MARKER_ACCENT = "#FF3B30"
MARKER_FOREGROUND = "#F2F5F8"
MARKER_WRAP_CHARS = 46
MARKER_FONT = "Helvetica Neue" if sys.platform == "darwin" else "Segoe UI"


# The three geometry helpers below are copies of the canonical ones in
# `src/llm/tools/computer/overlay.py` (kept in sync by a unit test) — this
# file must stay runnable as a child script without importing `src.*`.


def wrap_note(note: str, width: int) -> list[str]:
    """Wrap a note to `width` characters, keeping author-written line breaks."""
    if width <= 0:
        raise ValueError("width must be positive")
    lines: list[str] = []
    for paragraph in note.splitlines():
        if not paragraph.strip():
            continue
        lines.extend(textwrap.wrap(paragraph.strip(), width=width) or [])
    return lines


def tooltip_placement(anchor, tooltip_size, screen_size, *, offset: int = 24, margin: int = 12):
    """Place the tooltip next to `anchor`, flipping it to stay on screen."""
    anchor_x, anchor_y = anchor
    tooltip_width, tooltip_height = tooltip_size
    screen_width, screen_height = screen_size

    x = anchor_x + offset
    if x + tooltip_width + margin > screen_width:
        x = anchor_x - offset - tooltip_width
    y = anchor_y + offset
    if y + tooltip_height + margin > screen_height:
        y = anchor_y - offset - tooltip_height

    max_x = max(margin, screen_width - tooltip_width - margin)
    max_y = max(margin, screen_height - tooltip_height - margin)
    return (min(max(x, margin), max_x), min(max(y, margin), max_y))


def connector_corner(anchor, position, tooltip_size):
    """Corner of the tooltip that faces `anchor`, for the connector line."""
    anchor_x, anchor_y = anchor
    x, y = position
    width, height = tooltip_size
    return (
        x if anchor_x < x else x + width,
        y if anchor_y < y else y + height,
    )


def scale_coords(values, scale: float) -> list[float]:
    """Capture-pixel coordinates → Tk points (retina capture is 2x the grid)."""
    return [value / scale for value in values]


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

    def __init__(
        self,
        screen_size=None,
        child_factory=None,
        pill_only: bool = False,
        marker_scale: float = 1.0,
        island: bool = False,
    ):
        self.screen_size = screen_size
        self.pill_only = pill_only  # fallback child mode: small pill window only
        self.marker_scale = marker_scale  # capture pixels per Tk point (retina: 2)
        self.island = island  # macOS: notch island at top center instead of the pill
        self.notch_width = 0  # physical notch width in points; 0 = none detected
        self.voice_state: str | None = None  # None | "listening" | "syncing"
        self.voice_shown_at = 0.0  # monotonic marks driving the pop/retract animation
        self.voice_hidden_at = 0.0
        self.computer_until = 0.0  # monotonic deadline of the border layer
        self.marker: dict | None = None  # screen_mark highlight, capture coords
        self.marker_until = 0.0
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
        self._dispatch("computer", linger)

    def show_marker(self, marker: dict) -> None:
        """Show a screen_mark highlight; `marker` is the wire dict (see Marker)."""
        self._dispatch("marker", marker)

    def hide_marker(self) -> None:
        if not self._is_active():
            return  # nothing shown — don't boot a backend just to hide
        self._dispatch("marker", None)

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
            if payload is not None and self.voice_state is None:
                self.voice_shown_at = now  # pop out (state swaps while visible don't re-pop)
            elif payload is None and self.voice_state is not None:
                self.voice_hidden_at = now  # retract
            self.voice_state = payload
        elif command == "computer":
            self.computer_until = max(self.computer_until, now + payload)
        elif command == "marker":
            self.marker = payload
            self.marker_until = 0.0 if payload is None else now + float(payload.get("duration") or 6.0)

    def computer_active(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return now < self.computer_until

    def marker_active(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.marker is not None and now < self.marker_until

    def island_progress(self, now: float | None = None) -> float:
        """0..1 pop-out amount of the island: eases out on show, back in on hide."""
        now = time.monotonic() if now is None else now
        if self.voice_state is not None:
            return ease_out_cubic((now - self.voice_shown_at) / ISLAND_ANIM_SECONDS)
        if self.voice_hidden_at <= 0:
            return 0.0
        return 1.0 - ease_out_cubic((now - self.voice_hidden_at) / ISLAND_ANIM_SECONDS)

    def _has_content(self, now: float) -> bool:
        """Whether anything must be on screen; border and markers need the fullscreen window."""
        if self.voice_state is not None:
            return True
        if self.island and self.voice_hidden_at > 0 and now - self.voice_hidden_at < ISLAND_ANIM_SECONDS:
            return True  # keep drawing the retract animation
        if self.pill_only:
            return False
        return self.computer_active(now) or self.marker_active(now)

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

    def _pick_background(self, root, tkinter) -> str:
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
                # Empty areas stay fully clear; drawn content is translucent —
                # except the island, which must merge with the black notch.
                root.attributes("-alpha", ISLAND_ALPHA if self.island else OVERLAY_ALPHA)
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
            active = self._has_content(now)
            if active:
                canvas.delete("all")
                self._safe_draw(canvas, width, height, now)
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

    def _safe_draw(self, canvas, width: int, height: int, now: float) -> None:
        """One bad frame must never freeze the overlay.

        tick() reschedules itself with root.after — an exception escaping the
        draw would break that chain and leave the last frame stuck on screen.
        """
        try:
            self._draw(canvas, width, height, now)
        except Exception:
            logger.exception("Overlay frame failed")

    def _draw(self, canvas, width: int, height: int, now: float) -> None:
        if self.computer_active(now) and not self.pill_only:
            self._draw_border(canvas, width, height, now)
        if self.marker_active(now) and not self.pill_only:
            self._draw_marker(canvas, width, height)
        if self.island:
            if self.island_progress(now) > 0:
                self._draw_island(canvas, width, now)
        elif self.voice_state is not None:
            self._draw_pill(canvas, width, height, now)

    def _draw_island(self, canvas, width: int, now: float) -> None:
        """The notch island: a black rounded-bottom slab sliding out of the top edge."""
        progress = self.island_progress(now)
        left, right, height = island_geometry(width, self.notch_width or NOTCH_FALLBACK_WIDTH, progress)
        if height <= 2:
            return
        radius = min(ISLAND_RADIUS, height // 2)
        canvas.create_rectangle(left, 0, right, height - radius, fill=ISLAND_BACKGROUND, outline="")
        canvas.create_rectangle(
            left + radius, height - radius, right - radius, height, fill=ISLAND_BACKGROUND, outline=""
        )
        for corner_left in (left, right - 2 * radius):
            canvas.create_oval(
                corner_left, height - 2 * radius, corner_left + 2 * radius, height, fill=ISLAND_BACKGROUND, outline=""
            )
        if progress < 0.7 or self.voice_state is None:
            return  # content fades in only once the slab is mostly out
        accent = LISTENING_COLOR if self.voice_state == "listening" else SYNCING_COLOR
        center_x = (left + right) // 2
        # The physical notch is black glass — draw in the strip below it.
        content_top = NOTCH_BOTTOM if self.notch_width else height // 3
        center_y = (content_top + height) // 2
        if self.voice_state == "listening":
            self._draw_level_bars(canvas, center_x, center_y, accent, now)
        else:
            self._draw_sync_dots(canvas, center_x, center_y, accent, now)

    def _draw_marker(self, canvas, width: int, height: int) -> None:
        """The screen_mark highlight: outlined box, connector, tooltip.

        Mirrors `TkOverlay._draw` in `src/llm/tools/computer/overlay.py`;
        coordinates arrive in capture pixels and are scaled to Tk points.
        """
        from tkinter import font as tkinter_font

        marker = self.marker
        x1, y1, x2, y2 = scale_coords(marker.get("box") or (0, 0, 0, 0), self.marker_scale)
        canvas.create_rectangle(x1, y1, x2, y2, outline=MARKER_ACCENT, width=3)

        if marker.get("anchor"):
            anchor_x, anchor_y = scale_coords(marker["anchor"], self.marker_scale)
        else:
            anchor_x = min(max((x1 + x2) / 2, 0), width)
            anchor_y = min(max((y1 + y2) / 2, 0), height)
        anchor = (anchor_x, anchor_y)

        title = str(marker.get("title") or "")
        note_lines = wrap_note(str(marker.get("note") or ""), MARKER_WRAP_CHARS)
        title_font = tkinter_font.Font(family=MARKER_FONT, size=12, weight="bold")
        note_font = tkinter_font.Font(family=MARKER_FONT, size=10)
        padding, line_gap = 12, 6

        title_height = title_font.metrics("linespace")
        note_height = note_font.metrics("linespace")
        content_width = max([title_font.measure(title)] + [note_font.measure(line) for line in note_lines])
        content_height = title_height + (line_gap + note_height * len(note_lines) if note_lines else 0)
        tooltip_size = (content_width + padding * 2, content_height + padding * 2)
        position = tooltip_placement(anchor, tooltip_size, (width, height))

        corner = connector_corner(anchor, position, tooltip_size)
        canvas.create_line(*anchor, *corner, fill=MARKER_ACCENT, width=2)
        canvas.create_oval(
            anchor_x - 5, anchor_y - 5, anchor_x + 5, anchor_y + 5, fill=MARKER_ACCENT, outline=MARKER_ACCENT
        )

        left, top = position
        canvas.create_rectangle(
            left,
            top,
            left + tooltip_size[0],
            top + tooltip_size[1],
            fill=PILL_BACKGROUND,
            outline=MARKER_ACCENT,
            width=2,
        )
        text_x, text_y = left + padding, top + padding
        canvas.create_text(text_x, text_y, anchor="nw", text=title, fill=MARKER_ACCENT, font=title_font)
        text_y += title_height + line_gap
        for line in note_lines:
            canvas.create_text(text_x, text_y, anchor="nw", text=line, fill=MARKER_FOREGROUND, font=note_font)
            text_y += note_height

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
            canvas.create_oval(*shape, fill=PILL_BACKGROUND, outline=accent, width=1)
        canvas.create_rectangle(left + radius, top, right - radius, bottom, fill=PILL_BACKGROUND, outline="")
        for y in (top, bottom):
            canvas.create_line(left + radius, y, right - radius, y, fill=accent, width=1)

        center_y = (top + bottom) // 2
        if self.voice_state == "listening":
            self._draw_level_bars(canvas, width // 2, center_y, accent, now)
        else:
            self._draw_sync_dots(canvas, width // 2, center_y, accent, now)

    @staticmethod
    def _draw_level_bars(canvas, center_x: int, center_y: int, accent: str, now: float) -> None:
        bars, spacing = 7, 9
        start_x = center_x - (bars - 1) * spacing // 2
        for index in range(bars):
            wave = abs(math.sin(now * 6.0 + index * 0.9))
            half = 2 + wave * 8
            x = start_x + index * spacing
            canvas.create_line(x, center_y - half, x, center_y + half, fill=accent, width=3, capstyle="round")

    @staticmethod
    def _draw_sync_dots(canvas, center_x: int, center_y: int, accent: str, now: float) -> None:
        for index in range(3):
            wave = 0.5 + 0.5 * math.sin(now * 5.0 - index * 0.9)
            radius_dot = 2.5 + wave * 3
            x = center_x + (index - 1) * 16
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


def _macos_appkit_available() -> bool:
    """Click-through is mandatory for a fullscreen window — without AppKit the
    child must stay in the small pill window that cannot block the desktop."""
    try:
        import AppKit  # noqa: F401

        return True
    except Exception:
        return False


def _macos_backing_scale() -> float:
    """Capture pixels per Tk point of the main screen (2.0 on retina)."""
    try:
        import AppKit

        return float(AppKit.NSScreen.mainScreen().backingScaleFactor()) or 1.0
    except Exception:
        return 1.0


def _macos_notch_width() -> int:
    """Width of the physical notch in points; 0 when this screen has none."""
    try:
        import AppKit

        screen = AppKit.NSScreen.mainScreen()
        if not hasattr(screen, "safeAreaInsets") or screen.safeAreaInsets().top <= 0:
            return 0
        full = screen.frame().size.width
        left = screen.auxiliaryTopLeftArea().size.width
        right = screen.auxiliaryTopRightArea().size.width
        return max(0, int(full - left - right))
    except Exception:
        return 0


def run_overlay_child() -> None:
    """Entry point of the overlay child: Tk on THIS process's main thread."""
    _ensure_tcl_env()
    fullscreen = sys.platform != "darwin" or _macos_appkit_available()
    overlay = StatusOverlay(
        pill_only=not fullscreen,
        marker_scale=_macos_backing_scale(),
        island=fullscreen and sys.platform == "darwin",
    )
    if overlay.island:
        overlay.notch_width = _macos_notch_width()
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
