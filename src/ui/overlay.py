"""Always-on-top status indication drawn over the whole desktop.

One transparent click-through Tk window (the same technique as the
computer-use marker overlay) renders two independent layers:

- the **voice pill** at the bottom of the screen — animated level bars while
  recording ("listening"), pulsing dots from send until the reply lands
  ("syncing");
- the **computer border** — a breathing accent frame around the screen while
  the agent drives the desktop: a "hands off, the AI is working" signal,
  refreshed by every screen tool call and fading out a few seconds after the
  last one.

One Tk loop thread owns the window; `set_voice`/`ping_computer` only queue
commands and never block. Every failure is swallowed and logged — status
indication must never break a turn. `_apply`/`computer_active` are pure and
unit-tested without Tk.
"""

import atexit
import gc
import math
import queue
import threading
import time

from loguru import logger

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


class StatusOverlay:
    """Queue-driven overlay window; safe to call from any thread."""

    def __init__(self, screen_size=None):
        self.screen_size = screen_size
        self.voice_state: str | None = None  # None | "listening" | "syncing"
        self.computer_until = 0.0  # monotonic deadline of the border layer
        self._commands: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._failed = False

    # ------------------------------------------------------------ public API

    def set_voice(self, state: str | None) -> None:
        if self._start():
            self._commands.put(("voice", state))

    def ping_computer(self, linger: float = COMPUTER_LINGER) -> None:
        if self._start():
            self._commands.put(("computer", linger))

    def close(self) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
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

    def _start(self) -> bool:
        if self._failed:
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._thread = threading.Thread(target=self._serve, name="status-overlay", daemon=True)
            self._thread.start()
            atexit.register(self.close)
            return True

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

    def _run_loop(self, root, tkinter) -> None:
        from src.llm.tools.computer.overlay import enable_click_through

        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        width, height = self.screen_size or (root.winfo_screenwidth(), root.winfo_screenheight())
        root.geometry(f"{width}x{height}+0+0")
        try:
            root.attributes("-transparentcolor", TRANSPARENT_KEY)
            background = TRANSPARENT_KEY
        except tkinter.TclError:
            root.attributes("-alpha", 0.85)
            background = PILL_BACKGROUND
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
                    try:
                        enable_click_through(int(root.winfo_id()))
                    except (OSError, AttributeError, ValueError):
                        pass
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
        if self.computer_active(now):
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
        top = height - PILL_MARGIN - PILL_HEIGHT
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
