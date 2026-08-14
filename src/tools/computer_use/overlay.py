import atexit
import ctypes
import gc
import queue
import sys
import textwrap
import threading
from dataclasses import dataclass, field

from .detection import box_center
from .types import Marker, Point, Size


GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000


@dataclass(frozen=True)
class OverlayStyle:
    accent: str = "#FF3B30"
    background: str = "#11161C"
    foreground: str = "#F2F5F8"
    outline_width: int = 3
    padding: int = 12
    line_gap: int = 6
    offset: int = 24
    margin: int = 12
    wrap_chars: int = 46
    font_family: str = "Segoe UI"
    title_size: int = 12
    note_size: int = 10
    transparent_key: str = "#0B1F0B"
    poll_interval_ms: int = 50


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


def tooltip_placement(
    anchor: Point,
    tooltip_size: Size,
    screen_size: Size,
    *,
    offset: int = 24,
    margin: int = 12,
) -> Point:
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


def connector_corner(anchor: Point, position: Point, tooltip_size: Size) -> Point:
    """Corner of the tooltip that faces `anchor`, for the connector line."""
    anchor_x, anchor_y = anchor
    x, y = position
    width, height = tooltip_size
    return (
        x if anchor_x < x else x + width,
        y if anchor_y < y else y + height,
    )


def enable_click_through(window_handle: int, *, user32=None) -> None:
    """Let mouse events pass through the overlay to the app underneath."""
    library = user32
    if library is None:
        if sys.platform != "win32":
            return
        library = ctypes.windll.user32

    styles = library.GetWindowLongW(window_handle, GWL_EXSTYLE)
    library.SetWindowLongW(
        window_handle,
        GWL_EXSTYLE,
        styles
        | WS_EX_LAYERED
        | WS_EX_TRANSPARENT
        | WS_EX_TOOLWINDOW
        | WS_EX_NOACTIVATE,
    )


class NullOverlay:
    """Overlay backend that records markers instead of drawing them."""

    def __init__(self) -> None:
        self.markers: list[Marker] = []
        self.visible = False
        self.closed = False

    def show(self, marker: Marker) -> None:
        self.markers.append(marker)
        self.visible = True

    def hide(self) -> None:
        self.visible = False

    def close(self) -> None:
        self.visible = False
        self.closed = True


@dataclass
class TkOverlay:
    """Borderless, click-through highlight drawn on top of the desktop.

    One Tk loop is created on first use and owns the window for the life of the
    process: showing a marker only sends a command to that loop, so it never
    blocks the caller and never pays window-creation cost twice. The marker
    hides itself after `Marker.duration` seconds.

    Tk objects must be created and destroyed on the same thread, so the loop
    thread also performs teardown, triggered by `close()` or at interpreter
    exit.
    """

    style: OverlayStyle = field(default_factory=OverlayStyle)
    screen_size: Size | None = None

    def __post_init__(self) -> None:
        self._commands: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._failed = False

    # ----------------------------------------------------------- public API

    def show(self, marker: Marker) -> None:
        if self._start():
            self._commands.put(("show", marker))

    def hide(self) -> None:
        if self._is_running():
            self._commands.put(("hide", None))

    def close(self) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is None or not thread.is_alive():
            return
        self._commands.put(("stop", None))
        thread.join(timeout=5.0)

    # -------------------------------------------------------------- internals

    def _is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _start(self) -> bool:
        if self._failed:
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._thread = threading.Thread(
                target=self._serve, name="computer-use-overlay", daemon=True
            )
            self._thread.start()
            atexit.register(self.close)
            return True

    def _serve(self) -> None:
        try:
            import tkinter
            from tkinter import font as tkinter_font
        except ImportError:
            self._failed = True
            return

        try:
            root = tkinter.Tk()
        except Exception:
            self._failed = True
            return

        canvas = None
        try:
            root.withdraw()
            root.overrideredirect(True)
            root.attributes("-topmost", True)

            width, height = self.screen_size or (
                root.winfo_screenwidth(),
                root.winfo_screenheight(),
            )
            root.geometry(f"{width}x{height}+0+0")

            try:
                root.attributes("-transparentcolor", self.style.transparent_key)
                canvas_background = self.style.transparent_key
            except tkinter.TclError:
                root.attributes("-alpha", 0.9)
                canvas_background = self.style.background

            canvas = tkinter.Canvas(
                root,
                width=width,
                height=height,
                bg=canvas_background,
                highlightthickness=0,
                borderwidth=0,
            )
            canvas.pack(fill="both", expand=True)

            hide_job: list[str | None] = [None]

            def cancel_hide() -> None:
                if hide_job[0] is not None:
                    root.after_cancel(hide_job[0])
                    hide_job[0] = None

            def hide() -> None:
                cancel_hide()
                canvas.delete("all")
                root.withdraw()

            def show(marker: Marker) -> None:
                cancel_hide()
                canvas.delete("all")
                self._draw(canvas, marker, (width, height), tkinter_font)
                root.deiconify()
                root.attributes("-topmost", True)
                root.update_idletasks()
                try:
                    enable_click_through(int(root.winfo_id()))
                except (OSError, AttributeError, ValueError):
                    pass
                hide_job[0] = root.after(max(1, int(marker.duration * 1000)), hide)

            def pump() -> None:
                try:
                    while True:
                        command, payload = self._commands.get_nowait()
                        if command == "stop":
                            cancel_hide()
                            root.quit()
                            return
                        if command == "show":
                            show(payload)
                        elif command == "hide":
                            hide()
                except queue.Empty:
                    pass
                root.after(self.style.poll_interval_ms, pump)

            root.after(0, pump)
            root.mainloop()
        finally:
            try:
                root.destroy()
            except Exception:
                pass
            # Release every Tk reference on this thread: finalizing the
            # interpreter from another thread aborts the process.
            del canvas
            del root
            gc.collect()

    def _draw(self, canvas, marker: Marker, screen_size: Size, tkinter_font) -> None:
        style = self.style
        x1, y1, x2, y2 = marker.box
        canvas.create_rectangle(
            x1, y1, x2, y2, outline=style.accent, width=style.outline_width
        )

        anchor = marker.anchor or box_center(marker.box, screen_size)
        title_font = tkinter_font.Font(
            family=style.font_family, size=style.title_size, weight="bold"
        )
        note_font = tkinter_font.Font(family=style.font_family, size=style.note_size)

        note_lines = wrap_note(marker.note, style.wrap_chars)
        title_height = title_font.metrics("linespace")
        note_height = note_font.metrics("linespace")
        content_width = max(
            [title_font.measure(marker.title)]
            + [note_font.measure(line) for line in note_lines]
        )
        content_height = title_height + (
            style.line_gap + note_height * len(note_lines) if note_lines else 0
        )
        tooltip_size = (
            content_width + style.padding * 2,
            content_height + style.padding * 2,
        )
        position = tooltip_placement(
            anchor,
            tooltip_size,
            screen_size,
            offset=style.offset,
            margin=style.margin,
        )

        corner = connector_corner(anchor, position, tooltip_size)
        canvas.create_line(*anchor, *corner, fill=style.accent, width=2)
        canvas.create_oval(
            anchor[0] - 5,
            anchor[1] - 5,
            anchor[0] + 5,
            anchor[1] + 5,
            fill=style.accent,
            outline=style.accent,
        )

        left, top = position
        canvas.create_rectangle(
            left,
            top,
            left + tooltip_size[0],
            top + tooltip_size[1],
            fill=style.background,
            outline=style.accent,
            width=2,
        )
        text_x = left + style.padding
        text_y = top + style.padding
        canvas.create_text(
            text_x,
            text_y,
            anchor="nw",
            text=marker.title,
            fill=style.accent,
            font=title_font,
        )
        text_y += title_height + style.line_gap
        for line in note_lines:
            canvas.create_text(
                text_x,
                text_y,
                anchor="nw",
                text=line,
                fill=style.foreground,
                font=note_font,
            )
            text_y += note_height
