import ctypes
import sys
from collections.abc import Callable
from typing import Any

from PIL import Image, ImageGrab

from ...types import Size

SM_CXSCREEN = 0
SM_CYSCREEN = 1
E_ACCESSDENIED = ctypes.c_int32(0x80070005).value


def enable_dpi_awareness(*, platform: str = sys.platform, libraries: Any | None = None) -> None:
    """Make the process per-monitor DPI aware so screenshot pixels and cursor
    coordinates share one coordinate system. No-op outside Windows."""
    if platform != "win32":
        return

    windows_libraries = libraries if libraries is not None else ctypes.windll
    try:
        set_process_dpi_awareness = windows_libraries.shcore.SetProcessDpiAwareness
        if libraries is None:
            set_process_dpi_awareness.argtypes = [ctypes.c_int]
            set_process_dpi_awareness.restype = ctypes.c_long
        hresult = set_process_dpi_awareness(2)
    except (AttributeError, OSError):
        should_fall_back = True
    else:
        signed_hresult = ctypes.c_int32(hresult).value
        # E_ACCESSDENIED means DPI awareness was already configured elsewhere.
        if signed_hresult == E_ACCESSDENIED:
            return
        should_fall_back = signed_hresult < 0

    if not should_fall_back:
        return

    set_process_dpi_aware = windows_libraries.user32.SetProcessDPIAware
    if libraries is None:
        set_process_dpi_aware.argtypes = []
        set_process_dpi_aware.restype = ctypes.c_int
    if not set_process_dpi_aware():
        raise OSError("SetProcessDPIAware failed")


def capture_primary_screen(*, grabber: Callable[..., Image.Image] = ImageGrab.grab) -> Image.Image:
    screenshot = grabber(all_screens=False)
    if screenshot.mode == "RGB":
        return screenshot
    return screenshot.convert("RGB")


def primary_screen_size(
    *,
    platform: str = sys.platform,
    get_system_metrics: Callable[[int], int] | None = None,
    grabber: Callable[..., Image.Image] = ImageGrab.grab,
) -> Size:
    """Primary display size in pixels, without paying for a full screenshot."""
    if get_system_metrics is None and platform == "win32":
        get_system_metrics = ctypes.windll.user32.GetSystemMetrics

    if get_system_metrics is not None:
        width = int(get_system_metrics(SM_CXSCREEN))
        height = int(get_system_metrics(SM_CYSCREEN))
        if width > 0 and height > 0:
            return (width, height)

    return capture_primary_screen(grabber=grabber).size


class PrimaryScreen:
    """Screen backend bound to the primary display."""

    def __init__(
        self,
        *,
        grabber: Callable[..., Image.Image] = ImageGrab.grab,
        size_reader: Callable[[], Size] | None = None,
    ) -> None:
        self._grabber = grabber
        self._size_reader = size_reader

    def capture(self) -> Image.Image:
        return capture_primary_screen(grabber=self._grabber)

    def size(self) -> Size:
        if self._size_reader is not None:
            return self._size_reader()
        return primary_screen_size(grabber=self._grabber)
