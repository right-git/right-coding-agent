"""Primary-monitor capture through the cross-platform `mss` library."""

from PIL import Image

from ...types import Size


class MssScreen:
    """Screen backend bound to the primary display, via mss."""

    def capture(self) -> Image.Image:
        import mss

        with mss.mss() as grabber:
            shot = grabber.grab(grabber.monitors[1])
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def size(self) -> Size:
        import mss

        with mss.mss() as grabber:
            monitor = grabber.monitors[1]
        return (int(monitor["width"]), int(monitor["height"]))
