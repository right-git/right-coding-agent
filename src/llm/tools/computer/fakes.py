"""Inert backends for dry runs, headless environments, and tests."""

from collections.abc import Sequence

from PIL import Image

from .types import Detection, MouseButton, Point, ScrollDirection, Size


class StaticScreen:
    """Screen backend that serves prepared images instead of the desktop."""

    def __init__(self, images: Sequence[Image.Image], size: Size | None = None) -> None:
        if not images:
            raise ValueError("StaticScreen needs at least one image")
        self.images = list(images)
        self.captures = 0
        self._size = size or self.images[0].size

    def capture(self) -> Image.Image:
        index = min(self.captures, len(self.images) - 1)
        self.captures += 1
        return self.images[index]

    def size(self) -> Size:
        return self._size


class RecordingPointer:
    """Pointer backend that records requested input without performing it."""

    def __init__(self, position: Point = (0, 0)) -> None:
        self.moves: list[Point] = []
        self.clicks: list[tuple[str, int, tuple[str, ...]]] = []
        self.buttons: list[tuple[str, str]] = []
        self.drags: list[tuple[Point, Point, str]] = []
        self.scrolls: list[tuple[str, int, tuple[str, ...]]] = []
        self.typed: list[str] = []
        self.keys: list[str] = []
        self.holds: list[tuple[str, float]] = []
        self._position = position

    def move(self, x: int, y: int) -> None:
        self._position = (x, y)
        self.moves.append((x, y))

    def position(self) -> Point:
        return self._position

    def click(
        self,
        button: MouseButton = "left",
        count: int = 1,
        modifiers: Sequence[str] = (),
    ) -> None:
        self.clicks.append((button, count, tuple(modifiers)))

    def mouse_down(self, button: MouseButton = "left") -> None:
        self.buttons.append(("down", button))

    def mouse_up(self, button: MouseButton = "left") -> None:
        self.buttons.append(("up", button))

    def drag(self, start: Point, end: Point, *, button: MouseButton = "left") -> None:
        self.drags.append((start, end, button))

    def scroll(
        self,
        direction: ScrollDirection,
        amount: int = 3,
        modifiers: Sequence[str] = (),
    ) -> None:
        self.scrolls.append((direction, amount, tuple(modifiers)))

    def type_text(self, text: str) -> None:
        self.typed.append(text)

    def key(self, combination: str) -> None:
        self.keys.append(combination)

    def hold_key(self, combination: str, duration: float) -> None:
        self.holds.append((combination, duration))


class ScriptedLocator:
    """Locator that replays prepared detections for each call."""

    def __init__(
        self,
        responses: Sequence[Sequence[Detection]],
        *,
        max_image_side: int = 768,
    ) -> None:
        self.responses = [list(response) for response in responses]
        self.max_image_side = max_image_side
        self.calls: list[tuple[Size, str]] = []

    def locate(self, image: Image.Image, description: str) -> list[Detection]:
        self.calls.append((image.size, description))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        if not self.responses:
            return []
        return list(self.responses[index])


class MemoryClipboard:
    """Clipboard double for dry runs and tests."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.reads = 0
        self.writes: list[str] = []

    def read_text(self) -> str:
        self.reads += 1
        return self.text

    def write_text(self, text: str) -> None:
        self.text = text
        self.writes.append(text)
