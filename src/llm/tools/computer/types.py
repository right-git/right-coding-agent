from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from PIL import Image

Box = tuple[int, int, int, int]
Point = tuple[int, int]
Size = tuple[int, int]

TargetMode = Literal["first", "all"]
MouseButton = Literal["left", "right", "middle"]
ScrollDirection = Literal["up", "down", "left", "right"]


@dataclass(frozen=True)
class Detection:
    """A region of the screen that matched a natural-language description."""

    label: str
    box: Box


@dataclass(frozen=True)
class Marker:
    """A highlight the user sees on screen: an outlined box plus a tooltip."""

    box: Box
    title: str
    note: str = ""
    duration: float = 6.0
    anchor: Point | None = None


class Locator(Protocol):
    def locate(self, image: Image.Image, description: str) -> Sequence[Detection]: ...


class ScreenBackend(Protocol):
    def capture(self) -> Image.Image: ...

    def size(self) -> Size: ...


class PointerBackend(Protocol):
    def move(self, x: int, y: int) -> None: ...

    def position(self) -> Point: ...

    def click(
        self,
        button: MouseButton = "left",
        count: int = 1,
        modifiers: Sequence[str] = (),
    ) -> None: ...

    def mouse_down(self, button: MouseButton = "left") -> None: ...

    def mouse_up(self, button: MouseButton = "left") -> None: ...

    def drag(
        self,
        start: Point,
        end: Point,
        *,
        button: MouseButton = "left",
    ) -> None: ...

    def scroll(
        self,
        direction: ScrollDirection,
        amount: int = 3,
        modifiers: Sequence[str] = (),
    ) -> None: ...

    def type_text(self, text: str) -> None: ...

    def key(self, combination: str) -> None: ...

    def hold_key(self, combination: str, duration: float) -> None: ...


class OverlayBackend(Protocol):
    def show(self, marker: Marker) -> None: ...

    def hide(self) -> None: ...

    def close(self) -> None: ...
