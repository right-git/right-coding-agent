import hashlib
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

from src.config.logging import logger

from .annotation import (
    DEFAULT_OUTPUT_PATH,
    annotate,
    image_to_base64,
    save_annotated_image,
)
from .detection import (
    box_center,
    clamp_box,
    clamp_point,
    describe_detections,
    expand_box,
    inference_scale,
    needs_refinement,
    screen_thumbnail,
    screens_roughly_equal,
    select_targets,
    shift_box,
)
from . import platforms
from .overlay import default_overlay
from .types import (
    Box,
    Detection,
    Locator,
    Marker,
    MouseButton,
    OverlayBackend,
    Point,
    PointerBackend,
    ScreenBackend,
    ScrollDirection,
    Size,
    TargetMode,
)

DEFAULT_INFERENCE_SIDE = 768
REFINE_MARGIN = 160
MIN_TARGET_PX = 40
REFINE_LIMIT = 3


class ComputerUse:
    """Screen understanding plus mouse and keyboard control.

    Perception comes from a vision locator that turns a natural-language
    description into on-screen boxes; control goes through a pointer backend.
    `mark_object` combines both to show the user where an element is instead of
    clicking it.

    Locating small controls takes two passes: a coarse one over the whole
    screen to find the neighbourhood, then a second one over a crop of that
    neighbourhood at native resolution. Downscaling the full screen turns a
    40-pixel-tall text field into a 17-pixel smear, and the resulting box drifts
    far enough to click outside the element; the crop keeps the pixels.
    """

    def __init__(
        self,
        *,
        locator: Locator | None = None,
        screen: ScreenBackend | None = None,
        pointer: PointerBackend | None = None,
        overlay: OverlayBackend | None = None,
        clipboard=None,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
        mark_duration: float = 6.0,
        dpi_aware: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        refine_margin: int = REFINE_MARGIN,
        min_target_px: int = MIN_TARGET_PX,
        refine_limit: int = REFINE_LIMIT,
        cache_detections: bool = True,
        cache_ttl: float = 15.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if dpi_aware:
            platforms.enable_dpi_awareness()

        self._locator = locator
        self.screen: ScreenBackend = screen or platforms.default_screen()
        self.pointer: PointerBackend = pointer or platforms.default_pointer()
        self.overlay: OverlayBackend = overlay or default_overlay()
        self._clipboard = clipboard
        self.output_path = Path(output_path)
        self.mark_duration = mark_duration
        self._sleep = sleep

        self.refine_margin = refine_margin
        self.min_target_px = min_target_px
        self.refine_limit = refine_limit
        self.cache_detections = cache_detections
        self.cache_ttl = cache_ttl
        self._clock = clock or time.monotonic

        self.last_screenshot: Image.Image | None = None
        self.last_detections: list[Detection] = []
        self.inference_calls = 0
        self._cache: dict[tuple, list[Detection]] = {}
        self._cache_key: bytes | None = None
        self._cache_thumb: Image.Image | None = None
        self._cache_size: tuple[int, int] | None = None  # boxes live in this pixel space
        self._cache_time = 0.0

    # ------------------------------------------------------------------ setup

    @property
    def locator(self) -> Locator:
        """The vision locator, imported and constructed on first use."""
        if self._locator is None:
            from .locator import LocateAnythingLocator

            logger.info("Loading the screen locator model")
            self._locator = LocateAnythingLocator()
        return self._locator

    def screen_size(self) -> Size:
        return self.screen.size()

    def active_window(self):
        """The window currently receiving keyboard input, or None."""
        from .platforms import foreground_window

        return foreground_window()

    def focus_window(self, title_contains: str):
        """Bring a window to the front and confirm it got there.

        Call this before typing: input goes to whatever is focused, and a
        sequence that assumes the wrong window types into someone else's
        document.
        """
        from .platforms import focus_window

        window = focus_window(title_contains)
        logger.info("Focused window [{}]", window.title)
        return window

    def require_window(self, title_contains: str):
        """Raise unless the focused window matches, without changing focus."""
        window = self.active_window()
        if window is None or title_contains.casefold() not in window.title.casefold():
            raise RuntimeError(
                f"expected {title_contains!r} to be focused, but it is " f"{(window.title if window else '<none>')!r}"
            )
        return window

    def close(self) -> None:
        """Release the overlay window and its Tk loop."""
        self.overlay.close()

    # ------------------------------------------------------------- perception

    def get_screenshot(self, *, save_to: str | Path | None = None) -> Image.Image:
        """Capture the primary display and remember it as the current view."""
        screenshot = self.screen.capture()
        self.last_screenshot = screenshot
        if save_to is not None:
            screenshot.save(save_to)
        return screenshot

    def screenshot_base64(
        self,
        *,
        max_side: int | None = 1280,
        quality: int = 80,
        fresh: bool = True,
    ) -> str:
        """The screen as base64 JPEG; `fresh=False` reuses the last capture."""
        if fresh or self.last_screenshot is None:
            image = self.get_screenshot()
        else:
            image = self.last_screenshot
        return image_to_base64(image, max_side=max_side, quality=quality)

    def annotated_base64(
        self,
        image: Image.Image | None = None,
        detections: Sequence[Detection] | None = None,
        *,
        max_side: int | None = 1280,
        quality: int = 80,
    ) -> str:
        """The current view with detection boxes drawn, as base64 JPEG."""
        if image is None:
            image = self.last_screenshot
        if image is None:
            raise RuntimeError("No screenshot to annotate; call get_screenshot first")
        annotated = annotate(image, self.last_detections if detections is None else detections)
        return image_to_base64(annotated, max_side=max_side, quality=quality)

    @property
    def inference_side(self) -> int:
        """Longest side of the image the locator actually looks at.

        The Locator protocol does not require this attribute, so anything may
        be there; fall back to the default rather than trusting it.
        """
        side = getattr(self.locator, "max_image_side", DEFAULT_INFERENCE_SIDE)
        if isinstance(side, int) and not isinstance(side, bool) and side > 0:
            return side
        return DEFAULT_INFERENCE_SIDE

    def _locate(self, image: Image.Image, description: str) -> list[Detection]:
        self.inference_calls += 1
        return list(self.locator.locate(image, description))

    def _refine(
        self,
        image: Image.Image,
        description: str,
        detection: Detection,
        scale: float,
    ) -> Detection:
        """Look again at a crop around `detection` when it is too small to trust.

        `scale` is how much the previous pass shrank its input: a search that
        already ran at native resolution needs no second look.
        """
        if not needs_refinement(detection.box, scale, self.min_target_px):
            return detection

        area = expand_box(detection.box, self.refine_margin, image.size)
        found = self._locate(image.crop(area), description)
        if not found:
            logger.debug("Refinement found nothing for [{}]; keeping the coarse box", description)
            return detection
        return Detection(label=found[0].label, box=shift_box(found[0].box, (area[0], area[1])))

    def locate_object(
        self,
        description: str,
        *,
        screenshot: Image.Image | None = None,
        mode: TargetMode = "all",
        region: Box | None = None,
        refine: bool = True,
        annotate_to: str | Path | None = None,
    ) -> list[Detection]:
        """Find every region matching `description` on a fresh screenshot.

        `region` limits the search to part of the screen, which is both faster
        and more accurate than searching the whole display — pass it whenever
        the neighbourhood is already known (an app window, a panel found
        earlier). Small targets are located twice; see the class docstring.
        """
        started = time.perf_counter()
        image = screenshot if screenshot is not None else self.get_screenshot()
        self.last_screenshot = image

        cache_key = (description, region, mode, refine)
        cached = self._lookup(image, cache_key)
        if cached is not None:
            self.last_detections = cached
            return list(cached)

        if region is None:
            search_image, origin = image, (0, 0)
        else:
            area = clamp_box(region, image.size)
            search_image, origin = image.crop(area), (area[0], area[1])

        detections = [
            Detection(label=detection.label, box=shift_box(detection.box, origin))
            for detection in self._locate(search_image, description)
        ]
        scale = inference_scale(search_image.size, self.inference_side)

        detections = select_targets(detections, mode)
        if refine:
            detections = [
                (self._refine(image, description, detection, scale) if index < self.refine_limit else detection)
                for index, detection in enumerate(detections)
            ]

        self.last_detections = detections
        self._store(image, cache_key, detections)
        logger.info(
            "Located [{}] region(s) for description [{}] in [{:.1f}s]",
            len(detections),
            description,
            time.perf_counter() - started,
        )

        if annotate_to is not None:
            save_annotated_image(image, detections, annotate_to)

        return detections

    # ------------------------------------------------------------------ cache

    @staticmethod
    def _digest(image: Image.Image) -> bytes:
        return hashlib.blake2b(image.tobytes(), digest_size=16).digest()

    def _lookup(self, image: Image.Image, key: tuple) -> list[Detection] | None:
        """Reuse a result while the screen is byte-identical — or, within
        `cache_ttl` seconds of the last inference, merely near-identical.

        A live desktop never stays byte-identical between two captures (the
        menu-bar clock and terminal spinners always tick), and on CPU/MPS a
        full locate costs many seconds — so a locate followed by a mark or a
        click must not pay twice for a few changed pixels. Big changes (a
        scroll, a moved window) still miss, and the TTL runs from the last
        real inference, so fuzzy hits cannot chain stale results forever.
        """
        if not self.cache_detections:
            return None
        if self._digest(image) == self._cache_key:
            return self._cache.get(key)
        if (
            self._cache_thumb is not None
            and image.size == self._cache_size  # boxes are in the old capture's pixel space
            and self._clock() - self._cache_time <= self.cache_ttl
            and screens_roughly_equal(screen_thumbnail(image), self._cache_thumb)
        ):
            return self._cache.get(key)
        return None

    def _store(self, image: Image.Image, key: tuple, detections: list[Detection]) -> None:
        if not self.cache_detections:
            return
        digest = self._digest(image)
        if digest != self._cache_key:
            self._cache_key = digest
            self._cache = {}
        self._cache_thumb = screen_thumbnail(image)
        self._cache_size = image.size
        self._cache_time = self._clock()
        self._cache[key] = list(detections)

    def find_object(
        self,
        description: str,
        *,
        screenshot: Image.Image | None = None,
        region: Box | None = None,
    ) -> Detection | None:
        """The first region matching `description`, or None."""
        detections = self.locate_object(description, screenshot=screenshot, mode="first", region=region)
        return detections[0] if detections else None

    def locate_point(
        self,
        description: str,
        *,
        screenshot: Image.Image | None = None,
        region: Box | None = None,
    ) -> Point | None:
        """Clickable center of the first region matching `description`."""
        detection = self.find_object(description, screenshot=screenshot, region=region)
        if detection is None:
            return None
        return self.center_of(detection)

    def center_of(self, detection: Detection) -> Point:
        image_size = self.last_screenshot.size if self.last_screenshot is not None else self.screen_size()
        return box_center(detection.box, image_size)

    def describe(self, detections: Sequence[Detection] | None = None) -> str:
        """Human- and LLM-readable report of the latest detections."""
        image_size = self.last_screenshot.size if self.last_screenshot is not None else self.screen_size()
        return describe_detections(self.last_detections if detections is None else detections, image_size)

    def save_annotated(
        self,
        image: Image.Image | None = None,
        detections: Sequence[Detection] | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """Write the current view with boxes drawn on it."""
        if image is None:
            image = self.last_screenshot
        if image is None:
            raise RuntimeError("No screenshot to annotate; call get_screenshot first")
        return save_annotated_image(
            image,
            self.last_detections if detections is None else detections,
            self.output_path if output_path is None else output_path,
        )

    # ------------------------------------------------------------------ mouse

    def move_mouse(self, x: int, y: int) -> Point:
        target = clamp_point((int(x), int(y)), self.screen_size())
        self.pointer.move(*target)
        return target

    def mouse_position(self) -> Point:
        return self.pointer.position()

    def _click(
        self,
        button: MouseButton,
        count: int,
        x: int | None,
        y: int | None,
        modifiers: Sequence[str],
    ) -> None:
        if x is not None and y is not None:
            self.move_mouse(x, y)
        logger.info(
            "Mouse click button [{}] count [{}] at [{}]",
            button,
            count,
            (x, y) if x is not None else "current position",
        )
        self.pointer.click(button, count, modifiers)

    def left_click(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        modifiers: Sequence[str] = (),
    ) -> None:
        self._click("left", 1, x, y, modifiers)

    def right_click(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        modifiers: Sequence[str] = (),
    ) -> None:
        self._click("right", 1, x, y, modifiers)

    def middle_click(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        modifiers: Sequence[str] = (),
    ) -> None:
        self._click("middle", 1, x, y, modifiers)

    def double_click(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        modifiers: Sequence[str] = (),
    ) -> None:
        self._click("left", 2, x, y, modifiers)

    def triple_click(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        modifiers: Sequence[str] = (),
    ) -> None:
        self._click("left", 3, x, y, modifiers)

    def mouse_down(self, button: MouseButton = "left") -> None:
        self.pointer.mouse_down(button)

    def mouse_up(self, button: MouseButton = "left") -> None:
        self.pointer.mouse_up(button)

    def drag(self, start: Point, end: Point, *, button: MouseButton = "left") -> None:
        screen_size = self.screen_size()
        self.pointer.drag(
            clamp_point(start, screen_size),
            clamp_point(end, screen_size),
            button=button,
        )

    def scroll(
        self,
        direction: ScrollDirection,
        amount: int = 3,
        x: int | None = None,
        y: int | None = None,
        *,
        modifiers: Sequence[str] = (),
    ) -> None:
        if x is not None and y is not None:
            self.move_mouse(x, y)
        self.pointer.scroll(direction, amount, modifiers)

    # --------------------------------------------------------------- keyboard

    def type_text(self, text: str) -> None:
        logger.info("Typing [{}] character(s)", len(text))
        self.pointer.type_text(text)

    def read_clipboard(self) -> str:
        """Current clipboard text."""
        if self._clipboard is None:
            self._clipboard = platforms.default_clipboard()
        return self._clipboard.read_text()

    def paste_text(self, text: str, *, settle: float = 0.15) -> None:
        """Put `text` on the clipboard and paste it into the focused field.

        Use this instead of `type_text` for anything long or multi-line: a
        newline typed into a chat box sends the message early, and SendInput
        emits two events per character.
        """
        self.write_clipboard(text)
        self._sleep(settle)
        self.key("ctrl+v")
        self._sleep(settle)

    def write_clipboard(self, text: str) -> None:
        if self._clipboard is None:
            self._clipboard = platforms.default_clipboard()
        self._clipboard.write_text(text)

    def copy(self, *, settle: float = 0.15) -> str:
        """Press Ctrl+C and return what landed in the clipboard.

        Reading the value back beats transcribing it from a screenshot: a URL
        or an id has no redundancy, so one misread character is silent
        corruption.
        """
        self.key("ctrl+c")
        self._sleep(settle)
        return self.read_clipboard()

    def copy_address_bar(self, *, settle: float = 0.15) -> str:
        """Focus the browser address bar and return the current URL."""
        self.key("ctrl+l")
        self._sleep(settle)
        return self.copy(settle=settle)

    def key(self, combination: str) -> None:
        logger.info("Pressing keys [{}]", combination)
        self.pointer.key(combination)

    def hold_key(self, combination: str, duration: float) -> None:
        self.pointer.hold_key(combination, duration)

    def wait(self, duration: float) -> None:
        if duration < 0:
            raise ValueError("duration must not be negative")
        self._sleep(duration)

    # ------------------------------------------------------- described action

    def move_to_object(
        self,
        description: str,
        *,
        screenshot: Image.Image | None = None,
        region: Box | None = None,
    ) -> Detection | None:
        """Move the pointer onto the first matching element without clicking."""
        detection = self.find_object(description, screenshot=screenshot, region=region)
        if detection is None:
            return None
        self.move_mouse(*self.center_of(detection))
        return detection

    def click_object(
        self,
        description: str,
        *,
        button: MouseButton = "left",
        count: int = 1,
        modifiers: Sequence[str] = (),
        screenshot: Image.Image | None = None,
        region: Box | None = None,
    ) -> Detection | None:
        """Locate an element by description and click its center."""
        detection = self.find_object(description, screenshot=screenshot, region=region)
        if detection is None:
            logger.warning("Nothing matched [{}]; no click was made", description)
            return None
        x, y = self.center_of(detection)
        self._click(button, count, x, y, modifiers)
        return detection

    def type_into_object(
        self,
        description: str,
        text: str,
        *,
        clear: bool = False,
        screenshot: Image.Image | None = None,
    ) -> Detection | None:
        """Click a field found by description, then type into it."""
        detection = self.click_object(description, screenshot=screenshot)
        if detection is None:
            return None
        if clear:
            self.key("ctrl+a")
            self.key("delete")
        self.type_text(text)
        return detection

    # --------------------------------------------------------------- markers

    def mark_box(
        self,
        box: Box,
        title: str,
        note: str = "",
        *,
        duration: float | None = None,
        move_pointer: bool = True,
        anchor: Point | None = None,
    ) -> Point:
        """Outline a region and show an explanatory tooltip next to it."""
        image_size = self.last_screenshot.size if self.last_screenshot is not None else self.screen_size()
        target = anchor or box_center(box, image_size)
        if move_pointer:
            target = self.move_mouse(*target)

        self.overlay.show(
            Marker(
                box=box,
                title=title,
                note=note,
                duration=self.mark_duration if duration is None else duration,
                anchor=target,
            )
        )
        logger.info("Marked [{}] at [{}]", title, target)
        return target

    def mark_point(
        self,
        x: int,
        y: int,
        title: str,
        note: str = "",
        *,
        radius: int = 24,
        duration: float | None = None,
        move_pointer: bool = True,
    ) -> Point:
        """Mark a bare coordinate when the box is already known."""
        return self.mark_box(
            (x - radius, y - radius, x + radius, y + radius),
            title,
            note,
            duration=duration,
            move_pointer=move_pointer,
            anchor=(x, y),
        )

    def mark_object(
        self,
        description: str,
        note: str = "",
        *,
        title: str | None = None,
        duration: float | None = None,
        move_pointer: bool = True,
        screenshot: Image.Image | None = None,
        region: Box | None = None,
    ) -> Detection | None:
        """Show the user where an element is.

        Finds the element described in plain language, moves the pointer onto
        it, outlines it, and shows a tooltip with `note` — what the element is
        and what happens when it is used. Nothing is clicked.
        """
        detection = self.find_object(description, screenshot=screenshot, region=region)
        if detection is None:
            logger.warning("Nothing matched [{}]; no marker was shown", description)
            return None

        self.mark_box(
            detection.box,
            title or detection.label or description,
            note,
            duration=duration,
            move_pointer=move_pointer,
        )
        return detection

    def clear_marks(self) -> None:
        self.overlay.hide()
