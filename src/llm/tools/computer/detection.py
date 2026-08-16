import re
from collections.abc import Sequence

from PIL import Image, ImageChops, ImageStat

from .types import Box, Detection, Point, Size, TargetMode

NORMALIZED_SCALE = 1000
THUMBNAIL_SIDE = 64
SCREEN_SIMILARITY_TOLERANCE = 3.0  # mean abs pixel diff on the thumbnail, 0..255


REGION_STRIP_FRACTION = 0.15
_REGION_NUMBERS = re.compile(r"^\s*(-?\d+)\s*[,;\s]\s*(-?\d+)\s*[,;\s]\s*(-?\d+)\s*[,;\s]\s*(-?\d+)\s*$")


def region_names() -> list[str]:
    """The named-region vocabulary understood by `parse_region`."""
    cells = [f"{row}-{column}" for row in ("top", "middle", "bottom") for column in ("left", "center", "right")]
    return cells + ["center", "top-bar", "bottom-bar", "left-bar", "right-bar"]


def parse_region(spec: str | None, screen_size: Size) -> Box | None:
    """A region spec string → pixel box on this screen, or None for "everywhere".

    Accepts the 3x3 grid the locator's own reports use ("top-right",
    "middle-center", plain "center"), thin edge strips for bars and docks
    ("top-bar", "bottom-bar", "left-bar", "right-bar"), or explicit pixels
    "l,t,r,b" in the same coordinate space locate results are reported in.
    Unknown specs raise ValueError naming the vocabulary, so a tool error
    teaches the caller the valid forms.
    """
    if not spec:
        return None
    width, height = screen_size
    text = str(spec).strip().lower()

    numbers = _REGION_NUMBERS.match(text)
    if numbers:
        return clamp_box(tuple(int(value) for value in numbers.groups()), screen_size)

    columns = {"left": (0, width // 3), "center": (width // 3, 2 * width // 3), "right": (2 * width // 3, width)}
    rows = {"top": (0, height // 3), "middle": (height // 3, 2 * height // 3), "bottom": (2 * height // 3, height)}
    named: dict[str, Box] = {
        f"{row}-{column}": (x1, y1, x2, y2) for row, (y1, y2) in rows.items() for column, (x1, x2) in columns.items()
    }
    named["center"] = named["middle-center"]
    strip_width, strip_height = round(width * REGION_STRIP_FRACTION), round(height * REGION_STRIP_FRACTION)
    named["top-bar"] = (0, 0, width, strip_height)
    named["bottom-bar"] = (0, height - strip_height, width, height)
    named["left-bar"] = (0, 0, strip_width, height)
    named["right-bar"] = (width - strip_width, 0, width, height)

    if text in named:
        return named[text]
    raise ValueError(f"Unknown region {spec!r}; use one of {', '.join(region_names())}, or pixel bounds 'l,t,r,b'")


def screen_thumbnail(image: Image.Image, side: int = THUMBNAIL_SIDE) -> Image.Image:
    """A tiny grayscale of the screen, cheap to compare for near-equality."""
    height = max(1, round(side * image.height / image.width))
    return image.convert("L").resize((side, height))


def screens_roughly_equal(
    thumb_a: Image.Image,
    thumb_b: Image.Image,
    tolerance: float = SCREEN_SIMILARITY_TOLERANCE,
) -> bool:
    """Whether two screen thumbnails differ only by small local changes.

    A live desktop is never byte-identical between two captures — the menu-bar
    clock and terminal spinners always tick — but those change a fraction of a
    percent of pixels, while a scroll or window move shifts most of them.
    """
    if thumb_a.size != thumb_b.size:
        return False
    difference = ImageChops.difference(thumb_a, thumb_b)
    return ImageStat.Stat(difference).mean[0] <= tolerance


_TOKEN_PATTERN = re.compile(
    r"<ref>(?P<label>(?:(?!<ref>|<box>).)*?)</ref>" r"|(?P<box><box>(?:(?!<ref>|<box>).)*?</box>)",
    re.DOTALL,
)
_BOX_PATTERN = re.compile(
    r"<box>\s*"
    r"<\s*([+-]?\d+)\s*>\s*"
    r"<\s*([+-]?\d+)\s*>\s*"
    r"<\s*([+-]?\d+)\s*>\s*"
    r"<\s*([+-]?\d+)\s*>\s*"
    r"</box>"
)


def resize_for_inference(image: Image.Image, max_side: int = 768) -> Image.Image:
    if max_side <= 0:
        raise ValueError("max_side must be positive")

    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def scale_normalized_box(box: Sequence[int], image_size: Size) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width, height = image_size
    return (
        x1 / NORMALIZED_SCALE * width,
        y1 / NORMALIZED_SCALE * height,
        x2 / NORMALIZED_SCALE * width,
        y2 / NORMALIZED_SCALE * height,
    )


def parse_detections(model_output: str, image_size: Size) -> list[Detection]:
    """Turn `<ref>label</ref><box><x1><y1><x2><y2></box>` output into boxes.

    Coordinates are emitted by the model on a 0..1000 grid and are rescaled to
    the pixel space of `image_size`. Malformed boxes are skipped.
    """
    detections: list[Detection] = []
    detection_label = "object"

    for token_match in _TOKEN_PATTERN.finditer(model_output):
        raw_label = token_match.group("label")
        if raw_label is not None:
            detection_label = (raw_label or "").strip() or "object"
            continue

        coordinate_match = _BOX_PATTERN.fullmatch(token_match.group("box"))
        if coordinate_match is None:
            continue

        normalized_box = tuple(int(value) for value in coordinate_match.groups())
        x1, y1, x2, y2 = scale_normalized_box(normalized_box, image_size)
        detections.append(
            Detection(
                label=detection_label,
                box=(round(x1), round(y1), round(x2), round(y2)),
            )
        )

    return detections


def clamp_point(point: Point, image_size: Size) -> Point:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    x, y = point
    return (
        min(max(x, 0), width - 1),
        min(max(y, 0), height - 1),
    )


def box_center(box: Box, image_size: Size) -> Point:
    x1, y1, x2, y2 = box
    return clamp_point(((x1 + x2) // 2, (y1 + y2) // 2), image_size)


def clamp_box(box: Box, image_size: Size) -> Box:
    """Trim a box to the image, keeping corners ordered."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    x1, y1, x2, y2 = box
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return (
        min(max(left, 0), width),
        min(max(top, 0), height),
        min(max(right, 0), width),
        min(max(bottom, 0), height),
    )


def expand_box(box: Box, margin: int, image_size: Size) -> Box:
    """Grow a box by `margin` on every side, clipped to the image."""
    if margin < 0:
        raise ValueError("margin must not be negative")

    x1, y1, x2, y2 = box
    return clamp_box((x1 - margin, y1 - margin, x2 + margin, y2 + margin), image_size)


def shift_box(box: Box, offset: Point) -> Box:
    """Move a box from crop coordinates back into full-image coordinates."""
    offset_x, offset_y = offset
    x1, y1, x2, y2 = box
    return (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)


def inference_scale(image_size: Size, max_side: int) -> float:
    """How much an image shrinks before the model sees it."""
    if max_side <= 0:
        raise ValueError("max_side must be positive")

    longest = max(image_size)
    if longest <= 0:
        raise ValueError("image dimensions must be positive")
    return min(1.0, max_side / longest)


def needs_refinement(box: Box, scale: float, min_target_px: int) -> bool:
    """True when the target is too small in the model's frame to be located
    precisely, so it is worth looking again at a crop.

    Measured: at roughly 30 model-pixels the box drifts by ~100 screen pixels,
    at 42 it lands within 5. A pass that already ran at native resolution
    (`scale` of 1) is never refined — a tighter crop cannot add pixels that
    were never lost.
    """
    if scale >= 1.0:
        return False

    x1, y1, x2, y2 = box
    return min(abs(x2 - x1), abs(y2 - y1)) * scale < min_target_px


def select_targets(detections: Sequence[Detection], mode: TargetMode) -> list[Detection]:
    if mode == "first":
        return list(detections[:1])
    if mode == "all":
        return list(detections)
    raise ValueError(f"Unsupported target mode: {mode}")


def position_label(center: Point, image_size: Size) -> str:
    """Coarse position of a point on screen: "top-left" … "bottom-right".

    Reports include it so matches can be told apart, and refined queries can
    be phrased the way people talk about a screen.
    """
    x, y = center
    width, height = image_size
    if x < width / 3:
        column = "left"
    elif x < 2 * width / 3:
        column = "center"
    else:
        column = "right"
    if y < height / 3:
        row = "top"
    elif y < 2 * height / 3:
        row = "middle"
    else:
        row = "bottom"
    return f"{row}-{column}"


def describe_detections(detections: Sequence[Detection], image_size: Size) -> str:
    """Render detections as a compact report an LLM (or a human) can read."""
    if not detections:
        return "No matching region was found on the screen."

    lines = []
    for index, detection in enumerate(detections, start=1):
        center = box_center(detection.box, image_size)
        lines.append(
            f"{index}. {detection.label}: "
            f"box={detection.box}, center={center}, "
            f"at {position_label(center, image_size)}"
        )
    return "\n".join(lines)
