import re
from collections.abc import Sequence

from PIL import Image

from .types import Box, Detection, Point, Size, TargetMode


NORMALIZED_SCALE = 1000

_TOKEN_PATTERN = re.compile(
    r"<ref>(?P<label>(?:(?!<ref>|<box>).)*?)</ref>"
    r"|(?P<box><box>(?:(?!<ref>|<box>).)*?</box>)",
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


def scale_normalized_box(
    box: Sequence[int], image_size: Size
) -> tuple[float, float, float, float]:
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


def select_targets(
    detections: Sequence[Detection], mode: TargetMode
) -> list[Detection]:
    if mode == "first":
        return list(detections[:1])
    if mode == "all":
        return list(detections)
    raise ValueError(f"Unsupported target mode: {mode}")


def describe_detections(
    detections: Sequence[Detection], image_size: Size
) -> str:
    """Render detections as a compact report an LLM (or a human) can read."""
    if not detections:
        return "No matching region was found on the screen."

    lines = []
    for index, detection in enumerate(detections, start=1):
        center = box_center(detection.box, image_size)
        lines.append(
            f"{index}. {detection.label}: "
            f"box={detection.box}, center={center}"
        )
    return "\n".join(lines)
