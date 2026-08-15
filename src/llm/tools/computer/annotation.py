import base64
import io
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .types import Detection

DEFAULT_OUTPUT_PATH = Path("output_boxes.jpg")
COLORS = ("#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE", "#00C7BE")


def image_to_base64(
    image: Image.Image,
    *,
    max_side: int | None = 1280,
    format: str = "JPEG",
    quality: int = 80,
) -> str:
    """Encode an image as base64 for transport to a multimodal model.

    `max_side` bounds the longest edge — a full-resolution desktop capture
    costs several times more vision tokens than a 1280px one and reads just
    as well for layout questions.
    """
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
        )
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def annotate(image: Image.Image, detections: Sequence[Detection]) -> Image.Image:
    """Return a copy of `image` with a labelled box drawn for each detection."""
    annotated_image = image.copy()
    if annotated_image.mode != "RGB":
        annotated_image = annotated_image.convert("RGB")

    draw = ImageDraw.Draw(annotated_image)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, image.width // 60))
    except OSError:
        font = ImageFont.load_default()

    label_colors: dict[str, str] = {}
    line_width = max(2, image.width // 400)
    for detection in detections:
        label = detection.label
        color = label_colors.setdefault(label, COLORS[len(label_colors) % len(COLORS)])
        x1, y1, x2, y2 = detection.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = y1 - text_height - 6 if y1 - text_height - 6 > 0 else y1 + 2
        draw.rectangle([x1, text_y, x1 + text_width + 8, text_y + text_height + 6], fill=color)
        draw.text((x1 + 4, text_y + 2), label, fill="white", font=font)

    return annotated_image


def save_annotated_image(
    image: Image.Image,
    detections: Sequence[Detection],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    annotated_image = annotate(image, detections)
    destination = Path(output_path)
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)
    annotated_image.save(destination, format="JPEG", quality=95)
    return destination
