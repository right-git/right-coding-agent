from collections.abc import Sequence

from PIL import Image


def resize_for_inference(image: Image.Image, max_side: int = 768) -> Image.Image:
    if max_side <= 0:
        raise ValueError("max_side must be positive")

    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def scale_normalized_box(
    box: Sequence[int], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width, height = image_size
    return (
        x1 / 1000 * width,
        y1 / 1000 * height,
        x2 / 1000 * width,
        y2 / 1000 * height,
    )
