"""Reading images from the system clipboard for chat input.

`PIL.ImageGrab.grabclipboard()` covers Windows and macOS natively and Linux
via xclip/wl-paste. It returns a PIL image for pixel data, a list of file
paths when files were copied, or None — `grab_clipboard_image` normalizes
all three into "an image or None", and `encode_image` turns the image into
the `image_url`-ready payload the model providers accept.
"""

from pathlib import Path

from PIL import Image

from src.config.logging import logger
from src.llm.tools.computer import image_to_base64

MAX_IMAGE_SIDE = 1568  # larger buys tokens, not comprehension
IMAGE_FILE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


def _default_grabber():
    from PIL import ImageGrab

    return ImageGrab.grabclipboard()


def grab_clipboard_image(*, grabber=_default_grabber) -> Image.Image | None:
    """The clipboard's image, or None when it holds no usable image."""
    try:
        data = grabber()
    except Exception:
        logger.exception("Reading the clipboard image failed")
        return None

    if isinstance(data, Image.Image):
        return data
    if isinstance(data, list):  # copied files: take the first image among them
        for entry in data:
            path = Path(str(entry))
            if path.suffix.lower() in IMAGE_FILE_SUFFIXES and path.is_file():
                try:
                    return Image.open(path)
                except Exception:
                    logger.exception("Opening copied image file [{}] failed", path)
    return None


def encode_image(image: Image.Image) -> dict:
    """`{base64_data, mime_type, width, height}` ready for an image_url block."""
    width, height = image.size
    return {
        "base64_data": image_to_base64(image, max_side=MAX_IMAGE_SIDE, quality=90),
        "mime_type": "image/jpeg",
        "width": width,
        "height": height,
    }
