"""Agent middlewares: cross-cutting steps that run around every model call.

One file per middleware. `attachments.py` surfaces tool-captured screenshots
as vision messages; `message_log.py` writes the request/response traffic to
the log as scrubbed JSON lines.
"""

from .attachments import ATTACHMENT_MARKER, AttachedImagesMiddleware, image_blocks
from .message_log import (
    MAX_TEXT_CHARS,
    MessageLogMiddleware,
    scrub,
    scrub_text,
    serialize_message,
)

__all__ = [
    "ATTACHMENT_MARKER",
    "AttachedImagesMiddleware",
    "MAX_TEXT_CHARS",
    "MessageLogMiddleware",
    "image_blocks",
    "scrub",
    "scrub_text",
    "serialize_message",
]
