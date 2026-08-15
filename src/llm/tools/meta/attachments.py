"""The tool-side channel that carries captured images out of a script run.

Base64 inside a tool's text result is invisible to the model, so images
travel out of band: while a `run_tools` script executes, any tool may call
`attach_image()`; the image lands in a per-call bucket (a ContextVar, so
concurrent runs and `parallel(...)` branches stay isolated but share one
bucket per run). `run_tools` opens the bucket with `collecting_images()` and
returns its content as the ToolMessage artifact, which
`src.llm.middlewares.attachments.AttachedImagesMiddleware` later surfaces to
the model as a vision message.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_pending: ContextVar[list[dict] | None] = ContextVar("pending_image_attachments", default=None)


def attach_image(base64_data: str, mime_type: str = "image/jpeg", label: str = "") -> bool:
    """Queue an image for the model to see; False when no run is collecting."""
    bucket = _pending.get()
    if bucket is None:
        return False
    bucket.append({"base64_data": base64_data, "mime_type": mime_type, "label": label})
    return True


@contextmanager
def collecting_images():
    """Open a bucket for one run_tools call and yield it."""
    bucket: list[dict] = []
    token = _pending.set(bucket)
    try:
        yield bucket
    finally:
        _pending.reset(token)
