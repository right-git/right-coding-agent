"""Route images captured by tools into the model's field of view.

Base64 inside a tool's text result is invisible to the model: providers only
read images from `image_url`-style content blocks, and those are not allowed
in tool-role messages on OpenAI-compatible APIs. So images travel out of band:

1. While a `run_tools` script executes, any tool may call `attach_image()`;
   the image lands in a per-call bucket (a ContextVar, so concurrent runs
   and `parallel(...)` branches stay isolated but share one bucket per run).
2. `run_tools` returns `(text, images)` — the images become the ToolMessage's
   `artifact`, which is carried in state but never sent to the provider.
3. `AttachedImagesMiddleware.before_model` finds artifacts produced by the
   latest round of tool calls and appends ONE HumanMessage whose content is
   `image_url` data-URI blocks — the shape every multimodal provider accepts.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config.logging import logger


ATTACHMENT_MARKER = "attached_images"

_pending: ContextVar[list[dict] | None] = ContextVar(
    "pending_image_attachments", default=None
)


def attach_image(
    base64_data: str, mime_type: str = "image/jpeg", label: str = ""
) -> bool:
    """Queue an image for the model to see; False when no run is collecting."""
    bucket = _pending.get()
    if bucket is None:
        return False
    bucket.append(
        {"base64_data": base64_data, "mime_type": mime_type, "label": label}
    )
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


def _image_artifacts(artifact: Any) -> list[dict]:
    if not isinstance(artifact, list):
        return []
    return [
        item
        for item in artifact
        if isinstance(item, dict) and item.get("base64_data")
    ]


def image_blocks(images: list[dict]) -> list[dict]:
    """The attachments as message content blocks providers can render."""
    blocks: list[dict] = [
        {
            "type": "text",
            "text": "Images captured by the tool calls above:",
        }
    ]
    for image in images:
        label = str(image.get("label") or "").strip()
        if label:
            blocks.append({"type": "text", "text": label})
        mime_type = str(image.get("mime_type") or "image/jpeg")
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image['base64_data']}"
                },
            }
        )
    return blocks


class AttachedImagesMiddleware(AgentMiddleware):
    """Surface tool-produced screenshots as a vision message before the model runs."""

    def before_model(self, state, runtime=None) -> dict[str, Any] | None:
        messages = state["messages"]

        last_ai_index = None
        for index, message in enumerate(messages):
            if isinstance(message, AIMessage):
                last_ai_index = index
        if last_ai_index is None:
            return None

        tail = messages[last_ai_index + 1 :]
        for message in tail:
            if isinstance(message, HumanMessage) and message.additional_kwargs.get(
                ATTACHMENT_MARKER
            ):
                return None  # this round's images were already surfaced

        images: list[dict] = []
        for message in tail:
            if isinstance(message, ToolMessage):
                images.extend(_image_artifacts(getattr(message, "artifact", None)))
        if not images:
            return None

        logger.info("Surfacing [{}] tool image(s) to the model", len(images))
        return {
            "messages": [
                HumanMessage(
                    content=image_blocks(images),
                    additional_kwargs={ATTACHMENT_MARKER: True},
                )
            ]
        }
