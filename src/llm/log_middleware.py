"""Structured logging of every request the agent sends to the model.

`MessageLogMiddleware.before_model` serializes the message list to one JSON
log line per model call. Payloads that would drown the log are stripped
before serialization: data URIs and long base64 runs become short
placeholders that keep the length, and any remaining long text is truncated
with the overflow noted. Registered last, it sees the conversation exactly
as the model will (the system prompt is injected by the agent separately and
is not part of state).
"""

import json
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents.middleware.types import AgentMiddleware

from src.config.logging import logger


MAX_TEXT_CHARS = 400

_DATA_URI = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,[A-Za-z0-9+/=]+")
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{200,}")


def scrub_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Strip binary payloads, then truncate whatever is still too long."""
    text = _DATA_URI.sub(
        lambda match: f"<data-uri stripped, {len(match.group(0))} chars>", text
    )
    text = _BASE64_RUN.sub(
        lambda match: f"<base64 stripped, {len(match.group(0))} chars>", text
    )
    if len(text) > max_chars:
        text = text[:max_chars] + f"… [+{len(text) - max_chars} chars]"
    return text


def scrub(value: Any, max_chars: int = MAX_TEXT_CHARS) -> Any:
    """`value` with every string scrubbed, safe for json.dumps."""
    if isinstance(value, str):
        return scrub_text(value, max_chars)
    if isinstance(value, dict):
        return {str(key): scrub(item, max_chars) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(item, max_chars) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return scrub_text(repr(value), max_chars)


def serialize_message(message: Any, max_chars: int = MAX_TEXT_CHARS) -> dict:
    entry: dict[str, Any] = {"type": getattr(message, "type", type(message).__name__)}
    identifier = getattr(message, "id", None)
    if identifier:
        entry["id"] = identifier
    entry["content"] = scrub(getattr(message, "content", None), max_chars)

    if isinstance(message, AIMessage):
        if message.tool_calls:
            entry["tool_calls"] = scrub(message.tool_calls, max_chars)
        usage = getattr(message, "usage_metadata", None)
        if usage:
            entry["usage"] = {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }
    if isinstance(message, ToolMessage):
        entry["tool_call_id"] = message.tool_call_id
        if message.name:
            entry["name"] = message.name
        entry["status"] = message.status

    return entry


class MessageLogMiddleware(AgentMiddleware):
    """Log model traffic as JSON lines: the request before every call, the
    response — including its finish_reason — after it."""

    def __init__(
        self,
        *,
        max_text_chars: int = MAX_TEXT_CHARS,
        level: str = "INFO",
        emit: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.max_text_chars = max_text_chars
        self.level = level
        self._emit = emit or (lambda line: logger.log(self.level, "{}", line))

    def before_model(self, state, runtime=None) -> None:
        try:
            messages = state["messages"]
            payload = {
                "event": "model_request",
                "message_count": len(messages),
                "messages": [
                    serialize_message(message, self.max_text_chars)
                    for message in messages
                ],
            }
            self._emit(json.dumps(payload, ensure_ascii=False, default=repr))
        except Exception:
            logger.exception("Failed to log the model request")
        return None

    def after_model(self, state, runtime=None) -> None:
        try:
            messages = state["messages"]
            last = messages[-1] if messages else None
            if not isinstance(last, AIMessage):
                return None
            entry = serialize_message(last, self.max_text_chars)
            metadata = getattr(last, "response_metadata", None)
            if metadata:
                entry["response_metadata"] = scrub(metadata, self.max_text_chars)
            payload = {"event": "model_response", "message": entry}
            self._emit(json.dumps(payload, ensure_ascii=False, default=repr))
        except Exception:
            logger.exception("Failed to log the model response")
        return None
