"""Small shared helpers used across the MCP layer."""

from typing import Any


def read_field(value: Any, name: str, default=None):
    """Read an attribute or dict key, tolerating either shape.

    MCP SDK results arrive as either typed objects or plain dicts depending
    on the transport/version, so every caller that walks a result needs this
    same tolerant lookup; shared here instead of duplicated per module.
    """
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
