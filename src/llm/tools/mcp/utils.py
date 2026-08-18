"""Small shared helpers used across the MCP layer."""

import re
from typing import Any

_CAMEL_BOUNDARY_RE = re.compile(r"(?<!^)(?=[A-Z])")
_MISSING = object()


def _snake(name: str) -> str:
    return _CAMEL_BOUNDARY_RE.sub("_", name).lower()


def read_field(value: Any, name: str, default=None):
    """Read an attribute or dict key, tolerating either shape AND either casing.

    Callers name fields in the MCP wire format (camelCase: `inputSchema`,
    `isError`, `mimeType`), but what they receive varies: SDK 2.0 pydantic
    models expose snake_case attributes (`input_schema`), plain dicts from
    older transports keep the wire casing, and tests use fakes of both
    shapes. A camelCase miss therefore retries the snake_case spelling —
    without it every real SDK 2.0 tool registered with an empty schema
    while the camelCase fakes kept the suite green.
    """
    for candidate in (name, _snake(name)):
        if isinstance(value, dict):
            found = value.get(candidate, _MISSING)
        else:
            found = getattr(value, candidate, _MISSING)
        if found is not _MISSING:
            return found
    return default
