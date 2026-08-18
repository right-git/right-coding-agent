"""Shared identifier building for dynamically-registered tools (MCP, skills)."""

import hashlib
import re

_HASH_LENGTH = 8
MAX_TOOL_NAME_LENGTH = 64
_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9]+")


def safe_part(value: str) -> str:
    """Normalize a component to alphanumeric + underscores; fallback to 'x'."""
    normalized = _SAFE_PART_RE.sub("_", str(value or "").strip()).strip("_")
    return normalized or "x"


def hashed_identifier(readable: str, raw_key: str) -> str:
    """`readable` unchanged when short; hash-suffixed truncation past the ceiling."""
    if len(readable) <= MAX_TOOL_NAME_LENGTH:
        return readable
    digest = hashlib.sha256(raw_key.encode()).hexdigest()[:_HASH_LENGTH]
    suffix = f"_{digest}"
    return readable[: MAX_TOOL_NAME_LENGTH - len(suffix)].rstrip("_") + suffix
