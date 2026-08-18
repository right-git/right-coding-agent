"""The per-run delivery channel for skill bodies.

Mirrors the get_tool contracts mechanic: a pseudo-tool records the rendered
body here instead of returning it, run_tools ships the bucket as the result
JSON's `skills` field, and history compaction carries it verbatim. When no
channel is open (library use outside run_tools) `record_skill_body` returns
False and the caller falls back to returning the body inline.
"""

from contextlib import contextmanager
from contextvars import ContextVar

MAX_SKILL_BODY_CHARS = 12_000

_BUCKET: ContextVar[dict | None] = ContextVar("skill_bodies", default=None)


@contextmanager
def collecting_skill_bodies():
    bucket: dict[str, str] = {}
    token = _BUCKET.set(bucket)
    try:
        yield bucket
    finally:
        _BUCKET.reset(token)


def record_skill_body(slug: str, body: str) -> bool:
    bucket = _BUCKET.get()
    if bucket is None:
        return False
    if len(body) > MAX_SKILL_BODY_CHARS:
        dropped = len(body) - MAX_SKILL_BODY_CHARS
        body = body[:MAX_SKILL_BODY_CHARS] + f"… [+{dropped} chars truncated — the skill file exceeds the budget]"
    bucket[slug] = body
    return True
