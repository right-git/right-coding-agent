"""Helpers for turning fetched HTML/Markdown into clean text."""

from typing import Any


def extract_front_matter(markdown: str) -> str:
    """The leading `---` front-matter block of `markdown`, or "" when absent."""
    lines = markdown.splitlines()
    if not lines or lines[0] != "---":
        return ""

    for index, line in enumerate(lines[1:], start=1):
        if line == "---":
            return "\n".join(lines[: index + 1])

    return ""


def as_content_text(result: Any) -> str:
    """Whatever the markdown converter returned, as plain text.

    `html_to_markdown.convert` returns a plain string in some versions
    and a `{"content": ...}` mapping in others; callers get a string
    either way.
    """
    if isinstance(result, dict):
        content = result.get("content")
        return content if isinstance(content, str) else ""
    return result if isinstance(result, str) else str(result or "")
