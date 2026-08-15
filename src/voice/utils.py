"""Small pure helpers of the voice layer."""

import re
from pathlib import Path

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MARKUP = re.compile(r"[*_#>|~]+")


def default_models_dir() -> Path:
    """Repo-root `models/` — shared with the sandbox scripts and gitignored."""
    return Path(__file__).resolve().parents[2] / "models"


class SentenceBuffer:
    """Accumulates streamed tokens and releases complete sentences.

    This is what lets TTS start speaking while the model is still generating:
    `feed` returns every sentence completed by the incoming token, `flush`
    returns whatever tail remains when the stream ends.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        self._buffer += token
        parts = _SENTENCE_BOUNDARY.split(self._buffer)
        if len(parts) == 1:
            return []
        *done, self._buffer = parts
        return [part.strip() for part in done if part.strip()]

    def flush(self) -> str:
        tail, self._buffer = self._buffer.strip(), ""
        return tail


class SpeakableFilter:
    """Turns markdown-ish streamed sentences into speakable plain text.

    Stateful across sentences on purpose: everything inside ``` fences is
    dropped entirely, even when a fence spans many sentences — code must
    never be read aloud. Inline backticks, emphasis marks, headers, and
    links are reduced to their readable text.
    """

    def __init__(self) -> None:
        self._in_code = False

    def filter(self, sentence: str) -> str:
        parts = sentence.split("```")
        if len(parts) == 1:
            kept = "" if self._in_code else sentence
        else:
            spoken = []
            inside = self._in_code
            for index, part in enumerate(parts):
                if not inside:
                    spoken.append(part)
                if index < len(parts) - 1:  # a fence marker sits between parts
                    inside = not inside
            self._in_code = inside
            kept = " ".join(spoken)
        kept = _LINK.sub(r"\1", kept)
        kept = _INLINE_CODE.sub(r"\1", kept)
        kept = _MARKUP.sub("", kept)
        return re.sub(r"\s+", " ", kept).strip()
