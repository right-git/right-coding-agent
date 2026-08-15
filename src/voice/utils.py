"""Small pure helpers of the voice layer."""

import re
from pathlib import Path

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


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
