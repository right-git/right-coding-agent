"""Dataclasses and protocols of the voice layer."""

from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field


class VoiceProvider(BaseModel):
    """Names a speech provider the way `LLMProvider` names an LLM one.

    The `local` provider ignores the api_* fields; cloud providers (fish,
    elevenlabs) authenticate with them but keep separate keys in settings —
    their APIs are structurally different, so unification happens at the
    `Transcriber`/`Speaker` protocols, not at the transport.
    """

    provider_name: str = Field(description='The provider name: "local", "fish", or "elevenlabs".')
    model_name: str | None = Field(default=None, description="ASR or TTS model identifier of that provider.")
    api_key: str | None = Field(default=None, description="API key for cloud providers.")
    api_base: str | None = Field(default=None, description="Optional base URL override for cloud providers.")
    voice: str | None = Field(default=None, description="TTS voice: Silero speaker or cloud voice/reference id.")
    language: str | None = Field(default=None, description="Fixed language code; None means auto-detect.")


class TranscriptionSession(Protocol):
    """Incremental transcription over one growing push-to-talk recording."""

    committed: str

    def step(self, audio: np.ndarray) -> str:
        """Transcribe newly recorded audio; returns the uncommitted draft tail."""
        ...

    def finalize(self, audio: np.ndarray) -> str:
        """Transcribe the remaining tail; returns the full utterance text."""
        ...


class Transcriber(Protocol):
    """ASR provider: loads lazily, opens one session per recording."""

    def load(self) -> object: ...

    def open_session(self) -> TranscriptionSession: ...


class Speaker(Protocol):
    """TTS provider: synthesizes one sentence/chunk per call."""

    def load(self) -> object: ...

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Returns (mono float32 audio, sample rate)."""
        ...
