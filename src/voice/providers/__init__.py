"""Factories mapping a `VoiceProvider` config onto concrete ASR/TTS implementations.

`local` is implemented (faster-whisper + Silero); `fish` and `elevenlabs` are
planned cloud adapters behind the same `Transcriber`/`Speaker` protocols — a new
provider is a new module here plus a branch in these factories, nothing else.
"""

from ..types import Speaker, Transcriber, VoiceProvider
from .silero import SileroSpeaker
from .whisper import WhisperTranscriber

PLANNED_PROVIDERS = frozenset({"fish", "elevenlabs"})


def build_transcriber(provider: VoiceProvider) -> Transcriber:
    if provider.provider_name == "local":
        return WhisperTranscriber(model_name=provider.model_name or "large-v3-turbo", language=provider.language)
    if provider.provider_name in PLANNED_PROVIDERS:
        raise NotImplementedError(f"ASR provider {provider.provider_name!r} is planned but not implemented yet")
    raise ValueError(f"Unknown ASR provider {provider.provider_name!r}; available: local")


def build_speaker(provider: VoiceProvider) -> Speaker:
    if provider.provider_name == "local":
        return SileroSpeaker(speaker=provider.voice or "xenia")
    if provider.provider_name in PLANNED_PROVIDERS:
        raise NotImplementedError(f"TTS provider {provider.provider_name!r} is planned but not implemented yet")
    raise ValueError(f"Unknown TTS provider {provider.provider_name!r}; available: local")
