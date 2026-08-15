"""The voice layer — push-to-talk ASR and pipelined TTS behind provider configs.

Deliberately shaped like the LLM layer: `VoiceProvider` (`types.py`) names a
provider the way `LLMProvider` does, and `providers/` maps it onto
implementations — `local` today (faster-whisper ASR + Silero TTS),
`fish`/`elevenlabs` planned. `audio.py` owns sounddevice I/O, `hotkey.py` the
global toggle, `utils.SentenceBuffer` the token→sentence pipelining for TTS.
Heavy deps (torch, faster-whisper, pynput, sounddevice) are imported lazily —
importing this package is cheap. REPL wiring lives in `src/ui` / `src/main.py`.
"""

from .audio import AudioPlayer, MicrophoneRecorder
from .hotkey import HotkeyListener, parse_hotkey
from .providers import PLANNED_PROVIDERS, build_speaker, build_transcriber
from .providers.silero import SPEAKERS, SileroSpeaker
from .providers.whisper import WhisperSession, WhisperTranscriber
from .types import Speaker, Transcriber, TranscriptionSession, VoiceProvider
from .utils import SentenceBuffer, SpeakableFilter, default_models_dir

__all__ = [
    "AudioPlayer",
    "HotkeyListener",
    "MicrophoneRecorder",
    "PLANNED_PROVIDERS",
    "SPEAKERS",
    "SentenceBuffer",
    "SileroSpeaker",
    "SpeakableFilter",
    "Speaker",
    "Transcriber",
    "TranscriptionSession",
    "VoiceProvider",
    "WhisperSession",
    "WhisperTranscriber",
    "build_speaker",
    "build_transcriber",
    "default_models_dir",
    "parse_hotkey",
]
