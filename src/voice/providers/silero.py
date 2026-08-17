"""Local TTS provider: Silero — Russian v4_ru plus English v3_en, on CPU."""

import re
import time
from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.downloads import format_download, report_progress
from src.utils.silence import silenced

from ..utils import default_models_dir

MODEL_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
ENGLISH_MODEL_URL = "https://models.silero.ai/models/tts/en/v3_en.pt"
SPEAKERS = ("aidar", "baya", "kseniya", "xenia", "eugene", "random")

# language → (download url, file name, warm-up phrase)
MODELS = {
    "ru": (MODEL_URL, "v4_ru.pt", "Привет."),
    "en": (ENGLISH_MODEL_URL, "v3_en.pt", "Hello."),
}
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def detect_language(text: str) -> str:
    """ "ru" when the sentence contains any Cyrillic, else "en".

    Mixed sentences lean Russian — v4_ru copes with embedded Latin tech terms,
    while a fully-English sentence through the Russian model comes out as a
    fraction of a second of noise (measured 0.33s for a 9-word reply).
    """
    return "ru" if _CYRILLIC.search(text) else "en"


class SileroSpeaker:
    """Synthesizes speech with Silero, routing each sentence by its language.

    Russian goes to v4_ru, English to v3_en (each ~40-60 MB, downloaded on
    first use). Runs on CPU on purpose — the GPU is busy with the vision model
    and whisper, and Silero is faster than real time on CPU anyway. `loader`
    is the test seam returning a model object with `apply_tts` used for every
    language.
    """

    def __init__(
        self,
        speaker: str = "xenia",
        english_speaker: str = "en_0",
        sample_rate: int = 48_000,
        models_dir: Path | None = None,
        loader=None,
        stream_factory=None,
    ):
        self.speaker = speaker
        self.english_speaker = english_speaker
        self.sample_rate = sample_rate
        self.models_dir = models_dir or default_models_dir() / "silero"
        self._loader = loader
        self._stream_factory = stream_factory
        self._models: dict[str, object] = {}

    def load(self, language: str = "ru"):
        if language not in self._models:
            if self._loader is not None:
                self._models[language] = self._loader()
            else:
                self._models[language] = self._default_loader(language)
        return self._models[language]

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        language = detect_language(text)
        options = {"speaker": self.speaker, "put_accent": True, "put_yo": True}
        if language == "en":
            # v3_en rejects the Russian-only options and has its own voices.
            options = {"speaker": self.english_speaker}
        audio = self.load(language).apply_tts(ssml_text=text, sample_rate=self.sample_rate, **options)
        return audio.cpu().numpy(), self.sample_rate

    def _default_loader(self, language: str):
        # Runs on a warm-up thread while the prompt is live — mute this
        # thread's torch/package chatter so it never lands on the prompt.
        with silenced():
            import torch

            url, filename, warmup = MODELS[language]
            path = self.models_dir / filename
            if not path.exists():
                self._download(path, url)
            started = time.perf_counter()
            model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
            model.to("cpu")
            speaker = self.english_speaker if language == "en" else self.speaker
            model.apply_tts(text=warmup, speaker=speaker, sample_rate=24_000)  # warm-up
            logger.info("Silero {} ready on cpu in {:.1f}s", filename, time.perf_counter() - started)
            return model

    def _download(self, path: Path, url: str = MODEL_URL) -> None:
        stream = self._stream_factory
        if stream is None:
            import httpx

            stream = httpx.stream

        logger.info("Downloading Silero TTS model from {} to {}", url, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with stream("GET", url, follow_redirects=True, timeout=None) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0) or None
            done = 0
            with open(path, "wb") as file:
                for chunk in response.iter_bytes(1 << 20):
                    file.write(chunk)
                    done += len(chunk)
                    report_progress(format_download(done, total))
        logger.info("Silero TTS model downloaded")
