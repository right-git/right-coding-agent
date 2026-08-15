"""Local TTS provider: Silero v4_ru — real-time synthesis on CPU."""

import time
from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.downloads import format_download, report_progress
from src.utils.silence import silenced

from ..utils import default_models_dir

MODEL_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
SPEAKERS = ("aidar", "baya", "kseniya", "xenia", "eugene", "random")


class SileroSpeaker:
    """Synthesizes Russian speech with Silero v4 (~38 MB, downloaded on first use).

    Runs on CPU on purpose — the GPU is busy with the vision model and whisper,
    and Silero is faster than real time on CPU anyway. `loader` is the test
    seam returning a model object with `apply_tts`.
    """

    def __init__(
        self,
        speaker: str = "xenia",
        sample_rate: int = 48_000,
        models_dir: Path | None = None,
        loader=None,
        stream_factory=None,
    ):
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.models_dir = models_dir or default_models_dir() / "silero"
        self._loader = loader
        self._stream_factory = stream_factory
        self._model = None

    def load(self):
        if self._model is None:
            self._model = (self._loader or self._default_loader)()
        return self._model

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        audio = self.load().apply_tts(
            text=text, speaker=self.speaker, sample_rate=self.sample_rate, put_accent=True, put_yo=True
        )
        return audio.cpu().numpy(), self.sample_rate

    def _default_loader(self):
        # Runs on a warm-up thread while the prompt is live — mute this
        # thread's torch/package chatter so it never lands on the prompt.
        with silenced():
            import torch

            path = self.models_dir / "v4_ru.pt"
            if not path.exists():
                self._download(path)
            started = time.perf_counter()
            model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
            model.to("cpu")
            model.apply_tts(text="Привет.", speaker=self.speaker, sample_rate=24_000)  # warm-up
            logger.info("Silero v4_ru ready on cpu in {:.1f}s", time.perf_counter() - started)
            return model

    def _download(self, path: Path) -> None:
        stream = self._stream_factory
        if stream is None:
            import httpx

            stream = httpx.stream

        logger.info("Downloading Silero TTS model (~38 MB) from {} to {}", MODEL_URL, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with stream("GET", MODEL_URL, follow_redirects=True, timeout=None) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0) or None
            done = 0
            with open(path, "wb") as file:
                for chunk in response.iter_bytes(1 << 20):
                    file.write(chunk)
                    done += len(chunk)
                    report_progress(format_download(done, total))
        logger.info("Silero TTS model downloaded")
