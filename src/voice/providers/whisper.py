"""Local ASR provider: faster-whisper with incremental push-to-talk sessions."""

import os
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.silence import silenced, suppress_native_stderr

from ..utils import default_models_dir

SAMPLE_RATE = 16_000
MIN_STEP_SAMPLES = SAMPLE_RATE  # don't touch the model until a second of new audio accumulated
CONTEXT_CHARS = 200
DEFAULT_HOTWORDS = "git, bash, Python, uv, npm, run.sh, commit, push, merge, branch, deploy, docker, API, README"
REQUIRED_MODEL_FILES = ("model.bin", "config.json")


def resolve_repo(model_name: str) -> str:
    """Hub repo id for a faster-whisper alias; full `org/name` ids pass through."""
    if "/" in model_name:
        return model_name
    try:
        from faster_whisper.utils import _MODELS

        return _MODELS.get(model_name, model_name)
    except Exception:
        return model_name


def _add_torch_cuda_dlls() -> None:
    """ctranslate2 on Windows looks for cuDNN/cuBLAS on PATH — expose torch's bundled DLLs."""
    if sys.platform != "win32":
        return
    try:
        import torch

        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
    except Exception as error:
        logger.warning("Could not expose torch CUDA DLLs to ctranslate2: {}", error)


class WhisperTranscriber:
    """Loads faster-whisper lazily (CUDA, falling back to CPU int8) and opens sessions.

    beam_size defaults to 5 on purpose: greedy decoding garbles mixed
    Russian/English speech, and `hotwords` biases known tech terms to Latin
    script. `loader` is the test seam returning a model object.
    """

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        language: str | None = None,
        hotwords: str | None = DEFAULT_HOTWORDS,
        beam_size: int = 5,
        models_dir: Path | None = None,
        loader=None,
        predownload=None,
    ):
        self.model_name = model_name
        self.language = language
        self.hotwords = hotwords
        self.beam_size = beam_size
        self.models_dir = models_dir or default_models_dir() / "faster-whisper"
        self._loader = loader
        self._predownload = predownload
        self._model = None

    def load(self):
        if self._model is None:
            self._model = (self._loader or self._default_loader)()
        return self._model

    def open_session(self) -> "WhisperSession":
        return WhisperSession(self)

    def _model_source(self) -> str:
        """A local snapshot path when possible, else the model name.

        Downloading ourselves (sequentially, on this thread) keeps the exit
        clean and feeds byte progress to the prompt's status line; on any
        failure faster-whisper's own downloader takes over unchanged.
        """
        predownload = self._predownload
        if predownload is None:
            from src.utils.downloads import download_hf_model

            predownload = download_hf_model
        try:
            return predownload(
                resolve_repo(self.model_name), cache_dir=self.models_dir, required_files=REQUIRED_MODEL_FILES
            )
        except Exception as error:
            logger.warning("Whisper pre-download failed ({}); faster-whisper will download the model itself", error)
            return self.model_name

    def _default_loader(self):
        # The load runs on a warm-up thread while the prompt is live:
        # silenced() drops Python-side chatter (hf progress bars, warnings),
        # suppress_native_stderr() the objc duplicate-class noise from the av
        # dylib load — see silence.py for why both are needed.
        with silenced():
            with suppress_native_stderr():
                from faster_whisper import WhisperModel

            _add_torch_cuda_dlls()
            source = self._model_source()
            return self._load_on_best_device(WhisperModel, source)

    def _load_on_best_device(self, whisper_model_cls, source: str):
        for device, compute_type in (("cuda", "float16"), ("cpu", "int8")):
            started = time.perf_counter()
            try:
                with suppress_native_stderr():
                    model = whisper_model_cls(
                        source, device=device, compute_type=compute_type, download_root=str(self.models_dir)
                    )
                    # Warm-up: the first call compiles kernels; without it the first real
                    # utterance pays seconds of extra latency.
                    list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="ru", beam_size=1)[0])
            except Exception as error:
                logger.warning("Whisper on {}/{} failed: {}", device, compute_type, error)
                continue
            logger.info(
                "Whisper {} ready on {}/{} in {:.1f}s",
                self.model_name,
                device,
                compute_type,
                time.perf_counter() - started,
            )
            return model
        raise RuntimeError(f"Could not load whisper model {self.model_name!r} on any device")


class WhisperSession:
    """Committed-prefix incremental transcription over one growing recording.

    Every pass transcribes only the uncommitted window; all returned segments
    except the last (it may still grow) are committed and the window advances
    past them. `finalize` therefore only processes a small tail, which is what
    keeps release-to-text latency flat regardless of utterance length.
    """

    def __init__(self, transcriber: WhisperTranscriber):
        self._transcriber = transcriber
        self.language = transcriber.language
        self.committed = ""
        self.window_start = 0

    def step(self, audio: np.ndarray) -> str:
        if len(audio) - self.window_start < MIN_STEP_SAMPLES:
            return ""
        segments = self._transcribe_window(audio)
        if len(segments) > 1:
            *done, last = segments
            self.committed += "".join(segment.text for segment in done)
            self.window_start += int(done[-1].end * SAMPLE_RATE)
            return last.text
        return segments[0].text if segments else ""

    def finalize(self, audio: np.ndarray) -> str:
        if len(audio) > self.window_start:
            self.committed += "".join(segment.text for segment in self._transcribe_window(audio))
        return self.committed.strip()

    def _transcribe_window(self, audio: np.ndarray) -> list:
        model = self._transcriber.load()
        segments, info = model.transcribe(
            audio[self.window_start :],
            language=self.language,
            beam_size=self._transcriber.beam_size,
            vad_filter=True,
            initial_prompt=self.committed[-CONTEXT_CHARS:] or None,
            hotwords=self._transcriber.hotwords,
        )
        segments = list(segments)
        if self.language is None:
            self.language = getattr(info, "language", None)
        return segments
