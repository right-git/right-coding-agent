"""Microphone capture and interruptible playback. sounddevice is imported lazily."""

import threading

import numpy as np


class MicrophoneRecorder:
    """Collects mono float32 chunks from the default input device.

    `stream_factory(on_chunk)` is the test seam: it must return an object with
    `start()`/`stop()`/`close()` that delivers `(frames, 1)` float32 arrays to
    `on_chunk`. `snapshot()` may be called from any thread while recording.
    """

    def __init__(self, sample_rate: int = 16_000, stream_factory=None):
        self.sample_rate = sample_rate
        self._stream_factory = stream_factory
        self._stream = None
        self._chunks: list[np.ndarray] = []

    def start(self) -> None:
        if self._stream is not None:
            return
        self._chunks = []
        factory = self._stream_factory or self._default_stream
        self._stream = factory(self._chunks.append)
        self._stream.start()

    def snapshot(self) -> np.ndarray:
        taken = list(self._chunks)
        if not taken:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(taken)
        return audio[:, 0] if audio.ndim == 2 else audio

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self.snapshot()

    @property
    def duration(self) -> float:
        return sum(len(chunk) for chunk in self._chunks) / self.sample_rate

    def _default_stream(self, on_chunk):
        import sounddevice as sd

        def callback(indata, frames, time_info, status) -> None:
            on_chunk(indata.copy())

        return sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=callback)


class AudioPlayer:
    """Block-wise blocking playback that another thread can interrupt with `stop()`.

    Interruptibility is the point: barge-in (pressing push-to-talk while the
    agent is speaking) must cut speech within one block, not one sentence.
    `stream_factory(sample_rate)` is the test seam returning a context manager
    with `write(block)`.
    """

    BLOCK_FRAMES = 4096

    def __init__(self, stream_factory=None):
        self._stream_factory = stream_factory
        self._interrupted = threading.Event()

    def play(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Plays to the end and returns True, or False when interrupted."""
        self._interrupted.clear()
        audio = audio.astype(np.float32, copy=False)
        factory = self._stream_factory or self._default_stream
        with factory(sample_rate) as stream:
            for start in range(0, len(audio), self.BLOCK_FRAMES):
                if self._interrupted.is_set():
                    return False
                stream.write(audio[start : start + self.BLOCK_FRAMES])
        return True

    def stop(self) -> None:
        self._interrupted.set()

    @staticmethod
    def _default_stream(sample_rate: int):
        import sounddevice as sd

        return sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
