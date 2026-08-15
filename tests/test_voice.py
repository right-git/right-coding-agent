import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.voice import (
    AudioPlayer,
    HotkeyListener,
    MicrophoneRecorder,
    SentenceBuffer,
    SileroSpeaker,
    VoiceProvider,
    WhisperTranscriber,
    build_speaker,
    build_transcriber,
    parse_hotkey,
)
from src.voice.providers.whisper import SAMPLE_RATE


class SentenceBufferTests(unittest.TestCase):
    def test_sentences_release_as_they_complete(self):
        buffer = SentenceBuffer()

        self.assertEqual(buffer.feed("Привет"), [])
        self.assertEqual(buffer.feed(", мир. Как"), ["Привет, мир."])
        self.assertEqual(buffer.feed(" дела? Хорошо."), ["Как дела?"])
        self.assertEqual(buffer.flush(), "Хорошо.")

    def test_flush_empties_the_buffer(self):
        buffer = SentenceBuffer()
        buffer.feed("хвост без точки")

        self.assertEqual(buffer.flush(), "хвост без точки")
        self.assertEqual(buffer.flush(), "")

    def test_blank_fragments_are_skipped(self):
        buffer = SentenceBuffer()

        self.assertEqual(buffer.feed("Раз! Два! "), ["Раз!", "Два!"])


class HotkeyTests(unittest.TestCase):
    def test_alt_r_also_matches_altgr(self):
        from pynput import keyboard

        keys = parse_hotkey("alt_r")

        self.assertIn(keyboard.Key.alt_r, keys)
        self.assertIn(keyboard.Key.alt_gr, keys)

    def test_named_key_and_single_character(self):
        from pynput import keyboard

        self.assertEqual(parse_hotkey("f8"), {keyboard.Key.f8})
        self.assertEqual(parse_hotkey("g"), {keyboard.KeyCode.from_char("g")})

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            parse_hotkey("no_such_key")

    def test_listener_fires_only_on_configured_key(self):
        from pynput import keyboard

        class FakeListener:
            def __init__(self, on_press):
                self.on_press = on_press

            def start(self):
                pass

            def stop(self):
                pass

        toggles = []
        listener = HotkeyListener("f8", on_toggle=lambda: toggles.append(1), listener_factory=FakeListener)
        listener.start()

        listener._listener.on_press(keyboard.Key.f8)
        listener._listener.on_press(keyboard.Key.f9)

        self.assertEqual(len(toggles), 1)
        listener.stop()


class ProviderFactoryTests(unittest.TestCase):
    def test_local_providers(self):
        asr = build_transcriber(VoiceProvider(provider_name="local", model_name="small", language="ru"))
        tts = build_speaker(VoiceProvider(provider_name="local", voice="eugene"))

        self.assertIsInstance(asr, WhisperTranscriber)
        self.assertEqual(asr.model_name, "small")
        self.assertEqual(asr.language, "ru")
        self.assertIsInstance(tts, SileroSpeaker)
        self.assertEqual(tts.speaker, "eugene")

    def test_planned_providers_raise_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            build_speaker(VoiceProvider(provider_name="fish"))
        with self.assertRaises(NotImplementedError):
            build_transcriber(VoiceProvider(provider_name="elevenlabs"))

    def test_unknown_provider_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_speaker(VoiceProvider(provider_name="nope"))


class FakeSegment:
    def __init__(self, text, end):
        self.text = text
        self.end = end


class FakeInfo:
    language = "ru"


class FakeWhisperModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((len(audio), kwargs))
        return iter(self.responses.pop(0)), FakeInfo()


class WhisperSessionTests(unittest.TestCase):
    def test_step_commits_all_but_last_segment_and_advances_window(self):
        model = FakeWhisperModel([[FakeSegment("Привет,", 2.0), FakeSegment(" мир.", 4.5)]])
        session = WhisperTranscriber(loader=lambda: model).open_session()

        draft = session.step(np.zeros(5 * SAMPLE_RATE, dtype=np.float32))

        self.assertEqual(draft, " мир.")
        self.assertEqual(session.committed, "Привет,")
        self.assertEqual(session.window_start, 2 * SAMPLE_RATE)

    def test_finalize_transcribes_only_the_tail(self):
        model = FakeWhisperModel(
            [
                [FakeSegment("Привет,", 2.0), FakeSegment(" мир", 4.5)],
                [FakeSegment(" мир и точка.", 3.0)],
            ]
        )
        session = WhisperTranscriber(loader=lambda: model).open_session()
        session.step(np.zeros(5 * SAMPLE_RATE, dtype=np.float32))

        text = session.finalize(np.zeros(6 * SAMPLE_RATE, dtype=np.float32))

        self.assertEqual(text, "Привет, мир и точка.")
        tail_length, kwargs = model.calls[-1]
        self.assertEqual(tail_length, 4 * SAMPLE_RATE)  # окно сдвинулось на 2с из 6с
        self.assertEqual(kwargs["initial_prompt"], "Привет,")

    def test_step_skips_model_until_enough_new_audio(self):
        model = FakeWhisperModel([])
        session = WhisperTranscriber(loader=lambda: model).open_session()

        self.assertEqual(session.step(np.zeros(SAMPLE_RATE // 2, dtype=np.float32)), "")
        self.assertEqual(model.calls, [])

    def test_language_detected_once_then_fixed(self):
        model = FakeWhisperModel([[FakeSegment("Привет.", 1.0), FakeSegment(" Ещё.", 2.0)]])
        session = WhisperTranscriber(loader=lambda: model).open_session()
        session.step(np.zeros(3 * SAMPLE_RATE, dtype=np.float32))

        self.assertEqual(session.language, "ru")

    def test_hotwords_and_beam_reach_the_model(self):
        model = FakeWhisperModel([[FakeSegment("x", 1.0)]])
        transcriber = WhisperTranscriber(hotwords="lint.sh", beam_size=3, loader=lambda: model)
        transcriber.open_session().step(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))

        _, kwargs = model.calls[0]
        self.assertEqual(kwargs["hotwords"], "lint.sh")
        self.assertEqual(kwargs["beam_size"], 3)


class FakeWave:
    def cpu(self):
        return self

    def numpy(self):
        return np.zeros(120, dtype=np.float32)


class FakeTtsModel:
    def __init__(self):
        self.calls = []

    def apply_tts(self, **kwargs):
        self.calls.append(kwargs)
        return FakeWave()


class SileroSpeakerTests(unittest.TestCase):
    def test_synthesize_returns_audio_and_rate(self):
        model = FakeTtsModel()
        speaker = SileroSpeaker(speaker="baya", sample_rate=24_000, loader=lambda: model)

        audio, rate = speaker.synthesize("Привет.")

        self.assertEqual(rate, 24_000)
        self.assertEqual(len(audio), 120)
        self.assertEqual(model.calls[0]["speaker"], "baya")
        self.assertEqual(model.calls[0]["text"], "Привет.")

    def test_model_loads_once(self):
        loads = []

        def loader():
            loads.append(1)
            return FakeTtsModel()

        speaker = SileroSpeaker(loader=loader)
        speaker.synthesize("Раз.")
        speaker.synthesize("Два.")

        self.assertEqual(len(loads), 1)


class FakeInputStream:
    def __init__(self):
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def close(self):
        pass


class MicrophoneRecorderTests(unittest.TestCase):
    def test_chunks_accumulate_into_mono_snapshot(self):
        sink = {}

        def factory(on_chunk):
            sink["on_chunk"] = on_chunk
            return FakeInputStream()

        recorder = MicrophoneRecorder(stream_factory=factory)
        recorder.start()
        sink["on_chunk"](np.ones((10, 1), dtype=np.float32))
        sink["on_chunk"](np.ones((6, 1), dtype=np.float32))

        self.assertEqual(recorder.snapshot().shape, (16,))
        self.assertEqual(recorder.stop().shape, (16,))
        self.assertAlmostEqual(recorder.duration, 16 / 16_000)

    def test_empty_recording_gives_empty_audio(self):
        recorder = MicrophoneRecorder(stream_factory=lambda on_chunk: FakeInputStream())
        recorder.start()

        self.assertEqual(len(recorder.stop()), 0)


class FakeOutputStream:
    def __init__(self, player=None):
        self.blocks = []
        self.player = player

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, block):
        self.blocks.append(block)
        if self.player is not None:
            self.player.stop()


class AudioPlayerTests(unittest.TestCase):
    def test_plays_all_blocks(self):
        stream = FakeOutputStream()
        player = AudioPlayer(stream_factory=lambda rate: stream)

        finished = player.play(np.zeros(AudioPlayer.BLOCK_FRAMES * 2 + 10, dtype=np.float32), 48_000)

        self.assertTrue(finished)
        self.assertEqual(len(stream.blocks), 3)

    def test_stop_interrupts_between_blocks(self):
        player = AudioPlayer()
        stream = FakeOutputStream(player=player)
        player._stream_factory = lambda rate: stream

        finished = player.play(np.zeros(AudioPlayer.BLOCK_FRAMES * 3, dtype=np.float32), 48_000)

        self.assertFalse(finished)
        self.assertEqual(len(stream.blocks), 1)


if __name__ == "__main__":
    unittest.main()
