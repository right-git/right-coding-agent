import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.prompts import Prompts
from src.voice import (
    SpeakableFilter,
)
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

    def test_vk_resolution_for_named_keys(self):
        from src.voice.hotkey import resolve_vk

        self.assertEqual(resolve_vk("alt_r"), {0xA5})
        self.assertEqual(resolve_vk("f8"), {0x77})
        self.assertIsNone(resolve_vk("no_such_key"))

    def test_poller_fires_only_on_down_edges(self):
        from src.voice.hotkey import KeyStatePoller

        states = iter([False, True, True, False, True])
        fired = []
        poller = KeyStatePoller({0xA5}, lambda: fired.append(1), state_reader=lambda: next(states))

        for _ in range(5):
            poller._step()

        self.assertEqual(len(fired), 2)  # два нажатия, удержание не дублирует

    @unittest.skipUnless(sys.platform == "win32", "raw input — только Windows")
    def test_windows_listener_prefers_the_raw_input_sink(self):
        from src.voice.rawinput import RawKeyboardSink

        listener = HotkeyListener("alt_r", on_toggle=lambda: None)
        listener.start()
        try:
            self.assertIsInstance(listener._listener, RawKeyboardSink)
        finally:
            listener.stop()

    @unittest.skipUnless(sys.platform == "win32", "raw input — только Windows")
    def test_raw_modifiers_normalize_to_sided_codes(self):
        from src.voice.rawinput import RI_KEY_E0, normalize_vk

        self.assertEqual(normalize_vk(0x12, RI_KEY_E0, 0x38), 0xA5)  # правый Alt
        self.assertEqual(normalize_vk(0x12, 0, 0x38), 0xA4)  # левый Alt
        self.assertEqual(normalize_vk(0x10, 0, 0x36), 0xA1)  # правый Shift по скан-коду
        self.assertEqual(normalize_vk(0x41, 0, 0), 0x41)  # обычные клавиши как есть

    @unittest.skipUnless(sys.platform == "win32", "raw input — только Windows")
    def test_raw_decoder_fires_once_per_hold_and_swallows_autorepeat(self):
        from src.voice.rawinput import RI_KEY_BREAK, RI_KEY_E0, KeyEdgeDecoder

        fired = []
        decoder = KeyEdgeDecoder({0xA5}, lambda: fired.append(1))

        decoder.handle(0x12, RI_KEY_E0, 0x38)  # нажатие
        decoder.handle(0x12, RI_KEY_E0, 0x38)  # autorepeat при удержании
        decoder.handle(0x12, RI_KEY_E0 | RI_KEY_BREAK, 0x38)  # отпускание
        decoder.handle(0x12, RI_KEY_E0, 0x38)  # второе нажатие
        decoder.handle(0x12, 0, 0x38)  # левый Alt — не наша клавиша

        self.assertEqual(len(fired), 2)

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

    def test_denied_input_access_raises_instead_of_starting_pynput(self):
        # Starting an untrusted event tap on macOS prints "This process is not
        # trusted!" straight over the prompt — preflight and explain instead.
        factory_calls = []

        listener = HotkeyListener(
            "f8",
            listener_factory=lambda on_press: factory_calls.append(on_press),
            access_checker=lambda: False,
        )

        with self.assertRaises(PermissionError) as raised:
            listener.start()

        self.assertIn("Accessibility", str(raised.exception))
        self.assertIn("Input Monitoring", str(raised.exception))
        self.assertEqual(factory_calls, [])

    def test_macos_access_needs_both_permissions(self):
        from src.voice import hotkey as hotkey_module

        cases = [(True, True, True), (True, False, False), (False, True, False)]
        for accessibility, monitoring, expected in cases:
            with (
                patch.object(hotkey_module, "_accessibility_granted", return_value=accessibility),
                patch.object(hotkey_module, "_input_monitoring_granted", return_value=monitoring),
            ):
                self.assertEqual(hotkey_module.macos_input_access(platform="darwin"), expected)

    def test_access_is_assumed_outside_macos(self):
        from src.voice.hotkey import macos_input_access

        self.assertTrue(macos_input_access(platform="win32"))

    def test_darwin_keycodes_resolve_for_named_keys(self):
        from src.voice.hotkey import resolve_darwin_keycode

        self.assertEqual(resolve_darwin_keycode("alt_r"), {61})
        self.assertEqual(resolve_darwin_keycode("alt_gr"), {61})  # same physical key
        self.assertEqual(resolve_darwin_keycode("f8"), {100})
        self.assertIsNone(resolve_darwin_keycode("no_such_key"))
        self.assertIsNone(resolve_darwin_keycode("g"))  # characters stay on pynput

    def test_darwin_state_reader_reports_any_watched_key(self):
        from src.voice.hotkey import darwin_state_reader

        down = {61}
        reader = darwin_state_reader({58, 61}, key_state=lambda code: code in down)
        self.assertTrue(reader())

        down.clear()
        self.assertFalse(reader())

    @unittest.skipUnless(sys.platform == "darwin", "poller backend — только macOS")
    def test_macos_listener_prefers_the_key_state_poller(self):
        # A pynput event tap dies silently when macOS hits
        # kCGEventTapDisabledByTimeout under CPU load (whisper steps during
        # recording) and pynput never re-enables it — the second press is
        # lost and the pill sticks on "listening". Polling has no tap to lose.
        from src.voice.hotkey import KeyStatePoller

        listener = HotkeyListener("alt_r", on_toggle=lambda: None, access_checker=lambda: True)
        listener.start()
        try:
            self.assertIsInstance(listener._listener, KeyStatePoller)
        finally:
            listener.stop()

    def test_hotkey_gets_a_human_label_on_macos(self):
        from src.voice.hotkey import describe_hotkey

        self.assertEqual(describe_hotkey("alt_r", platform="darwin"), "alt_r (right Option ⌥)")
        self.assertEqual(describe_hotkey("alt_r", platform="win32"), "alt_r")
        self.assertEqual(describe_hotkey("f8", platform="darwin"), "f8")

    def test_granted_input_access_starts_the_listener(self):
        class FakeListener:
            def __init__(self, on_press):
                self.on_press = on_press

            def start(self):
                pass

            def stop(self):
                pass

        listener = HotkeyListener("f8", listener_factory=FakeListener, access_checker=lambda: True)
        listener.start()

        self.assertIsInstance(listener._listener, FakeListener)
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


class WhisperModelSourceTests(unittest.TestCase):
    def test_known_alias_maps_to_its_hub_repo(self):
        from src.voice.providers.whisper import resolve_repo

        self.assertIn("/", resolve_repo("large-v3-turbo"))
        self.assertNotEqual(resolve_repo("large-v3-turbo"), "large-v3-turbo")

    def test_full_repo_id_passes_through(self):
        from src.voice.providers.whisper import resolve_repo

        self.assertEqual(resolve_repo("org/custom-model"), "org/custom-model")

    def test_model_source_prefers_the_predownloaded_snapshot(self):
        from src.voice.providers.whisper import WhisperTranscriber

        calls = []

        def predownload(repo_id, cache_dir=None, required_files=()):
            calls.append((repo_id, str(cache_dir)))
            return "/snapshots/whisper"

        transcriber = WhisperTranscriber(
            model_name="large-v3-turbo", models_dir=Path("/models/fw"), predownload=predownload
        )

        self.assertEqual(transcriber._model_source(), "/snapshots/whisper")
        self.assertEqual(len(calls), 1)
        self.assertIn("/", calls[0][0])
        self.assertEqual(calls[0][1], "/models/fw")

    def test_model_source_falls_back_to_the_name_when_predownload_fails(self):
        from src.voice.providers.whisper import WhisperTranscriber

        def predownload(repo_id, cache_dir=None, required_files=()):
            raise OSError("offline")

        transcriber = WhisperTranscriber(model_name="large-v3-turbo", predownload=predownload)

        self.assertEqual(transcriber._model_source(), "large-v3-turbo")


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

    def test_cyrillic_routes_to_russian_and_latin_to_english(self):
        from src.voice.providers.silero import detect_language

        self.assertEqual(detect_language("Привет, как дела?"), "ru")
        self.assertEqual(detect_language("Готово: file saved"), "ru")  # mixed leans ru
        self.assertEqual(detect_language("I have closed the video."), "en")
        self.assertEqual(detect_language(""), "en")

    def test_synthesis_picks_the_speaker_and_options_per_language(self):
        # v4_ru speaks 0.3s of noise for an English sentence — English must go
        # to the English model with its own speaker (and without the
        # Russian-only put_accent/put_yo options, which it rejects).
        model = FakeTtsModel()
        speaker = SileroSpeaker(speaker="xenia", english_speaker="en_0", loader=lambda: model)

        speaker.synthesize("Привет.")
        speaker.synthesize("Hello there.")

        russian, english = model.calls
        self.assertEqual(russian["speaker"], "xenia")
        self.assertTrue(russian["put_accent"])
        self.assertEqual(english["speaker"], "en_0")
        self.assertNotIn("put_accent", english)
        self.assertNotIn("put_yo", english)

    def test_download_reports_progress_and_writes_the_file(self):
        from contextlib import contextmanager
        from tempfile import TemporaryDirectory

        from src.utils.downloads import reporting_progress

        chunks = [b"x" * 19_000_000, b"x" * 19_000_000]

        class FakeResponse:
            headers = {"content-length": str(sum(len(c) for c in chunks))}

            def raise_for_status(self):
                pass

            def iter_bytes(self, _size):
                yield from chunks

        @contextmanager
        def fake_stream(method, url, **kwargs):
            yield FakeResponse()

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "v4_ru.pt"
            speaker = SileroSpeaker(models_dir=Path(tmp), stream_factory=fake_stream)
            seen = []
            with reporting_progress(seen.append):
                speaker._download(target)

            self.assertEqual(target.stat().st_size, 38_000_000)
            self.assertIn("↓ 19/38 MB 50%", seen)
            self.assertEqual(seen[-1], "↓ 38/38 MB 100%")


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


class SpeakableFilterTests(unittest.TestCase):
    def test_inline_markdown_becomes_plain_text(self):
        cleaner = SpeakableFilter()

        self.assertEqual(cleaner.filter("Запусти `lint.sh` — **важно**."), "Запусти lint.sh — важно.")
        self.assertEqual(cleaner.filter("Смотри [доку](https://a.b)."), "Смотри доку.")

    def test_fenced_code_is_never_spoken(self):
        cleaner = SpeakableFilter()

        self.assertEqual(cleaner.filter("Вот код: ```python"), "Вот код:")
        self.assertEqual(cleaner.filter("x = 1."), "")  # внутри fence
        self.assertEqual(cleaner.filter("``` Готово."), "Готово.")


class VoicePromptTests(unittest.TestCase):
    def test_voice_mode_appends_tts_instructions(self):
        base = Prompts.coding_system(False)
        voiced = Prompts.coding_system(True)

        self.assertTrue(voiced.startswith(base))
        self.assertIn("text-to-speech", voiced)
        self.assertNotIn("text-to-speech", base)


class FakeHotkeyListener:
    def __init__(self):
        self.started = False
        self.key_spec = "f8"

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


class FakeVoiceRecorder:
    def __init__(self, audio):
        self.audio = audio
        self.recording = False

    def start(self):
        self.recording = True

    def snapshot(self):
        return self.audio

    def stop(self):
        self.recording = False
        return self.audio


class FakeVoiceSpeaker:
    def load(self):
        return self

    def synthesize(self, text):
        return np.zeros(10, dtype=np.float32), 24_000


class FakeVoicePlayer:
    def __init__(self):
        self.stopped = False

    def play(self, audio, rate):
        return True

    def stop(self):
        self.stopped = True


class FakeStatusOverlay:
    def __init__(self):
        self.voice_states = []
        self.pings = 0
        self.prewarmed = False

    def set_voice(self, state):
        self.voice_states.append(state)

    def ping_computer(self, linger=3.0):
        self.pings += 1

    def prewarm(self):
        self.prewarmed = True


class VoiceControllerTests(unittest.TestCase):
    def make_controller(self, text="привет мир"):
        from src.ui.voice import VoiceController

        model = FakeWhisperModel([[FakeSegment(text, 2.0)]])
        return VoiceController(
            SimpleNamespace(prompt_session=None),
            transcriber=WhisperTranscriber(loader=lambda: model),
            speaker=FakeVoiceSpeaker(),
            recorder=FakeVoiceRecorder(np.zeros(3 * SAMPLE_RATE, dtype=np.float32)),
            player=FakeVoicePlayer(),
            listener=FakeHotkeyListener(),
            step_interval=999,
            status_overlay=FakeStatusOverlay(),
        )

    def test_start_input_prewarms_the_overlay(self):
        # The macOS overlay child takes ~a second to boot; spawning it with the
        # REPL means the pill shows the instant the hotkey is first pressed.
        controller = self.make_controller()
        controller.start_input()
        self.addCleanup(controller.shutdown)

        self.assertTrue(controller._status_overlay.prewarmed)

    def test_toggle_records_then_submits_the_transcript(self):
        controller = self.make_controller()
        controller.start_input()
        self.addCleanup(controller.shutdown)

        controller.toggle()
        self.assertTrue(controller._recording)
        self.assertTrue(controller._recorder.recording)

        with patch("src.ui.sound.play_done_sound") as cue:
            controller.toggle()
            controller._finalize_worker.join(timeout=2)
        self.assertFalse(controller._recording)
        self.assertEqual(controller.take_pending_text(), "привет мир")
        self.assertIsNone(controller.take_pending_text())
        cue.assert_called_once()  # звук — подтверждение отправки из любого окна

    def test_second_press_stops_the_animation_before_transcription_finishes(self):
        # Finalize takes seconds on CPU — the pill and the prompt stopwatch
        # must react to the key instantly, not after the transcription.
        import threading as threading_module

        from src.ui.voice import VoiceController

        gate = threading_module.Event()

        class GatedSession:
            committed = ""

            def step(self, audio):
                return ""

            def finalize(self, audio):
                gate.wait(2)
                return "готово"

        class GatedTranscriber:
            def load(self):
                return None

            def open_session(self):
                return GatedSession()

        overlay = FakeStatusOverlay()
        controller = VoiceController(
            SimpleNamespace(prompt_session=None, sound_enabled=False),
            transcriber=GatedTranscriber(),
            speaker=FakeVoiceSpeaker(),
            recorder=FakeVoiceRecorder(np.zeros(SAMPLE_RATE, dtype=np.float32)),
            player=FakeVoicePlayer(),
            listener=FakeHotkeyListener(),
            step_interval=999,
            status_overlay=overlay,
        )
        controller.started = True
        self.addCleanup(controller.shutdown)

        controller.toggle()
        controller.toggle()  # must return immediately, before finalize is done

        self.assertFalse(controller._recording)
        self.assertEqual(overlay.voice_states[-1], "syncing")
        self.assertEqual(controller.status(), "")  # the 🎙 stopwatch is gone at once
        self.assertIsNone(controller.take_pending_text())  # still transcribing

        gate.set()
        controller._finalize_worker.join(timeout=2)
        self.assertEqual(controller.take_pending_text(), "готово")

    def test_empty_transcript_is_not_submitted_and_stays_silent(self):
        controller = self.make_controller()
        controller._transcriber = WhisperTranscriber(loader=lambda: FakeWhisperModel([[]]))
        controller.started = True

        controller.toggle()
        with patch("src.ui.sound.play_done_sound") as cue:
            controller.toggle()
            controller._finalize_worker.join(timeout=2)

        self.assertIsNone(controller.take_pending_text())
        cue.assert_not_called()

    def test_streamed_tokens_become_queued_sentences(self):
        controller = self.make_controller()
        controller.speak_replies = True  # без start_input(): очередь не разбирается воркером

        wrap = controller.wrap_on_token(None)
        wrap("Привет")
        wrap(" мир. Хвост")
        controller.finish_turn()

        self.assertEqual(controller._sentences.get_nowait(), "Привет мир.")
        self.assertEqual(controller._sentences.get_nowait(), "Хвост")
        self.assertTrue(controller._sentences.empty())

    def test_code_fences_are_not_queued_for_speech(self):
        controller = self.make_controller()
        controller.speak_replies = True

        controller.wrap_on_token(None)("Смотри. ```py x = 1. ``` Готово. ")
        controller.finish_turn()

        self.assertEqual(controller._sentences.get_nowait(), "Смотри.")
        self.assertEqual(controller._sentences.get_nowait(), "Готово.")
        self.assertTrue(controller._sentences.empty())

    def test_transcript_is_typed_into_the_live_prompt(self):
        class FakeBuffer:
            text = "поправь тесты:"
            cursor_position = 0
            accepted = False

            def validate_and_handle(self):
                self.accepted = True

        buffer = FakeBuffer()
        app = SimpleNamespace(is_running=True, current_buffer=buffer)
        controller = self.make_controller()
        controller.ui = SimpleNamespace(prompt_session=SimpleNamespace(app=app))
        controller._loop = SimpleNamespace(call_soon_threadsafe=lambda fn: fn())

        controller._submit("запусти линт")

        self.assertEqual(buffer.text, "поправь тесты: запусти линт")
        self.assertTrue(buffer.accepted)  # программный Enter — текст виден и уходит обычным путём
        self.assertIsNone(controller.take_pending_text())

    def test_toggle_barges_in_on_running_speech(self):
        controller = self.make_controller()
        controller.started = True
        controller._sentences.put("недоговорённое")

        controller.toggle()
        self.addCleanup(controller.shutdown)

        self.assertTrue(controller._recording)
        self.assertTrue(controller._sentences.empty())
        self.assertTrue(controller._player.stopped)

    def test_not_started_controller_ignores_tokens_and_toggles(self):
        controller = self.make_controller()

        controller.feed_token("Привет. ")
        controller.toggle()

        self.assertTrue(controller._sentences.empty())
        self.assertFalse(controller._recording)

    def test_overlay_walks_listening_syncing_hidden(self):
        controller = self.make_controller()
        controller.started = True
        overlay = controller._status_overlay

        controller.toggle()  # запись
        with patch("src.ui.sound.play_done_sound"):
            controller.toggle()  # стоп + отправка
            controller._finalize_worker.join(timeout=2)
        controller.finish_turn()  # ответ пришёл
        self.addCleanup(controller.shutdown)

        self.assertEqual(overlay.voice_states, ["listening", "syncing", None])

    def test_overlay_hides_when_nothing_was_recognized(self):
        controller = self.make_controller()
        controller._transcriber = WhisperTranscriber(loader=lambda: FakeWhisperModel([[]]))
        controller.started = True
        overlay = controller._status_overlay

        controller.toggle()
        controller.toggle()
        controller._finalize_worker.join(timeout=2)

        # "syncing" flips on immediately at the second press; the empty
        # transcript then hides the pill.
        self.assertEqual(overlay.voice_states, ["listening", "syncing", None])

    def test_voice_off_mutes_replies_but_keeps_push_to_talk(self):
        controller = self.make_controller()
        controller.started = True
        controller.speak_replies = True
        controller.wrap_on_token(None)("Раз. ")
        self.assertEqual(controller._sentences.get_nowait(), "Раз.")

        controller.set_speaking(False)
        controller.wrap_on_token(None)("Два. ")

        self.assertTrue(controller._sentences.empty())
        controller.toggle()  # push-to-talk всё ещё работает
        self.addCleanup(controller.shutdown)
        self.assertTrue(controller._recording)


class StatusOverlayTests(unittest.TestCase):
    def test_voice_state_and_border_deadline(self):
        from src.ui.overlay import StatusOverlay

        overlay = StatusOverlay()
        overlay._apply("voice", "listening")
        self.assertEqual(overlay.voice_state, "listening")

        overlay._apply("computer", 3.0, now=100.0)
        self.assertTrue(overlay.computer_active(now=102.9))
        self.assertFalse(overlay.computer_active(now=103.1))

    def test_border_deadline_only_extends(self):
        from src.ui.overlay import StatusOverlay

        overlay = StatusOverlay()
        overlay._apply("computer", 5.0, now=100.0)
        overlay._apply("computer", 1.0, now=100.5)  # короткий пинг не укорачивает

        self.assertTrue(overlay.computer_active(now=104.9))

    def test_blend_mixes_hex_colors(self):
        from src.ui.overlay import blend

        self.assertEqual(blend("#000000", "#ff0000", 0.0), "#000000")
        self.assertEqual(blend("#000000", "#ff0000", 1.0), "#ff0000")
        self.assertEqual(blend("#000000", "#ff0000", 0.5), "#800000")

    def test_one_bad_frame_never_freezes_the_overlay(self):
        # tick() reschedules itself with root.after; an exception escaping the
        # draw would break that chain and freeze the pill on its last frame.
        from unittest.mock import Mock

        from src.ui.overlay import StatusOverlay

        overlay = StatusOverlay()
        with patch.object(StatusOverlay, "_draw", side_effect=RuntimeError("boom")):
            overlay._safe_draw(Mock(), 100, 100, now=0.0)  # must not raise

    def test_hidden_voice_state_never_starts_the_window(self):
        from src.ui.overlay import StatusOverlay

        overlay = StatusOverlay()
        with patch("src.ui.overlay.tk_thread_supported", return_value=True):
            overlay.set_voice(None)  # already hidden — nothing to show

        self.assertIsNone(overlay._thread)


class FakeOverlayChild:
    """Stands in for the overlay child process (subprocess.Popen shape)."""

    def __init__(self):
        self.lines: list[str] = []
        self.exit_code = None
        self.terminated = False
        self.stdin = SimpleNamespace(
            write=self.lines.append,
            flush=lambda: None,
            close=lambda: None,
        )

    def poll(self):
        return self.exit_code

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        self.terminated = True


class OverlayChildProcessTests(unittest.TestCase):
    """On macOS the overlay runs in a child process: Tk windows are only legal
    on a main thread, and AppKit aborts the whole app otherwise."""

    def make_child_overlay(self):
        from src.ui.overlay import StatusOverlay

        child = FakeOverlayChild()
        return StatusOverlay(child_factory=lambda: child), child

    def test_commands_roundtrip_through_the_wire_format(self):
        from src.ui.overlay import decode_command, encode_command

        cases = [
            ("voice", "listening"),
            ("voice", None),
            ("computer", 3.0),
            ("marker", {"box": [10, 20, 30, 40], "title": "Кнопка", "note": "", "duration": 6.0, "anchor": None}),
            ("marker", None),
            ("stop", None),
        ]
        for command, payload in cases:
            self.assertEqual(decode_command(encode_command(command, payload)), (command, payload))
        self.assertIsNone(decode_command("not json"))
        self.assertIsNone(decode_command('{"command": "unknown"}'))

    def test_macos_backend_sends_voice_commands_to_the_child(self):
        from src.ui.overlay import decode_command

        overlay, child = self.make_child_overlay()
        with patch("src.ui.overlay.tk_thread_supported", return_value=False):
            overlay.set_voice("listening")
            overlay.set_voice(None)

        self.assertIsNone(overlay._thread)
        decoded = [decode_command(line) for line in child.lines]
        self.assertEqual(decoded, [("voice", "listening"), ("voice", None)])

    def test_computer_pings_reach_the_child(self):
        from src.ui.overlay import decode_command

        overlay, child = self.make_child_overlay()
        with patch("src.ui.overlay.tk_thread_supported", return_value=False):
            overlay.ping_computer(2.5)

        self.assertEqual([decode_command(line) for line in child.lines], [("computer", 2.5)])

    def test_markers_are_forwarded_and_hidden_through_the_child(self):
        from src.ui.overlay import decode_command

        overlay, child = self.make_child_overlay()
        payload = {"box": [10, 20, 110, 60], "title": "Кнопка", "note": "тут", "duration": 6.0, "anchor": None}
        with patch("src.ui.overlay.tk_thread_supported", return_value=False):
            overlay.show_marker(payload)
            overlay.hide_marker()

        self.assertEqual(
            [decode_command(line) for line in child.lines],
            [("marker", payload), ("marker", None)],
        )

    def test_hiding_a_marker_with_nothing_running_spawns_nothing(self):
        overlay, child = self.make_child_overlay()
        with patch("src.ui.overlay.tk_thread_supported", return_value=False):
            overlay.hide_marker()

        self.assertEqual(child.lines, [])

    def test_marker_state_applies_and_expires(self):
        from src.ui.overlay import StatusOverlay

        overlay = StatusOverlay()
        overlay._apply("marker", {"box": [1, 2, 3, 4], "duration": 5.0}, now=100.0)
        self.assertTrue(overlay.marker_active(now=104.9))
        self.assertFalse(overlay.marker_active(now=105.1))

        overlay._apply("marker", None, now=104.0)
        self.assertFalse(overlay.marker_active(now=104.0))

    def test_marker_coordinates_are_scaled_from_capture_pixels(self):
        # Retina capture is 2x the Tk point grid — the child divides by the
        # screen's backing scale factor so boxes land on the right element.
        from src.ui.overlay import scale_coords

        self.assertEqual(scale_coords([100, 220, 300, 440], 2.0), [50.0, 110.0, 150.0, 220.0])
        self.assertEqual(scale_coords([10, 20], 1.0), [10.0, 20.0])

    def test_marker_geometry_helpers_stay_in_sync_with_the_canonical_copies(self):
        from src.llm.tools.computer import overlay as marker_overlay
        from src.ui import overlay as ui_overlay

        self.assertEqual(
            ui_overlay.wrap_note("one two three\n\nfour", 7),
            marker_overlay.wrap_note("one two three\n\nfour", 7),
        )
        self.assertEqual(
            ui_overlay.tooltip_placement((100, 100), (200, 80), (1920, 1080), offset=20, margin=10),
            marker_overlay.tooltip_placement((100, 100), (200, 80), (1920, 1080), offset=20, margin=10),
        )
        self.assertEqual(
            ui_overlay.connector_corner((0, 0), (50, 60), (200, 80)),
            marker_overlay.connector_corner((0, 0), (50, 60), (200, 80)),
        )

    def test_dead_child_disables_the_overlay(self):
        overlay, child = self.make_child_overlay()
        child.stdin.write = Mock(side_effect=BrokenPipeError)
        with patch("src.ui.overlay.tk_thread_supported", return_value=False):
            overlay.set_voice("listening")
            overlay.set_voice("listening")  # must be a silent no-op now

        self.assertTrue(overlay._failed)

    def test_prewarm_spawns_the_child_before_the_first_command(self):
        overlay, child = self.make_child_overlay()
        with patch("src.ui.overlay.tk_thread_supported", return_value=False):
            overlay.prewarm()

        self.assertEqual(child.lines, [])
        with patch("src.ui.overlay.tk_thread_supported", return_value=False):
            overlay.set_voice("listening")
        self.assertEqual(len(child.lines), 1)

    def test_child_reader_feeds_the_command_queue_until_stop(self):
        import io
        import queue as queue_module

        from src.ui.overlay import _read_commands, encode_command

        lines = [
            encode_command("voice", "listening"),
            "garbage",
            encode_command("computer", 3.0),
            encode_command("stop", None),
            encode_command("voice", "syncing"),  # after stop — must not be read
        ]
        commands: queue_module.Queue = queue_module.Queue()
        _read_commands(io.StringIO("\n".join(lines) + "\n"), commands)

        drained = []
        while not commands.empty():
            drained.append(commands.get())
        self.assertEqual(drained, [("voice", "listening"), ("computer", 3.0), ("stop", None)])

    def test_child_reader_stops_on_eof(self):
        import io
        import queue as queue_module

        from src.ui.overlay import _read_commands

        commands: queue_module.Queue = queue_module.Queue()
        _read_commands(io.StringIO(""), commands)

        self.assertEqual(commands.get_nowait(), ("stop", None))

    def test_pill_sits_above_the_bottom_on_big_screens_and_centers_in_small_windows(self):
        from src.ui.overlay import PILL_HEIGHT, PILL_MARGIN, pill_top

        self.assertEqual(pill_top(1080), 1080 - PILL_MARGIN - PILL_HEIGHT)
        small_window = PILL_HEIGHT + 32  # the pill-only fallback window
        self.assertEqual(pill_top(small_window), (small_window - PILL_HEIGHT) // 2)

    def test_tcl_env_is_derived_from_the_base_interpreter(self):
        import os
        import tempfile
        from pathlib import Path

        from src.ui.overlay import _ensure_tcl_env

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "lib" / "tcl8.6").mkdir(parents=True)
            (Path(tmp) / "lib" / "tk8.6").mkdir(parents=True)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TCL_LIBRARY", None)
                os.environ.pop("TK_LIBRARY", None)
                _ensure_tcl_env(base_prefix=tmp)
                self.assertEqual(os.environ["TCL_LIBRARY"], str(Path(tmp) / "lib" / "tcl8.6"))
                self.assertEqual(os.environ["TK_LIBRARY"], str(Path(tmp) / "lib" / "tk8.6"))

            with patch.dict(os.environ, {"TCL_LIBRARY": "/custom"}, clear=False):
                _ensure_tcl_env(base_prefix=tmp)
                self.assertEqual(os.environ["TCL_LIBRARY"], "/custom")  # never overrides


class VoiceUiTests(unittest.TestCase):
    def test_notify_done_is_silent_in_voice_mode(self):
        from src.ui.chat import ChatUI

        ui = ChatUI(model="m")
        with patch("src.ui.chat.play_done_sound") as sound:
            ui.voice = SimpleNamespace(speak_replies=True, status=lambda: "")
            ui.notify_done()
            sound.assert_not_called()

            ui.voice = None
            ui.notify_done()
            sound.assert_called_once()

    def test_model_status_line_shows_loading_and_hides_stale_ready(self):
        import time as time_module

        from src.ui.chat import ChatUI

        ui = ChatUI(model="m")
        now = time_module.monotonic()
        ui.model_status["vision"] = ("loading", now - 3, None)
        ui.model_status["voice asr"] = ("ready", now, None)
        ui.model_status["voice tts"] = ("ready", now - 60, None)  # давно готова — не показываем
        ui.model_status["broken"] = ("failed", now - 60, None)  # провал висит, пока его видно

        text = ui._model_status_text()

        self.assertIn("⏳ vision 3s", text)
        self.assertIn("✓ voice asr", text)
        self.assertNotIn("voice tts", text)
        self.assertIn("✗ broken", text)

    def test_model_status_shows_download_detail_while_loading(self):
        from src.ui.chat import ChatUI

        ui = ChatUI(model="m")
        ui.set_model_status("voice asr", "loading", "↓ 1.2/6.4 GB 19%")

        self.assertIn("⏳ voice asr ↓ 1.2/6.4 GB 19%", ui._model_status_text())

    def test_progress_updates_do_not_reset_the_loading_stopwatch(self):
        from src.ui.chat import ChatUI

        ui = ChatUI(model="m")
        ui.set_model_status("vision", "loading")
        started = ui.model_status["vision"][1]
        ui.model_status["vision"] = ("loading", started - 5, None)  # age the entry

        ui.set_model_status("vision", "loading", "↓ 1 GB")

        self.assertEqual(ui.model_status["vision"][1], started - 5)
        self.assertNotEqual(
            ui.set_model_status("vision", "ready") or ui.model_status["vision"][1],
            started - 5,
            "a state change must restart the clock",
        )

    def test_welcome_line_uses_the_human_key_label(self):
        from src.ui.chat import ChatUI
        from src.voice.hotkey import describe_hotkey

        ui = ChatUI(model="m")
        ui.voice = SimpleNamespace(key_spec="alt_r", speak_replies=False)

        self.assertIn(f"push-to-talk {describe_hotkey('alt_r')}", ui._voice_welcome_line())

    def test_rprompt_combines_voice_and_model_status(self):
        from src.ui.chat import ChatUI

        ui = ChatUI(model="m")
        ui.voice = SimpleNamespace(status=lambda: "🎙 2s")
        ui.set_model_status("vision", "loading")

        status = ui._rprompt_status()

        self.assertIn("🎙 2s", status)
        self.assertIn("⏳ vision", status)

    def test_warm_up_reports_model_status(self):
        statuses = []
        controller = self.make_controller_with_ui(SimpleNamespace(set_model_status=lambda *args: statuses.append(args)))

        controller._warm_up_asr()

        self.assertEqual(statuses, [("voice asr", "loading"), ("voice asr", "ready")])

    def test_warm_up_asr_forwards_download_progress(self):
        from src.utils.downloads import report_progress
        from src.ui.voice import VoiceController

        statuses = []
        controller = VoiceController(
            SimpleNamespace(set_model_status=lambda *args: statuses.append(args)),
            transcriber=SimpleNamespace(load=lambda: report_progress("↓ 5/38 MB 13%")),
            speaker=FakeVoiceSpeaker(),
            recorder=FakeVoiceRecorder(np.zeros(SAMPLE_RATE, dtype=np.float32)),
            player=FakeVoicePlayer(),
            listener=FakeHotkeyListener(),
            status_overlay=FakeStatusOverlay(),
        )

        controller._warm_up_asr()

        self.assertIn(("voice asr", "loading", "↓ 5/38 MB 13%"), statuses)

    def test_warm_up_tts_forwards_download_progress(self):
        from src.utils.downloads import report_progress
        from src.ui.voice import VoiceController

        statuses = []
        controller = VoiceController(
            SimpleNamespace(set_model_status=lambda *args: statuses.append(args)),
            transcriber=SimpleNamespace(load=lambda: None),
            speaker=SimpleNamespace(load=lambda: report_progress("↓ 38 MB")),
            recorder=FakeVoiceRecorder(np.zeros(SAMPLE_RATE, dtype=np.float32)),
            player=FakeVoicePlayer(),
            listener=FakeHotkeyListener(),
            status_overlay=FakeStatusOverlay(),
        )

        controller._warm_up_tts()

        self.assertIn(("voice tts", "loading", "↓ 38 MB"), statuses)

    def make_controller_with_ui(self, ui):
        from src.ui.voice import VoiceController

        return VoiceController(
            ui,
            transcriber=WhisperTranscriber(loader=lambda: FakeWhisperModel([])),
            speaker=FakeVoiceSpeaker(),
            recorder=FakeVoiceRecorder(np.zeros(SAMPLE_RATE, dtype=np.float32)),
            player=FakeVoicePlayer(),
            listener=FakeHotkeyListener(),
            status_overlay=FakeStatusOverlay(),
        )

    def test_vision_preload_reports_model_status(self):
        from unittest.mock import Mock

        from src.main import preload_vision_model

        ui = Mock()
        with patch("src.llm.tools.warm_up_computer"):
            preload_vision_model(ui)

        ui.set_model_status.assert_any_call("vision", "loading")
        ui.set_model_status.assert_called_with("vision", "ready")

    def test_vision_preload_forwards_download_progress(self):
        from unittest.mock import Mock

        from src.main import preload_vision_model
        from src.utils.downloads import report_progress

        ui = Mock()
        with patch("src.llm.tools.warm_up_computer", side_effect=lambda: report_progress("↓ 2/7 GB 29%")):
            preload_vision_model(ui)

        ui.set_model_status.assert_any_call("vision", "loading", "↓ 2/7 GB 29%")

    def test_voice_command_toggles_spoken_replies(self):
        from src.ui.chat import ChatUI

        ui = ChatUI(model="m")
        with patch.object(ChatUI, "set_voice_replies") as replies:
            ui.handle_command("/voice")
        replies.assert_called_once_with(True)

        ui.voice = SimpleNamespace(speak_replies=True, key_spec="alt_r")
        with patch.object(ChatUI, "set_voice_replies") as replies:
            ui.handle_command("/voice")
        replies.assert_called_once_with(False)


if __name__ == "__main__":
    unittest.main()
