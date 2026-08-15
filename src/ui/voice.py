"""Voice wiring for the REPL: always-on push-to-talk in, toggleable spoken replies out.

`VoiceController` glues the `src.voice` layer to the chat loop. Push-to-talk
starts with the REPL (`start_input()`) and is always available; `/voice` only
flips `speak_replies` — whether the agent's answers are spoken (and whether
the system prompt gets the TTS-friendly suffix). Thread map: the pynput
listener thread calls `toggle()`; a recording worker steps the incremental
ASR session while the user talks; a speech worker synthesizes and plays
queued sentences; the asyncio main loop receives finished transcripts via
`call_soon_threadsafe` (exiting the active prompt with the text, or parking
it in `take_pending_text` when no prompt is up). Every callback path swallows
and logs its own failures — voice must never break a turn or kill a listener
thread.
"""

import asyncio
import queue
import threading
import time

from loguru import logger

from src.voice import AudioPlayer, HotkeyListener, MicrophoneRecorder, SentenceBuffer, SpeakableFilter

STEP_INTERVAL = 2.0
DRAFT_CHARS = 48


class VoiceController:
    """Owns the microphone, hotkey, ASR session, and TTS pipeline of voice mode."""

    def __init__(
        self,
        ui,
        *,
        transcriber=None,
        speaker=None,
        recorder=None,
        player=None,
        listener=None,
        loop=None,
        step_interval: float = STEP_INTERVAL,
        status_overlay=None,
    ):
        self.ui = ui
        self._status_overlay = status_overlay
        self.started = False  # push-to-talk running (listener + workers)
        self.speak_replies = False  # /voice: read answers aloud
        self.key_spec = "alt_r"
        self.step_interval = step_interval
        self._transcriber = transcriber
        self._speaker = speaker
        self._recorder = recorder
        self._player = player
        self._listener = listener
        self._loop = loop
        self._session = None
        self._recording = False
        self._record_started = 0.0
        self._draft = ""
        self._record_stop = threading.Event()
        self._record_worker: threading.Thread | None = None
        self._sentences: queue.Queue = queue.Queue()
        self._speech_stop = threading.Event()
        self._speech_worker: threading.Thread | None = None
        self._speaking = threading.Event()
        self._pending_text: str | None = None
        self._pending_lock = threading.Lock()
        self._buffer = SentenceBuffer()
        self._filter = SpeakableFilter()

    # ------------------------------------------------------------- lifecycle

    def start_input(self) -> None:
        """Start push-to-talk: hotkey listener, speech worker, ASR warm-up."""
        if self.started:
            return
        self._build_components()
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None
        self.started = True
        self._speech_stop.clear()
        self._speech_worker = threading.Thread(target=self._speech_loop, daemon=True)
        self._speech_worker.start()
        self._listener.start()
        threading.Thread(target=self._warm_up_asr, daemon=True).start()

    def set_speaking(self, on: bool) -> None:
        """`/voice`: turn spoken replies on or off; push-to-talk is unaffected."""
        self.speak_replies = on
        if on:
            threading.Thread(target=self._warm_up_tts, daemon=True).start()
        else:
            self._interrupt_speech()
            self._buffer = SentenceBuffer()
            self._filter = SpeakableFilter()

    def shutdown(self) -> None:
        """Full stop of push-to-talk and speech (tests and process exit)."""
        self.started = False
        self.speak_replies = False
        if self._listener is not None:
            self._listener.stop()
        if self._recording:
            self._recording = False
            self._record_stop.set()
            try:
                self._recorder.stop()
            except Exception:
                logger.exception("Failed to stop the microphone")
        self._interrupt_speech()
        self._speech_stop.set()
        self._sentences.put(None)  # unblock the speech worker so it can exit
        self._draft = ""
        self._buffer = SentenceBuffer()
        self._filter = SpeakableFilter()
        self._set_overlay_voice(None)

    def _set_overlay_voice(self, state: str | None) -> None:
        """Update the on-screen voice pill (listening/syncing/hidden)."""
        try:
            overlay = self._status_overlay
            if overlay is None:
                from src.ui.overlay import get_status_overlay

                overlay = self._status_overlay = get_status_overlay()
            overlay.set_voice(state)
        except Exception:
            logger.exception("Status overlay update failed")

    def _build_components(self) -> None:
        from src.config.settings import settings
        from src.voice import VoiceProvider, build_speaker, build_transcriber

        self.key_spec = getattr(self._listener, "key_spec", None) or settings.voice_ptt_key
        if self._transcriber is None:
            self._transcriber = build_transcriber(
                VoiceProvider(
                    provider_name="local", model_name=settings.voice_asr_model, language=settings.voice_language
                )
            )
        if self._speaker is None:
            self._speaker = build_speaker(VoiceProvider(provider_name="local", voice=settings.voice_tts_speaker))
        if self._recorder is None:
            self._recorder = MicrophoneRecorder()
        if self._player is None:
            self._player = AudioPlayer()
        if self._listener is None:
            self._listener = HotkeyListener(self.key_spec, on_toggle=self.toggle)

    def _warm_up_asr(self) -> None:
        set_status = getattr(self.ui, "set_model_status", None) or (lambda *_: None)
        try:
            set_status("voice asr", "loading")
            self._transcriber.load()
            set_status("voice asr", "ready")
            logger.info("Push-to-talk ASR model is warm")
        except Exception:
            set_status("voice asr", "failed")
            logger.exception("ASR warm-up failed; the model will retry lazily")

    def _warm_up_tts(self) -> None:
        set_status = getattr(self.ui, "set_model_status", None) or (lambda *_: None)
        try:
            set_status("voice tts", "loading")
            self._speaker.load()
            set_status("voice tts", "ready")
            logger.info("TTS model is warm")
        except Exception:
            set_status("voice tts", "failed")
            logger.exception("TTS warm-up failed; the model will retry lazily")

    # ------------------------------------------------- push-to-talk (hotkey)

    def toggle(self) -> None:
        """One hotkey press: start recording, or stop-and-send. Barge-in included."""
        if not self.started:
            return
        try:
            if self._recording:
                self._finish_recording()
            else:
                self._interrupt_speech()
                self._start_recording()
        except Exception:
            logger.exception("Push-to-talk toggle failed")

    def _start_recording(self) -> None:
        self._session = self._transcriber.open_session()
        self._recorder.start()
        self._recording = True
        self._record_started = time.monotonic()
        self._draft = ""
        self._record_stop.clear()
        self._record_worker = threading.Thread(target=self._record_loop, daemon=True)
        self._record_worker.start()
        logger.info("PTT: recording started")
        self._set_overlay_voice("listening")
        self._invalidate()

    def _record_loop(self) -> None:
        while not self._record_stop.wait(self.step_interval):
            try:
                draft = self._session.step(self._recorder.snapshot())
                self._draft = (self._session.committed + draft).strip()
            except Exception:
                logger.exception("Incremental transcription step failed")
            self._invalidate()

    def _finish_recording(self) -> None:
        self._recording = False
        self._record_stop.set()
        if self._record_worker is not None:
            self._record_worker.join(timeout=15)
        audio = self._recorder.stop()
        started = time.monotonic()
        text = ""
        try:
            text = self._session.finalize(audio)
        except Exception:
            logger.exception("Transcription finalize failed")
        logger.info(
            "PTT: recording stopped — audio {:.1f}s, finalize {:.2f}s, text_chars {}",
            len(audio) / 16_000,
            time.monotonic() - started,
            len(text),
        )
        self._draft = ""
        self._set_overlay_voice("syncing" if text else None)
        self._invalidate()
        if text:
            self._submit(text)
            self._play_sent_cue()

    # --------------------------------------------------- transcript delivery

    def _submit(self, text: str) -> None:
        """Hand the transcript to the main loop: type it into the live prompt, or park it.

        The transcript goes through the prompt buffer and `validate_and_handle`
        (a programmatic Enter) instead of `app.exit(result=...)` so the user
        SEES what was recognized: the text shows in the prompt line, stays in
        the scrollback, and enters the input history. Text already typed in
        the buffer is kept — the transcript is appended after it.
        """
        session = getattr(self.ui, "prompt_session", None)
        app = getattr(session, "app", None) if session is not None else None
        if app is None or self._loop is None:
            self._set_pending(text)
            return

        def push() -> None:
            try:
                if app.is_running:
                    buffer = app.current_buffer
                    existing = buffer.text.strip()
                    buffer.text = f"{existing} {text}" if existing else text
                    buffer.cursor_position = len(buffer.text)
                    buffer.validate_and_handle()
                    return
            except Exception:
                logger.exception("Could not deliver the transcript into the prompt")
            self._set_pending(text)

        self._loop.call_soon_threadsafe(push)

    def _play_sent_cue(self) -> None:
        """Audible confirmation that the utterance went off to the model.

        Push-to-talk works with any window focused, so the terminal may not
        be visible — without the cue the user cannot tell a sent message from
        a failed transcription. Respects the /sound toggle.
        """
        if not getattr(self.ui, "sound_enabled", True):
            return
        try:
            from src.ui.sound import play_done_sound

            play_done_sound()
        except Exception:
            logger.exception("Send-confirmation sound failed")

    def _set_pending(self, text: str) -> None:
        with self._pending_lock:
            self._pending_text = text

    def take_pending_text(self) -> str | None:
        """A transcript that arrived while no prompt was active, once."""
        with self._pending_lock:
            text, self._pending_text = self._pending_text, None
            return text

    # -------------------------------------------------------- spoken replies

    def wrap_on_token(self, inner=None):
        """A token callback that feeds both the turn stream and the TTS pipeline."""

        def forward(piece) -> None:
            if inner is not None:
                inner(piece)
            self.feed_token(str(piece))

        return forward

    def feed_token(self, text: str) -> None:
        if not self.speak_replies:
            return
        try:
            for sentence in self._buffer.feed(text):
                self._enqueue(sentence)
        except Exception:
            logger.exception("Failed to queue streamed text for speech")

    def cancel_turn(self) -> None:
        """A cancelled turn must not speak its tail: drop everything buffered."""
        self._interrupt_speech()
        self._buffer = SentenceBuffer()
        self._filter = SpeakableFilter()
        self._set_overlay_voice(None)

    def finish_turn(self) -> None:
        """Flush the sentence tail and reset per-turn markdown state."""
        try:
            tail = self._buffer.flush()
            if self.speak_replies and tail:
                self._enqueue(tail)
        except Exception:
            logger.exception("Failed to flush speech at turn end")
        self._filter = SpeakableFilter()
        self._set_overlay_voice(None)  # ответ пришёл — «syncing» снимается

    def _enqueue(self, sentence: str) -> None:
        speakable = self._filter.filter(sentence)
        if speakable:
            self._sentences.put(speakable)

    def _speech_loop(self) -> None:
        while not self._speech_stop.is_set():
            sentence = self._sentences.get()
            if sentence is None:
                continue
            self._speaking.set()
            self._invalidate()
            try:
                audio, rate = self._speaker.synthesize(sentence)
                self._player.play(audio, rate)
            except Exception:
                logger.exception("Speech synthesis or playback failed")
            finally:
                if self._sentences.empty():
                    self._speaking.clear()
                    self._invalidate()

    def _interrupt_speech(self) -> None:
        while True:
            try:
                self._sentences.get_nowait()
            except queue.Empty:
                break
        if self._player is not None:
            self._player.stop()
        self._speaking.clear()

    # ---------------------------------------------------------------- status

    def status(self) -> str:
        """The prompt's right-side status: recording stopwatch + draft, or speaker."""
        if self._recording:
            seconds = time.monotonic() - self._record_started
            tail = self._draft[-DRAFT_CHARS:]
            return f"🎙 {seconds:.0f}s {tail}".rstrip()
        if self._speaking.is_set():
            return "🔊"
        return ""

    def _invalidate(self) -> None:
        """Threadsafe prompt redraw so the rprompt status stays current."""
        session = getattr(self.ui, "prompt_session", None)
        app = getattr(session, "app", None) if session is not None else None
        if app is None or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(app.invalidate)
        except Exception:
            logger.exception("Prompt invalidate failed")
