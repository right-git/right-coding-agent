"""Global push-to-talk toggle key. pynput is imported lazily so the module stays light."""

from collections.abc import Callable

from loguru import logger


def parse_hotkey(spec: str) -> set:
    """A pynput key name (alt_r, f8, pause...) or single character → set of key objects to match.

    Right Alt is special-cased: on some Windows layouts it arrives as AltGr,
    so both are accepted.
    """
    from pynput import keyboard

    spec = spec.strip().lower()
    try:
        key = keyboard.Key[spec]
    except KeyError:
        if len(spec) == 1:
            return {keyboard.KeyCode.from_char(spec)}
        raise ValueError(f"Unknown hotkey {spec!r}; use a pynput key name (alt_r, f8, pause) or a single character")
    if key is keyboard.Key.alt_r:
        return {keyboard.Key.alt_r, keyboard.Key.alt_gr}
    return {key}


class HotkeyListener:
    """Fires `on_toggle` on every press of the configured key, from the listener thread.

    `listener_factory(on_press)` is the test seam; the default builds a global
    pynput keyboard listener. Errors in `on_toggle` are logged, never raised —
    a broken callback must not kill the listener thread.
    """

    def __init__(self, key_spec: str = "alt_r", on_toggle: Callable[[], None] = lambda: None, listener_factory=None):
        self.key_spec = key_spec
        self.on_toggle = on_toggle
        self._listener_factory = listener_factory
        self._listener = None

    def start(self) -> None:
        if self._listener is not None:
            return
        keys = parse_hotkey(self.key_spec)

        def on_press(key) -> None:
            if key in keys:
                try:
                    self.on_toggle()
                except Exception as error:
                    logger.error("Push-to-talk toggle callback failed: {}", error)

        factory = self._listener_factory or self._default_listener
        self._listener = factory(on_press)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @staticmethod
    def _default_listener(on_press):
        from pynput import keyboard

        return keyboard.Listener(on_press=on_press)
