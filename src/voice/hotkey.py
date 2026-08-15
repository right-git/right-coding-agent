"""Global push-to-talk toggle key.

On Windows the key is read by POLLING `GetAsyncKeyState` instead of a
pynput keyboard hook: low-level hooks have a ~300 ms callback timeout, and
when a game loads the machine while our Python threads hold the GIL,
Windows silently removes the hook — the hotkey "randomly" dies in games.
Polling has no hook to lose and sees the key regardless of raw-input games
or focus. pynput remains the portable fallback (macOS/Linux, exotic keys)
and is imported lazily so the module stays light.
"""

import sys
import threading
from collections.abc import Callable

from loguru import logger

VK_CODES = {
    "alt_r": 0xA5,
    "alt_gr": 0xA5,  # AltGr is the same physical VK as right Alt
    "alt_l": 0xA4,
    "ctrl_r": 0xA3,
    "ctrl_l": 0xA2,
    "shift_r": 0xA1,
    "shift_l": 0xA0,
    "pause": 0x13,
    "caps_lock": 0x14,
    "space": 0x20,
    "home": 0x24,
    "end": 0x23,
    "insert": 0x2D,
    "scroll_lock": 0x91,
    **{f"f{n}": 0x70 + n - 1 for n in range(1, 25)},
}


def resolve_vk(spec: str) -> set[int] | None:
    """Windows virtual-key codes for a key spec, or None when unmapped."""
    spec = spec.strip().lower()
    if spec in VK_CODES:
        return {VK_CODES[spec]}
    if len(spec) == 1 and sys.platform == "win32":
        import ctypes

        code = ctypes.windll.user32.VkKeyScanW(ord(spec))
        if code != -1:
            return {code & 0xFF}
    return None


class KeyStatePoller:
    """Fires on every down-edge of the watched keys via GetAsyncKeyState."""

    INTERVAL_SECONDS = 0.03

    def __init__(self, vk_codes: set[int], on_press_edge: Callable[[], None], state_reader=None):
        self.vk_codes = vk_codes
        self.on_press_edge = on_press_edge
        self._state_reader = state_reader or self._default_state_reader()
        self._was_down = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _default_state_reader(self):
        import ctypes

        user32 = ctypes.windll.user32
        return lambda: any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in self.vk_codes)

    def _step(self) -> None:
        down = bool(self._state_reader())
        if down and not self._was_down:
            self.on_press_edge()
        self._was_down = down

    def _run(self) -> None:
        while not self._stop.wait(self.INTERVAL_SECONDS):
            try:
                self._step()
            except Exception:
                logger.exception("Hotkey polling failed")
                return

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ptt-hotkey-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


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
    """Fires `on_toggle` on every press of the configured key, from a watcher thread.

    On Windows with a mappable key this is a `KeyStatePoller`; otherwise a
    global pynput listener. `listener_factory(on_press)` is the test seam and
    forces the pynput-style path. Errors in `on_toggle` are logged, never
    raised — a broken callback must not kill the watcher.
    """

    def __init__(self, key_spec: str = "alt_r", on_toggle: Callable[[], None] = lambda: None, listener_factory=None):
        self.key_spec = key_spec
        self.on_toggle = on_toggle
        self._listener_factory = listener_factory
        self._listener = None

    def _fire(self) -> None:
        try:
            self.on_toggle()
        except Exception as error:
            logger.error("Push-to-talk toggle callback failed: {}", error)

    def start(self) -> None:
        if self._listener is not None:
            return

        if self._listener_factory is None and sys.platform == "win32":
            vk_codes = resolve_vk(self.key_spec)
            if vk_codes:
                self._listener = KeyStatePoller(vk_codes, self._fire)
                self._listener.start()
                return

        keys = parse_hotkey(self.key_spec)

        def on_press(key) -> None:
            if key in keys:
                self._fire()

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
