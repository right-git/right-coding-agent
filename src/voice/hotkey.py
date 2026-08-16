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

# macOS virtual keycodes (kVK_*) for the same key-spec names.
DARWIN_KEYCODES = {
    "alt_r": 61,
    "alt_gr": 61,  # AltGr arrives as right Option
    "alt_l": 58,
    "cmd_r": 54,
    "cmd_l": 55,
    "ctrl_r": 62,
    "ctrl_l": 59,
    "shift_r": 60,
    "shift_l": 56,
    "caps_lock": 57,
    "space": 49,
    "home": 115,
    "end": 119,
    **dict(
        zip(
            [f"f{n}" for n in range(1, 21)],
            [122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111, 105, 107, 113, 106, 64, 79, 80, 90],
        )
    ),
}


def resolve_darwin_keycode(spec: str) -> set[int] | None:
    """macOS keycodes for a key spec, or None when only pynput can watch it."""
    spec = spec.strip().lower()
    if spec in DARWIN_KEYCODES:
        return {DARWIN_KEYCODES[spec]}
    return None


def darwin_state_reader(keycodes: set[int], key_state=None):
    """A KeyStatePoller state reader over Quartz.CGEventSourceKeyState.

    Reads the session's HID key state directly — there is no event tap for
    macOS to disable, so it keeps working through the CPU spikes that kill
    pynput's tap (`key_state` is the test seam).
    """
    if key_state is None:
        import Quartz

        def key_state(code):
            return Quartz.CGEventSourceKeyState(Quartz.kCGEventSourceStateCombinedSessionState, code)

    return lambda: any(key_state(code) for code in keycodes)


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


ACCESS_HINT = (
    "push-to-talk needs macOS permissions: System Settings → Privacy & Security → "
    "enable your terminal app under BOTH Accessibility and Input Monitoring, then restart"
)

DARWIN_KEY_LABELS = {
    "alt_r": "right Option ⌥",
    "alt_l": "left Option ⌥",
    "alt_gr": "right Option ⌥",
    "cmd_r": "right Command ⌘",
    "cmd_l": "left Command ⌘",
    "ctrl_r": "right Control ⌃",
    "ctrl_l": "left Control ⌃",
}


def describe_hotkey(spec: str, platform: str | None = None) -> str:
    """The key spec with its macOS name — "alt_r" alone reads as the wrong key there."""
    spec = spec.strip().lower()
    label = DARWIN_KEY_LABELS.get(spec) if (platform or sys.platform) == "darwin" else None
    return f"{spec} ({label})" if label else spec


def _accessibility_granted() -> bool:
    """Accessibility (AXIsProcessTrusted); raises the system dialog once when missing."""
    try:
        import ApplicationServices

        if ApplicationServices.AXIsProcessTrusted():
            return True
        try:
            options = {ApplicationServices.kAXTrustedCheckOptionPrompt: True}
            ApplicationServices.AXIsProcessTrustedWithOptions(options)
        except Exception:
            logger.debug("Could not raise the macOS accessibility prompt")
        return False
    except Exception:
        return True


def _input_monitoring_granted() -> bool:
    """Input Monitoring (CGPreflightListenEventAccess); requests it once when missing.

    A listen-only keyboard tap on modern macOS needs this in addition to
    Accessibility — without it the tap starts with no error and then silently
    receives nothing, which reads as "the key does nothing".
    """
    try:
        import Quartz

        if Quartz.CGPreflightListenEventAccess():
            return True
        try:
            Quartz.CGRequestListenEventAccess()
        except Exception:
            logger.debug("Could not raise the macOS input-monitoring prompt")
        return False
    except Exception:
        return True


def macos_input_access(platform: str | None = None) -> bool:
    """True when this process may watch global input; asks macOS to grant it once.

    Starting an untrusted event tap makes the OS print "This process is not
    trusted!" straight to stderr (over the live prompt) and the listener
    silently receives nothing — so preflight both permissions, trigger the
    system dialogs, and let the caller explain instead. When a check itself is
    unavailable, report access so pynput still gets its chance.
    """
    if (platform or sys.platform) != "darwin":
        return True
    return _accessibility_granted() and _input_monitoring_granted()


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

    def __init__(
        self,
        key_spec: str = "alt_r",
        on_toggle: Callable[[], None] = lambda: None,
        listener_factory=None,
        access_checker=None,
    ):
        self.key_spec = key_spec
        self.on_toggle = on_toggle
        self._listener_factory = listener_factory
        self._access_checker = access_checker
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
                sink = self._try_raw_input(vk_codes)
                if sink is not None:
                    self._listener = sink
                    logger.info("Push-to-talk backend: raw-input sink ({})", self.key_spec)
                    return
                self._listener = KeyStatePoller(vk_codes, self._fire)
                self._listener.start()
                logger.info("Push-to-talk backend: GetAsyncKeyState poller ({})", self.key_spec)
                return

        # The explicit checker is always honoured; the platform default only
        # guards the real pynput path (test factories need no OS permission).
        if self._access_checker is not None:
            allowed = self._access_checker()
        else:
            allowed = self._listener_factory is not None or macos_input_access()
        if not allowed:
            raise PermissionError(ACCESS_HINT)

        if self._listener_factory is None and sys.platform == "darwin":
            # Primary macOS backend: poll the HID key state. A pynput event
            # tap is silently disabled by kCGEventTapDisabledByTimeout when
            # whisper's CPU spikes starve the callback (observed live: the
            # stop press never arrived and recording ran forever), and pynput
            # never re-enables it. Polling has no tap to lose — the same cure
            # as GetAsyncKeyState on Windows.
            keycodes = resolve_darwin_keycode(self.key_spec)
            if keycodes:
                self._listener = KeyStatePoller(keycodes, self._fire, state_reader=darwin_state_reader(keycodes))
                self._listener.start()
                logger.info("Push-to-talk backend: macOS key-state poller ({})", self.key_spec)
                return

        keys = parse_hotkey(self.key_spec)
        logger.info("Push-to-talk backend: pynput listener ({})", self.key_spec)

        def on_press(key) -> None:
            if key in keys:
                self._fire()

        factory = self._listener_factory or self._default_listener
        self._listener = factory(on_press)
        self._listener.start()

    def _try_raw_input(self, vk_codes: set[int]):
        """The primary Windows backend; None when it could not start."""
        try:
            from .rawinput import KeyEdgeDecoder, RawKeyboardSink

            sink = RawKeyboardSink(KeyEdgeDecoder(vk_codes, self._fire))
            if sink.start():
                return sink
            sink.stop()
        except Exception:
            logger.exception("Raw-input backend unavailable, falling back to polling")
        return None

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @staticmethod
    def _default_listener(on_press):
        from pynput import keyboard

        return keyboard.Listener(on_press=on_press)
