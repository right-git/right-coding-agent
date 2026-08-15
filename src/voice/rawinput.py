"""Windows Raw Input sink for the push-to-talk key (the Wispr-Flow-grade backend).

A hidden message-only window registers for raw keyboard input with
`RIDEV_INPUTSINK`, so Windows delivers WM_INPUT for every keystroke of the
session directly to our thread — no low-level hook (nothing for the OS to
time-out and silently remove under game load), and raw-input games cannot
starve it. `KeyEdgeDecoder`/`normalize_vk` are pure and unit-tested; the
window/message plumbing is exercised by injecting keys through SendInput.

Windows-only: import this module lazily (`hotkey.py` does), never at package
import time.
"""

import ctypes
import threading
from ctypes import wintypes

from loguru import logger

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

WM_INPUT = 0x00FF
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
HWND_MESSAGE = -3
RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIM_TYPEKEYBOARD = 1

RI_KEY_BREAK = 0x01
RI_KEY_E0 = 0x02
VK_SHIFT, VK_CONTROL, VK_MENU = 0x10, 0x11, 0x12
SC_RIGHT_SHIFT = 0x36


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG),
    ]


class RAWINPUT(ctypes.Structure):
    # The union's other arms (mouse/HID) are larger, but we only ever read
    # the keyboard arm, and only when dwType says it is one.
    _fields_ = [("header", RAWINPUTHEADER), ("keyboard", RAWKEYBOARD)]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


def normalize_vk(vkey: int, flags: int, make_code: int) -> int:
    """Raw input reports generic modifier VKs; map them to the sided codes."""
    if vkey == VK_MENU:
        return 0xA5 if flags & RI_KEY_E0 else 0xA4
    if vkey == VK_CONTROL:
        return 0xA3 if flags & RI_KEY_E0 else 0xA2
    if vkey == VK_SHIFT:
        return 0xA1 if make_code == SC_RIGHT_SHIFT else 0xA0
    return vkey


class KeyEdgeDecoder:
    """Fires `on_press_edge` on down-edges of the watched keys; swallows autorepeat."""

    def __init__(self, vk_codes: set[int], on_press_edge):
        self.vk_codes = vk_codes
        self.on_press_edge = on_press_edge
        self._down: set[int] = set()

    def handle(self, vkey: int, flags: int, make_code: int) -> None:
        vk = normalize_vk(vkey, flags, make_code)
        if vk not in self.vk_codes:
            return
        if flags & RI_KEY_BREAK:
            self._down.discard(vk)
            return
        if vk not in self._down:  # autorepeat re-sends down events while held
            self._down.add(vk)
            try:
                self.on_press_edge()
            except Exception:
                logger.exception("Push-to-talk edge callback failed")


class RawKeyboardSink:
    """Message-only window + pump thread receiving session-wide WM_INPUT."""

    def __init__(self, decoder: KeyEdgeDecoder):
        self.decoder = decoder
        self._hwnd: int | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._wndproc = None  # held so ctypes does not GC the callback

    def start(self) -> bool:
        """Start the pump; True when the sink is registered and receiving."""
        self._thread = threading.Thread(target=self._run, name="ptt-rawinput", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3.0)
        return self._ready.is_set() and not self._failed.is_set()

    def stop(self) -> None:
        if self._hwnd:
            ctypes.windll.user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        try:
            self._pump()
        except Exception:
            logger.exception("Raw-input sink crashed")
            self._failed.set()
        finally:
            self._ready.set()

    @staticmethod
    def _bind_prototypes(user32, kernel32) -> None:
        """Explicit argtypes/restype: without them ctypes squeezes 64-bit handles into c_int."""
        kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        user32.DefWindowProcW.restype = LRESULT
        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.RegisterClassW.restype = wintypes.ATOM
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        user32.RegisterRawInputDevices.restype = wintypes.BOOL
        user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = ctypes.c_int
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DispatchMessageW.restype = LRESULT
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.PostQuitMessage.argtypes = [ctypes.c_int]
        user32.GetRawInputData.restype = wintypes.UINT
        user32.GetRawInputData.argtypes = [
            wintypes.HANDLE,
            wintypes.UINT,
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.UINT),
            wintypes.UINT,
        ]

    def _pump(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._bind_prototypes(user32, kernel32)

        self._wndproc = WNDPROC(self._window_proc)
        class_name = f"right_code_ptt_{id(self)}"
        window_class = WNDCLASSW()
        window_class.lpfnWndProc = self._wndproc
        window_class.hInstance = kernel32.GetModuleHandleW(None)
        window_class.lpszClassName = class_name
        if not user32.RegisterClassW(ctypes.byref(window_class)):
            raise ctypes.WinError()

        hwnd = user32.CreateWindowExW(
            0, class_name, None, 0, 0, 0, 0, 0, wintypes.HWND(HWND_MESSAGE), None, window_class.hInstance, None
        )
        if not hwnd:
            raise ctypes.WinError()

        device = RAWINPUTDEVICE(0x01, 0x06, RIDEV_INPUTSINK, hwnd)  # generic desktop / keyboard
        if not user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(RAWINPUTDEVICE)):
            user32.DestroyWindow(hwnd)
            raise ctypes.WinError()

        self._hwnd = hwnd
        self._ready.set()
        logger.info("Raw-input push-to-talk sink is listening")

        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))

        user32.UnregisterClassW(class_name, window_class.hInstance)
        self._hwnd = None

    def _window_proc(self, hwnd, msg, wparam, lparam):
        user32 = ctypes.windll.user32
        if msg == WM_INPUT:
            try:
                self._read_input(lparam)
            except Exception:
                logger.exception("Failed to read raw input")
            return 0
        if msg == WM_CLOSE:
            user32.DestroyWindow(hwnd)
            return 0
        if msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _read_input(self, lparam) -> None:
        user32 = ctypes.windll.user32
        size = wintypes.UINT(0)
        header_size = ctypes.sizeof(RAWINPUTHEADER)
        user32.GetRawInputData(ctypes.c_void_p(lparam), RID_INPUT, None, ctypes.byref(size), header_size)
        if not size.value:
            return
        buffer = ctypes.create_string_buffer(size.value)
        read = user32.GetRawInputData(ctypes.c_void_p(lparam), RID_INPUT, buffer, ctypes.byref(size), header_size)
        if read != size.value:
            return
        raw = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
        if raw.header.dwType == RIM_TYPEKEYBOARD:
            keyboard = raw.keyboard
            self.decoder.handle(keyboard.VKey, keyboard.Flags, keyboard.MakeCode)
