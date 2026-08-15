"""Native Windows backends: SendInput, Win32 clipboard, window focus, DPI.

Import these modules only on Windows (`pointer.py`, `clipboard.py`, and
`focus.py` use `ctypes.wintypes`, which does not import elsewhere) — the
factory in `..` does that check for you.
"""
