"""Cross-platform backends for macOS and Linux.

Built on pynput (mouse/keyboard), mss (screen capture), and pyperclip
(clipboard). Third-party imports happen inside the classes, so this package
imports cleanly even where those libraries cannot initialize (e.g. Linux
without a display); construction is where a missing backend fails, with the
library's own error.

OS prerequisites: on macOS grant the process Accessibility (input) and
Screen Recording (capture) permissions; on Linux X11 is expected (on
Wayland, capture works via mss/pipewire portals, input support varies).
"""
