"""macOS permission checks for `/check`: report state and raise every consent dialog.

The app needs four TCC permissions on macOS, and each one fails SILENTLY when
missing (the hotkey receives nothing, screenshots show only the wallpaper,
recordings come back empty), so `/check` probes them all at once and — with
`trigger=True` — makes the OS show every outstanding permission dialog in one
sitting. Each prober is guarded: a missing pyobjc framework means "unknown"
(None), never a crash. On other platforms there is nothing to check.

The microphone has no readable status without the AVFoundation bridge (not a
dependency), so its prober only opens the input stream for an instant — that
is what makes macOS raise the microphone dialog on first use — and always
reports "unknown".
"""

import sys
from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class PermissionStatus:
    name: str
    granted: bool | None  # None — cannot be determined on this machine
    settings_pane: str  # where to enable it inside System Settings
    purpose: str  # what the app uses it for


def _accessibility(trigger: bool) -> bool | None:
    try:
        import ApplicationServices

        if ApplicationServices.AXIsProcessTrusted():
            return True
        if trigger:
            try:
                options = {ApplicationServices.kAXTrustedCheckOptionPrompt: True}
                ApplicationServices.AXIsProcessTrustedWithOptions(options)
            except Exception:
                logger.debug("Could not raise the accessibility prompt")
        return False
    except Exception:
        return None


def _input_monitoring(trigger: bool) -> bool | None:
    try:
        import Quartz

        if Quartz.CGPreflightListenEventAccess():
            return True
        if trigger:
            try:
                Quartz.CGRequestListenEventAccess()
            except Exception:
                logger.debug("Could not raise the input-monitoring prompt")
        return False
    except Exception:
        return None


def _screen_recording(trigger: bool) -> bool | None:
    try:
        import Quartz

        if Quartz.CGPreflightScreenCaptureAccess():
            return True
        if trigger:
            try:
                Quartz.CGRequestScreenCaptureAccess()
            except Exception:
                logger.debug("Could not raise the screen-recording prompt")
        return False
    except Exception:
        return None


def _microphone(trigger: bool) -> bool | None:
    if not trigger:
        return None
    try:
        import sounddevice

        stream = sounddevice.InputStream(channels=1, samplerate=16_000)
        stream.start()
        stream.stop()
        stream.close()
    except Exception:
        logger.debug("Microphone probe failed (no device, or access denied)")
    return None


def check_permissions(platform: str | None = None, trigger: bool = True) -> list[PermissionStatus]:
    """Every macOS permission the app depends on, probed (and prompted) once."""
    if (platform or sys.platform) != "darwin":
        return []
    return [
        PermissionStatus(
            "Accessibility",
            _accessibility(trigger),
            "Privacy & Security → Accessibility",
            "push-to-talk hotkey",
        ),
        PermissionStatus(
            "Input Monitoring",
            _input_monitoring(trigger),
            "Privacy & Security → Input Monitoring",
            "push-to-talk hotkey",
        ),
        PermissionStatus(
            "Screen Recording",
            _screen_recording(trigger),
            "Privacy & Security → Screen Recording",
            "screenshots for the screen tools",
        ),
        PermissionStatus(
            "Microphone",
            _microphone(trigger),
            "Privacy & Security → Microphone",
            "voice input",
        ),
    ]
