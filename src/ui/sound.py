"""Completion sound for finished turns.

Playback must be instant and non-blocking. On Windows that rules out
spawning PowerShell with WPF's MediaPlayer — its cold start delays the
sound by seconds, which reads as "the toggle is broken" — so the clip is
played in-process through the winmm MCI API (mp3-capable, returns
immediately, plays in the background). macOS uses `afplay`, Linux the
first of mpg123/ffplay/cvlc, both as fire-and-forget processes. Every OS
call sits behind a seam (`sender`, `spawner`, `which`), so tests never
make noise, and failures only log — a broken audio setup must not touch
the chat loop.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from src.config.logging import logger

DONE_SOUND = Path(__file__).resolve().parents[2] / "assets" / "sounds" / "done.mp3"
MCI_ALIAS = "right_code_done_sound"

LINUX_PLAYERS = (
    ["mpg123", "-q"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["cvlc", "--play-and-exit", "--intf", "dummy"],
)


def build_command(path: Path, *, platform: str = sys.platform, which=shutil.which) -> list[str] | None:
    """The player invocation for macOS/Linux, or None when no player exists."""
    if platform == "darwin":
        return ["afplay", str(path)]
    for player in LINUX_PLAYERS:
        if which(player[0]):
            return [*player, str(path)]
    return None


def _default_mci_sender():
    import ctypes

    return ctypes.windll.winmm.mciSendStringW


def _play_windows(path: Path, *, sender=None) -> bool:
    """Play in-process via winmm MCI: instant, mp3-capable, non-blocking."""
    try:
        send = sender or _default_mci_sender()
    except Exception:
        logger.exception("winmm is unavailable")
        return False

    send(f"close {MCI_ALIAS}", None, 0, None)  # drop the previous clip, if any
    if send(f'open "{path}" type mpegvideo alias {MCI_ALIAS}', None, 0, None) != 0:
        logger.warning("MCI could not open the done sound [{}]", path)
        return False
    if send(f"play {MCI_ALIAS} from 0", None, 0, None) != 0:
        logger.warning("MCI could not play the done sound [{}]", path)
        return False
    return True


def play_done_sound(
    path: Path = DONE_SOUND,
    *,
    platform: str = sys.platform,
    which=shutil.which,
    spawner=subprocess.Popen,
    sender=None,
) -> bool:
    """Start playback in the background; False when it could not start."""
    path = Path(path)
    if not path.is_file():
        logger.warning("Done sound file missing: [{}]", path)
        return False

    if platform == "win32":
        return _play_windows(path, sender=sender)

    command = build_command(path, platform=platform, which=which)
    if command is None:
        logger.warning("No audio player found for the done sound")
        return False
    try:
        spawner(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        logger.exception("Playing the done sound failed")
        return False
