"""Completion sound for finished turns.

Playback is fire-and-forget through an OS player process, so the chat loop
never blocks and a broken audio setup costs nothing but a log line:
`afplay` on macOS, the first of mpg123/ffplay/cvlc on Linux, and a hidden
PowerShell `MediaPlayer` on Windows (mp3-capable, unlike `winsound`; the
trailing sleep keeps the process alive while the clip plays). Every OS call
sits behind a seam (`spawner`, `which`), so tests never make noise.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from src.config.logging import logger

DONE_SOUND = Path(__file__).resolve().parents[2] / "assets" / "sounds" / "done.mp3"
WINDOWS_PLAY_SECONDS = 3  # keeps the hidden player process alive for the clip

LINUX_PLAYERS = (
    ["mpg123", "-q"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["cvlc", "--play-and-exit", "--intf", "dummy"],
)


def build_command(path: Path, *, platform: str = sys.platform, which=shutil.which) -> list[str] | None:
    """The player invocation for this OS, or None when no player exists."""
    if platform == "win32":
        script = (
            "Add-Type -AssemblyName PresentationCore; "
            "$player = New-Object System.Windows.Media.MediaPlayer; "
            f"$player.Open([Uri]'{path}'); $player.Play(); "
            f"Start-Sleep -Seconds {WINDOWS_PLAY_SECONDS}"
        )
        return ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script]
    if platform == "darwin":
        return ["afplay", str(path)]
    for player in LINUX_PLAYERS:
        if which(player[0]):
            return [*player, str(path)]
    return None


def play_done_sound(
    path: Path = DONE_SOUND,
    *,
    platform: str = sys.platform,
    which=shutil.which,
    spawner=subprocess.Popen,
) -> bool:
    """Start playback in the background; False when it could not start."""
    path = Path(path)
    if not path.is_file():
        logger.warning("Done sound file missing: [{}]", path)
        return False
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
