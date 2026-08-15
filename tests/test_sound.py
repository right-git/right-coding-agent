import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ui import ChatUI
from src.ui.chat import theme
from src.ui.sound import DONE_SOUND, build_command, play_done_sound


def make_ui():
    ui = ChatUI(model="google/gemini-3.7-flash")
    ui.console = Console(file=StringIO(), record=True, force_terminal=False, width=200, theme=theme)
    return ui


class RecordingSpawner:
    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))


class RecordingSender:
    """Scripted stand-in for winmm's mciSendStringW."""

    def __init__(self, fail_on: str | None = None):
        self.commands: list[str] = []
        self.fail_on = fail_on

    def __call__(self, command, buffer, length, callback):
        self.commands.append(command)
        return 1 if self.fail_on and command.startswith(self.fail_on) else 0


class WindowsMciTests(unittest.TestCase):
    def test_playback_closes_opens_and_plays_in_process(self):
        sender = RecordingSender()

        started = play_done_sound(DONE_SOUND, platform="win32", sender=sender)

        self.assertTrue(started)
        self.assertEqual(len(sender.commands), 3)
        self.assertTrue(sender.commands[0].startswith("close"))
        self.assertIn(str(DONE_SOUND), sender.commands[1])
        self.assertIn("type mpegvideo", sender.commands[1])
        self.assertTrue(sender.commands[2].startswith("play"))
        self.assertIn("from 0", sender.commands[2])

    def test_open_failure_reports_false(self):
        sender = RecordingSender(fail_on="open")

        self.assertFalse(play_done_sound(DONE_SOUND, platform="win32", sender=sender))


class BuildCommandTests(unittest.TestCase):
    def test_macos_uses_afplay(self):
        command = build_command(Path("/x/done.mp3"), platform="darwin")

        self.assertEqual(command[0], "afplay")

    def test_linux_picks_the_first_available_player(self):
        command = build_command(Path("/x/done.mp3"), platform="linux", which=lambda name: name == "ffplay")

        self.assertEqual(command[0], "ffplay")

    def test_linux_without_players_yields_none(self):
        self.assertIsNone(build_command(Path("/x/done.mp3"), platform="linux", which=lambda name: None))


class PlayDoneSoundTests(unittest.TestCase):
    def test_bundled_sound_file_exists(self):
        self.assertTrue(DONE_SOUND.is_file(), f"missing asset: {DONE_SOUND}")

    def test_playback_spawns_the_player_in_the_background(self):
        spawner = RecordingSpawner()

        started = play_done_sound(DONE_SOUND, platform="darwin", spawner=spawner)

        self.assertTrue(started)
        self.assertEqual(spawner.commands, [["afplay", str(DONE_SOUND)]])

    def test_missing_file_plays_nothing(self):
        spawner = RecordingSpawner()

        started = play_done_sound(Path("nope/missing.mp3"), platform="darwin", spawner=spawner)

        self.assertFalse(started)
        self.assertEqual(spawner.commands, [])

    def test_spawn_failure_is_swallowed(self):
        def broken(command, **kwargs):
            raise OSError("no player")

        self.assertFalse(play_done_sound(DONE_SOUND, platform="darwin", spawner=broken))


class NotifyDoneTests(unittest.TestCase):
    def test_notify_plays_when_enabled(self):
        ui = make_ui()

        with patch("src.ui.chat.play_done_sound") as play:
            ui.notify_done()

        play.assert_called_once()

    def test_notify_is_silent_when_disabled(self):
        ui = make_ui()
        ui.sound_enabled = False

        with patch("src.ui.chat.play_done_sound") as play:
            ui.notify_done()

        play.assert_not_called()


class SoundCommandTests(unittest.TestCase):
    def test_bare_command_toggles(self):
        ui = make_ui()

        ui.handle_command("/sound")
        self.assertFalse(ui.sound_enabled)
        self.assertIn("completion sound off", ui.console.export_text())

        ui.handle_command("/sound")
        self.assertTrue(ui.sound_enabled)

    def test_explicit_on_off(self):
        ui = make_ui()

        ui.handle_command("/sound off")
        self.assertFalse(ui.sound_enabled)

        ui.handle_command("/sound on")
        self.assertTrue(ui.sound_enabled)

    def test_invalid_value_is_rejected(self):
        ui = make_ui()

        ui.handle_command("/sound loud")

        self.assertTrue(ui.sound_enabled)
        self.assertIn("invalid value", ui.console.export_text())


if __name__ == "__main__":
    unittest.main()
