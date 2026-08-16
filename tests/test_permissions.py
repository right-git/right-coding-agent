import unittest
from io import StringIO
from unittest.mock import patch

from rich.console import Console

from src.ui import ChatUI
from src.ui.chat import theme
from src.utils import permissions
from src.utils.permissions import PermissionStatus, check_permissions


def make_ui():
    ui = ChatUI(model="m")
    ui.console = Console(file=StringIO(), record=True, force_terminal=False, width=200, theme=theme)
    return ui


class CheckPermissionsTests(unittest.TestCase):
    def test_nothing_to_check_outside_macos(self):
        self.assertEqual(check_permissions(platform="win32"), [])
        self.assertEqual(check_permissions(platform="linux"), [])

    def test_reports_every_macos_permission_with_its_state(self):
        with (
            patch.object(permissions, "_accessibility", return_value=True) as accessibility,
            patch.object(permissions, "_input_monitoring", return_value=False),
            patch.object(permissions, "_screen_recording", return_value=None),
            patch.object(permissions, "_microphone", return_value=None),
        ):
            statuses = check_permissions(platform="darwin", trigger=False)

        self.assertEqual(
            [(status.name, status.granted) for status in statuses],
            [
                ("Accessibility", True),
                ("Input Monitoring", False),
                ("Screen Recording", None),
                ("Microphone", None),
            ],
        )
        accessibility.assert_called_once_with(False)
        for status in statuses:
            self.assertTrue(status.settings_pane)
            self.assertTrue(status.purpose)

    def test_probers_never_raise_when_the_frameworks_are_missing(self):
        # Guarded probers: an import error means "unknown", not a crash.
        with patch.dict("sys.modules", {"ApplicationServices": None, "Quartz": None, "sounddevice": None}):
            self.assertIsNone(permissions._accessibility(False))
            self.assertIsNone(permissions._input_monitoring(False))
            self.assertIsNone(permissions._screen_recording(False))
            self.assertIsNone(permissions._microphone(False))


class CheckCommandTests(unittest.TestCase):
    def test_check_command_lists_permissions_and_hints(self):
        ui = make_ui()
        statuses = [
            PermissionStatus("Accessibility", True, "Privacy & Security → Accessibility", "push-to-talk hotkey"),
            PermissionStatus("Screen Recording", False, "Privacy & Security → Screen Recording", "screenshots"),
            PermissionStatus("Microphone", None, "Privacy & Security → Microphone", "voice input"),
        ]
        with patch.object(permissions, "check_permissions", return_value=statuses) as check:
            ui.handle_command("/check")

        check.assert_called_once()
        rendered = ui.console.export_text()
        self.assertIn("✓ Accessibility", rendered)
        self.assertIn("✗ Screen Recording", rendered)
        self.assertIn("? Microphone", rendered)
        self.assertIn("Privacy & Security → Screen Recording", rendered)
        self.assertNotIn("Privacy & Security → Accessibility", rendered)  # granted needs no hint
        self.assertIn("restart", rendered)

    def test_check_command_outside_macos_says_nothing_is_needed(self):
        ui = make_ui()
        ui.commands._run_check(platform="win32")

        self.assertIn("no OS permission setup needed", ui.console.export_text())

    def test_check_appears_in_help(self):
        ui = make_ui()
        ui.handle_command("/help")

        self.assertIn("/check", ui.console.export_text())


if __name__ == "__main__":
    unittest.main()
