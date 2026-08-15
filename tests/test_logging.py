import unittest
from io import StringIO
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from rich.console import Console

from src.ui import ChatUI
from src.ui.chat import theme


class LoggingManagerTests(unittest.TestCase):
    def test_logging_manager_accepts_runtime_level_changes(self):
        from src.config.logging import LoggingManager

        with (
            patch("src.config.logging.logger.remove") as remove_mock,
            patch(
                "src.config.logging.logger.add",
                side_effect=[1, 2, 3, 4],
            ) as add_mock,
        ):
            manager = LoggingManager(
                log_file="tmp/test.log",
                level="info",
                rotation="10 KB",
                compression="zip",
            )

            manager.configure()
            new_level = manager.set_level("debug")

        self.assertEqual(new_level, "DEBUG")
        self.assertEqual(manager.get_level(), "DEBUG")
        self.assertGreaterEqual(remove_mock.call_count, 2)
        self.assertGreaterEqual(add_mock.call_count, 2)

    def test_logging_manager_rejects_invalid_levels(self):
        from src.config.logging import LoggingManager

        manager = LoggingManager()

        with self.assertRaisesRegex(ValueError, "Invalid log level"):
            manager.set_level("loud")


class ChatUILoggingCommandTests(unittest.TestCase):
    def test_log_level_command_updates_logging_manager(self):
        with patch("src.ui.commands.app_logging") as app_logging:
            app_logging.set_level = Mock(return_value="DEBUG")
            ui = ChatUI(model="openai/gpt-5.1-codex-mini")
            ui.console = Console(
                file=StringIO(),
                record=True,
                force_terminal=False,
                width=120,
                theme=theme,
            )

            result = ui.handle_command("/log-level debug")

            self.assertIsNone(result)
            app_logging.set_level.assert_called_once_with("debug")
            self.assertIn("log level set to DEBUG", ui.console.export_text())

    def test_has_visible_output_detects_empty_turn(self):
        ui = ChatUI(model="openai/gpt-5.1-codex-mini")

        self.assertFalse(ui.has_visible_output([AIMessage(content="")]))
        self.assertTrue(ui.has_visible_output([AIMessage(content=[{"type": "text", "text": "done"}])]))
