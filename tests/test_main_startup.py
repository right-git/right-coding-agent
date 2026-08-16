import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch


class MainStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_can_start_and_await_async_input(self):
        from src import main as main_module

        ui = Mock()
        ui.model = "openai/gpt-5.1-codex-mini"
        ui.get_input = AsyncMock(return_value="/quit")
        ui.handle_command = Mock(side_effect=SystemExit(0))

        catalog = Mock()
        catalog.models = AsyncMock(return_value={})

        with (
            patch("src.main.ChatUI", return_value=ui),
            patch("src.main.OpenRouterCatalog", return_value=catalog),
            patch("src.main.preload_vision_model"),
            # Иначе main() навесит реальный оверлей-слушатель на screen-тулзы,
            # и следующие тесты стартовали бы настоящее Tk-окно.
            patch("src.llm.tools.computer.set_activity_listener"),
            patch("src.llm.agents.Agents") as agents_cls,
        ):
            agents_cls.return_value = Mock()

            with self.assertRaises(SystemExit):
                await main_module.main()

        ui.print_welcome.assert_called_once()
        ui.get_input.assert_awaited_once()

    async def test_vision_preload_runs_on_a_daemon_thread(self):
        # A non-daemon loader thread (asyncio.to_thread uses the default
        # executor) blocks interpreter exit for as long as a first-run model
        # download takes — /quit must never wait for it.
        from src import main as main_module

        ui = Mock()
        ui.model = "openai/gpt-5.1-codex-mini"
        ui.get_input = AsyncMock(return_value="/quit")
        ui.handle_command = Mock(side_effect=SystemExit(0))

        catalog = Mock()
        catalog.models = AsyncMock(return_value={})

        seen = {}
        ran = threading.Event()

        def record_thread(_ui):
            seen["daemon"] = threading.current_thread().daemon
            ran.set()

        with (
            patch("src.main.ChatUI", return_value=ui),
            patch("src.main.OpenRouterCatalog", return_value=catalog),
            patch("src.main.preload_vision_model", side_effect=record_thread),
            patch("src.llm.tools.computer.set_activity_listener"),
            patch("src.llm.agents.Agents", return_value=Mock()),
        ):
            with self.assertRaises(SystemExit):
                await main_module.main()

        self.assertTrue(ran.wait(2), "vision preload never ran")
        self.assertTrue(seen["daemon"], "vision preload must run on a daemon thread")

    async def test_missing_input_permission_is_explained_to_the_user(self):
        from src import main as main_module

        ui = Mock()
        ui.model = "openai/gpt-5.1-codex-mini"
        ui.get_input = AsyncMock(return_value="/quit")
        ui.handle_command = Mock(side_effect=SystemExit(0))
        ui.start_voice_input = Mock(side_effect=PermissionError("needs the macOS Accessibility permission"))

        catalog = Mock()
        catalog.models = AsyncMock(return_value={})

        with (
            patch("src.main.ChatUI", return_value=ui),
            patch("src.main.OpenRouterCatalog", return_value=catalog),
            patch("src.main.preload_vision_model"),
            patch("src.llm.tools.computer.set_activity_listener"),
            patch("src.llm.agents.Agents", return_value=Mock()),
            patch.object(main_module.settings, "enable_voice_model", True),
        ):
            with self.assertRaises(SystemExit):
                await main_module.main()

        ui.print_warning.assert_any_call("needs the macOS Accessibility permission")

    async def test_voice_startup_is_skipped_while_disabled(self):
        # ENABLE_VOICE_MODEL off (the default) must not touch the voice layer
        # at all — no hotkey listener, no whisper warm-up.
        from src import main as main_module

        ui = Mock()
        ui.model = "openai/gpt-5.1-codex-mini"
        ui.get_input = AsyncMock(return_value="/quit")
        ui.handle_command = Mock(side_effect=SystemExit(0))

        catalog = Mock()
        catalog.models = AsyncMock(return_value={})

        with (
            patch("src.main.ChatUI", return_value=ui),
            patch("src.main.OpenRouterCatalog", return_value=catalog),
            patch("src.main.preload_vision_model"),
            patch("src.llm.tools.computer.set_activity_listener"),
            patch("src.llm.agents.Agents", return_value=Mock()),
            patch.object(main_module.settings, "enable_voice_model", False),
        ):
            with self.assertRaises(SystemExit):
                await main_module.main()

        ui.start_voice_input.assert_not_called()
