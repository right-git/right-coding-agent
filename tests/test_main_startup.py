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
