import unittest
from io import StringIO
from unittest.mock import AsyncMock, Mock, patch

from rich.console import Console

from src.ui import ChatUI
from src.ui.chat import theme


class ChatUIInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_input_uses_prompt_async(self):
        ui = ChatUI(model="openai/gpt-5.1-codex-mini")
        ui.prompt_session = Mock()
        ui.prompt_session.prompt_async = AsyncMock(return_value="hello")

        value = await ui.get_input()

        self.assertEqual(value, "hello")
        ui.prompt_session.prompt_async.assert_awaited_once_with("> ")

    async def test_get_input_returns_quit_on_interrupt(self):
        ui = ChatUI(model="openai/gpt-5.1-codex-mini")
        ui.console = Console(
            file=StringIO(),
            record=True,
            force_terminal=False,
            width=120,
            theme=theme,
        )
        ui.prompt_session = Mock()
        ui.prompt_session.prompt_async = AsyncMock(side_effect=KeyboardInterrupt())

        value = await ui.get_input()

        self.assertEqual(value, "/quit")

    async def test_get_tool_approval_uses_prompt_async(self):
        ui = ChatUI(model="openai/gpt-5.1-codex-mini")
        ui.console = Console(
            file=StringIO(),
            record=True,
            force_terminal=False,
            width=120,
            theme=theme,
        )
        ui.prompt_session = Mock()
        ui.prompt_session.prompt_async = AsyncMock(return_value="y")

        result = await ui.get_tool_approval([{"name": "execute", "args": {"command": "rm file.txt"}}])

        self.assertEqual(result, {"decisions": [{"type": "approve"}]})
        ui.prompt_session.prompt_async.assert_awaited_once_with("  > ")

    async def test_module_uses_prompt_toolkit_session(self):
        with patch("src.ui.chat.PromptSession") as prompt_session_cls:
            prompt_session_cls.return_value = Mock()
            ui = ChatUI(model="openai/gpt-5.1-codex-mini")
            session = ui._get_prompt_session()

        self.assertIs(session, prompt_session_cls.return_value)
