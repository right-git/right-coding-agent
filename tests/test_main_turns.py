import unittest
from contextlib import nullcontext
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.main import process_user_turn


class ProcessUserTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_turn_logs_and_surfaces_error(self):
        agents = Mock()
        agents.right_code = AsyncMock(side_effect=RuntimeError("boom"))

        ui = Mock()
        ui.loading.return_value = nullcontext()

        with patch("src.main.logger") as logger:
            updated_messages = await process_user_turn(
                agents=agents,
                ui=ui,
                messages=[],
                model="openai/gpt-5.1-codex-mini",
                user_content="test",
            )

        self.assertEqual(updated_messages, [])
        ui.print_error.assert_called_once()
        ui.print_response.assert_not_called()
        ui.print_warning.assert_not_called()
        logger.exception.assert_called()

    async def test_silent_turn_logs_and_surfaces_warning(self):
        agents = Mock()
        agents.right_code = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage("test"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "execute",
                                "args": {"command": "echo hi"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            }
        )

        ui = Mock()
        ui.loading.return_value = nullcontext()
        ui.has_visible_output.return_value = False

        with patch("src.main.logger") as logger:
            updated_messages = await process_user_turn(
                agents=agents,
                ui=ui,
                messages=[],
                model="openai/gpt-5.1-codex-mini",
                user_content="test",
            )

        self.assertEqual(updated_messages, [HumanMessage("test")])
        ui.print_warning.assert_called_once()
        ui.print_response.assert_not_called()
        logger.warning.assert_called()

    async def test_visible_turn_prints_response(self):
        agents = Mock()
        response_messages = [
            HumanMessage("test"),
            AIMessage(content="done"),
        ]
        agents.right_code = AsyncMock(return_value={"messages": response_messages})

        ui = Mock()
        ui.loading.return_value = nullcontext()
        ui.has_visible_output.return_value = True

        with patch("src.main.logger"):
            updated_messages = await process_user_turn(
                agents=agents,
                ui=ui,
                messages=[],
                model="openai/gpt-5.1-codex-mini",
                user_content="test",
            )

        self.assertEqual(updated_messages, response_messages)
        ui.print_response.assert_called_once_with([AIMessage(content="done")])
        ui.print_warning.assert_not_called()
        ui.print_error.assert_not_called()
