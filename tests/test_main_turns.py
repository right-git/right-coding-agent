import unittest
from contextlib import nullcontext
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.llm.providers.openrouter import ModelInfo
from src.llm.statistics import SessionUsage
from src.main import EMPTY_RESPONSE_NUDGE, process_user_turn
from src.llm.utils import is_empty_final_response


class ProcessUserTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_turn_logs_and_surfaces_error(self):
        agents = Mock()
        agents.right_coding_agent = AsyncMock(side_effect=RuntimeError("boom"))

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
        agents.right_coding_agent = AsyncMock(
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
        agents.right_coding_agent = AsyncMock(return_value={"messages": response_messages})

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


class EmptyFinalResponseTests(unittest.TestCase):
    def test_empty_or_whitespace_text_is_empty(self):
        self.assertTrue(is_empty_final_response([AIMessage(content="")]))
        self.assertTrue(is_empty_final_response([AIMessage(content="  \n")]))

    def test_blocks_without_text_are_empty(self):
        message = AIMessage(content=[{"type": "text", "text": ""}, {"type": "reasoning", "summary": []}])
        self.assertTrue(is_empty_final_response([message]))

    def test_text_or_reasoning_summaries_are_not_empty(self):
        self.assertFalse(is_empty_final_response([AIMessage(content="done")]))
        self.assertFalse(
            is_empty_final_response([AIMessage(content=[{"type": "reasoning", "summary": [{"text": "thought"}]}])])
        )

    def test_tool_calls_and_non_ai_tails_are_not_empty(self):
        with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "run_tools", "args": {}, "id": "c1", "type": "tool_call"}],
        )
        self.assertFalse(is_empty_final_response([with_tools]))
        self.assertFalse(is_empty_final_response([HumanMessage("hi")]))
        self.assertFalse(is_empty_final_response([]))


class EmptyResponseRetryTests(unittest.IsolatedAsyncioTestCase):
    def make_ui(self):
        ui = Mock()
        ui.loading.return_value = nullcontext()
        ui.has_visible_output.return_value = True
        return ui

    async def test_empty_final_response_is_retried_with_a_nudge(self):
        empty_response = {"messages": [HumanMessage("test"), AIMessage(content="", id="empty")]}
        good_response = {
            "messages": [
                HumanMessage("test"),
                HumanMessage(EMPTY_RESPONSE_NUDGE),
                AIMessage(content="done", id="final"),
            ]
        }
        agents = Mock()
        agents.right_coding_agent = AsyncMock(side_effect=[empty_response, good_response])
        ui = self.make_ui()

        with patch("src.main.logger"):
            updated = await process_user_turn(
                agents=agents,
                ui=ui,
                messages=[],
                model="google/gemini-3.7-flash",
                user_content="test",
            )

        self.assertEqual(agents.right_coding_agent.await_count, 2)
        retry_messages = agents.right_coding_agent.await_args_list[1].kwargs["messages"]
        self.assertEqual(retry_messages[-1], HumanMessage(EMPTY_RESPONSE_NUDGE))
        self.assertNotIn("empty", [getattr(m, "id", None) for m in retry_messages])
        self.assertEqual(updated[-1], AIMessage(content="done", id="final"))
        ui.print_warning.assert_not_called()

    async def test_two_empty_responses_surface_a_warning(self):
        empty_response = {"messages": [HumanMessage("test"), AIMessage(content="")]}
        agents = Mock()
        agents.right_coding_agent = AsyncMock(side_effect=[empty_response, empty_response])
        ui = self.make_ui()

        with patch("src.main.logger"):
            await process_user_turn(
                agents=agents,
                ui=ui,
                messages=[],
                model="google/gemini-3.7-flash",
                user_content="test",
            )

        self.assertEqual(agents.right_coding_agent.await_count, 2)
        ui.print_warning.assert_called_once()


class UsageReportingTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_reports_usage_with_openrouter_pricing(self):
        history = [
            AIMessage(
                content="old",
                id="prev",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                },
            )
        ]
        response_messages = [
            *history,
            HumanMessage("test"),
            AIMessage(
                content="step",
                id="new_1",
                tool_calls=[
                    {
                        "name": "run_tools",
                        "args": {"code": "return 1"},
                        "id": "tc1",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
            ),
            ToolMessage(
                content='{"result": 1, "tool_calls": 4}',
                tool_call_id="tc1",
                name="run_tools",
            ),
            AIMessage(
                content="done",
                id="new_2",
                usage_metadata={
                    "input_tokens": 130,
                    "output_tokens": 30,
                    "total_tokens": 160,
                },
            ),
        ]
        agents = Mock()
        agents.right_coding_agent = AsyncMock(return_value={"messages": response_messages})

        ui = Mock()
        ui.loading.return_value = nullcontext()
        ui.has_visible_output.return_value = True

        info = ModelInfo(
            id="google/gemini-3.7-flash",
            name="Gemini",
            context_length=1_048_576,
            prompt_price=7.5e-8,
            completion_price=3e-7,
        )
        catalog = Mock()
        catalog.get = AsyncMock(return_value=info)
        session = SessionUsage()

        with patch("src.main.logger"):
            await process_user_turn(
                agents=agents,
                ui=ui,
                messages=history,
                model="google/gemini-3.7-flash",
                user_content="test",
                catalog=catalog,
                session_usage=session,
            )

        ui.print_usage.assert_called_once()
        turn, model_info, cost, session_arg, duration = ui.print_usage.call_args.args
        self.assertGreaterEqual(duration, 0)
        self.assertAlmostEqual(session.duration, duration)
        self.assertEqual(turn.input_tokens, 230)
        self.assertEqual(turn.output_tokens, 50)
        self.assertEqual(turn.context_tokens, 160)
        self.assertEqual(turn.calls, 2)
        self.assertEqual(turn.tool_calls, 1)
        self.assertEqual(turn.script_tool_calls, 4)
        self.assertIs(model_info, info)
        self.assertAlmostEqual(cost, 230 * 7.5e-8 + 50 * 3e-7)
        self.assertIs(session_arg, session)
        self.assertEqual(session.total_tokens, 280)

    async def test_usage_reporting_is_skipped_without_a_catalog(self):
        agents = Mock()
        agents.right_coding_agent = AsyncMock(return_value={"messages": [HumanMessage("test"), AIMessage("done")]})

        ui = Mock()
        ui.loading.return_value = nullcontext()
        ui.has_visible_output.return_value = True

        with patch("src.main.logger"):
            await process_user_turn(
                agents=agents,
                ui=ui,
                messages=[],
                model="google/gemini-3.7-flash",
                user_content="test",
            )

        ui.print_usage.assert_not_called()
