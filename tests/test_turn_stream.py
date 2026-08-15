import sys
import unittest
from io import StringIO
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.client import LLMClient
from src.ui import ChatUI
from src.ui.chat import theme
from src.ui.stream import TurnStream


def make_ui():
    ui = ChatUI(model="google/gemini-3.7-flash")
    ui.console = Console(file=StringIO(), record=True, force_terminal=False, width=200, theme=theme)
    return ui


def tool_call_message(identifier="ai-1"):
    return AIMessage(
        content="",
        id=identifier,
        tool_calls=[{"name": "search_tools", "args": {"query": "web"}, "id": "c1", "type": "tool_call"}],
    )


class TurnStreamTests(unittest.TestCase):
    def test_tool_calls_and_results_print_live_and_are_recorded(self):
        ui = make_ui()
        stream = TurnStream(ui)

        stream.on_message(tool_call_message())
        stream.on_message(ToolMessage(content="found 3 tools", tool_call_id="c1", id="tool-1"))

        rendered = ui.console.export_text()
        self.assertIn("Search tools", rendered)
        self.assertIn("found 3 tools", rendered)
        self.assertIn("thought for", rendered)
        self.assertEqual(stream.printed_ids, {"ai-1", "tool-1"})

    def test_duplicate_messages_print_once(self):
        ui = make_ui()
        stream = TurnStream(ui)

        stream.on_message(tool_call_message())
        stream.on_message(tool_call_message())

        self.assertEqual(ui.console.export_text().count("Search tools"), 1)

    def test_final_answer_is_not_recorded_as_printed(self):
        ui = make_ui()
        stream = TurnStream(ui)

        stream.on_message(AIMessage(content="the answer", id="ai-2"))

        rendered = ui.console.export_text()
        self.assertIn("thought for", rendered)
        self.assertNotIn("the answer", rendered)
        self.assertEqual(stream.printed_ids, set())

    def test_tool_result_shows_duration_only_after_a_tool_round(self):
        ui = make_ui()
        stream = TurnStream(ui)

        stream.on_message(tool_call_message())
        self.assertEqual(stream.ticker.label, "running tools")
        stream.on_message(ToolMessage(content="ok", tool_call_id="c1", id="tool-2"))
        self.assertEqual(stream.ticker.label, "thinking")

    def test_streamed_text_feeds_the_live_tail(self):
        stream = TurnStream(make_ui())

        stream.on_token("hello ")
        stream.on_token("world")

        self.assertEqual(stream._text, "hello world")

    def test_human_messages_are_ignored(self):
        ui = make_ui()
        stream = TurnStream(ui)

        stream.on_message(HumanMessage(content="attached images", id="h1"))

        self.assertEqual(stream.printed_ids, set())
        self.assertNotIn("attached", ui.console.export_text())


class PrintResponseSkipTests(unittest.TestCase):
    def test_messages_already_streamed_are_skipped(self):
        ui = make_ui()
        messages = [
            tool_call_message("ai-1"),
            ToolMessage(content="tool output", tool_call_id="c1", id="tool-1"),
            AIMessage(content="final answer", id="ai-2"),
        ]

        ui.print_response(messages, skip_ids={"ai-1", "tool-1"})

        rendered = ui.console.export_text()
        self.assertNotIn("Search tools", rendered)
        self.assertNotIn("tool output", rendered)
        self.assertIn("final answer", rendered)


class FakeStreamingAgent:
    """Yields a scripted (mode, chunk) sequence like langgraph's astream."""

    def __init__(self, events):
        self._events = events

    async def astream(self, *, input, context, config, stream_mode):
        self.requested_modes = stream_mode
        for event in self._events:
            yield event


class StreamAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_messages_and_returns_final_state(self):
        ai = tool_call_message("ai-1")
        tool = ToolMessage(content="ok", tool_call_id="c1", id="tool-1")
        final_state = {"messages": [HumanMessage("hi"), ai, tool, AIMessage("done", id="ai-2")]}
        agent = FakeStreamingAgent(
            [
                ("updates", {"model": {"messages": [ai]}}),
                ("values", {"messages": [ai]}),
                ("updates", {"tools": {"messages": [tool]}}),
                ("updates", {"SummarizationMiddleware.before_model": None}),
                ("values", final_state),
            ]
        )
        client = LLMClient(providers=[])
        received = []

        response = await client._stream_agent(agent, {"messages": []}, None, None, received.append, None)

        self.assertEqual(response, final_state)
        self.assertEqual(received, [ai, tool])
        self.assertEqual(agent.requested_modes, ["updates", "values"])

    async def test_token_mode_is_requested_only_with_a_token_callback(self):
        agent = FakeStreamingAgent([("values", {"messages": []})])
        client = LLMClient(providers=[])

        await client._stream_agent(agent, {"messages": []}, None, None, None, lambda piece: None)

        self.assertEqual(agent.requested_modes, ["updates", "values", "messages"])

    async def test_tokens_come_only_from_the_model_node_without_warnings(self):
        import warnings

        from langchain_core.messages import AIMessageChunk

        agent = FakeStreamingAgent(
            [
                ("messages", (AIMessageChunk(content="Hel"), {"langgraph_node": "model"})),
                ("messages", (AIMessageChunk(content="lo"), {"langgraph_node": "model"})),
                (
                    "messages",
                    (AIMessageChunk(content="summary"), {"langgraph_node": "SummarizationMiddleware.before_model"}),
                ),
                ("values", {"messages": []}),
            ]
        )
        client = LLMClient(providers=[])
        pieces = []

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await client._stream_agent(agent, {"messages": []}, None, None, None, pieces.append)

        self.assertEqual(pieces, ["Hel", "lo"])

    async def test_missing_final_state_raises(self):
        agent = FakeStreamingAgent([("updates", {"model": {"messages": []}})])
        client = LLMClient(providers=[])

        with self.assertRaisesRegex(RuntimeError, "final state"):
            await client._stream_agent(agent, {"messages": []}, None, None, None, None)

    async def test_callback_errors_do_not_kill_the_stream(self):
        ai = tool_call_message("ai-1")
        agent = FakeStreamingAgent(
            [
                ("updates", {"model": {"messages": [ai]}}),
                ("values", {"messages": [ai]}),
            ]
        )
        client = LLMClient(providers=[])

        def broken(message):
            raise RuntimeError("boom")

        response = await client._stream_agent(agent, {"messages": []}, None, None, broken, None)

        self.assertEqual(response, {"messages": [ai]})


if __name__ == "__main__":
    unittest.main()
