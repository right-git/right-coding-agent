import sys
import unittest
from io import StringIO
from pathlib import Path

from langchain_core.messages import AIMessageChunk
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.client import LLMClient
from src.llm.providers.reasoning import ReasoningChatOpenAI, reasoning_delta
from src.ui import ChatUI
from src.ui.chat import theme
from src.ui.stream import TurnStream


def make_ui():
    ui = ChatUI(model="moonshotai/kimi-k2.7-code")
    ui.console = Console(file=StringIO(), record=True, force_terminal=False, width=200, theme=theme)
    return ui


def raw_chunk(delta):
    return {"id": "c", "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


class ReasoningDeltaTests(unittest.TestCase):
    def test_reads_openrouter_reasoning_field(self):
        self.assertEqual(reasoning_delta(raw_chunk({"content": "", "reasoning": "We need"})), "We need")

    def test_reads_the_deepseek_spelling(self):
        self.assertEqual(reasoning_delta(raw_chunk({"reasoning_content": "hmm"})), "hmm")

    def test_falls_back_to_reasoning_details(self):
        delta = {"reasoning_details": [{"type": "reasoning.text", "text": "step "}, {"text": "two"}]}
        self.assertEqual(reasoning_delta(raw_chunk(delta)), "step two")

    def test_absent_reasoning_is_empty(self):
        self.assertEqual(reasoning_delta(raw_chunk({"content": "hi"})), "")
        self.assertEqual(reasoning_delta({"choices": []}), "")
        self.assertEqual(reasoning_delta({}), "")


class ReasoningChatOpenAITests(unittest.TestCase):
    def setUp(self):
        self.model = ReasoningChatOpenAI(model="m", api_key="test-key", base_url="https://example.invalid/v1")

    def convert(self, delta):
        return self.model._convert_chunk_to_generation_chunk(raw_chunk(delta), AIMessageChunk, None)

    def test_reasoning_is_kept_in_additional_kwargs(self):
        chunk = self.convert({"content": "", "reasoning": "We need"})

        self.assertEqual(chunk.message.additional_kwargs["reasoning"], "We need")
        self.assertEqual(str(chunk.message.text), "")

    def test_answer_text_is_untouched_and_carries_no_reasoning_key(self):
        chunk = self.convert({"content": "42"})

        self.assertEqual(str(chunk.message.text), "42")
        self.assertNotIn("reasoning", chunk.message.additional_kwargs)


class FakeStreamingAgent:
    def __init__(self, events):
        self._events = events

    async def astream(self, *, input, context, config, stream_mode):
        self.requested_modes = stream_mode
        for event in self._events:
            yield event


class StreamReasoningTests(unittest.IsolatedAsyncioTestCase):
    def make_agent(self):
        return FakeStreamingAgent(
            [
                (
                    "messages",
                    (
                        AIMessageChunk(content="", additional_kwargs={"reasoning": "We need "}),
                        {"langgraph_node": "model"},
                    ),
                ),
                ("messages", (AIMessageChunk(content="42"), {"langgraph_node": "model"})),
                ("values", {"messages": []}),
            ]
        )

    async def test_reasoning_goes_to_its_own_callback_never_to_on_token(self):
        agent = self.make_agent()
        client = LLMClient(providers=[])
        tokens, thoughts = [], []

        await client._stream_agent(agent, {"messages": []}, None, None, None, tokens.append, thoughts.append)

        self.assertEqual(tokens, ["42"])
        self.assertEqual(thoughts, ["We need "])

    async def test_token_mode_is_requested_for_a_reasoning_callback_alone(self):
        agent = FakeStreamingAgent([("values", {"messages": []})])
        client = LLMClient(providers=[])

        await client._stream_agent(agent, {"messages": []}, None, None, None, None, lambda piece: None)

        self.assertEqual(agent.requested_modes, ["updates", "values", "messages"])


class TurnStreamReasoningTests(unittest.TestCase):
    def test_reasoning_is_shown_while_thinking(self):
        ui = make_ui()
        stream = TurnStream(ui)

        stream.on_reasoning("We need to multiply 17 by 23")
        ui.console.print(stream)

        rendered = ui.console.export_text()
        self.assertIn("We need to multiply 17 by 23", rendered)
        self.assertIn("thinking", rendered)

    def test_reasoning_does_not_end_the_thinking_phase(self):
        ui = make_ui()
        stream = TurnStream(ui)

        stream.on_reasoning("still weighing options")

        self.assertEqual(stream.ticker.label, "thinking")
        self.assertNotIn("thought for", ui.console.export_text())

    def test_the_answer_replaces_the_reasoning(self):
        ui = make_ui()
        stream = TurnStream(ui)

        stream.on_reasoning("scratch work")
        stream.on_token("The answer is 391")
        ui.console.print(stream)

        rendered = ui.console.export_text()
        self.assertIn("The answer is 391", rendered)
        self.assertNotIn("scratch work", rendered)
        self.assertEqual(stream.ticker.label, "responding")

    def test_a_tool_round_clears_the_previous_reasoning(self):
        ui = make_ui()
        stream = TurnStream(ui)
        from langchain_core.messages import AIMessage

        stream.on_reasoning("I should search")
        stream.on_message(
            AIMessage(
                content="",
                id="ai-1",
                tool_calls=[{"name": "run_tools", "args": {"code": "x"}, "id": "c1", "type": "tool_call"}],
            )
        )
        ui.console.print(stream)

        self.assertNotIn("I should search", ui.console.export_text())

    def test_reasoning_failures_never_break_the_turn(self):
        stream = TurnStream(make_ui())

        stream.on_reasoning(None)

        self.assertEqual(stream.ticker.label, "thinking")


if __name__ == "__main__":
    unittest.main()
