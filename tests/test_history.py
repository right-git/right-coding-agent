import sys
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.history import (
    RECAP_CODE_CHARS,
    RECAP_MARKER,
    compact_finished_turn,
)
from src.llm.middlewares.attachments import ATTACHMENT_MARKER


def tool_call_ai(name, args, call_id, identifier=None):
    return AIMessage(
        content="",
        id=identifier,
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def run_tools_round(code, result, call_id):
    return [
        tool_call_ai("run_tools", {"code": code}, call_id),
        ToolMessage(content=result, tool_call_id=call_id, name="run_tools"),
    ]


class CompactFinishedTurnTests(unittest.TestCase):
    def test_turn_without_tools_is_unchanged(self):
        messages = [HumanMessage("hi"), AIMessage("hello")]

        self.assertEqual(compact_finished_turn(messages), messages)

    def test_turn_without_a_final_answer_is_unchanged(self):
        messages = [HumanMessage("hi"), *run_tools_round("x = 1", "ok", "c1")]

        self.assertEqual(compact_finished_turn(messages), messages)

    def test_tool_rounds_collapse_into_one_recap_pair(self):
        messages = [
            HumanMessage("сколько строк в файле?"),
            *run_tools_round("a = web_search('https://x')", '{"result": "page text"}', "c1"),
            *run_tools_round("return len(a)", '{"result": 42}', "c2"),
            AIMessage("42 строки"),
        ]

        compacted = compact_finished_turn(messages)

        self.assertEqual(len(compacted), 4)
        user, recap_call, recap_result, final = compacted
        self.assertEqual(user.content, "сколько строк в файле?")
        self.assertEqual(final.content, "42 строки")
        self.assertTrue(recap_call.additional_kwargs[RECAP_MARKER])
        self.assertTrue(recap_result.additional_kwargs[RECAP_MARKER])
        code = recap_call.tool_calls[0]["args"]["code"]
        self.assertIn("web_search", code)
        self.assertIn("return len(a)", code)
        self.assertEqual(recap_call.tool_calls[0]["id"], recap_result.tool_call_id)
        self.assertIn("2×run_tools", recap_result.content)
        self.assertIn("page text", recap_result.content)

    def test_discovery_tools_survive_only_as_counters(self):
        messages = [
            HumanMessage("go"),
            tool_call_ai("search_tools", {"query": "screen"}, "c1"),
            ToolMessage(content="screen_click(...) — clicks", tool_call_id="c1", name="search_tools"),
            *run_tools_round("screen_click('button')", "clicked", "c2"),
            AIMessage("done"),
        ]

        compacted = compact_finished_turn(messages)

        recap_result = compacted[2]
        self.assertIn("1×search_tools", recap_result.content)
        self.assertIn("1×run_tools", recap_result.content)
        self.assertNotIn("screen_click(...) — clicks", recap_result.content)

    def test_attached_image_messages_are_kept_after_the_recap(self):
        image = HumanMessage(content=[{"type": "text", "text": "img"}], additional_kwargs={ATTACHMENT_MARKER: True})
        messages = [
            HumanMessage("что на экране?"),
            *run_tools_round("screen_screenshot()", "attached", "c1"),
            image,
            AIMessage("вижу браузер"),
        ]

        compacted = compact_finished_turn(messages)

        self.assertIs(compacted[3], image)
        self.assertEqual(compacted[4].content, "вижу браузер")

    def test_previous_turns_and_their_recaps_stay_untouched(self):
        first_turn = compact_finished_turn(
            [
                HumanMessage("turn one"),
                *run_tools_round("x = 1", "ok", "c1"),
                AIMessage("first answer"),
            ]
        )
        messages = [
            *first_turn,
            HumanMessage("turn two"),
            *run_tools_round("y = 2", "ok", "c2"),
            AIMessage("second answer"),
        ]

        compacted = compact_finished_turn(messages)

        self.assertEqual(compacted[:4], first_turn)
        self.assertEqual(len(compacted), 8)
        self.assertIn("1×run_tools", compacted[6].content)
        self.assertIn("y = 2", compacted[5].tool_calls[0]["args"]["code"])
        self.assertNotIn("x = 1", compacted[5].tool_calls[0]["args"]["code"])

    def test_merged_code_is_capped(self):
        long_code = "x = 1\n" * 2000
        messages = [
            HumanMessage("go"),
            *run_tools_round(long_code, "ok", "c1"),
            AIMessage("done"),
        ]

        compacted = compact_finished_turn(messages)

        code = compacted[1].tool_calls[0]["args"]["code"]
        self.assertLessEqual(len(code), RECAP_CODE_CHARS + 40)
        self.assertIn("chars]", code)

    def test_recap_messages_carry_ids_for_usage_accounting(self):
        messages = [
            HumanMessage("go"),
            *run_tools_round("x = 1", "ok", "c1"),
            AIMessage("done"),
        ]

        compacted = compact_finished_turn(messages)

        self.assertTrue(compacted[1].id)
        self.assertTrue(compacted[2].id)


if __name__ == "__main__":
    unittest.main()
