import json
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

    def test_in_script_contracts_are_kept_for_reuse(self):
        contract = "web_search(url)\n\nFetch a web page and return Markdown.\n\nArgument schema: {...}"
        discovery_result = json.dumps({"result": "ok", "logs": [], "error": None, "contracts": [contract]})
        messages = [
            HumanMessage("go"),
            *run_tools_round("get_tool(['web_search'])", discovery_result, "c1"),
            *run_tools_round("web_search('https://x')", "ok", "c2"),
            AIMessage("done"),
        ]

        compacted = compact_finished_turn(messages)

        recap_result = compacted[2]
        self.assertIn("tool contracts", recap_result.content)
        self.assertIn("Fetch a web page and return Markdown.", recap_result.content)
        # kept once, in the contracts section — not again in the result heads
        self.assertEqual(recap_result.content.count("Fetch a web page and return Markdown."), 1)

    def test_identical_contracts_from_two_runs_are_kept_once(self):
        contract = "screen_click(description)\n\nClick an element."
        discovery_result = json.dumps({"result": "ok", "logs": [], "error": None, "contracts": [contract]})
        messages = [
            HumanMessage("go"),
            *run_tools_round("get_tool(['screen_click'])", discovery_result, "c1"),
            *run_tools_round("get_tool(['screen_click'])", discovery_result, "c2"),
            *run_tools_round("screen_click('x')", "ok", "c3"),
            AIMessage("done"),
        ]

        compacted = compact_finished_turn(messages)

        self.assertEqual(compacted[2].content.count("Click an element."), 1)

    def test_non_json_run_tools_results_pass_through_untouched(self):
        messages = [
            HumanMessage("go"),
            *run_tools_round("boom()", "Tool call failed, error: kaput", "c1"),
            AIMessage("done"),
        ]

        compacted = compact_finished_turn(messages)

        self.assertIn("Tool call failed, error: kaput", compacted[2].content)
        self.assertNotIn("tool contracts", compacted[2].content)

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


def tool_image_message(identifier, count=1):
    blocks = [{"type": "text", "text": "Images captured by the tool calls above:"}]
    for index in range(count):
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,IMG{identifier}{index}"}})
    return HumanMessage(content=blocks, id=identifier, additional_kwargs={ATTACHMENT_MARKER: True})


def user_image_message(identifier, text="смотри"):
    return HumanMessage(
        content=[
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,USR{identifier}"}},
        ],
        id=identifier,
    )


def image_urls(message):
    return [block["image_url"]["url"] for block in message.content if block.get("type") == "image_url"]


class PruneImagesTests(unittest.TestCase):
    def test_only_the_newest_tool_screenshot_survives(self):
        from src.llm.history import TOOL_IMAGE_STUB, prune_images

        messages = [
            tool_image_message("shot1"),
            AIMessage("step one"),
            tool_image_message("shot2"),
            AIMessage("step two"),
        ]

        pruned = prune_images(messages, keep_tool=1)

        self.assertEqual(image_urls(pruned[0]), [])
        self.assertIn({"type": "text", "text": TOOL_IMAGE_STUB}, pruned[0].content)
        self.assertEqual(len(image_urls(pruned[2])), 1)

    def test_user_images_keep_their_own_budget(self):
        from src.llm.history import USER_IMAGE_STUB, prune_images

        messages = [
            user_image_message("u1"),
            user_image_message("u2"),
            tool_image_message("shot1"),
            user_image_message("u3"),
        ]

        pruned = prune_images(messages, keep_tool=1, keep_user=2)

        self.assertIn({"type": "text", "text": USER_IMAGE_STUB}, pruned[0].content)
        self.assertEqual(len(image_urls(pruned[1])), 1)
        self.assertEqual(len(image_urls(pruned[2])), 1)  # tool budget untouched by user images
        self.assertEqual(len(image_urls(pruned[3])), 1)

    def test_ids_and_markers_survive_the_rewrite(self):
        from src.llm.history import prune_images

        messages = [tool_image_message("old"), tool_image_message("new")]

        pruned = prune_images(messages, keep_tool=1)

        self.assertEqual(pruned[0].id, "old")
        self.assertTrue(pruned[0].additional_kwargs[ATTACHMENT_MARKER])

    def test_pruning_is_idempotent(self):
        from src.llm.history import prune_images

        messages = [tool_image_message("a"), tool_image_message("b"), tool_image_message("c")]

        once = prune_images(messages, keep_tool=1)
        twice = prune_images(once, keep_tool=1)

        self.assertEqual([m.content for m in once], [m.content for m in twice])

    def test_plain_text_messages_are_untouched(self):
        from src.llm.history import prune_images

        messages = [HumanMessage("привет"), AIMessage("привет!")]

        self.assertEqual(prune_images(messages), messages)


class FinalizeTurnHistoryTests(unittest.TestCase):
    def test_compacts_tools_and_prunes_old_images(self):
        from src.llm.history import finalize_turn_history

        messages = [
            tool_image_message("old"),
            HumanMessage("что на экране?"),
            *run_tools_round("screen_screenshot()", "attached", "c1"),
            tool_image_message("fresh"),
            AIMessage("вижу браузер"),
        ]

        finalized = finalize_turn_history(messages)

        self.assertIn({"type": "text", "text": "[screenshot removed to save context]"}, finalized[0].content)
        self.assertTrue(finalized[2].additional_kwargs["tool_recap"])
        self.assertEqual(len(image_urls(finalized[4])), 1)
        self.assertEqual(finalized[5].content, "вижу браузер")


if __name__ == "__main__":
    unittest.main()
