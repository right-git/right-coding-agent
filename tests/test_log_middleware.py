import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.llm.middlewares.message_log import (
    MessageLogMiddleware,
    scrub,
    scrub_text,
    serialize_message,
)


class ScrubTests(unittest.TestCase):
    def test_data_uris_become_placeholders(self):
        uri = "data:image/png;base64," + "B" * 300

        self.assertEqual(
            scrub_text(f"look: {uri} done"),
            f"look: <data-uri stripped, {len(uri)} chars> done",
        )

    def test_long_base64_runs_become_placeholders(self):
        self.assertEqual(
            scrub_text("payload: " + "Q" * 5000),
            "payload: <base64 stripped, 5000 chars>",
        )

    def test_long_plain_text_is_truncated_with_overflow_noted(self):
        text = "word " * 100

        self.assertEqual(
            scrub_text(text, max_chars=10),
            text[:10] + f"… [+{len(text) - 10} chars]",
        )

    def test_short_text_is_untouched(self):
        self.assertEqual(scrub_text("hello"), "hello")

    def test_scrub_recurses_into_containers(self):
        value = {"a": ["x" * 300, {"b": 5, "c": None, "d": True}]}

        scrubbed = scrub(value, max_chars=50)

        self.assertIn("<base64 stripped, 300 chars>", scrubbed["a"][0])
        self.assertEqual(scrubbed["a"][1], {"b": 5, "c": None, "d": True})

    def test_unserializable_values_fall_back_to_repr(self):
        self.assertTrue(scrub(object()).startswith("<object"))


class SerializeMessageTests(unittest.TestCase):
    def test_tool_message_keeps_call_id_name_and_status(self):
        entry = serialize_message(ToolMessage(content="ok", tool_call_id="c1", name="run_tools"))

        self.assertEqual(entry["type"], "tool")
        self.assertEqual(entry["tool_call_id"], "c1")
        self.assertEqual(entry["name"], "run_tools")
        self.assertEqual(entry["status"], "success")

    def test_usage_carries_cache_reads_when_present(self):
        message = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_token_details": {"cache_read": 80},
            },
        )

        entry = serialize_message(message)

        self.assertEqual(entry["usage"], {"input_tokens": 100, "output_tokens": 10, "cache_read": 80})

    def test_ai_message_keeps_tool_calls_and_usage(self):
        entry = serialize_message(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_tools",
                        "args": {"code": "return 1"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                },
            )
        )

        self.assertEqual(entry["tool_calls"][0]["name"], "run_tools")
        self.assertEqual(entry["usage"], {"input_tokens": 10, "output_tokens": 2})


class MessageLogMiddlewareTests(unittest.TestCase):
    def test_logs_the_scrubbed_conversation_as_one_json_line(self):
        lines = []
        middleware = MessageLogMiddleware(emit=lines.append, max_text_chars=100)
        base64_payload = "Q" * 5000
        long_code = "print(1) " * 60
        messages = [
            HumanMessage("hello", id="h1"),
            AIMessage(
                content="",
                id="a1",
                tool_calls=[
                    {
                        "name": "run_tools",
                        "args": {"code": long_code},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=f"base64 JPEG: {base64_payload}",
                tool_call_id="c1",
                name="run_tools",
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Images captured:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_payload}"},
                    },
                ]
            ),
        ]

        result = middleware.before_model({"messages": messages})

        self.assertIsNone(result)
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["event"], "model_request")
        self.assertEqual(payload["message_count"], 4)

        human, ai, tool, vision = payload["messages"]
        self.assertEqual(human, {"type": "human", "id": "h1", "content": "hello"})
        self.assertIn("… [+", ai["tool_calls"][0]["args"]["code"])
        self.assertEqual(tool["content"], "base64 JPEG: <base64 stripped, 5000 chars>")
        self.assertEqual(
            vision["content"][1]["image_url"]["url"],
            f"<data-uri stripped, {23 + 5000} chars>",
        )

    def test_logs_the_model_response_with_finish_reason(self):
        lines = []
        middleware = MessageLogMiddleware(emit=lines.append)
        response = AIMessage(
            content="",
            response_metadata={"finish_reason": "stop", "model_name": "gemini"},
            usage_metadata={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )

        result = middleware.after_model({"messages": [HumanMessage("hi"), response]})

        self.assertIsNone(result)
        payload = json.loads(lines[0])
        self.assertEqual(payload["event"], "model_response")
        self.assertEqual(payload["message"]["response_metadata"]["finish_reason"], "stop")
        self.assertEqual(
            payload["message"]["usage"],
            {"input_tokens": 0, "output_tokens": 0},
        )

    def test_response_logging_skips_non_ai_tails(self):
        lines = []
        middleware = MessageLogMiddleware(emit=lines.append)

        middleware.after_model({"messages": [HumanMessage("hi")]})
        middleware.after_model({"messages": []})

        self.assertEqual(lines, [])

    def test_logging_failures_never_propagate(self):
        def explode(_line):
            raise RuntimeError("sink is broken")

        middleware = MessageLogMiddleware(emit=explode)

        self.assertIsNone(middleware.before_model({"messages": [HumanMessage("hi")]}))


if __name__ == "__main__":
    unittest.main()
