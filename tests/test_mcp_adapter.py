import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp.adapter import (
    MAX_TOOL_NAME_LENGTH,
    build_prompt_command,
    build_tool_name,
    normalize_tool_arguments,
)


class TestToolNames(unittest.TestCase):
    def test_simple_name(self):
        self.assertEqual(build_tool_name("playwright", "click"), "mcp__playwright__click")

    def test_sanitizes_non_identifier_chars(self):
        name = build_tool_name("my-server", "do.thing")
        self.assertEqual(name, "mcp__my_server__do_thing")
        self.assertTrue(name.isidentifier())

    def test_long_name_truncates_with_stable_hash(self):
        long_tool = "extremely_" * 12
        first = build_tool_name("srv", long_tool)
        second = build_tool_name("srv", long_tool)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), MAX_TOOL_NAME_LENGTH)
        self.assertTrue(first.isidentifier())

    def test_long_names_differing_only_in_tail_stay_distinct(self):
        a = build_tool_name("srv", "x" * 80 + "a")
        b = build_tool_name("srv", "x" * 80 + "b")
        self.assertNotEqual(a, b)

    def test_prompt_command(self):
        self.assertEqual(build_prompt_command("srv", "code-review"), "/mcp__srv__code_review")


class TestNormalizeArguments(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "properties": {
            "flag": {"type": "boolean"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "items": {"type": "array"},
            "config": {"type": "object"},
            "text": {"type": "string"},
        },
        "required": ["text"],
    }

    def test_string_booleans_and_numbers_coerce(self):
        result = normalize_tool_arguments({"flag": "true", "count": "3", "ratio": "0.5", "text": "x"}, self.SCHEMA)
        self.assertEqual(result, {"flag": True, "count": 3, "ratio": 0.5, "text": "x"})

    def test_json_string_array_and_object_coerce(self):
        result = normalize_tool_arguments({"items": '["a", 1]', "config": '{"k": 2}', "text": "x"}, self.SCHEMA)
        self.assertEqual(result["items"], ["a", 1])
        self.assertEqual(result["config"], {"k": 2})

    def test_comma_string_becomes_array(self):
        result = normalize_tool_arguments({"items": "a, b; c", "text": "x"}, self.SCHEMA)
        self.assertEqual(result["items"], ["a", "b", "c"])

    def test_empty_optional_values_dropped_required_kept(self):
        result = normalize_tool_arguments({"count": "", "text": ""}, self.SCHEMA)
        self.assertNotIn("count", result)
        self.assertIn("text", result)

    def test_no_schema_passes_through(self):
        self.assertEqual(normalize_tool_arguments({"a": "1"}, None), {"a": 1})


if __name__ == "__main__":
    unittest.main()
