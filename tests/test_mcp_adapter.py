import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp.adapter import (
    MAX_TOOL_NAME_LENGTH,
    build_mcp_tool,
    build_prompt_command,
    build_tool_name,
    normalize_tool_arguments,
    serialize_call_result,
)
from src.llm.tools.meta.attachments import collecting_images


def call_result(*content, structured=None, is_error=False):
    return SimpleNamespace(content=list(content), structuredContent=structured, isError=is_error)


def text_item(text):
    return SimpleNamespace(type="text", text=text)


def image_item(data="aGk=", mime="image/png"):
    return SimpleNamespace(type="image", data=data, mimeType=mime)


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


class TestSerializeCallResult(unittest.TestCase):
    def test_single_text_returns_plain(self):
        out = serialize_call_result(call_result(text_item("hello")), server="s", tool_name="t")
        self.assertEqual(out, "hello")

    def test_error_flag_prefixes(self):
        out = serialize_call_result(call_result(text_item("boom"), is_error=True), server="s", tool_name="t")
        self.assertIn("[mcp error]", out)
        self.assertIn("boom", out)

    def test_structured_content_serialized(self):
        out = serialize_call_result(call_result(structured={"k": 1}), server="s", tool_name="t")
        self.assertIn('"k": 1', out)

    def test_image_goes_to_attachment_channel(self):
        with collecting_images() as images:
            out = serialize_call_result(call_result(image_item()), server="s", tool_name="t")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["mime_type"], "image/png")
        self.assertIn("image attached", out)

    def test_image_without_channel_reports_stub(self):
        out = serialize_call_result(call_result(image_item()), server="s", tool_name="t")
        self.assertIn("image", out.lower())

    def test_resource_link_summarized(self):
        link = SimpleNamespace(
            type="resource_link", name="doc", title=None, uri="res://x", description=None, mimeType=None, size=None
        )
        out = serialize_call_result(call_result(link), server="s", tool_name="t")
        self.assertIn("res://x", out)


class TestBuildMcpTool(unittest.TestCase):
    def remote_tool(self, annotations=None):
        return SimpleNamespace(
            name="click",
            description="Click an element.",
            inputSchema={
                "type": "object",
                "properties": {"selector": {"type": "string"}, "force": {"type": "boolean"}},
                "required": ["selector"],
            },
            annotations=annotations,
        )

    def build(self, annotations=None):
        self.calls = []

        async def call(name, args):
            self.calls.append((name, args))
            return call_result(text_item("ok"))

        return build_mcp_tool("playwright", self.remote_tool(annotations), call)

    def test_name_description_and_args(self):
        tool_obj = self.build()
        self.assertEqual(tool_obj.name, "mcp__playwright__click")
        self.assertIn("Click an element.", tool_obj.description)
        self.assertEqual(list(tool_obj.args), ["selector", "force"])

    def test_invoke_normalizes_and_calls(self):
        tool_obj = self.build()
        out = asyncio.run(tool_obj.ainvoke({"selector": "#a", "force": "true"}))
        self.assertEqual(out, "ok")
        self.assertEqual(self.calls, [("click", {"selector": "#a", "force": True})])

    def test_call_failure_returns_error_string(self):
        async def call(name, args):
            raise RuntimeError("gone")

        tool_obj = build_mcp_tool("playwright", self.remote_tool(), call)
        out = asyncio.run(tool_obj.ainvoke({"selector": "#a"}))
        self.assertIn("[mcp error]", out)
        self.assertIn("gone", out)

    def test_destructive_annotation_marks_description(self):
        tool_obj = self.build(annotations=SimpleNamespace(readOnlyHint=None, destructiveHint=True))
        self.assertIn("[DESTRUCTIVE]", tool_obj.description)

    def test_registry_integration_with_dict_schema(self):
        from src.llm.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(self.build(), source="mcp:playwright")
        signature = registry.signature(registry.get("mcp__playwright__click"))
        self.assertIn("selector", signature)
        self.assertIn("mcp__playwright__click", registry.document("mcp__playwright__click"))
        table = registry.callables()
        out = asyncio.run(table["mcp__playwright__click"]("#a"))
        self.assertEqual(out, "ok")


if __name__ == "__main__":
    unittest.main()
