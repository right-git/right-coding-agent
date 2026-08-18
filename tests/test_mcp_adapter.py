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


class TestRealSdkTypes(unittest.TestCase):
    """Adaptation against REAL mcp.types objects, not fakes.

    SDK 2.0 renamed every wire-format camelCase field to snake_case on its
    pydantic models (inputSchema -> input_schema, isError -> is_error, ...).
    The fakes elsewhere in this file use camelCase attributes and stayed
    green while every real MCP tool registered with an empty schema — this
    class exists so the fakes can never drift from the SDK again.
    """

    def real_tool(self):
        from mcp.types import Tool, ToolAnnotations

        return Tool.model_validate(
            {
                "name": "navigate",
                "description": "Navigate to a URL",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string", "description": "Target URL"}},
                    "required": ["url"],
                },
                "annotations": ToolAnnotations.model_validate({"destructiveHint": True}),
            }
        )

    def test_real_tool_schema_reaches_signature_and_document(self):
        from src.llm.tools import ToolRegistry

        registry = ToolRegistry()
        calls = []

        async def call(name, args):
            calls.append((name, args))
            return call_result(text_item("ok"))

        registry.register(build_mcp_tool("pw", self.real_tool(), call), source="mcp:pw")
        tool_obj = registry.get("mcp__pw__navigate")
        self.assertIn("url", registry.signature(tool_obj))
        self.assertIn('"url"', registry.document("mcp__pw__navigate"))

    def test_real_tool_annotations_mark_description(self):
        async def call(name, args):
            return call_result(text_item("ok"))

        tool_obj = build_mcp_tool("pw", self.real_tool(), call)
        self.assertIn("[DESTRUCTIVE]", tool_obj.description)

    def test_real_error_result_gets_error_prefix(self):
        from mcp.types import CallToolResult, TextContent

        result = CallToolResult.model_validate({"content": [{"type": "text", "text": "boom"}], "isError": True})
        out = serialize_call_result(result, server="pw", tool_name="navigate")
        self.assertIn("[mcp error]", out)
        self.assertIsInstance(result.content[0], TextContent)

    def test_real_structured_content_is_serialized(self):
        from mcp.types import CallToolResult

        result = CallToolResult.model_validate({"content": [], "structuredContent": {"k": 1}})
        out = serialize_call_result(result, server="pw", tool_name="navigate")
        self.assertIn('"k": 1', out)

    def test_real_image_content_attaches_with_its_mime(self):
        from mcp.types import CallToolResult

        result = CallToolResult.model_validate(
            {"content": [{"type": "image", "data": "aGk=", "mimeType": "image/webp"}]}
        )
        with collecting_images() as images:
            serialize_call_result(result, server="pw", tool_name="shot")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["mime_type"], "image/webp")


class TestLinkedImageFiles(unittest.TestCase):
    """Playwright-style results: a markdown link to a PNG on disk, no image block.

    The server saves the screenshot and returns only text like
    `- [Screenshot of viewport](.playwright-mcp/page-....png)` — without this
    path the model stays blind to its own screenshots.
    """

    PNG_BYTES = bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000100000001080600000000"
        "1f15c4890000000d49444154789c636000000002000155c2d37e00000000"
        "49454e44ae426082"
    )

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.png = Path(self.tmp.name) / "page-shot.png"
        self.png.write_bytes(self.PNG_BYTES)

    def serialize(self, text):
        with collecting_images() as images:
            out = serialize_call_result(call_result(text_item(text)), server="pw", tool_name="browser_take_screenshot")
        return out, images

    def test_markdown_link_to_existing_png_attaches(self):
        out, images = self.serialize(f"### Result\n- [Screenshot of viewport]({self.png})")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["mime_type"], "image/png")
        self.assertIn("attached", out)

    def test_missing_file_is_left_alone(self):
        out, images = self.serialize("- [Screenshot](/nowhere/gone.png)")
        self.assertEqual(images, [])
        self.assertIn("gone.png", out)

    def test_oversized_file_is_skipped(self):
        from unittest.mock import patch

        from src.llm.tools.mcp import adapter

        with patch.object(adapter, "MAX_LINKED_IMAGE_BYTES", 10):
            out, images = self.serialize(f"[shot]({self.png})")
        self.assertEqual(images, [])

    def test_non_image_links_ignored(self):
        report = Path(self.tmp.name) / "notes.txt"
        report.write_text("hi", encoding="utf-8")
        out, images = self.serialize(f"[notes]({report})")
        self.assertEqual(images, [])


if __name__ == "__main__":
    unittest.main()
