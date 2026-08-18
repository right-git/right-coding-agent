import asyncio
import sys
import unittest
from pathlib import Path

from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools import ToolRegistry, search_tools, set_registry


@tool(parse_docstring=True)
async def native_probe(x: str) -> str:
    """A native probe tool.

    Args:
        x: Anything.

    Returns:
        Echo.
    """
    return x


@tool(parse_docstring=True)
async def mcp__srv__probe(x: str) -> str:
    """A remote probe tool.

    Args:
        x: Anything.

    Returns:
        Echo.
    """
    return x


class TestRegistrySources(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(native_probe)
        self.registry.register(mcp__srv__probe, source="mcp:srv")

    def test_source_of(self):
        self.assertIsNone(self.registry.source_of("native_probe"))
        self.assertEqual(self.registry.source_of("mcp__srv__probe"), "mcp:srv")

    def test_brief_marks_mcp_tools(self):
        self.assertIn("[MCP: srv]", self.registry.brief(mcp__srv__probe))
        self.assertNotIn("[MCP", self.registry.brief(native_probe))

    def test_search_source_filter(self):
        names = [t.name for t in self.registry.search("probe", source_prefix="mcp:")]
        self.assertEqual(names, ["mcp__srv__probe"])

    def test_unregister(self):
        self.assertTrue(self.registry.unregister("mcp__srv__probe"))
        self.assertFalse(self.registry.unregister("mcp__srv__probe"))
        self.assertIsNone(self.registry.get("mcp__srv__probe"))
        self.assertIsNone(self.registry.source_of("mcp__srv__probe"))


class TestSearchToolsOnlyMcp(unittest.TestCase):
    def setUp(self):
        registry = ToolRegistry()
        registry.register(native_probe)
        registry.register(mcp__srv__probe, source="mcp:srv")
        set_registry(registry)
        self.addCleanup(set_registry, None)

    def test_only_mcp_filters(self):
        listing = asyncio.run(search_tools("probe", only_mcp=True))
        self.assertIn("mcp__srv__probe", listing)
        self.assertNotIn("native_probe", listing)

    def test_default_lists_both_with_marker(self):
        listing = asyncio.run(search_tools("probe"))
        self.assertIn("native_probe", listing)
        self.assertIn("[MCP: srv]", listing)


if __name__ == "__main__":
    unittest.main()


def make_tool(name: str, description: str):
    """A minimal StructuredTool with a one-arg dict schema, for bulk fixtures."""
    from langchain_core.tools import StructuredTool

    async def run(**kwargs):
        return "ok"

    return StructuredTool(
        name=name,
        description=description,
        args_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        coroutine=run,
    )


class TestGroupedMcpDiscovery(unittest.TestCase):
    """Big MCP servers collapse to one aggregate line; server= drills in."""

    def setUp(self):
        registry = ToolRegistry()
        registry.register(make_tool("web_fetch", "Fetch a browser page."))
        for i in range(5):
            registry.register(make_tool(f"mcp__pw__browser_{i}", f"Browser action {i}."), source="mcp:pw")
        registry.register(make_tool("mcp__tiny__browser_open", "Open a browser."), source="mcp:tiny")
        set_registry(registry)
        self.addCleanup(set_registry, None)

    def listing(self, *args, **kwargs):
        return asyncio.run(search_tools(*args, **kwargs))

    def test_big_server_collapses_to_aggregate_line(self):
        out = self.listing("browser")
        self.assertIn("web_fetch", out)
        self.assertIn("MCP server 'pw': 5 matching tools", out)
        self.assertIn('server="pw"', out)
        self.assertNotIn("mcp__pw__browser_0", out)

    def test_small_server_stays_inline(self):
        out = self.listing("browser")
        self.assertIn("mcp__tiny__browser_open", out)

    def test_server_param_lists_everything_ungrouped(self):
        out = self.listing("", server="pw")
        for i in range(5):
            self.assertIn(f"mcp__pw__browser_{i}", out)
        self.assertNotIn("matching tools", out)

    def test_server_param_with_query_filters(self):
        out = self.listing("action 3", server="pw")
        self.assertIn("mcp__pw__browser_3", out)

    def test_unknown_server_lists_known_ones(self):
        out = self.listing("browser", server="nope")
        self.assertIn("Unknown MCP server 'nope'", out)
        self.assertIn("pw", out)
        self.assertIn("tiny", out)

    def test_empty_query_groups_instead_of_truncating_silently(self):
        out = self.listing("")
        self.assertIn("MCP server 'pw': 5", out)
        self.assertIn("web_fetch", out)


class TestTruncationHint(unittest.TestCase):
    def test_long_native_listing_reports_hidden_count(self):
        registry = ToolRegistry()
        for i in range(12):
            registry.register(make_tool(f"native_probe_{i}", "A probing helper."))
        set_registry(registry)
        self.addCleanup(set_registry, None)
        out = asyncio.run(search_tools("probing"))
        self.assertIn("4 more", out)
