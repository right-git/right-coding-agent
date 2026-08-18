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
