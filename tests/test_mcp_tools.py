import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.tools.mcp import tool as mcp_tool
from src.llm.tools.mcp.manager import set_mcp_manager


class FakeManager:
    def __init__(self):
        self.read_requests = []

    async def list_resources(self, server=None):
        return [{"server": "srv", "uri": "res://a", "name": "A", "description": "", "mime_type": "text/plain"}]

    async def read_resource(self, server, uri):
        self.read_requests.append((server, uri))
        return SimpleNamespace(contents=[SimpleNamespace(uri=uri, text="the body", mimeType="text/plain")])


class TestResourceTools(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        set_mcp_manager(self.manager)
        self.addCleanup(set_mcp_manager, None)

    def test_list_resources(self):
        out = asyncio.run(mcp_tool.mcp_list_resources.ainvoke({}))
        self.assertIn("res://a", out)

    def test_read_resource_text(self):
        out = asyncio.run(mcp_tool.mcp_read_resource.ainvoke({"server": "srv", "uri": "res://a"}))
        self.assertIn("the body", out)
        self.assertEqual(self.manager.read_requests, [("srv", "res://a")])

    def test_errors_return_strings(self):
        async def boom(server, uri):
            raise RuntimeError("nope")

        self.manager.read_resource = boom
        out = asyncio.run(mcp_tool.mcp_read_resource.ainvoke({"server": "srv", "uri": "res://a"}))
        self.assertIn("[mcp error]", out)


class TestDefaultRegistryGating(unittest.TestCase):
    def test_service_tools_registered_only_with_servers_configured(self):
        from unittest.mock import patch

        from src.llm.tools.meta import defaults

        with patch.object(defaults, "_mcp_servers_configured", return_value=False):
            names = {t.name for t in defaults.default_tools()}
            self.assertNotIn("mcp_list_resources", names)
        with patch.object(defaults, "_mcp_servers_configured", return_value=True):
            names = {t.name for t in defaults.default_tools()}
            self.assertIn("mcp_list_resources", names)
            self.assertIn("mcp_read_resource", names)


if __name__ == "__main__":
    unittest.main()
