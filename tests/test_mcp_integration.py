"""Real-server integration test for the MCP layer (opt-in, needs node).

Skipped unless RUN_MCP_TESTS=1: it shells out to `npx -y
@modelcontextprotocol/server-everything`, a real stdio MCP server, and drives
the whole stack — config, persistent connection, tool adaptation, the
registry — against it, so it never runs in the normal suite (no node/network
dependency there). Gated the same way as tests/test_vision_integration.py.

    RUN_MCP_TESTS=1 uv run python -m unittest tests.test_mcp_integration -v
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUN = os.environ.get("RUN_MCP_TESTS") == "1"


@unittest.skipUnless(RUN, "set RUN_MCP_TESTS=1 to run against a live npx MCP server")
class TestLiveStdioServer(unittest.TestCase):
    def test_everything_server_end_to_end(self):
        from src.llm.tools import ToolRegistry
        from src.llm.tools.mcp.config import McpServerConfig
        from src.llm.tools.mcp.manager import McpManager

        async def scenario():
            registry = ToolRegistry()
            manager = McpManager(
                configs={
                    "everything": McpServerConfig(
                        name="everything",
                        command="npx",
                        args=["-y", "@modelcontextprotocol/server-everything"],
                    )
                },
                registry=registry,
            )
            await manager.start()
            try:
                names = {t.name for t in registry.all_tools(source_prefix="mcp:")}
                self.assertIn("mcp__everything__echo", names)
                echo = registry.get("mcp__everything__echo")
                out = await echo.ainvoke({"message": "round-trip"})
                self.assertIn("round-trip", out)
            finally:
                await manager.stop()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
