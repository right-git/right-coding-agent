import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage, ToolMessage
from PIL import Image

from evaluation.direct_agent import DIRECT_CODING_AGENT_SYS, DirectAgents
from evaluation.direct_tools import (
    DIRECT_TOOLS,
    schema_token_estimate,
)
from src.llm.tools.computer import screen_click, set_computer
from src.llm.middlewares.message_log import MessageLogMiddleware
from src.llm.tools import META_TOOLS
from src.llm.types import LLMProvider
from src.llm.tools.computer import ComputerUse, NullOverlay
from src.llm.tools.computer.fakes import (
    RecordingPointer,
    ScriptedLocator,
    StaticScreen,
)

EXPECTED_DIRECT_NAMES = [
    "web_search",
    "screen_locate",
    "screen_screenshot",
    "screen_mark",
    "screen_click",
    "screen_type",
    "screen_key",
    "screen_scroll",
]


class DirectToolsTests(unittest.TestCase):
    def test_direct_tools_cover_the_whole_registry(self):
        self.assertEqual([tool_obj.name for tool_obj in DIRECT_TOOLS], EXPECTED_DIRECT_NAMES)

    def test_wrappers_keep_the_original_schema_and_description(self):
        wrapped = next(t for t in DIRECT_TOOLS if t.name == "screen_click")

        self.assertEqual(wrapped.description, screen_click.description)
        self.assertEqual(wrapped.args, screen_click.args)

    def test_direct_schemas_cost_more_than_the_meta_surface(self):
        meta = schema_token_estimate(META_TOOLS)
        direct = schema_token_estimate(DIRECT_TOOLS)

        self.assertGreater(meta, 0)
        self.assertGreater(direct, meta)


class DirectToolImageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(set_computer, None)
        set_computer(
            ComputerUse(
                locator=ScriptedLocator([[]]),
                screen=StaticScreen([Image.new("RGB", (200, 100))], (200, 100)),
                pointer=RecordingPointer(),
                overlay=NullOverlay(),
                output_path=Path(directory.name) / "annotated.jpg",
                dpi_aware=False,
                min_target_px=0,
            )
        )

    async def test_direct_screenshot_carries_the_image_as_artifact(self):
        wrapped = next(t for t in DIRECT_TOOLS if t.name == "screen_screenshot")

        message = await wrapped.ainvoke(
            {
                "type": "tool_call",
                "name": "screen_screenshot",
                "args": {},
                "id": "call_1",
            }
        )

        self.assertIsInstance(message, ToolMessage)
        self.assertIn("attached as an image", message.content)
        self.assertNotIn("base64 JPEG:", message.content)
        self.assertEqual(len(message.artifact), 1)
        self.assertTrue(message.artifact[0]["base64_data"])


class DirectAgentContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_agent_exposes_every_tool_and_same_middlewares(self):
        agent = DirectAgents(
            [
                LLMProvider(
                    provider_name="openai",
                    api_key="test-key",
                    api_base="http://localhost",
                )
            ]
        )
        agent.ask_agent = AsyncMock(return_value={"messages": []})

        await agent.right_coding_agent(
            messages=[HumanMessage("click the button")],
            model="openai/gpt-4.1-mini",
        )

        kwargs = agent.ask_agent.await_args.kwargs
        self.assertEqual(
            [tool_obj.name for tool_obj in kwargs["tools"]],
            EXPECTED_DIRECT_NAMES,
        )
        self.assertEqual(kwargs["system_prompt"], DIRECT_CODING_AGENT_SYS)
        self.assertIsInstance(kwargs["middlewares"][-1], MessageLogMiddleware)


if __name__ == "__main__":
    unittest.main()
