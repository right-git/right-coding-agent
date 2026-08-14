import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from PIL import Image

from src.llm.attachments import (
    ATTACHMENT_MARKER,
    AttachedImagesMiddleware,
    attach_image,
    collecting_images,
    image_blocks,
)
from src.llm.computer_tools import screen_locate, screen_screenshot, set_computer
from src.llm.meta_tools import ToolRegistry, run_tools, set_registry
from src.tools.computer_use import (
    ComputerUse,
    Detection,
    NullOverlay,
    image_to_base64,
)
from src.tools.computer_use.fakes import (
    RecordingPointer,
    ScriptedLocator,
    StaticScreen,
)


def decode(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


class ImageEncodingTests(unittest.TestCase):
    def test_roundtrip_preserves_size_and_mode(self):
        image = Image.new("RGBA", (200, 100), (255, 0, 0, 255))

        decoded = decode(image_to_base64(image, max_side=None))

        self.assertEqual(decoded.size, (200, 100))
        self.assertEqual(decoded.mode, "RGB")

    def test_large_images_are_downscaled_to_max_side(self):
        image = Image.new("RGB", (4000, 2000))

        decoded = decode(image_to_base64(image, max_side=1000))

        self.assertEqual(decoded.size, (1000, 500))


class AttachmentChannelTests(unittest.TestCase):
    def test_attach_outside_a_run_is_refused(self):
        self.assertFalse(attach_image("QUJD"))

    def test_attach_inside_a_run_collects_and_closes(self):
        with collecting_images() as bucket:
            self.assertTrue(attach_image("QUJD", "image/png", label="shot"))

        self.assertEqual(
            bucket,
            [{"base64_data": "QUJD", "mime_type": "image/png", "label": "shot"}],
        )
        self.assertFalse(attach_image("after"))


class ComputerUseImageTests(unittest.TestCase):
    def make_computer(self, responses=((),)):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return ComputerUse(
            locator=ScriptedLocator(responses),
            screen=StaticScreen([Image.new("RGB", (200, 100), "blue")], (200, 100)),
            pointer=RecordingPointer(),
            overlay=NullOverlay(),
            output_path=Path(directory.name) / "annotated.jpg",
            dpi_aware=False,
            min_target_px=0,
        )

    def test_screenshot_base64_captures_and_encodes(self):
        computer = self.make_computer()

        decoded = decode(computer.screenshot_base64())

        self.assertEqual(decoded.size, (200, 100))
        self.assertEqual(computer.screen.captures, 1)

    def test_annotated_base64_requires_a_screenshot(self):
        computer = self.make_computer()

        with self.assertRaisesRegex(RuntimeError, "No screenshot"):
            computer.annotated_base64()

    def test_annotated_base64_draws_the_last_detections(self):
        computer = self.make_computer(
            responses=[[Detection("save", (10, 20, 60, 60))]]
        )
        computer.locate_object("the save button")

        decoded = decode(computer.annotated_base64())

        self.assertEqual(decoded.size, (200, 100))


class ScreenToolAttachTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(set_computer, None)
        self.directory = Path(directory.name)

    def install(self, responses=((),)):
        computer = ComputerUse(
            locator=ScriptedLocator(responses),
            screen=StaticScreen([Image.new("RGB", (200, 100))], (200, 100)),
            pointer=RecordingPointer(),
            overlay=NullOverlay(),
            output_path=self.directory / "annotated.jpg",
            dpi_aware=False,
            min_target_px=0,
        )
        set_computer(computer)
        return computer

    async def test_screenshot_tool_attaches_the_image(self):
        self.install()

        with collecting_images() as bucket:
            result = await screen_screenshot.ainvoke({})

        self.assertIn("attached as an image", result)
        self.assertNotIn("base64 JPEG:", result)
        self.assertEqual(len(bucket), 1)
        self.assertEqual(decode(bucket[0]["base64_data"]).size, (200, 100))

    async def test_screenshot_tool_can_also_return_base64_text(self):
        self.install()

        with collecting_images():
            result = await screen_screenshot.ainvoke({"return_base64": True})

        self.assertIn("base64 JPEG:", result)

    async def test_screenshot_tool_falls_back_to_base64_without_a_channel(self):
        self.install()

        result = await screen_screenshot.ainvoke({})

        self.assertIn("base64 JPEG:", result)

    async def test_locate_with_return_screen_attaches_annotated_view(self):
        self.install(responses=[[Detection("save", (10, 20, 60, 60))]])

        with collecting_images() as bucket:
            result = await screen_locate.ainvoke(
                {"description": "save button", "return_screen": True}
            )

        self.assertIn("save: box=", result)
        self.assertIn("annotated screenshot attached", result)
        self.assertEqual(len(bucket), 1)
        self.assertEqual(bucket[0]["label"], "screen_locate: save button")

    async def test_locate_without_return_screen_attaches_nothing(self):
        self.install(responses=[[Detection("save", (10, 20, 60, 60))]])

        with collecting_images() as bucket:
            await screen_locate.ainvoke({"description": "save button"})

        self.assertEqual(bucket, [])


class RunToolsImageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(set_registry, None)

        @tool(parse_docstring=True)
        async def snap() -> str:
            """Capture something and attach it.

            Returns:
                Confirmation.
            """
            attach_image("QUJD", label="snap")
            return "snapped"

        set_registry(ToolRegistry([snap]))

    async def test_plain_invoke_reports_attached_images_in_the_json(self):
        result = await run_tools.ainvoke({"code": 'snap()\nreturn "ok"'})

        outcome = json.loads(result)
        self.assertEqual(outcome["result"], "ok")
        self.assertEqual(outcome["attached_images"], 1)

    async def test_tool_call_invoke_carries_images_in_the_artifact(self):
        message = await run_tools.ainvoke(
            {
                "type": "tool_call",
                "name": "run_tools",
                "args": {"code": 'snap()\nreturn "ok"'},
                "id": "call_1",
            }
        )

        self.assertIsInstance(message, ToolMessage)
        self.assertEqual(len(message.artifact), 1)
        self.assertEqual(message.artifact[0]["base64_data"], "QUJD")
        self.assertEqual(json.loads(message.content)["attached_images"], 1)

    async def test_attachments_are_capped_and_drops_reported(self):
        message = await run_tools.ainvoke(
            {
                "type": "tool_call",
                "name": "run_tools",
                "args": {"code": "for i in range(8):\n    snap()\nreturn i"},
                "id": "call_2",
            }
        )

        outcome = json.loads(message.content)
        self.assertEqual(outcome["attached_images"], 6)
        self.assertEqual(outcome["dropped_images"], 2)
        self.assertEqual(len(message.artifact), 6)


class MiddlewareTests(unittest.TestCase):
    def tool_round(self, artifact):
        return [
            HumanMessage("look at my screen"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_tools",
                        "args": {"code": "..."},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="{}", tool_call_id="call_1", artifact=artifact),
        ]

    def test_images_are_surfaced_as_a_vision_message(self):
        middleware = AttachedImagesMiddleware()
        messages = self.tool_round(
            [{"base64_data": "QUJD", "mime_type": "image/png", "label": "shot"}]
        )

        update = middleware.before_model({"messages": messages})

        (injected,) = update["messages"]
        self.assertIsInstance(injected, HumanMessage)
        self.assertTrue(injected.additional_kwargs[ATTACHMENT_MARKER])
        image_urls = [
            block["image_url"]["url"]
            for block in injected.content
            if block["type"] == "image_url"
        ]
        self.assertEqual(image_urls, ["data:image/png;base64,QUJD"])
        texts = [
            block["text"]
            for block in injected.content
            if block["type"] == "text"
        ]
        self.assertIn("shot", texts)

    def test_surfacing_is_idempotent(self):
        middleware = AttachedImagesMiddleware()
        messages = self.tool_round([{"base64_data": "QUJD"}])
        messages.extend(middleware.before_model({"messages": messages})["messages"])

        self.assertIsNone(middleware.before_model({"messages": messages}))

    def test_rounds_without_images_are_ignored(self):
        middleware = AttachedImagesMiddleware()

        self.assertIsNone(
            middleware.before_model({"messages": self.tool_round(None)})
        )
        self.assertIsNone(
            middleware.before_model({"messages": [HumanMessage("hi")]})
        )

    def test_image_blocks_include_labels(self):
        blocks = image_blocks([{"base64_data": "QUJD", "label": "screen 1"}])

        self.assertEqual(blocks[1], {"type": "text", "text": "screen 1"})
        self.assertEqual(
            blocks[2]["image_url"]["url"], "data:image/jpeg;base64,QUJD"
        )


if __name__ == "__main__":
    unittest.main()
