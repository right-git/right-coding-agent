"""Real-model integration tests for the vision locator (opt-in, slow).

Skipped unless RUN_VISION_TESTS=1: they load the actual LocateAnything-3B
weights (minutes, gigabytes) and run real inference on a synthetic screen,
so they never run in the normal suite. Quantization follows the regular
VISION_QUANTIZATION setting, so run them twice to compare:

    RUN_VISION_TESTS=1 uv run python -m unittest tests.test_vision_integration -v
    RUN_VISION_TESTS=1 VISION_QUANTIZATION=int8 uv run python -m unittest tests.test_vision_integration -v

The screen and pointer are fakes (no desktop is touched); only the model is
real. Timings are printed so runs double as a benchmark.
"""

import os
import sys
import time
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUN = os.environ.get("RUN_VISION_TESTS") == "1"

SCREEN_SIZE = (1600, 1000)
SAVE_BOX = (1200, 850, 1340, 910)  # red button, bottom-right
CANCEL_BOX = (1000, 850, 1140, 910)  # gray button next to it


def draw_fixture() -> Image.Image:
    """A synthetic desktop: toolbar, text, and two buttons bottom-right."""
    image = Image.new("RGB", SCREEN_SIZE, "#ECECEC")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(28)
    small = ImageFont.load_default(20)

    draw.rectangle((0, 0, SCREEN_SIZE[0], 44), fill="#2B2B2B")
    draw.text((20, 8), "File   Edit   View   Help", fill="white", font=small)

    draw.rectangle((60, 120, 900, 700), fill="white", outline="#B5B5B5", width=2)
    draw.text((90, 150), "Report draft", fill="#222222", font=font)

    draw.rounded_rectangle(CANCEL_BOX, radius=8, fill="#D8D8D8", outline="#9A9A9A", width=2)
    draw.text((CANCEL_BOX[0] + 28, CANCEL_BOX[1] + 14), "Cancel", fill="#333333", font=font)
    draw.rounded_rectangle(SAVE_BOX, radius=8, fill="#D02020", outline="#8E1414", width=2)
    draw.text((SAVE_BOX[0] + 38, SAVE_BOX[1] + 14), "Save", fill="white", font=font)
    return image


def overlap_ratio(box, expected) -> float:
    """Intersection area over the expected button's area."""
    left = max(box[0], expected[0])
    top = max(box[1], expected[1])
    right = min(box[2], expected[2])
    bottom = min(box[3], expected[3])
    if right <= left or bottom <= top:
        return 0.0
    expected_area = (expected[2] - expected[0]) * (expected[3] - expected[1])
    return (right - left) * (bottom - top) / expected_area


@unittest.skipUnless(RUN, "set RUN_VISION_TESTS=1 to run the real-model vision integration tests")
class VisionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    computer = None
    timings: dict[str, float] = {}

    @classmethod
    def setUpClass(cls):
        from src.llm.tools.computer import ComputerUse, NullOverlay
        from src.llm.tools.computer.fakes import RecordingPointer, StaticScreen
        from src.llm.tools.computer.locator import LocateAnythingLocator, configured_quantization

        print(f"\n[vision-integration] quantization = {configured_quantization()}", flush=True)
        cls.overlay = NullOverlay()
        cls.pointer = RecordingPointer()
        locator = LocateAnythingLocator(warmup=False)
        started = time.perf_counter()
        locator.load()
        cls.timings["model load"] = time.perf_counter() - started
        print(f"[vision-integration] model load: {cls.timings['model load']:.1f}s", flush=True)
        cls.computer = ComputerUse(
            locator=locator,
            screen=StaticScreen([draw_fixture()], SCREEN_SIZE),
            pointer=cls.pointer,
            overlay=cls.overlay,
            dpi_aware=False,
            cache_detections=False,  # every test measures a real inference
        )

    @classmethod
    def tearDownClass(cls):
        if cls.timings:
            report = " · ".join(f"{name}: {seconds:.1f}s" for name, seconds in cls.timings.items())
            print(f"[vision-integration] {report}", flush=True)

    def locate(self, name, description, **kwargs):
        started = time.perf_counter()
        detections = self.computer.locate_object(description, **kwargs)
        self.timings[name] = time.perf_counter() - started
        print(f"[vision-integration] {name}: {self.timings[name]:.1f}s -> {detections}", flush=True)
        return detections

    def test_finds_the_save_button_on_the_full_screen(self):
        detections = self.locate("full-screen locate", "the red Save button in the bottom right corner")

        self.assertTrue(detections, "the model found nothing")
        self.assertGreater(overlap_ratio(detections[0].box, SAVE_BOX), 0.25)

    def test_region_narrows_the_search_and_still_finds_the_button(self):
        detections = self.locate("region locate", "the red Save button", mode="first", region="bottom-right")

        self.assertTrue(detections, "the model found nothing inside the region")
        self.assertGreater(overlap_ratio(detections[0].box, SAVE_BOX), 0.25)

    async def test_merged_locate_and_mark_points_at_the_button_in_one_call(self):
        from src.llm.tools.computer import screen_locate, set_computer

        set_computer(self.computer)
        self.addCleanup(set_computer, None)

        started = time.perf_counter()
        result = await screen_locate.ainvoke(
            {
                "description": "the red Save button in the bottom right corner",
                "mark": True,
                "note": "Saves the draft",
                "title": "Save",
                "region": "bottom-right",
            }
        )
        type(self).timings["tool locate+mark"] = time.perf_counter() - started
        print(f"[vision-integration] tool locate+mark: {result}", flush=True)

        self.assertIn("Marked", result)
        self.assertTrue(self.overlay.markers, "no marker was drawn")
        self.assertGreater(overlap_ratio(self.overlay.markers[-1].box, SAVE_BOX), 0.25)
        self.assertEqual(self.pointer.clicks, [])


if __name__ == "__main__":
    unittest.main()
