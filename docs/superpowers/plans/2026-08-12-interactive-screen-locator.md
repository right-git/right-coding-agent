# Interactive Screen Locator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `test.py` load LocateAnything once, repeatedly capture the Windows primary screen, locate a requested UI region, print box centers, save an annotated screenshot, and move the pointer without clicking.

**Architecture:** Add a focused `screen_locator.py` module for pure detection geometry, loop control, and thin Windows adapters. Keep model construction, model inference, and image annotation in `test.py`; its `main()` wires those operations into the reusable loop so tests can substitute harmless callables for screen capture and pointer movement.

**Tech Stack:** Python 3.12, Pillow, standard-library `ctypes`, PyTorch, Transformers, `unittest`, `unittest.mock`, uv

---

## File Structure

- Create `screen_locator.py`: detection value object, model-output parsing, center/target selection, interactive command loop, primary-screen capture, DPI setup, and pointer movement.
- Create `tests/test_screen_locator.py`: unit tests for parsing, geometry, modes, loop behavior, and Windows adapter contracts without touching the real desktop or mouse.
- Modify `test.py`: extract persistent runtime loading, one-query inference, annotation, and a small `main()` composition root.
- Modify `tests/test_inference_device.py`: retain existing model configuration tests and add integration-contract tests for one-time runtime initialization and annotated output.

### Task 1: Detection Parsing and Target Geometry

**Files:**
- Create: `screen_locator.py`
- Create: `tests/test_screen_locator.py`

- [ ] **Step 1: Write the failing pure-function tests**

Create `tests/test_screen_locator.py` with these tests:

```python
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screen_locator import (
    Detection,
    box_center,
    parse_detections,
    select_targets,
)


class DetectionParsingTests(unittest.TestCase):
    def test_parses_reference_label_and_box_in_screenshot_coordinates(self):
        detections = parse_detections(
            "<ref>outline button</ref><box><100><200><300><400></box>",
            (1920, 1080),
        )

        self.assertEqual(
            detections,
            [Detection(label="outline button", box=(192, 216, 576, 432))],
        )

    def test_parses_multiple_boxes_in_model_order(self):
        detections = parse_detections(
            "<ref>item</ref><box><0><0><100><100></box>"
            "<box><500><500><1000><1000></box>",
            (1000, 800),
        )

        self.assertEqual(
            detections,
            [
                Detection(label="item", box=(0, 0, 100, 80)),
                Detection(label="item", box=(500, 400, 1000, 800)),
            ],
        )

    def test_ignores_malformed_box(self):
        detections = parse_detections(
            "<ref>broken</ref><box><10><20><bad><40></box>",
            (1000, 1000),
        )

        self.assertEqual(detections, [])


class TargetGeometryTests(unittest.TestCase):
    def test_box_center_returns_integer_midpoint(self):
        self.assertEqual(box_center((10, 20, 30, 50), (100, 100)), (20, 35))

    def test_box_center_is_clamped_to_screenshot(self):
        self.assertEqual(box_center((-40, -20, 240, 220), (100, 80)), (99, 79))

    def test_first_mode_selects_only_first_detection(self):
        detections = [
            Detection("first", (0, 0, 10, 10)),
            Detection("second", (10, 10, 20, 20)),
        ]

        self.assertEqual(select_targets(detections, "first"), detections[:1])

    def test_all_mode_preserves_detection_order(self):
        detections = [
            Detection("first", (0, 0, 10, 10)),
            Detection("second", (10, 10, 20, 20)),
        ]

        self.assertEqual(select_targets(detections, "all"), detections)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported target mode"):
            select_targets([], "nearest")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the expected RED state**

Run:

```powershell
uv run --frozen python -m unittest discover -s tests -p "test_screen_locator.py" -v
```

Expected: import failure `ModuleNotFoundError: No module named 'screen_locator'`.

- [ ] **Step 3: Implement the detection primitives**

Create `screen_locator.py`:

```python
import re
from dataclasses import dataclass
from typing import Literal, Sequence

from image_optimization import scale_normalized_box


TargetMode = Literal["first", "all"]
Box = tuple[int, int, int, int]
Point = tuple[int, int]

REFERENCE_GROUP = re.compile(
    r"(?:<ref>(?P<label>[^<]+)</ref>\s*)?"
    r"(?P<boxes>(?:<box><-?\d+><-?\d+><-?\d+><-?\d+></box>\s*)+)"
)
BOX_PATTERN = re.compile(
    r"<box><(-?\d+)><(-?\d+)><(-?\d+)><(-?\d+)></box>"
)


@dataclass(frozen=True)
class Detection:
    label: str
    box: Box


def parse_detections(answer: str, image_size: tuple[int, int]) -> list[Detection]:
    detections: list[Detection] = []
    for group in REFERENCE_GROUP.finditer(answer):
        label = (group.group("label") or "object").strip()
        for match in BOX_PATTERN.finditer(group.group("boxes")):
            normalized_box = tuple(int(value) for value in match.groups())
            scaled_box = scale_normalized_box(normalized_box, image_size)
            detections.append(
                Detection(label=label, box=tuple(round(value) for value in scaled_box))
            )
    return detections


def box_center(box: Box, image_size: tuple[int, int]) -> Point:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = box
    center_x = min(max((x1 + x2) // 2, 0), width - 1)
    center_y = min(max((y1 + y2) // 2, 0), height - 1)
    return center_x, center_y


def select_targets(
    detections: Sequence[Detection], mode: TargetMode
) -> list[Detection]:
    if mode == "first":
        return list(detections[:1])
    if mode == "all":
        return list(detections)
    raise ValueError(f"Unsupported target mode: {mode}")
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
uv run --frozen python -m unittest discover -s tests -p "test_screen_locator.py" -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit only the Task 1 files**

```powershell
git add -- screen_locator.py tests/test_screen_locator.py
git commit -m "feat: add screen detection geometry"
```

### Task 2: Interactive Query Loop and Mode Commands

**Files:**
- Modify: `screen_locator.py`
- Modify: `tests/test_screen_locator.py`

- [ ] **Step 1: Add failing command and loop tests**

Add these imports to `tests/test_screen_locator.py`:

```python
from unittest.mock import Mock, call

from PIL import Image
```

Add `parse_mode_command` and `run_interactive_loop` to its `screen_locator` import, then append:

```python
class ModeCommandTests(unittest.TestCase):
    def test_valid_mode_command_returns_selected_mode(self):
        self.assertEqual(parse_mode_command(":mode all"), "all")

    def test_invalid_mode_command_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Use :mode first or :mode all"):
            parse_mode_command(":mode nearest")


class InteractiveLoopTests(unittest.TestCase):
    @staticmethod
    def input_from(*values):
        commands = iter(values)
        return lambda _prompt: next(commands)

    def test_each_query_uses_a_fresh_screenshot(self):
        first_image = Image.new("RGB", (100, 100))
        second_image = Image.new("RGB", (200, 100))
        capture = Mock(side_effect=[first_image, second_image])
        locate = Mock(return_value=[])

        run_interactive_loop(
            locate=locate,
            capture_screen=capture,
            move_pointer=Mock(),
            save_result=Mock(),
            input_fn=self.input_from("first query", "second query", "exit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        self.assertEqual(capture.call_count, 2)
        self.assertEqual(
            locate.call_args_list,
            [call(first_image, "first query"), call(second_image, "second query")],
        )

    def test_default_mode_moves_only_to_first_center(self):
        image = Image.new("RGB", (100, 100))
        move = Mock()

        run_interactive_loop(
            locate=Mock(
                return_value=[
                    Detection("first", (10, 10, 30, 30)),
                    Detection("second", (40, 40, 60, 60)),
                ]
            ),
            capture_screen=Mock(return_value=image),
            move_pointer=move,
            save_result=Mock(),
            input_fn=self.input_from("button", "quit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        move.assert_called_once_with(20, 20)

    def test_all_mode_moves_to_every_center_in_order(self):
        image = Image.new("RGB", (100, 100))
        move = Mock()
        pause = Mock()

        run_interactive_loop(
            locate=Mock(
                return_value=[
                    Detection("first", (10, 10, 30, 30)),
                    Detection("second", (40, 40, 60, 60)),
                ]
            ),
            capture_screen=Mock(return_value=image),
            move_pointer=move,
            save_result=Mock(),
            input_fn=self.input_from(":mode all", "buttons", "exit"),
            output_fn=Mock(),
            pause_fn=pause,
        )

        self.assertEqual(move.call_args_list, [call(20, 20), call(50, 50)])
        pause.assert_called_once_with(0.35)

    def test_no_detection_saves_image_without_moving_pointer(self):
        image = Image.new("RGB", (100, 100))
        move = Mock()
        save = Mock()

        run_interactive_loop(
            locate=Mock(return_value=[]),
            capture_screen=Mock(return_value=image),
            move_pointer=move,
            save_result=save,
            input_fn=self.input_from("missing", "exit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        save.assert_called_once_with(image, [])
        move.assert_not_called()

    def test_empty_input_does_not_capture_screen(self):
        capture = Mock()

        run_interactive_loop(
            locate=Mock(),
            capture_screen=capture,
            move_pointer=Mock(),
            save_result=Mock(),
            input_fn=self.input_from("   ", "exit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        capture.assert_not_called()

    def test_prints_every_box_and_center(self):
        image = Image.new("RGB", (100, 100))
        output = Mock()

        run_interactive_loop(
            locate=Mock(return_value=[Detection("button", (10, 20, 30, 40))]),
            capture_screen=Mock(return_value=image),
            move_pointer=Mock(),
            save_result=Mock(),
            input_fn=self.input_from("button", "exit"),
            output_fn=output,
            pause_fn=Mock(),
        )

        output.assert_any_call("1. button: box=(10, 20, 30, 40), center=(20, 30)")

    def test_invalid_mode_command_keeps_default_first_mode(self):
        image = Image.new("RGB", (100, 100))
        move = Mock()

        run_interactive_loop(
            locate=Mock(
                return_value=[
                    Detection("first", (10, 10, 30, 30)),
                    Detection("second", (40, 40, 60, 60)),
                ]
            ),
            capture_screen=Mock(return_value=image),
            move_pointer=move,
            save_result=Mock(),
            input_fn=self.input_from(":mode nearest", "buttons", "exit"),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        move.assert_called_once_with(20, 20)

    def test_keyboard_interrupt_stops_without_capturing(self):
        capture = Mock()

        run_interactive_loop(
            locate=Mock(),
            capture_screen=capture,
            move_pointer=Mock(),
            save_result=Mock(),
            input_fn=Mock(side_effect=KeyboardInterrupt),
            output_fn=Mock(),
            pause_fn=Mock(),
        )

        capture.assert_not_called()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
uv run --frozen python -m unittest discover -s tests -p "test_screen_locator.py" -v
```

Expected: import failure because `parse_mode_command` and `run_interactive_loop` do not exist.

- [ ] **Step 3: Implement command parsing and the injected loop**

Extend the imports in `screen_locator.py`:

```python
import time
from collections.abc import Callable

from PIL import Image
```

Append:

```python
Locate = Callable[[Image.Image, str], Sequence[Detection]]
CaptureScreen = Callable[[], Image.Image]
MovePointer = Callable[[int, int], None]
SaveResult = Callable[[Image.Image, Sequence[Detection]], None]

EXIT_COMMANDS = {"exit", "quit"}
TARGET_MODES = {"first", "all"}


def parse_mode_command(command: str) -> TargetMode:
    parts = command.strip().lower().split()
    if len(parts) == 2 and parts[0] == ":mode" and parts[1] in TARGET_MODES:
        return parts[1]  # type: ignore[return-value]
    raise ValueError("Use :mode first or :mode all")


def run_interactive_loop(
    *,
    locate: Locate,
    capture_screen: CaptureScreen,
    move_pointer: MovePointer,
    save_result: SaveResult,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    pause_fn: Callable[[float], None] = time.sleep,
) -> None:
    mode: TargetMode = "first"
    output_fn("Commands: :mode first, :mode all, exit")

    while True:
        try:
            command = input_fn(f"Что найти? [{mode}] ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("Stopped.")
            return

        if not command:
            continue
        if command.lower() in EXIT_COMMANDS:
            return
        if command.lower().startswith(":mode"):
            try:
                mode = parse_mode_command(command)
                output_fn(f"Mode: {mode}")
            except ValueError as error:
                output_fn(str(error))
            continue

        try:
            screenshot = capture_screen()
            detections = list(locate(screenshot, command))
            save_result(screenshot, detections)

            if not detections:
                output_fn("Nothing found; pointer was not moved.")
                continue

            for index, detection in enumerate(detections, start=1):
                center = box_center(detection.box, screenshot.size)
                output_fn(
                    f"{index}. {detection.label}: "
                    f"box={detection.box}, center={center}"
                )

            targets = select_targets(detections, mode)
            for index, detection in enumerate(targets):
                move_pointer(*box_center(detection.box, screenshot.size))
                if index < len(targets) - 1:
                    pause_fn(0.35)
        except Exception as error:
            output_fn(f"Query failed: {error}")
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
uv run --frozen python -m unittest discover -s tests -p "test_screen_locator.py" -v
```

Expected: 18 tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- screen_locator.py tests/test_screen_locator.py
git commit -m "feat: add interactive locator loop"
```

### Task 3: Windows Screenshot and Pointer Adapters

**Files:**
- Modify: `screen_locator.py`
- Modify: `tests/test_screen_locator.py`

- [ ] **Step 1: Add failing adapter tests**

Add `capture_primary_screen`, `enable_dpi_awareness`, and `move_pointer` to the imports from `screen_locator`. Add these tests:

```python
class WindowsAdapterTests(unittest.TestCase):
    def test_capture_uses_only_primary_screen_and_returns_rgb(self):
        source = Image.new("RGBA", (20, 10))
        grabber = Mock(return_value=source)

        screenshot = capture_primary_screen(grabber=grabber)

        grabber.assert_called_once_with(all_screens=False)
        self.assertEqual(screenshot.mode, "RGB")
        self.assertEqual(screenshot.size, (20, 10))

    def test_move_pointer_calls_windows_position_api(self):
        setter = Mock(return_value=1)

        move_pointer(25, 40, set_cursor_position=setter)

        setter.assert_called_once_with(25, 40)

    def test_move_pointer_reports_windows_api_failure(self):
        with self.assertRaisesRegex(OSError, "SetCursorPos failed"):
            move_pointer(25, 40, set_cursor_position=Mock(return_value=0))

    def test_dpi_awareness_prefers_per_monitor_api(self):
        shcore = Mock()
        user32 = Mock()
        libraries = Mock(shcore=shcore, user32=user32)

        enable_dpi_awareness(platform="win32", libraries=libraries)

        shcore.SetProcessDpiAwareness.assert_called_once_with(2)
        user32.SetProcessDPIAware.assert_not_called()

    def test_dpi_awareness_falls_back_for_older_windows(self):
        shcore = Mock()
        shcore.SetProcessDpiAwareness.side_effect = OSError("unsupported")
        user32 = Mock()
        libraries = Mock(shcore=shcore, user32=user32)

        enable_dpi_awareness(platform="win32", libraries=libraries)

        user32.SetProcessDPIAware.assert_called_once_with()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
uv run --frozen python -m unittest discover -s tests -p "test_screen_locator.py" -v
```

Expected: import failure because the three adapter functions do not exist.

- [ ] **Step 3: Implement the dependency-free Windows adapters**

Extend `screen_locator.py` imports:

```python
import ctypes
import sys
from typing import Any

from PIL import Image, ImageGrab
```

Append:

```python
def enable_dpi_awareness(
    *, platform: str = sys.platform, libraries: Any | None = None
) -> None:
    if platform != "win32":
        return
    if libraries is None:
        libraries = ctypes.windll
    try:
        libraries.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        libraries.user32.SetProcessDPIAware()


def capture_primary_screen(
    *, grabber: Callable[..., Image.Image] = ImageGrab.grab
) -> Image.Image:
    return grabber(all_screens=False).convert("RGB")


def move_pointer(
    x: int,
    y: int,
    *,
    set_cursor_position: Callable[[int, int], int] | None = None,
) -> None:
    if set_cursor_position is None:
        if sys.platform != "win32":
            raise RuntimeError("Pointer movement is supported only on Windows")
        set_cursor_position = ctypes.windll.user32.SetCursorPos
    if not set_cursor_position(int(x), int(y)):
        raise OSError("SetCursorPos failed")
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
uv run --frozen python -m unittest discover -s tests -p "test_screen_locator.py" -v
```

Expected: 23 tests pass, with no real screenshot capture or pointer movement.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- screen_locator.py tests/test_screen_locator.py
git commit -m "feat: add Windows screen and pointer adapters"
```

### Task 4: Persistent Model Runtime and Application Wiring

**Files:**
- Modify: `test.py`
- Modify: `tests/test_inference_device.py`

- [ ] **Step 1: Add failing integration-contract tests**

Add these imports to `tests/test_inference_device.py`:

```python
from unittest.mock import Mock

from PIL import Image

from screen_locator import Detection
```

Append:

```python
class InteractiveApplicationTests(unittest.TestCase):
    @patch.object(inference_script, "run_interactive_loop")
    @patch.object(inference_script, "load_runtime")
    @patch.object(inference_script, "enable_dpi_awareness")
    def test_main_loads_runtime_once_before_starting_loop(
        self, enable_dpi, load_runtime, run_loop
    ):
        runtime = object()
        load_runtime.return_value = runtime

        inference_script.main()

        enable_dpi.assert_called_once_with()
        load_runtime.assert_called_once_with()
        run_loop.assert_called_once()

    def test_annotated_result_is_saved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.jpg"
            image = Image.new("RGB", (100, 100), "white")

            inference_script.save_annotated_image(
                image,
                [Detection("button", (10, 20, 40, 50))],
                output,
            )

            self.assertTrue(output.is_file())
            with Image.open(output) as result:
                self.assertEqual(result.size, (100, 100))
```

- [ ] **Step 2: Run the integration tests and verify RED**

Run:

```powershell
uv run --frozen python -m unittest discover -s tests -p "test_inference_device.py" -v
```

Expected: failure because `run_interactive_loop`, `load_runtime`, `enable_dpi_awareness`, and `save_annotated_image` are not exposed by `test.py`.

- [ ] **Step 3: Replace one-shot execution with persistent runtime wiring**

Update `test.py` so its imports, constants, helpers, and entry point are exactly:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from image_optimization import resize_for_inference
from screen_locator import (
    Detection,
    capture_primary_screen,
    enable_dpi_awareness,
    move_pointer,
    parse_detections,
    run_interactive_loop,
)


MODEL_ID = "nvidia/LocateAnything-3B"
OUTPUT_PATH = Path("output_boxes.jpg")
MAX_IMAGE_SIDE = 768
COLORS = ["#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE", "#00C7BE"]


def resolve_model_source(local_model: Path | None = None) -> str:
    if local_model is None:
        local_model = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--nvidia--LocateAnything-3B"
        )
    required_files = ("config.json", "model.safetensors.index.json")
    if all((local_model / filename).is_file() for filename in required_files):
        return str(local_model)
    return MODEL_ID


MODEL = resolve_model_source()


def build_gui_prompt(description: str) -> str:
    description = description.strip().rstrip(".")
    return f"Locate the region that matches the following description: {description}."


def generation_options() -> dict[str, object]:
    return {
        "max_new_tokens": 2048,
        "use_cache": True,
        "generation_mode": "hybrid",
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }


def select_inference_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@dataclass
class InferenceRuntime:
    tokenizer: Any
    processor: Any
    model: Any
    device: str
    dtype: torch.dtype


def load_runtime() -> InferenceRuntime:
    device, dtype = select_inference_device()
    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL, dtype=dtype, trust_remote_code=True)
        .to(device)
        .eval()
    )
    return InferenceRuntime(tokenizer, processor, model, device, dtype)


def locate_on_screen(
    runtime: InferenceRuntime, image: Image.Image, description: str
) -> list[Detection]:
    inference_image = resize_for_inference(image, MAX_IMAGE_SIDE)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": inference_image},
                {"type": "text", "text": build_gui_prompt(description)},
            ],
        }
    ]
    text = runtime.processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = runtime.processor.process_vision_info(messages)
    inputs = runtime.processor(
        text=[text], images=images, videos=videos, return_tensors="pt"
    ).to(runtime.device)

    with torch.inference_mode():
        response = runtime.model.generate(
            pixel_values=inputs["pixel_values"].to(runtime.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=runtime.tokenizer,
            **generation_options(),
        )

    answer = response[0] if isinstance(response, tuple) else response
    print(f"Raw output:\n{answer}")
    return parse_detections(answer, image.size)


def save_annotated_image(
    image: Image.Image,
    detections: Sequence[Detection],
    output_path: Path = OUTPUT_PATH,
) -> None:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, image.width // 60))
    except OSError:
        font = ImageFont.load_default()
    line_width = max(2, image.width // 400)
    label_colors: dict[str, str] = {}

    for detection in detections:
        color = label_colors.setdefault(
            detection.label, COLORS[len(label_colors) % len(COLORS)]
        )
        x1, y1, x2, y2 = detection.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        text_box = draw.textbbox((0, 0), detection.label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = y1 - text_height - 6 if y1 - text_height - 6 > 0 else y1 + 2
        draw.rectangle(
            [x1, text_y, x1 + text_width + 8, text_y + text_height + 6],
            fill=color,
        )
        draw.text((x1 + 4, text_y + 2), detection.label, fill="white", font=font)

    annotated.save(output_path, quality=95)
    print(f"Saved: {output_path}")


def main() -> None:
    enable_dpi_awareness()
    runtime = load_runtime()
    run_interactive_loop(
        locate=lambda image, description: locate_on_screen(
            runtime, image, description
        ),
        capture_screen=capture_primary_screen,
        move_pointer=move_pointer,
        save_result=save_annotated_image,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the inference and loop tests and verify GREEN**

Run:

```powershell
uv run --frozen python -m unittest discover -s tests -p "test_inference_device.py" -v
uv run --frozen python -m unittest discover -s tests -p "test_screen_locator.py" -v
```

Expected: all tests in both files pass; no model is loaded and no real pointer movement occurs during tests.

- [ ] **Step 5: Run the complete regression suite**

Run:

```powershell
$env:OPENAI_API_KEY='test-key'
$env:OPENAI_API_BASE='http://localhost'
uv run --frozen python -m unittest discover -s tests -v
```

Expected: all project tests pass.

- [ ] **Step 6: Perform a real Windows smoke test**

Run:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
uv run --frozen python test.py
```

At the prompt, enter `outline button`. Expected: a fresh screenshot is captured; one or more boxes and centers are printed; `output_boxes.jpg` is updated; the pointer moves to the first center without clicking. Then enter `:mode all`, submit another description, verify that every reported center is visited in order, and enter `exit`.

- [ ] **Step 7: Commit Task 4 without staging unrelated workspace files**

```powershell
git add -- test.py tests/test_inference_device.py
git commit -m "feat: run screen locator as persistent session"
```

### Task 5: Final Verification and Documentation Check

**Files:**
- Verify: `screen_locator.py`
- Verify: `test.py`
- Verify: `tests/test_screen_locator.py`
- Verify: `tests/test_inference_device.py`
- Verify: `docs/superpowers/specs/2026-08-12-interactive-screen-locator-design.md`

- [ ] **Step 1: Check formatting and whitespace errors**

Run:

```powershell
git diff --check c24c7db..HEAD
```

Expected: no output.

- [ ] **Step 2: Confirm the feature adds no package dependency**

Run:

```powershell
git diff c24c7db..HEAD -- pyproject.toml uv.lock
```

Expected: no feature-related diff; the locator uses Pillow and standard-library `ctypes` already present in the environment.

- [ ] **Step 3: Run the final regression suite once more**

Run:

```powershell
$env:OPENAI_API_KEY='test-key'
$env:OPENAI_API_BASE='http://localhost'
uv run --frozen python -m unittest discover -s tests -v
```

Expected: the full suite passes.
