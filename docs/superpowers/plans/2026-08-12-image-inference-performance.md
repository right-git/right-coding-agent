# Image Inference Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce LocateAnything-3B inference work by sending a maximum-768-pixel image to the model and selecting fast decoding while retaining the original-resolution annotated output.

**Architecture:** Put pure resizing and normalized-coordinate scaling in `image_optimization.py`. Refactor `test.py` behind `main()` so tests can import its module without loading the model; retain a full-resolution image for drawing and pass a resized copy only to the processor.

**Tech Stack:** Python 3.12, Pillow, PyTorch, Transformers, pytest, uv

---

## File structure

- Create `image_optimization.py`: pure Pillow resizing and coordinate-scaling helpers.
- Create `tests/test_image_optimization.py`: focused behavior tests for resizing, scaling, and import safety.
- Modify `test.py`: orchestration, model loading, inference, parsing, and original-resolution rendering.

### Task 1: Add the pure image optimization helpers

**Files:**
- Create: `tests/test_image_optimization.py`
- Create: `image_optimization.py`

- [x] **Step 1: Write failing resize and coordinate tests**

Create `tests/test_image_optimization.py` with:

```python
from PIL import Image
import pytest

from image_optimization import resize_for_inference, scale_normalized_box


@pytest.mark.parametrize(
    ("source_size", "expected_size"),
    [
        ((1200, 800), (768, 512)),
        ((800, 1200), (512, 768)),
        ((640, 480), (640, 480)),
    ],
)
def test_resize_for_inference_limits_longest_side_without_upscaling(
    source_size, expected_size
):
    source = Image.new("RGB", source_size, "white")

    resized = resize_for_inference(source, max_side=768)

    assert resized.size == expected_size
    assert source.size == source_size
    assert resized is not source


def test_resize_for_inference_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="max_side must be positive"):
        resize_for_inference(Image.new("RGB", (10, 10)), max_side=0)


def test_scale_normalized_box_uses_original_image_dimensions():
    assert scale_normalized_box((100, 200, 900, 800), (3024, 1964)) == pytest.approx(
        (302.4, 392.8, 2721.6, 1571.2)
    )
```

- [x] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
uv run --with pytest python -m pytest tests/test_image_optimization.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'image_optimization'`.

- [x] **Step 3: Implement the minimal pure helpers**

Create `image_optimization.py` with:

```python
from collections.abc import Sequence

from PIL import Image


def resize_for_inference(image: Image.Image, max_side: int = 768) -> Image.Image:
    if max_side <= 0:
        raise ValueError("max_side must be positive")

    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def scale_normalized_box(
    box: Sequence[int], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width, height = image_size
    return (
        x1 / 1000 * width,
        y1 / 1000 * height,
        x2 / 1000 * width,
        y2 / 1000 * height,
    )
```

- [x] **Step 4: Run the focused helper tests**

Run:

```bash
uv run --with pytest python -m pytest tests/test_image_optimization.py -q
```

Expected: `5 passed`.

### Task 2: Make `test.py` import-safe and use the fast inference image

**Files:**
- Modify: `tests/test_image_optimization.py`
- Modify: `test.py:1-89`

- [x] **Step 1: Add a failing import-safety test**

Append this test to `tests/test_image_optimization.py`:

```python
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def test_importing_script_does_not_load_the_model(monkeypatch):
    class ForbiddenLoader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("model loading ran during import")

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = ForbiddenLoader
    fake_transformers.AutoTokenizer = ForbiddenLoader
    fake_transformers.AutoProcessor = ForbiddenLoader
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    script_path = Path(__file__).parents[1] / "test.py"
    spec = importlib.util.spec_from_file_location("image_inference_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
```

- [x] **Step 2: Run the import-safety test and verify the expected failure**

Run:

```bash
uv run --with pytest python -m pytest tests/test_image_optimization.py::test_importing_script_does_not_load_the_model -q
```

Expected: failure with `AssertionError: model loading ran during import`.

- [x] **Step 3: Refactor `test.py` behind `main()` and wire in optimization**

Add these imports and constants to `test.py`:

```python
from PIL import Image, ImageDraw, ImageFont, ImageOps

from image_optimization import resize_for_inference, scale_normalized_box

MAX_IMAGE_SIDE = 768
```

Move execution into `main()` and prepare both image sizes before model preprocessing:

```python
def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else "example.jpg"
    prompt = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "Locate all the instances that matches the following description: {}"
    )

    with Image.open(image_path) as source:
        img = ImageOps.exif_transpose(source).convert("RGB")
    inference_img = resize_for_inference(img, MAX_IMAGE_SIDE)
    print(f"Image: {img.width}x{img.height} -> {inference_img.width}x{inference_img.height}")
```

Use `inference_img` in the processor message. Keep `img` for drawing and saving. Set the model call to:

```python
response = model.generate(
    pixel_values=inputs["pixel_values"].to(dtype),
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    image_grid_hws=inputs.get("image_grid_hws"),
    tokenizer=tokenizer,
    max_new_tokens=2048,
    use_cache=True,
    generation_mode="fast",
    do_sample=False,
)
```

Replace inline coordinate arithmetic with:

```python
"box": scale_normalized_box((x1, y1, x2, y2), img.size),
```

Finish the module with:

```python
if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run the focused test file**

Run:

```bash
uv run --with pytest python -m pytest tests/test_image_optimization.py -q
```

Expected: `6 passed`.

### Task 3: Verify correctness and measure the optimized workload

**Files:**
- Verify: `image_optimization.py`
- Verify: `test.py`
- Verify: `tests/test_image_optimization.py`

- [x] **Step 1: Compile the modified Python files**

Run:

```bash
uv run python -m py_compile test.py image_optimization.py tests/test_image_optimization.py
```

Expected: exit code 0 with no output.

- [x] **Step 2: Run the complete project test suite**

Run:

```bash
uv run --with pytest python -m pytest -q
```

Expected: all tests pass with no new failures.

- [x] **Step 3: Measure processor workload at both sizes**

Run a local benchmark that loads `screen.png`, prepares the original and 768-pixel copies with the cached model processor, and reports resize time, processor time, patch count, and visual-token count.

Expected for the current 3024x1964 sample: the original path produces roughly 6,500 visual tokens while the optimized path produces roughly 504 visual tokens, a reduction of about 92%.

- [x] **Step 4: Run one real optimized inference**

Run:

```bash
time uv run python test.py screen.png
```

Expected: the script logs `3024x1964 -> 768x499`, completes inference, and saves `output_boxes.jpg` at 3024x1964. Record elapsed time and parsed-box count. If the model does not complete within a practical window, stop it and report the measured phase and elapsed time without claiming an end-to-end result.

- [x] **Step 5: Inspect the final diff without committing**

Run:

```bash
git diff --check
git status --short
git diff -- test.py image_optimization.py tests/test_image_optimization.py docs/superpowers/specs/2026-08-12-image-inference-performance-design.md docs/superpowers/plans/2026-08-12-image-inference-performance.md
```

Expected: no whitespace errors; only the requested uncommitted files and the user's pre-existing changes remain.
