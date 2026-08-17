import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from image_optimization import resize_for_inference, scale_normalized_box


@pytest.mark.parametrize(
    ("source_size", "expected_size"),
    [
        ((1200, 800), (768, 512)),
        ((800, 1200), (512, 768)),
        ((640, 480), (640, 480)),
    ],
)
def test_resize_for_inference_limits_longest_side_without_upscaling(source_size, expected_size):
    source = Image.new("RGB", source_size, "white")

    resized = resize_for_inference(source, max_side=768)

    assert resized.size == expected_size
    assert source.size == source_size
    assert resized is not source


def test_resize_for_inference_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="max_side must be positive"):
        resize_for_inference(Image.new("RGB", (10, 10)), max_side=0)


def test_scale_normalized_box_uses_original_image_dimensions():
    assert scale_normalized_box((100, 200, 900, 800), (3024, 1964)) == pytest.approx((302.4, 392.8, 2721.6, 1571.2))


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
