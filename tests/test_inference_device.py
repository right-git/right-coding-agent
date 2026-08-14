import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tools.computer_use import Detection, save_annotated_image
from src.tools.computer_use import locator as locator_module
from src.tools.computer_use.locator import LocateAnythingLocator


class ModelSourceTests(unittest.TestCase):
    def test_existing_local_model_is_used_instead_of_repository_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            local_model = Path(temporary_directory)
            (local_model / "config.json").touch()
            (local_model / "model.safetensors.index.json").touch()

            source = locator_module.resolve_model_source(local_model)

        self.assertEqual(source, str(local_model))

    def test_missing_local_model_falls_back_to_the_repository_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = locator_module.resolve_model_source(Path(temporary_directory))

        self.assertEqual(source, locator_module.MODEL_ID)


class InferenceRequestTests(unittest.TestCase):
    def test_gui_description_is_wrapped_in_the_model_prompt(self):
        prompt = locator_module.build_gui_prompt("outline button")

        self.assertEqual(
            prompt,
            "Locate the region that matches the following description: outline button.",
        )

    def test_generation_is_greedy_so_the_same_screen_gives_the_same_box(self):
        options = locator_module.generation_options()

        self.assertEqual(
            options,
            {
                "max_new_tokens": 96,
                "use_cache": True,
                "generation_mode": "hybrid",
                "do_sample": False,
                "repetition_penalty": 1.1,
            },
        )
        # Sampling knobs must be absent, not merely unused: transformers warns
        # when they are set alongside do_sample=False.
        self.assertNotIn("temperature", options)
        self.assertNotIn("top_p", options)


class InferenceDeviceTests(unittest.TestCase):
    @patch.object(locator_module.torch.backends.mps, "is_available", return_value=True)
    @patch.object(locator_module.torch.cuda, "is_available", return_value=True)
    def test_cuda_is_preferred_when_available(self, _cuda_available, _mps_available):
        device, dtype = locator_module.select_inference_device()

        self.assertEqual(device, "cuda")
        self.assertIs(dtype, locator_module.torch.bfloat16)

    @patch.object(locator_module.torch.backends.mps, "is_available", return_value=True)
    @patch.object(locator_module.torch.cuda, "is_available", return_value=False)
    def test_mps_is_used_when_cuda_is_unavailable(self, _cuda_available, _mps_available):
        device, dtype = locator_module.select_inference_device()

        self.assertEqual(device, "mps")
        self.assertIs(dtype, locator_module.torch.float16)

    @patch.object(locator_module.torch.backends.mps, "is_available", return_value=False)
    @patch.object(locator_module.torch.cuda, "is_available", return_value=False)
    def test_cpu_uses_float32(self, _cuda_available, _mps_available):
        device, dtype = locator_module.select_inference_device()

        self.assertEqual(device, "cpu")
        self.assertIs(dtype, locator_module.torch.float32)


class RuntimeTests(unittest.TestCase):
    @patch.object(locator_module, "select_inference_device")
    @patch.object(locator_module.AutoModel, "from_pretrained")
    @patch.object(locator_module.AutoProcessor, "from_pretrained")
    @patch.object(locator_module.AutoTokenizer, "from_pretrained")
    def test_runtime_loads_each_model_component_once_on_the_selected_device(
        self,
        tokenizer_loader,
        processor_loader,
        model_loader,
        select_device,
    ):
        tokenizer = tokenizer_loader.return_value
        processor = processor_loader.return_value
        loaded_model = model_loader.return_value
        evaluated_model = loaded_model.to.return_value.eval.return_value
        select_device.return_value = ("cuda", locator_module.torch.bfloat16)

        runtime = locator_module.load_runtime()

        tokenizer_loader.assert_called_once_with(
            locator_module.MODEL, trust_remote_code=True
        )
        processor_loader.assert_called_once_with(
            locator_module.MODEL, trust_remote_code=True
        )
        model_loader.assert_called_once_with(
            locator_module.MODEL,
            dtype=locator_module.torch.bfloat16,
            trust_remote_code=True,
        )
        loaded_model.to.assert_called_once_with("cuda")
        loaded_model.to.return_value.eval.assert_called_once_with()
        self.assertEqual(
            runtime,
            locator_module.InferenceRuntime(
                tokenizer=tokenizer,
                processor=processor,
                model=evaluated_model,
                device="cuda",
                dtype=locator_module.torch.bfloat16,
            ),
        )

    def test_locate_scales_model_boxes_to_the_original_screenshot(self):
        class DeviceInputs(dict):
            def __init__(self, values):
                super().__init__(values)
                self.moved_to = []

            def to(self, device):
                self.moved_to.append(device)
                return self

        tokenizer = object()
        processor = MagicMock()
        model = MagicMock()
        pixel_values = MagicMock()
        cast_pixel_values = object()
        pixel_values.to.return_value = cast_pixel_values
        input_ids = object()
        attention_mask = object()
        image_grid_hws = object()
        inputs = DeviceInputs(
            {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_grid_hws": image_grid_hws,
            }
        )
        processor.py_apply_chat_template.return_value = "templated prompt"
        processor.process_vision_info.return_value = (["processed image"], None)
        processor.return_value = inputs
        model.generate.return_value = (
            "<ref>outline button</ref><box><100><200><500><600></box>"
        )
        runtime = locator_module.InferenceRuntime(
            tokenizer=tokenizer,
            processor=processor,
            model=model,
            device="cuda",
            dtype=locator_module.torch.float16,
        )
        screenshot = Image.new("RGB", (1600, 800), "white")

        detections = locator_module.locate_on_screen(
            runtime, screenshot, "outline button"
        )

        self.assertEqual(
            detections,
            [Detection(label="outline button", box=(160, 160, 800, 480))],
        )
        messages = processor.py_apply_chat_template.call_args.args[0]
        inference_image = messages[0]["content"][0]["image"]
        self.assertEqual(inference_image.size, (768, 384))
        self.assertEqual(screenshot.size, (1600, 800))
        self.assertEqual(
            messages[0]["content"][1]["text"],
            locator_module.build_gui_prompt("outline button"),
        )
        processor.py_apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor.process_vision_info.assert_called_once_with(messages)
        processor.assert_called_once_with(
            text=["templated prompt"],
            images=["processed image"],
            videos=None,
            return_tensors="pt",
        )
        self.assertEqual(inputs.moved_to, ["cuda"])
        pixel_values.to.assert_called_once_with(locator_module.torch.float16)
        model.generate.assert_called_once_with(
            pixel_values=cast_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_hws=image_grid_hws,
            tokenizer=tokenizer,
            **locator_module.generation_options(),
        )


class LocatorTests(unittest.TestCase):
    def test_loading_warms_the_model_up(self):
        loader = Mock(return_value=object())
        locator = LocateAnythingLocator(loader=loader)

        with patch.object(
            locator_module, "locate_on_screen", return_value=[]
        ) as locate_on_screen:
            locator.load()

        locate_on_screen.assert_called_once()
        self.assertEqual(locate_on_screen.call_args.args[2], "warmup")

    def test_a_failing_warm_up_does_not_break_loading(self):
        runtime = object()
        locator = LocateAnythingLocator(loader=Mock(return_value=runtime))

        with patch.object(
            locator_module, "locate_on_screen", side_effect=RuntimeError("no cuda")
        ):
            self.assertIs(locator.load(), runtime)

    def test_the_runtime_is_loaded_once_and_reused_for_every_query(self):
        runtime = object()
        loader = Mock(return_value=runtime)
        locator = LocateAnythingLocator(loader=loader, warmup=False)
        first_image = Image.new("RGB", (10, 10))
        second_image = Image.new("RGB", (20, 20))

        self.assertFalse(locator.is_loaded)
        with patch.object(
            locator_module, "locate_on_screen", return_value=[]
        ) as locate_on_screen:
            locator.locate(first_image, "first")
            locator.locate(second_image, "second")

        loader.assert_called_once_with()
        self.assertTrue(locator.is_loaded)
        self.assertEqual(
            [call.args for call in locate_on_screen.call_args_list],
            [(runtime, first_image, "first"), (runtime, second_image, "second")],
        )
        self.assertEqual(
            locate_on_screen.call_args.kwargs,
            {"max_image_side": locator_module.MAX_IMAGE_SIDE},
        )

    def test_an_injected_runtime_is_never_reloaded(self):
        loader = Mock()
        locator = LocateAnythingLocator(runtime=object(), loader=loader)

        locator.load()

        loader.assert_not_called()


class AnnotatedImageTests(unittest.TestCase):
    def test_save_annotated_image_preserves_source_and_dimensions(self):
        source = Image.new("RGB", (120, 80), "white")
        source_pixels = source.tobytes()
        detections = [
            Detection(label="button", box=(10, 15, 70, 50)),
            Detection(label="icon", box=(80, 15, 110, 60)),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "annotated.jpg"

            saved_path = save_annotated_image(source, detections, output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.is_file())
            with Image.open(output_path) as saved:
                self.assertEqual(saved.format, "JPEG")
                self.assertEqual(saved.size, source.size)
                saved_rgb = saved.convert("RGB")
                self.assertNotEqual(saved_rgb.getpixel((10, 15)), (255, 255, 255))
                self.assertNotEqual(saved_rgb.getpixel((80, 15)), (255, 255, 255))
            self.assertEqual(source.tobytes(), source_pixels)

    def test_save_annotated_image_writes_valid_jpeg_without_detections(self):
        source = Image.new("RGB", (32, 24), "navy")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "unannotated.jpg"

            save_annotated_image(source, [], output_path)

            with Image.open(output_path) as saved:
                saved.load()
                self.assertEqual(saved.format, "JPEG")
                self.assertEqual(saved.size, source.size)


if __name__ == "__main__":
    unittest.main()
