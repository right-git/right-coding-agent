import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test as inference_script
from screen_locator import Detection


class ModelSourceTests(unittest.TestCase):
    def test_existing_local_model_is_used_instead_of_repository_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            local_model = Path(temporary_directory)
            (local_model / "config.json").touch()
            (local_model / "model.safetensors.index.json").touch()

            source = inference_script.resolve_model_source(local_model)

        self.assertEqual(source, str(local_model))


class InferenceRequestTests(unittest.TestCase):
    def test_gui_description_is_wrapped_in_the_model_prompt(self):
        prompt = inference_script.build_gui_prompt("outline button")

        self.assertEqual(
            prompt,
            "Locate the region that matches the following description: outline button.",
        )

    def test_generation_options_use_the_stable_reference_configuration(self):
        self.assertEqual(
            inference_script.generation_options(),
            {
                "max_new_tokens": 2048,
                "use_cache": True,
                "generation_mode": "hybrid",
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            },
        )


class InferenceDeviceTests(unittest.TestCase):
    @patch.object(inference_script.torch.backends.mps, "is_available", return_value=True)
    @patch.object(inference_script.torch.cuda, "is_available", return_value=True)
    def test_cuda_is_preferred_when_available(self, _cuda_available, _mps_available):
        device, dtype = inference_script.select_inference_device()

        self.assertEqual(device, "cuda")
        self.assertIs(dtype, inference_script.torch.bfloat16)

    @patch.object(inference_script.torch.backends.mps, "is_available", return_value=True)
    @patch.object(inference_script.torch.cuda, "is_available", return_value=False)
    def test_mps_is_used_when_cuda_is_unavailable(self, _cuda_available, _mps_available):
        device, dtype = inference_script.select_inference_device()

        self.assertEqual(device, "mps")
        self.assertIs(dtype, inference_script.torch.float16)

    @patch.object(inference_script.torch.backends.mps, "is_available", return_value=False)
    @patch.object(inference_script.torch.cuda, "is_available", return_value=False)
    def test_cpu_uses_float32(self, _cuda_available, _mps_available):
        device, dtype = inference_script.select_inference_device()

        self.assertEqual(device, "cpu")
        self.assertIs(dtype, inference_script.torch.float32)


class RuntimeTests(unittest.TestCase):
    @patch.object(inference_script, "select_inference_device")
    @patch.object(inference_script.AutoModel, "from_pretrained")
    @patch.object(inference_script.AutoProcessor, "from_pretrained")
    @patch.object(inference_script.AutoTokenizer, "from_pretrained")
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
        select_device.return_value = ("cuda", inference_script.torch.bfloat16)

        runtime = inference_script.load_runtime()

        tokenizer_loader.assert_called_once_with(
            inference_script.MODEL, trust_remote_code=True
        )
        processor_loader.assert_called_once_with(
            inference_script.MODEL, trust_remote_code=True
        )
        model_loader.assert_called_once_with(
            inference_script.MODEL,
            dtype=inference_script.torch.bfloat16,
            trust_remote_code=True,
        )
        loaded_model.to.assert_called_once_with("cuda")
        loaded_model.to.return_value.eval.assert_called_once_with()
        self.assertEqual(
            runtime,
            inference_script.InferenceRuntime(
                tokenizer=tokenizer,
                processor=processor,
                model=evaluated_model,
                device="cuda",
                dtype=inference_script.torch.bfloat16,
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
        runtime = inference_script.InferenceRuntime(
            tokenizer=tokenizer,
            processor=processor,
            model=model,
            device="cuda",
            dtype=inference_script.torch.float16,
        )
        screenshot = Image.new("RGB", (1600, 800), "white")

        detections = inference_script.locate_on_screen(
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
            inference_script.build_gui_prompt("outline button"),
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
        pixel_values.to.assert_called_once_with(inference_script.torch.float16)
        model.generate.assert_called_once_with(
            pixel_values=cast_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_hws=image_grid_hws,
            tokenizer=tokenizer,
            **inference_script.generation_options(),
        )


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

            inference_script.save_annotated_image(source, detections, output_path)

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

            inference_script.save_annotated_image(source, [], output_path)

            with Image.open(output_path) as saved:
                saved.load()
                self.assertEqual(saved.format, "JPEG")
                self.assertEqual(saved.size, source.size)


class ApplicationCompositionTests(unittest.TestCase):
    @patch.object(inference_script, "run_interactive_loop")
    @patch.object(inference_script, "load_runtime")
    @patch.object(inference_script, "enable_dpi_awareness")
    def test_main_loads_one_runtime_and_reuses_it_for_every_query(
        self, enable_dpi_awareness, load_runtime, run_interactive_loop
    ):
        runtime = object()
        load_runtime.return_value = runtime

        inference_script.main()

        enable_dpi_awareness.assert_called_once_with()
        load_runtime.assert_called_once_with()
        run_interactive_loop.assert_called_once_with(
            locate=run_interactive_loop.call_args.kwargs["locate"],
            capture_screen=inference_script.capture_primary_screen,
            move_pointer=inference_script.move_pointer,
            save_result=inference_script.save_annotated_image,
        )
        locate = run_interactive_loop.call_args.kwargs["locate"]
        first_image = Image.new("RGB", (10, 10))
        second_image = Image.new("RGB", (20, 20))
        with patch.object(
            inference_script, "locate_on_screen", return_value=[]
        ) as locate_on_screen:
            locate(first_image, "first")
            locate(second_image, "second")

        locate_on_screen.assert_has_calls(
            [
                call(runtime, first_image, "first"),
                call(runtime, second_image, "second"),
            ]
        )


if __name__ == "__main__":
    unittest.main()
