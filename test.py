from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    print(
        f"Image: {image.width}x{image.height} -> "
        f"{inference_image.width}x{inference_image.height}"
    )
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

    generation_inputs = {
        "pixel_values": inputs["pixel_values"].to(runtime.dtype),
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "tokenizer": runtime.tokenizer,
    }
    if "image_grid_hws" in inputs:
        generation_inputs["image_grid_hws"] = inputs["image_grid_hws"]

    print("Running inference...")
    with torch.inference_mode():
        response = runtime.model.generate(
            **generation_inputs,
            **generation_options(),
        )

    answer = response[0] if isinstance(response, tuple) else response
    print("Raw output:\n", answer)
    return parse_detections(answer, image.size)


def save_annotated_image(
    image: Image.Image,
    detections: Sequence[Detection],
    output_path: str | Path = OUTPUT_PATH,
) -> None:
    annotated_image = image.copy()
    if annotated_image.mode != "RGB":
        annotated_image = annotated_image.convert("RGB")

    draw = ImageDraw.Draw(annotated_image)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, image.width // 60))
    except OSError:
        font = ImageFont.load_default()

    label_colors: dict[str, str] = {}
    line_width = max(2, image.width // 400)
    for detection in detections:
        label = detection.label
        color = label_colors.setdefault(label, COLORS[len(label_colors) % len(COLORS)])
        x1, y1, x2, y2 = detection.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = y1 - text_height - 6 if y1 - text_height - 6 > 0 else y1 + 2
        draw.rectangle(
            [x1, text_y, x1 + text_width + 8, text_y + text_height + 6], fill=color
        )
        draw.text((x1 + 4, text_y + 2), label, fill="white", font=font)

    annotated_image.save(output_path, format="JPEG", quality=95)
    print(f"Saved: {output_path}")


def main() -> None:
    enable_dpi_awareness()
    runtime = load_runtime()

    def locate(image: Image.Image, description: str) -> list[Detection]:
        return locate_on_screen(runtime, image, description)

    run_interactive_loop(
        locate=locate,
        capture_screen=capture_primary_screen,
        move_pointer=move_pointer,
        save_result=save_annotated_image,
    )


if __name__ == "__main__":
    main()
