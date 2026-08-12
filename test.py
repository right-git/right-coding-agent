import re
import sys

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from image_optimization import resize_for_inference, scale_normalized_box

MODEL = "nvidia/LocateAnything-3B"
OUTPUT_PATH = "output_boxes.jpg"
MAX_IMAGE_SIDE = 768
COLORS = ["#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE", "#00C7BE"]


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
    print(
        f"Image: {img.width}x{img.height} -> "
        f"{inference_img.width}x{inference_img.height}"
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16

    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL, torch_dtype=dtype, trust_remote_code=True)
        .to(device)
        .eval()
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": inference_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = processor.process_vision_info(messages)
    inputs = processor(
        text=[text], images=images, videos=videos, return_tensors="pt"
    ).to(device)

    print("Running inference...")
    with torch.no_grad():
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

    answer = response[0] if isinstance(response, tuple) else response
    print("Raw output:\n", answer)

    detections = []
    pattern = re.compile(r"([^<>]+?)?\s*((?:<box><\d+><\d+><\d+><\d+></box>\s*)+)")
    for match in pattern.finditer(answer):
        label = (match.group(1) or "").strip() or "object"
        for box in re.finditer(
            r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", match.group(2)
        ):
            coordinates = tuple(int(value) for value in box.groups())
            detections.append(
                {
                    "label": label,
                    "box": scale_normalized_box(coordinates, img.size),
                }
            )

    print(f"Parsed {len(detections)} boxes")

    label_colors = {}
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", size=max(14, img.width // 60)
        )
    except OSError:
        font = ImageFont.load_default()

    line_width = max(2, img.width // 400)
    for detection in detections:
        label = detection["label"]
        color = label_colors.setdefault(label, COLORS[len(label_colors) % len(COLORS)])
        x1, y1, x2, y2 = detection["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = y1 - text_height - 6 if y1 - text_height - 6 > 0 else y1 + 2
        draw.rectangle(
            [x1, text_y, x1 + text_width + 8, text_y + text_height + 6], fill=color
        )
        draw.text((x1 + 4, text_y + 2), label, fill="white", font=font)

    img.save(OUTPUT_PATH, quality=95)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
