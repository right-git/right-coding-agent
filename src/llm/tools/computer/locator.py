from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from src.config.logging import logger
from src.utils.silence import silenced

from .detection import parse_detections, resize_for_inference
from .types import Detection

MODEL_ID = "nvidia/LocateAnything-3B"
MAX_IMAGE_SIDE = 768
REQUIRED_MODEL_FILES = ("config.json", "model.safetensors.index.json")


def resolve_model_source(local_model: Path | None = None) -> str:
    """Prefer an already downloaded snapshot; otherwise use the hub id."""
    if local_model is None:
        local_model = Path.home() / ".cache" / "huggingface" / "hub" / "models--nvidia--LocateAnything-3B"

    if all((local_model / filename).is_file() for filename in REQUIRED_MODEL_FILES):
        return str(local_model)
    return MODEL_ID


MODEL = resolve_model_source()


def build_gui_prompt(description: str) -> str:
    description = description.strip().rstrip(".")
    return f"Locate the region that matches the following description: {description}."


def generation_options() -> dict[str, object]:
    """Greedy decoding: localisation wants the single best answer, not a sample.

    Measured on a query the model is unsure about: sampling returned five
    different boxes in five runs, scattered across the screen, while greedy
    returned the same — and correct — box every time. On queries the model is
    confident about, both agree. Determinism also makes results cacheable.
    """
    return {
        # A located box is ~110 characters and normally ends cleanly. When the
        # model is unsure it falls back to autoregressive decoding and can run
        # on repeating itself: measured 10579 characters and 16.8s against 0.8s
        # for the same query capped. The box is emitted first and came back
        # byte-identical at every cap from 256 down to 48, so cutting the tail
        # costs nothing on one target and bounds the worst case at ~2s. Raise it
        # if you ask for many objects at once and the later boxes go missing.
        "max_new_tokens": 96,
        "use_cache": True,
        # 'hybrid' lets the model re-decode coordinates it is unsure about.
        # 'fast' skips that and is ~7% quicker, but turned one 5px hit into a
        # 143px miss — not worth it.
        "generation_mode": "hybrid",
        "do_sample": False,
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


def quiet_transformers() -> None:
    """Silence transformers' chatter and progress bars before loading."""
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def load_runtime() -> InferenceRuntime:
    device, dtype = select_inference_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, dtype=dtype, trust_remote_code=True).to(device).eval()
    return InferenceRuntime(tokenizer, processor, model, device, dtype)


def locate_on_screen(
    runtime: InferenceRuntime,
    image: Image.Image,
    description: str,
    *,
    max_image_side: int = MAX_IMAGE_SIDE,
) -> list[Detection]:
    """Run one detection pass and return boxes in `image` pixel coordinates."""
    inference_image = resize_for_inference(image, max_image_side)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": inference_image},
                {"type": "text", "text": build_gui_prompt(description)},
            ],
        }
    ]

    text = runtime.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = runtime.processor.process_vision_info(messages)
    inputs = runtime.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(runtime.device)

    generation_inputs = {
        "pixel_values": inputs["pixel_values"].to(runtime.dtype),
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "tokenizer": runtime.tokenizer,
    }
    if "image_grid_hws" in inputs:
        generation_inputs["image_grid_hws"] = inputs["image_grid_hws"]

    with torch.inference_mode():
        response = runtime.model.generate(
            **generation_inputs,
            **generation_options(),
        )

    answer = response[0] if isinstance(response, tuple) else response
    return parse_detections(answer, image.size)


class LocateAnythingLocator:
    """Locator backed by `nvidia/LocateAnything-3B`, loaded once and reused."""

    def __init__(
        self,
        *,
        max_image_side: int = MAX_IMAGE_SIDE,
        runtime: InferenceRuntime | None = None,
        loader: Any = load_runtime,
        warmup: bool = True,
    ) -> None:
        self.max_image_side = max_image_side
        self._runtime = runtime
        self._loader = loader
        self._warmup = warmup

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    def load(self) -> InferenceRuntime:
        if self._runtime is None:
            # Loading and warm-up spew library warnings and progress bars;
            # they run on background threads, so mute this thread only.
            with silenced():
                quiet_transformers()
                self._runtime = self._loader()
                if self._warmup:
                    self.warm_up(self._runtime)
        return self._runtime

    def warm_up(self, runtime: InferenceRuntime) -> None:
        """Run one throwaway pass so the first real query is not the slow one.

        Kernel autotuning makes an unwarmed first call take about twice as long
        as the steady state.
        """
        try:
            locate_on_screen(
                runtime,
                Image.new("RGB", (64, 64)),
                "warmup",
                max_image_side=self.max_image_side,
            )
        except Exception:
            logger.warning("Locator warm-up failed; the first query will be slower")

    def locate(self, image: Image.Image, description: str) -> list[Detection]:
        runtime = self.load()
        with silenced():
            return locate_on_screen(
                runtime,
                image,
                description,
                max_image_side=self.max_image_side,
            )
