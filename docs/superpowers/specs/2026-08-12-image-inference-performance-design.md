# Faster Image Inference Design

## Goal

Reduce the per-image inference time of `test.py` by limiting the image data sent to LocateAnything-3B while preserving the original-resolution annotated output.

## Current bottleneck

The current 3024x1964 input is reduced by the model processor to roughly 6,500 visual tokens. A 768-pixel inference image produces roughly 504 visual tokens. The script also explicitly selects the model's slow autoregressive decoding mode.

## Design

1. Load the source image, apply its EXIF orientation, convert it to RGB, and keep this full-resolution image for drawing and saving.
2. Create a separate inference copy with the same aspect ratio and a maximum width or height of 768 pixels. Use Pillow's Lanczos resampling and never enlarge smaller images.
3. Pass only the resized copy to the processor and model.
4. Select `generation_mode="fast"` to match the requested maximum-speed profile. Keep the existing 2,048-token output limit and deterministic generation settings.
5. Parse the model's normalized 0-1000 coordinates as before, but scale them against the full-resolution image. The saved `output_boxes.jpg` therefore retains the source dimensions.
6. Print the original and inference dimensions so the applied optimization is visible when the script runs.

## Structure

Move execution behind a `main()` entry point and isolate the resize operation in a small pure function. This prevents model loading when the helper is imported by tests and keeps resizing independently testable.

The command-line contract remains unchanged:

```text
uv run python test.py [IMAGE_PATH] [PROMPT]
```

## Error handling

Keep Pillow's existing errors for missing or invalid images. Validate the resize limit inside the helper so zero or negative values fail clearly if the constant is changed incorrectly.

## Tests

Add focused tests that verify:

- landscape images are reduced to 768 pixels on their longest side;
- portrait images are reduced to 768 pixels on their longest side;
- images already within the limit are not enlarged;
- aspect ratio is preserved within integer rounding;
- the original image object and dimensions remain unchanged;
- normalized detection coordinates continue to map to the original dimensions.

Verification will include the focused test file, the existing project test suite, Python compilation, and a preprocessing benchmark comparing the current source input with the 768-pixel inference copy. Full model inference will be timed if it can complete within a practical verification window.

## Trade-off

The 768-pixel input and MTP-only fast decoding prioritize speed. Very small objects or complex scenes may have lower detection accuracy, and fast decoding has no autoregressive fallback for uncertain boxes. The full-resolution output is preserved, but it cannot restore visual detail omitted from the inference copy.
