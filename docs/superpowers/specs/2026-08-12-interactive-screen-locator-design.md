# Interactive Screen Locator Design

## Goal

Turn `test.py` into a persistent interactive screen locator. The model loads once, then every text query captures a fresh screenshot, locates matching regions, reports their centers, saves an annotated image, and moves the Windows mouse pointer without clicking.

## User Flow

1. The user starts `test.py` with no image argument.
2. The script makes the process DPI-aware and loads the tokenizer, processor, and model once.
3. The prompt `Что найти?` accepts a natural-language description.
4. After the user submits a description, the script captures the current primary screen.
5. The screenshot is resized only for inference. Returned normalized boxes are converted back to coordinates in the original screenshot.
6. The script prints every detected box and its center, saves all boxes to `output_boxes.jpg`, and moves the pointer according to the active target mode.
7. The prompt repeats without reloading the model.

The commands `exit` and `quit`, or `Ctrl+C`, stop the program cleanly. Empty input does nothing and shows the prompt again.

## Target Modes

- `first` is the default. The pointer moves to the center of the first parsed box.
- `all` moves the pointer through every parsed box in model output order.
- `:mode first` and `:mode all` change the active mode without restarting the program.
- An invalid `:mode` command prints the supported values and leaves the current mode unchanged.

Pointer movement never clicks, presses, or scrolls.

## Architecture

The existing model-loading and generation behavior remains in `test.py`, but reusable operations are separated into small functions:

- screen setup and capture: enable Windows DPI awareness and capture the primary display with Pillow;
- response parsing: turn model text into labels and screenshot-space boxes;
- coordinate calculation: compute an integer center `(x, y)` for each box;
- target selection: choose the first detection or all detections according to the active mode;
- pointer control: use the Windows `SetCursorPos` API through Python's standard-library `ctypes`;
- interactive loop: process commands and descriptions while reusing the loaded model.

Pillow and `ctypes` are already available, so the feature adds no package dependency and performs no download.

## Data Flow

For each description:

```text
description -> fresh primary-screen screenshot -> inference resize
            -> model generation -> parsed normalized boxes
            -> original screenshot coordinates -> centers
            -> console output + annotated JPEG + pointer movement
```

The screenshot used for inference and the screenshot used for coordinate scaling are the same capture. Moving the pointer happens only after inference, so the pointer's new location cannot affect that query's screenshot.

## Coordinate and Platform Behavior

The initial implementation targets the current Windows environment. DPI awareness is enabled before screen capture so Pillow screenshot pixels and `SetCursorPos` coordinates use the same coordinate system. The primary display is the capture area; multi-monitor desktop capture is outside this version's scope.

For a box `(x1, y1, x2, y2)`, its center is calculated as:

```text
((x1 + x2) // 2, (y1 + y2) // 2)
```

The calculated center is clamped to the screenshot boundaries before pointer movement, even if the model returns coordinates outside the normalized `0..1000` range.

## Errors and Safety

- If the model returns no valid box, the script reports that nothing was found, saves the unannotated screenshot, and does not move the pointer.
- A malformed box is ignored by the parser.
- Screenshot or pointer API failures are reported for the current query; the loop remains available when recovery is safe.
- Model loading failures stop startup because the loop cannot operate without the model.
- The pointer is never clicked automatically.

## Testing

Automated tests cover behavior without capturing the real desktop or moving the real pointer:

- parsing one and multiple boxes;
- center calculation;
- selecting `first` and `all` targets;
- mode command handling;
- empty input and exit commands;
- no pointer call when there are no detections;
- one pointer call for `first` and ordered pointer calls for `all`;
- model initialization occurs once while multiple queries are processed.

Screen capture, inference, and pointer movement are supplied to the loop as callable dependencies in tests. A final manual smoke test validates real Windows screenshot capture, printed coordinates, annotated output, and cursor movement.

## Acceptance Criteria

- Starting the script loads the model exactly once and opens the interactive prompt.
- Each ordinary prompt uses a newly captured primary-screen image.
- Every valid detection prints its screenshot-space box and center.
- `output_boxes.jpg` contains the boxes from the most recent query.
- Default `first` mode moves the pointer to only the first center.
- `all` mode visits all centers in output order.
- The script never clicks the mouse.
- No additional dependency or model download is required.
