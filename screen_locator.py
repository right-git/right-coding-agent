import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import image_optimization
from PIL import Image


TargetMode = Literal["first", "all"]
Box = tuple[int, int, int, int]
Point = tuple[int, int]


@dataclass(frozen=True)
class Detection:
    label: str
    box: Box


_TOKEN_PATTERN = re.compile(
    r"<ref>(?P<label>(?:(?!<ref>|<box>).)*?)</ref>"
    r"|(?P<box><box>(?:(?!<ref>|<box>).)*?</box>)",
    re.DOTALL,
)
_BOX_PATTERN = re.compile(
    r"<box>\s*"
    r"<\s*([+-]?\d+)\s*>\s*"
    r"<\s*([+-]?\d+)\s*>\s*"
    r"<\s*([+-]?\d+)\s*>\s*"
    r"<\s*([+-]?\d+)\s*>\s*"
    r"</box>"
)


def parse_detections(model_output: str, image_size: tuple[int, int]) -> list[Detection]:
    detections: list[Detection] = []
    detection_label = "object"

    for token_match in _TOKEN_PATTERN.finditer(model_output):
        raw_label = token_match.group("label")
        if raw_label is not None:
            detection_label = (raw_label or "").strip() or "object"
            continue

        coordinate_match = _BOX_PATTERN.fullmatch(token_match.group("box"))
        if coordinate_match is None:
            continue

        normalized_box = tuple(int(value) for value in coordinate_match.groups())
        x1, y1, x2, y2 = image_optimization.scale_normalized_box(
            normalized_box, image_size
        )
        detections.append(
            Detection(
                label=detection_label,
                box=(round(x1), round(y1), round(x2), round(y2)),
            )
        )

    return detections


def box_center(box: Box, image_size: tuple[int, int]) -> Point:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    x1, y1, x2, y2 = box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return (
        min(max(center_x, 0), width - 1),
        min(max(center_y, 0), height - 1),
    )


def select_targets(
    detections: Sequence[Detection], mode: TargetMode
) -> list[Detection]:
    if mode == "first":
        return list(detections[:1])
    if mode == "all":
        return list(detections)
    raise ValueError(f"Unsupported target mode: {mode}")


Locate = Callable[[Image.Image, str], Sequence[Detection]]
CaptureScreen = Callable[[], Image.Image]
MovePointer = Callable[[int, int], None]
SaveResult = Callable[[Image.Image, Sequence[Detection]], None]
Input = Callable[[str], str]
Output = Callable[[str], None]
Pause = Callable[[float], None]

_EXIT_COMMANDS = {"exit", "quit"}


def parse_mode_command(command: str) -> TargetMode:
    parts = command.strip().casefold().split()
    if parts == [":mode", "first"]:
        return "first"
    if parts == [":mode", "all"]:
        return "all"
    raise ValueError("Use :mode first or :mode all")


def run_interactive_loop(
    *,
    locate: Locate,
    capture_screen: CaptureScreen,
    move_pointer: MovePointer,
    save_result: SaveResult,
    input_fn: Input = input,
    output_fn: Output = print,
    pause_fn: Pause = time.sleep,
) -> None:
    mode: TargetMode = "first"
    output_fn("Commands: :mode first, :mode all, exit")

    while True:
        try:
            command = input_fn(f"[{mode}] Что найти? ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("Stopped.")
            return

        normalized_command = command.casefold()
        if not command:
            continue
        if normalized_command in _EXIT_COMMANDS:
            return
        if normalized_command.startswith(":mode"):
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
