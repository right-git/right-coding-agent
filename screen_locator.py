import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import image_optimization


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
