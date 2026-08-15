"""Screen understanding and desktop control for the agent.

Per-tool package layout: `service.py` holds `ComputerUse` — the facade that
captures the screen, finds elements by natural-language description, moves
and clicks the mouse, types, and can mark an element on screen with an
explanatory tooltip instead of acting on it — `tool.py` the `screen_*`
`@tool` functions the LLM sees, and the remaining modules the injected
OS-facing backends (locator, screen, pointer, overlay, clipboard, windows).

The heavy vision model is imported lazily, so importing this package stays
cheap until the first `locate_object` / `mark_object` call.
"""

from .annotation import (
    DEFAULT_OUTPUT_PATH,
    annotate,
    image_to_base64,
    save_annotated_image,
)
from .service import ComputerUse
from .tool import (
    COMPUTER_TOOLS,
    get_computer,
    screen_click,
    screen_key,
    screen_locate,
    screen_mark,
    screen_screenshot,
    screen_scroll,
    screen_type,
    set_computer,
    warm_up_computer,
)
from .detection import (
    box_center,
    clamp_point,
    describe_detections,
    parse_detections,
    position_label,
    resize_for_inference,
    scale_normalized_box,
    select_targets,
)
from .overlay import NullOverlay, OverlayStyle, TkOverlay
from .pointer import Pointer, Win32InputDispatcher, move_pointer
from .screen import (
    PrimaryScreen,
    capture_primary_screen,
    enable_dpi_awareness,
    primary_screen_size,
)
from .types import (
    Box,
    Detection,
    Marker,
    MouseButton,
    Point,
    ScrollDirection,
    Size,
    TargetMode,
)

__all__ = [
    "Box",
    "COMPUTER_TOOLS",
    "ComputerUse",
    "DEFAULT_OUTPUT_PATH",
    "Detection",
    "Marker",
    "MouseButton",
    "NullOverlay",
    "OverlayStyle",
    "Point",
    "Pointer",
    "PrimaryScreen",
    "ScrollDirection",
    "Size",
    "TargetMode",
    "TkOverlay",
    "Win32InputDispatcher",
    "annotate",
    "box_center",
    "capture_primary_screen",
    "clamp_point",
    "describe_detections",
    "enable_dpi_awareness",
    "get_computer",
    "image_to_base64",
    "move_pointer",
    "parse_detections",
    "position_label",
    "primary_screen_size",
    "resize_for_inference",
    "save_annotated_image",
    "scale_normalized_box",
    "screen_click",
    "screen_key",
    "screen_locate",
    "screen_mark",
    "screen_screenshot",
    "screen_scroll",
    "screen_type",
    "select_targets",
    "set_computer",
    "warm_up_computer",
]
