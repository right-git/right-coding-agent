"""Screen understanding and desktop control for the agent.

Per-tool package layout: `service.py` holds `ComputerUse` — the facade that
captures the screen, finds elements by natural-language description, moves
and clicks the mouse, types, and can mark an element on screen with an
explanatory tooltip instead of acting on it — `tool.py` the `screen_*`
`@tool` functions the LLM sees, and `platforms/` the OS backends (native
Win32 under `platforms/windows/`, a pynput/mss-based fallback for macOS and
Linux under `platforms/portable/`), picked lazily by `platforms.default_*`.

Only cross-platform names are re-exported here; OS-specific classes live in
their `platforms/` modules so importing this package works on any OS. The
heavy vision model is imported lazily, so importing this package stays
cheap until the first `locate_object` / `mark_object` call.
"""

from .annotation import (
    DEFAULT_OUTPUT_PATH,
    annotate,
    image_to_base64,
    save_annotated_image,
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
    "ScrollDirection",
    "Size",
    "TargetMode",
    "TkOverlay",
    "annotate",
    "box_center",
    "clamp_point",
    "describe_detections",
    "get_computer",
    "image_to_base64",
    "parse_detections",
    "position_label",
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
