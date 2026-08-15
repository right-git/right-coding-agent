"""Window focus outside Windows: not implemented yet.

`foreground_window` degrades to "unknown" instead of failing, so flows that
only check focus keep working; `focus_window` fails loudly, because typing
into an unfocused window is exactly how desktop automation goes wrong.
"""


def foreground_window():
    return None


def focus_window(title_contains: str):
    raise RuntimeError(
        "focus_window is not supported on this platform yet — bring the "
        f"target window ({title_contains!r}) to the foreground manually"
    )
