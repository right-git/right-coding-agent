"""Clipboard via pyperclip (pbcopy/pbpaste on macOS, xclip/xsel on Linux)."""


class PyperclipClipboard:
    """Clipboard backend on top of the cross-platform pyperclip library."""

    def read_text(self) -> str:
        """Current clipboard text, or an empty string when it holds none."""
        import pyperclip

        return pyperclip.paste() or ""

    def write_text(self, text: str) -> None:
        """Replace the clipboard contents with `text`."""
        import pyperclip

        pyperclip.copy(text)
