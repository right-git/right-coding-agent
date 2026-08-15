import base64
import io
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ui import ChatUI
from src.ui.chat import theme
from src.ui.clipboard import encode_image, grab_clipboard_image


def make_ui():
    ui = ChatUI(model="google/gemini-3.7-flash")
    ui.console = Console(file=StringIO(), record=True, force_terminal=False, width=200, theme=theme)
    return ui


class GrabClipboardImageTests(unittest.TestCase):
    def test_pixel_data_comes_back_as_is(self):
        image = Image.new("RGB", (12, 8), "red")

        self.assertIs(grab_clipboard_image(grabber=lambda: image), image)

    def test_copied_image_file_is_opened(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "shot.png"
        Image.new("RGB", (10, 10), "blue").save(path)

        grabbed = grab_clipboard_image(grabber=lambda: [str(path)])

        self.assertIsNotNone(grabbed)
        self.assertEqual(grabbed.size, (10, 10))

    def test_text_or_empty_clipboard_yields_none(self):
        self.assertIsNone(grab_clipboard_image(grabber=lambda: None))
        self.assertIsNone(grab_clipboard_image(grabber=lambda: ["notes.txt"]))

    def test_grabber_errors_yield_none(self):
        def broken():
            raise OSError("no clipboard")

        self.assertIsNone(grab_clipboard_image(grabber=broken))


class EncodeImageTests(unittest.TestCase):
    def test_payload_carries_size_mime_and_decodable_base64(self):
        payload = encode_image(Image.new("RGB", (40, 20), "green"))

        self.assertEqual((payload["width"], payload["height"]), (40, 20))
        self.assertEqual(payload["mime_type"], "image/jpeg")
        decoded = Image.open(io.BytesIO(base64.b64decode(payload["base64_data"])))
        self.assertEqual(decoded.size, (40, 20))


class TakeUserContentTests(unittest.TestCase):
    def test_plain_text_passes_through(self):
        ui = make_ui()

        self.assertEqual(ui.take_user_content("привет"), "привет")

    def test_pending_images_become_content_blocks_and_are_consumed(self):
        ui = make_ui()
        ui.pending_images.append({"base64_data": "QUJD", "mime_type": "image/jpeg", "width": 1, "height": 1})

        content = ui.take_user_content("что на скрине?")

        self.assertEqual(content[0], {"type": "text", "text": "что на скрине?"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/jpeg;base64,QUJD")
        self.assertEqual(ui.pending_images, [])
        self.assertEqual(ui.take_user_content("дальше"), "дальше")


class PasteCommandTests(unittest.TestCase):
    def test_paste_attaches_and_reports(self):
        ui = make_ui()
        with patch("src.ui.chat.grab_clipboard_image", return_value=Image.new("RGB", (30, 20))):
            ui.handle_command("/paste")

        self.assertEqual(len(ui.pending_images), 1)
        rendered = ui.console.export_text()
        self.assertIn("image attached (30×20)", rendered)
        self.assertIn("next message", rendered)

    def test_paste_without_an_image_explains(self):
        ui = make_ui()
        with patch("src.ui.chat.grab_clipboard_image", return_value=None):
            ui.handle_command("/paste")

        self.assertEqual(ui.pending_images, [])
        self.assertIn("no image in the clipboard", ui.console.export_text())

    def test_clear_drops_pending_images(self):
        ui = make_ui()
        ui.pending_images.append({"base64_data": "QUJD", "mime_type": "image/jpeg", "width": 1, "height": 1})

        ui.handle_command("/clear")

        self.assertEqual(ui.pending_images, [])


if __name__ == "__main__":
    unittest.main()
