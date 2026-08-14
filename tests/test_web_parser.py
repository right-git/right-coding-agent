import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.tools.web_parser import WebParser


class ContentTextTests(unittest.TestCase):
    def test_dict_results_yield_their_content_string(self):
        self.assertEqual(WebParser._as_content_text({"content": "hello"}), "hello")

    def test_dicts_without_usable_content_yield_empty_text(self):
        self.assertEqual(WebParser._as_content_text({}), "")
        self.assertEqual(WebParser._as_content_text({"content": None}), "")
        self.assertEqual(WebParser._as_content_text({"content": 5}), "")

    def test_strings_pass_through_and_none_becomes_empty(self):
        self.assertEqual(WebParser._as_content_text("markdown"), "markdown")
        self.assertEqual(WebParser._as_content_text(None), "")


class ParsePageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.parser = WebParser()

    async def test_parse_page_always_returns_a_string(self):
        html = (
            "<html><head><meta name='description' content='d'>"
            "<meta charset='utf-8'></head>"
            "<body><p>Hello world</p></body></html>"
        )
        self.parser.make_request = AsyncMock(return_value=Mock(text=html))

        result = await self.parser.parse_page("https://example.test")

        self.assertIsInstance(result, str)
        self.assertIn("Hello world", result)

    async def test_conversion_failure_falls_back_to_plain_text(self):
        html = "<html><body><p>Fallback text</p></body></html>"
        self.parser.make_request = AsyncMock(return_value=Mock(text=html))

        with patch.object(
            self.parser, "parse_html", side_effect=RuntimeError("boom")
        ):
            result = await self.parser.parse_page("https://example.test")

        self.assertIsInstance(result, str)
        self.assertIn("Fallback text", result)

    def test_valueless_meta_attributes_do_not_crash_parsing(self):
        html = "<html><head><meta name></head><body><p>ok</p></body></html>"

        result = self.parser.parse_html(html)

        self.assertIn("ok", WebParser._as_content_text(result))


if __name__ == "__main__":
    unittest.main()
