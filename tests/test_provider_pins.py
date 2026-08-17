import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.routing import DEFAULT_PROVIDER_PINS, load_provider_pins, provider_order_for


class LoadProviderPinsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, content: str) -> Path:
        path = self.root / "provider_pins.json"
        path.write_text(content, encoding="utf-8")
        return path

    def test_missing_file_falls_back_to_the_default_pins(self):
        pins = load_provider_pins(self.root / "nope.json")

        self.assertEqual(pins, DEFAULT_PROVIDER_PINS)

    def test_valid_file_is_parsed_and_comment_keys_are_ignored(self):
        path = self.write(json.dumps({"_comment": "why", "anthropic/": ["anthropic"], "google/": ["google-vertex"]}))

        pins = load_provider_pins(path)

        self.assertEqual(pins, {"anthropic/": ["anthropic"], "google/": ["google-vertex"]})

    def test_an_empty_object_means_no_pins_on_purpose(self):
        pins = load_provider_pins(self.write("{}"))

        self.assertEqual(pins, {})

    def test_broken_json_is_reported_and_falls_back_to_the_defaults(self):
        path = self.write("{not json")

        with patch("src.config.routing.logger") as logger:
            pins = load_provider_pins(path)

        self.assertEqual(pins, DEFAULT_PROVIDER_PINS)
        logger.warning.assert_called()

    def test_a_non_object_top_level_falls_back_to_the_defaults(self):
        with patch("src.config.routing.logger"):
            pins = load_provider_pins(self.write('["anthropic"]'))

        self.assertEqual(pins, DEFAULT_PROVIDER_PINS)

    def test_invalid_entries_are_skipped_and_valid_ones_kept(self):
        path = self.write(json.dumps({"anthropic/": ["anthropic"], "bad": "anthropic", "worse": [1, 2]}))

        with patch("src.config.routing.logger") as logger:
            pins = load_provider_pins(path)

        self.assertEqual(pins, {"anthropic/": ["anthropic"]})
        self.assertEqual(logger.warning.call_count, 2)


class ProviderOrderForTests(unittest.TestCase):
    def test_the_longest_matching_prefix_wins(self):
        pins = {"anthropic/": ["anthropic"], "anthropic/claude-opus": ["google-vertex"]}

        self.assertEqual(provider_order_for("anthropic/claude-haiku-4.5", pins), ["anthropic"])
        self.assertEqual(provider_order_for("anthropic/claude-opus-4.6", pins), ["google-vertex"])

    def test_no_matching_prefix_means_no_pin(self):
        self.assertIsNone(provider_order_for("openai/gpt-4.1-mini", {"anthropic/": ["anthropic"]}))

    def test_an_empty_order_overrides_a_broader_pin(self):
        pins = {"anthropic/": ["anthropic"], "anthropic/claude-3-haiku": []}

        self.assertIsNone(provider_order_for("anthropic/claude-3-haiku", pins))


if __name__ == "__main__":
    unittest.main()
