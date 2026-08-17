import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.client import LLMClient
from src.llm.types import LLMProvider


def make_client(api_base, pins=None):
    return LLMClient(
        [LLMProvider(provider_name="openai", api_key="key", api_base=api_base)],
        provider_pins=pins,
    )


class PromptCacheKwargsTests(unittest.TestCase):
    """Anthropic models via OpenRouter cache only on explicit request —
    the root-level cache_control flag is what turns full-price re-sends of
    the growing tool-loop prefix into 0.1x cache reads."""

    def test_anthropic_via_openrouter_requests_prompt_caching(self):
        client = make_client("https://openrouter.ai/api/v1", pins={})

        kwargs = client.build_client_kwargs(client.get_default_provider(), "anthropic/claude-haiku-4.5")

        self.assertEqual(kwargs["extra_body"], {"cache_control": {"type": "ephemeral"}})
        self.assertTrue(kwargs["stream_usage"])

    def test_non_anthropic_models_do_not_get_the_flag(self):
        client = make_client("https://openrouter.ai/api/v1", pins={})

        kwargs = client.build_client_kwargs(client.get_default_provider(), "openai/gpt-4.1-mini")

        self.assertNotIn("extra_body", kwargs)

    def test_non_openrouter_endpoints_do_not_get_the_flag(self):
        client = make_client("https://api.example.com/v1", pins={"anthropic/": ["anthropic"]})

        kwargs = client.build_client_kwargs(client.get_default_provider(), "anthropic/claude-haiku-4.5")

        self.assertNotIn("extra_body", kwargs)


class ProviderPinKwargsTests(unittest.TestCase):
    """Routing pins come from provider_pins.json: prompt caches are
    per-endpoint (Bedrock alone has three regional ones), so without a pin,
    load balancing turns cache hits into a lottery of full-price calls plus
    write premiums."""

    def test_pinned_models_get_the_provider_order(self):
        client = make_client("https://openrouter.ai/api/v1", pins={"anthropic/": ["anthropic"]})

        kwargs = client.build_client_kwargs(client.get_default_provider(), "anthropic/claude-haiku-4.5")

        self.assertEqual(kwargs["extra_body"]["provider"], {"order": ["anthropic"]})

    def test_pins_apply_to_models_without_explicit_caching_too(self):
        client = make_client("https://openrouter.ai/api/v1", pins={"google/": ["google-ai-studio"]})

        kwargs = client.build_client_kwargs(client.get_default_provider(), "google/gemini-3.7-flash")

        self.assertEqual(kwargs["extra_body"], {"provider": {"order": ["google-ai-studio"]}})

    def test_pins_are_ignored_off_openrouter(self):
        client = make_client("https://api.example.com/v1", pins={"google/": ["google-ai-studio"]})

        kwargs = client.build_client_kwargs(client.get_default_provider(), "google/gemini-3.7-flash")

        self.assertNotIn("extra_body", kwargs)

    def test_the_shipped_pins_file_keeps_anthropic_sticky(self):
        # Tripwire: the default client loads provider_pins.json from the repo
        # root; removing the anthropic pin there would quietly bring back the
        # cache-hit lottery, so this test names what would be lost.
        client = make_client("https://openrouter.ai/api/v1")

        kwargs = client.build_client_kwargs(client.get_default_provider(), "anthropic/claude-haiku-4.5")

        self.assertEqual(kwargs["extra_body"]["provider"], {"order": ["anthropic"]})


if __name__ == "__main__":
    unittest.main()
