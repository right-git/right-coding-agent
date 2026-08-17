import unittest

from src.llm.providers.openrouter import ModelInfo, OpenRouterCatalog, parse_models

PAYLOAD = {
    "data": [
        {
            "id": "google/gemini-3.7-flash",
            "name": "Google: Gemini 3.7 Flash",
            "context_length": 1048576,
            "pricing": {
                "prompt": "0.000000075",
                "completion": "0.0000003",
                "input_cache_read": "0.0000000075",
                "input_cache_write": "0.00000009375",
            },
            "supported_parameters": ["tools", "reasoning", "temperature"],
            "default_parameters": {"temperature": 1.0},
        },
        {
            "id": "openai/gpt-5.1-codex-mini",
            "name": "OpenAI: Codex Mini",
            "context_length": None,
            "top_provider": {"context_length": 400000},
            "pricing": {"prompt": "-1", "completion": "bogus"},
        },
        {"id": "", "pricing": {}},
        "garbage",
    ]
}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ParseModelsTests(unittest.TestCase):
    def test_valid_entries_are_parsed_and_junk_is_skipped(self):
        models = parse_models(PAYLOAD)

        self.assertEqual(
            sorted(models),
            ["google/gemini-3.7-flash", "openai/gpt-5.1-codex-mini"],
        )

    def test_prices_and_context_length_come_through(self):
        info = parse_models(PAYLOAD)["google/gemini-3.7-flash"]

        self.assertEqual(info.context_length, 1048576)
        self.assertAlmostEqual(info.prompt_price, 7.5e-8)
        self.assertAlmostEqual(info.completion_price, 3e-7)

    def test_context_falls_back_to_top_provider_and_bad_prices_become_none(self):
        info = parse_models(PAYLOAD)["openai/gpt-5.1-codex-mini"]

        self.assertEqual(info.context_length, 400000)
        self.assertIsNone(info.prompt_price)
        self.assertIsNone(info.completion_price)

    def test_supported_parameters_drive_capability_flags(self):
        gemini = parse_models(PAYLOAD)["google/gemini-3.7-flash"]
        codex = parse_models(PAYLOAD)["openai/gpt-5.1-codex-mini"]

        self.assertFalse(gemini.lacks_tools)
        self.assertTrue(gemini.supports_reasoning)
        self.assertFalse(gemini.lacks_temperature)
        # No capability list published: unknown, never treated as missing.
        self.assertFalse(codex.lacks_tools)
        self.assertFalse(codex.supports_reasoning)

    def test_default_temperature_is_parsed_when_published(self):
        models = parse_models(PAYLOAD)

        self.assertEqual(models["google/gemini-3.7-flash"].default_temperature, 1.0)
        self.assertIsNone(models["openai/gpt-5.1-codex-mini"].default_temperature)

    def test_garbage_payloads_parse_to_nothing(self):
        self.assertEqual(parse_models(None), {})
        self.assertEqual(parse_models({"data": "nope"}), {})
        self.assertEqual(parse_models([1, 2]), {})


class ModelInfoCostTests(unittest.TestCase):
    def test_cost_multiplies_tokens_by_per_token_prices(self):
        info = parse_models(PAYLOAD)["google/gemini-3.7-flash"]

        self.assertAlmostEqual(info.cost_of(1000, 100), 1000 * 7.5e-8 + 100 * 3e-7)

    def test_cost_is_none_without_pricing(self):
        info = parse_models(PAYLOAD)["openai/gpt-5.1-codex-mini"]

        self.assertIsNone(info.cost_of(1000, 100))

    def test_cache_prices_come_through(self):
        info = parse_models(PAYLOAD)["google/gemini-3.7-flash"]

        self.assertAlmostEqual(info.cache_read_price, 7.5e-9)
        self.assertAlmostEqual(info.cache_write_price, 9.375e-8)

    def test_cached_tokens_are_billed_at_the_cache_read_price(self):
        info = parse_models(PAYLOAD)["google/gemini-3.7-flash"]

        cost = info.cost_of(10_000, 100, cached_tokens=8_000)

        self.assertAlmostEqual(cost, 2_000 * 7.5e-8 + 8_000 * 7.5e-9 + 100 * 3e-7)

    def test_cached_tokens_fall_back_to_full_price_without_a_cache_price(self):
        info = ModelInfo("m", "M", 1000, 1e-6, 5e-6)

        self.assertAlmostEqual(info.cost_of(1_000, 0, cached_tokens=400), 1_000 * 1e-6)

    def test_cached_tokens_are_clamped_to_the_input_count(self):
        info = parse_models(PAYLOAD)["google/gemini-3.7-flash"]

        self.assertAlmostEqual(info.cost_of(1_000, 0, cached_tokens=5_000), 1_000 * 7.5e-9)

    def test_assume_cache_writes_bills_fresh_input_at_the_write_price(self):
        # Real generation from 2026-08-17 (req-1786954784…): 11,956 fresh
        # input + 1,073 output, caching requested, zero reads — OpenRouter
        # billed exactly $0.0203 (subtotal $0.0173 + cache write $0.00299).
        info = ModelInfo("anthropic/claude-haiku-4.5", "Haiku", 200_000, 1e-6, 5e-6, cache_write_price=1.25e-6)

        cost = info.cost_of(11_956, 1_073, cached_tokens=0, assume_cache_writes=True)

        self.assertAlmostEqual(cost, 11_956 * 1.25e-6 + 1_073 * 5e-6)
        self.assertAlmostEqual(cost, 0.0203, places=4)

    def test_assume_cache_writes_leaves_cached_tokens_at_the_read_price(self):
        info = parse_models(PAYLOAD)["google/gemini-3.7-flash"]

        cost = info.cost_of(10_000, 100, cached_tokens=8_000, assume_cache_writes=True)

        self.assertAlmostEqual(cost, 2_000 * 9.375e-8 + 8_000 * 7.5e-9 + 100 * 3e-7)

    def test_assume_cache_writes_falls_back_to_full_price_without_a_write_price(self):
        info = ModelInfo("m", "M", 1000, 1e-6, 5e-6)

        self.assertAlmostEqual(info.cost_of(1_000, 0, assume_cache_writes=True), 1_000 * 1e-6)


class OpenRouterCatalogTests(unittest.IsolatedAsyncioTestCase):
    def make_catalog(self, script, clock):
        calls = []

        async def fetch(url):
            calls.append(url)
            step = script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step

        catalog = OpenRouterCatalog(
            fetch_payload=fetch,
            ttl=3600,
            failure_cooldown=60,
            clock=clock,
        )
        return catalog, calls

    async def test_successful_fetch_is_cached(self):
        clock = FakeClock()
        catalog, calls = self.make_catalog([PAYLOAD], clock)

        first = await catalog.models()
        clock.advance(10)
        second = await catalog.models()

        self.assertEqual(len(calls), 1)
        self.assertIn("google/gemini-3.7-flash", first)
        self.assertIs(first, second)

    async def test_get_returns_one_model(self):
        clock = FakeClock()
        catalog, _ = self.make_catalog([PAYLOAD], clock)

        info = await catalog.get("google/gemini-3.7-flash")

        self.assertIsInstance(info, ModelInfo)
        self.assertIsNone(await catalog.get("missing/model"))

    async def test_failures_are_not_retried_within_the_cooldown(self):
        clock = FakeClock()
        catalog, calls = self.make_catalog([RuntimeError("offline"), PAYLOAD], clock)

        self.assertEqual(await catalog.models(), {})
        clock.advance(10)
        self.assertEqual(await catalog.models(), {})
        self.assertEqual(len(calls), 1)

        clock.advance(60)
        recovered = await catalog.models()
        self.assertEqual(len(calls), 2)
        self.assertIn("google/gemini-3.7-flash", recovered)

    async def test_stale_cache_survives_a_failed_refresh(self):
        clock = FakeClock()
        catalog, calls = self.make_catalog([PAYLOAD, RuntimeError("offline")], clock)

        first = await catalog.models()
        clock.advance(3601)
        second = await catalog.models()

        self.assertEqual(len(calls), 2)
        self.assertEqual(first, second)
        self.assertIn("google/gemini-3.7-flash", second)


if __name__ == "__main__":
    unittest.main()
