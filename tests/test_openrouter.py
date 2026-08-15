import unittest

from src.llm.providers.openrouter import ModelInfo, OpenRouterCatalog, parse_models

PAYLOAD = {
    "data": [
        {
            "id": "google/gemini-3.7-flash",
            "name": "Google: Gemini 3.7 Flash",
            "context_length": 1048576,
            "pricing": {"prompt": "0.000000075", "completion": "0.0000003"},
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
