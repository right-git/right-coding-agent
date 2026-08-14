import unittest
from io import StringIO

from rich.console import Console

from src.llm.openrouter import ModelInfo
from src.llm.usage import SessionUsage, TurnUsage
from src.ui import ChatUI
from src.ui.chat import theme


GEMINI = ModelInfo(
    id="google/gemini-3.7-flash",
    name="Google: Gemini 3.7 Flash",
    context_length=1_048_576,
    prompt_price=7.5e-8,
    completion_price=3e-7,
)
CODEX = ModelInfo(
    id="openai/gpt-5.1-codex-mini",
    name="OpenAI: Codex Mini",
    context_length=400_000,
    prompt_price=None,
    completion_price=None,
)


def make_ui(catalog=None, available=None, model="google/gemini-3.7-flash"):
    ui = ChatUI(
        model=model,
        available_models=available or ["google/gemini-3.7-flash"],
    )
    ui.console = Console(
        file=StringIO(),
        record=True,
        force_terminal=False,
        width=200,
        theme=theme,
    )
    ui.set_model_catalog(catalog)
    return ui


class ModelListingTests(unittest.TestCase):
    def test_models_listing_shows_context_and_prices(self):
        ui = make_ui(catalog={GEMINI.id: GEMINI})

        ui.handle_command("/models")

        rendered = ui.console.export_text()
        self.assertIn("google/gemini-3.7-flash", rendered)
        self.assertIn("ctx 1,048,576", rendered)
        self.assertIn("$0.075/M in", rendered)

    def test_current_model_outside_the_curated_list_is_still_shown(self):
        ui = make_ui(
            catalog={GEMINI.id: GEMINI, CODEX.id: CODEX},
            model="openai/gpt-5.1-codex-mini",
        )

        ui.handle_command("/models")

        self.assertIn("openai/gpt-5.1-codex-mini", ui.console.export_text())


class ModelSwitchTests(unittest.TestCase):
    def test_switching_to_any_catalog_model_works(self):
        ui = make_ui(catalog={GEMINI.id: GEMINI, CODEX.id: CODEX})

        ui.handle_command("/model openai/gpt-5.1-codex-mini")

        self.assertEqual(ui.model, "openai/gpt-5.1-codex-mini")
        self.assertIn("switched to", ui.console.export_text())

    def test_single_partial_match_switches(self):
        ui = make_ui(catalog={GEMINI.id: GEMINI, CODEX.id: CODEX})

        ui.handle_command("/model codex")

        self.assertEqual(ui.model, "openai/gpt-5.1-codex-mini")

    def test_ambiguous_partial_match_lists_candidates(self):
        llama_a = ModelInfo("meta/llama-x-large", "L", 8192, 1e-8, 2e-8)
        llama_b = ModelInfo("meta/llama-x-small", "S", 8192, 1e-8, 2e-8)
        ui = make_ui(catalog={llama_a.id: llama_a, llama_b.id: llama_b})

        ui.handle_command("/model llama-x")

        rendered = ui.console.export_text()
        self.assertEqual(ui.model, "google/gemini-3.7-flash")
        self.assertIn("2 models match", rendered)
        self.assertIn("meta/llama-x-large", rendered)
        self.assertIn("meta/llama-x-small", rendered)

    def test_unknown_model_with_catalog_is_rejected(self):
        ui = make_ui(catalog={GEMINI.id: GEMINI})

        ui.handle_command("/model nope/nothing")

        self.assertEqual(ui.model, "google/gemini-3.7-flash")
        self.assertIn("unknown model", ui.console.export_text())

    def test_unknown_model_without_catalog_switches_with_warning(self):
        ui = make_ui(catalog={})

        ui.handle_command("/model custom/self-hosted")

        self.assertEqual(ui.model, "custom/self-hosted")
        self.assertIn("not verified", ui.console.export_text())


class UsageFooterTests(unittest.TestCase):
    def test_footer_shows_context_fill_turn_cost_and_session_totals(self):
        ui = make_ui()
        session = SessionUsage()
        turn = TurnUsage(
            input_tokens=13_900,
            output_tokens=304,
            context_tokens=14_204,
            calls=2,
            tool_calls=2,
            script_tool_calls=5,
        )
        session.add(turn, 0.0012)

        ui.print_usage(turn, GEMINI, 0.0012, session)

        rendered = ui.console.export_text()
        self.assertIn("14,204/1,048,576 (1.4%)", rendered)
        self.assertIn("█" + "░" * 19, rendered)
        self.assertIn("turn 13,900 in + 304 out ($0.0012)", rendered)
        self.assertIn("tools 2 (+5 in scripts)", rendered)
        self.assertIn("session 14,204 tokens ($0.0012)", rendered)

    def test_footer_omits_the_tools_segment_when_none_were_called(self):
        ui = make_ui()
        session = SessionUsage()
        turn = TurnUsage(
            input_tokens=100, output_tokens=10, context_tokens=110, calls=1
        )
        session.add(turn, 0.001)

        ui.print_usage(turn, GEMINI, 0.001, session)

        self.assertNotIn("tools", ui.console.export_text())

    def test_context_bar_fills_and_changes_color_with_usage(self):
        ui = make_ui()

        self.assertEqual(
            ui._context_bar(0, 100), "[green]" + "[/]" + "░" * 20
        )
        self.assertEqual(
            ui._context_bar(50, 100), "[green]" + "█" * 10 + "[/]" + "░" * 10
        )
        self.assertEqual(
            ui._context_bar(80, 100), "[yellow]" + "█" * 16 + "[/]" + "░" * 4
        )
        self.assertEqual(ui._context_bar(100, 100), "[red]" + "█" * 20 + "[/]")
        self.assertEqual(ui._context_bar(250, 100), "[red]" + "█" * 20 + "[/]")

    def test_footer_without_model_info_marks_unknown_limit_and_price(self):
        ui = make_ui()
        session = SessionUsage()
        turn = TurnUsage(
            input_tokens=100, output_tokens=10, context_tokens=110, calls=1
        )
        session.add(turn, None)

        ui.print_usage(turn, None, None, session)

        rendered = ui.console.export_text()
        self.assertIn("ctx 110 (limit unknown)", rendered)
        self.assertIn("(price unknown)", rendered)
        self.assertIn("≈", rendered)

    def test_footer_reports_missing_usage_metadata(self):
        ui = make_ui()

        ui.print_usage(TurnUsage(), GEMINI, None, SessionUsage())

        self.assertIn(
            "provider reported no token counts", ui.console.export_text()
        )


if __name__ == "__main__":
    unittest.main()
