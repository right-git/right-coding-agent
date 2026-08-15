import sys
import unittest
from io import StringIO
from pathlib import Path

from prompt_toolkit.document import Document
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.providers.openrouter import ModelInfo
from src.ui import ChatUI
from src.ui.chat import theme
from src.ui.completer import CommandCompleter

REASONER = ModelInfo(
    id="openai/o5-mini",
    name="OpenAI: o5 mini",
    context_length=200_000,
    prompt_price=1e-7,
    completion_price=4e-7,
    supported_parameters=frozenset({"tools", "reasoning"}),
)
TOOLLESS = ModelInfo(
    id="meta/llama-chat",
    name="Meta: Llama Chat",
    context_length=8192,
    prompt_price=1e-8,
    completion_price=2e-8,
    supported_parameters=frozenset({"temperature"}),
)
BATCH = ModelInfo(
    id="openai/o5-mini:batch",
    name="OpenAI: o5 mini (batch)",
    context_length=200_000,
    prompt_price=5e-8,
    completion_price=2e-7,
    supported_parameters=frozenset({"tools"}),
)


def make_completer(catalog=None):
    ui = ChatUI(model="google/gemini-3.7-flash", available_models=["google/gemini-3.7-flash"])
    ui.console = Console(file=StringIO(), record=True, force_terminal=False, width=200, theme=theme)
    ui.set_model_catalog(catalog)
    return CommandCompleter(ui)


def complete(completer, text):
    return [completion.text for completion in completer.get_completions(Document(text), None)]


class CommandNameCompletionTests(unittest.TestCase):
    def test_slash_prefix_completes_command_names(self):
        completions = complete(make_completer(), "/mo")

        self.assertIn("/model", completions)
        self.assertIn("/models", completions)
        self.assertNotIn("/help", completions)

    def test_plain_text_gets_no_completions(self):
        self.assertEqual(complete(make_completer(), "напиши поэму"), [])


class ModelArgumentCompletionTests(unittest.TestCase):
    def test_model_ids_come_from_curated_list_and_catalog(self):
        completer = make_completer(catalog={REASONER.id: REASONER})

        completions = complete(completer, "/model o5")

        self.assertEqual(completions, ["openai/o5-mini"])

    def test_toolless_and_batch_models_are_not_offered(self):
        completer = make_completer(catalog={REASONER.id: REASONER, TOOLLESS.id: TOOLLESS, BATCH.id: BATCH})

        completions = complete(completer, "/model ")

        self.assertIn("openai/o5-mini", completions)
        self.assertNotIn("meta/llama-chat", completions)
        self.assertNotIn("openai/o5-mini:batch", completions)

    def test_second_model_argument_completes_effort_levels(self):
        completer = make_completer(catalog={REASONER.id: REASONER})

        completions = complete(completer, "/model openai/o5-mini h")

        self.assertEqual(completions, ["high"])


class OptionCompletionTests(unittest.TestCase):
    def test_effort_levels_complete(self):
        completions = complete(make_completer(), "/effort m")

        self.assertEqual(completions, ["minimal", "medium"])

    def test_log_levels_complete(self):
        completions = complete(make_completer(), "/log-level d")

        self.assertEqual(completions, ["debug"])

    def test_temperature_offers_none(self):
        self.assertEqual(complete(make_completer(), "/temperature n"), ["none"])


if __name__ == "__main__":
    unittest.main()
