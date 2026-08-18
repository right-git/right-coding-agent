import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_toolkit.document import Document  # noqa: E402
from rich.console import Console  # noqa: E402

from src.llm.tools.meta.registry import ToolRegistry  # noqa: E402
from src.llm.tools.skills.store import SkillStore, set_skill_store  # noqa: E402
from src.ui.chat import theme  # noqa: E402
from src.ui.commands import CommandHandler, SkillAction  # noqa: E402
from src.ui.completer import CommandCompleter  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402


class TestSkillCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        user_dir = Path(self.tmp.name)
        make_skill_dir(user_dir, "deploy", "---\ndescription: deploy it\nargument-hint: '[env]'\n---\nDeploy to $1.\n")
        make_skill_dir(user_dir, "hidden", "---\ndescription: d\nuser-invocable: false\n---\nbody\n")
        self.store = SkillStore(user_dir=user_dir, project_dirs=[], registry=ToolRegistry())
        self.store.scan()
        set_skill_store(self.store)
        self.addCleanup(set_skill_store, None)
        self.ui = MagicMock()
        self.ui.console = Console(record=True, width=120, theme=theme)
        self.handler = CommandHandler(self.ui)

    def output(self) -> str:
        return self.ui.console.export_text()

    def test_slug_returns_skill_action(self):
        result = self.handler.handle("/deploy prod")
        self.assertIsInstance(result, SkillAction)
        self.assertEqual(result.text, "Deploy to prod.\n")

    def test_hidden_skill_not_invocable(self):
        result = self.handler.handle("/hidden")
        self.assertIsNone(result)
        self.assertIn("unknown command", self.output())

    def test_unknown_slug_still_unknown_command(self):
        self.assertIsNone(self.handler.handle("/nosuch"))
        self.assertIn("unknown command", self.output())

    def test_builtin_wins_over_skill(self):
        make_skill_dir(Path(self.tmp.name), "help", "---\ndescription: shadowed\n---\nbody\n")
        self.store.scan()
        self.assertIsNone(self.handler.handle("/help"))  # built-in help printed, no SkillAction
        self.assertIn("/skills", self.output())

    def test_skills_table(self):
        self.assertIsNone(self.handler.handle("/skills"))
        text = self.output()
        self.assertIn("deploy", text)
        self.assertIn("user", text)
        self.assertIn("model", text)  # invocable-by column mentions model-only skill

    def test_skills_reload(self):
        make_skill_dir(Path(self.tmp.name), "fresh", "---\ndescription: d\n---\nbody\n")
        self.assertIsNone(self.handler.handle("/skills reload"))
        self.assertIsNotNone(self.store.get("fresh"))

    def test_hostile_description_is_markup_safe(self):
        make_skill_dir(Path(self.tmp.name), "evil", "---\ndescription: '[/bold red]boom[bold]'\n---\nbody\n")
        self.store.scan()
        self.assertIsNone(self.handler.handle("/skills"))  # must not raise MarkupError

    def test_no_store_degrades_gracefully(self):
        set_skill_store(None)
        self.assertIsNone(self.handler.handle("/skills"))
        self.assertIn("no skills", self.output().lower())

    def test_alias_commands_are_flagged_as_shadowed(self):
        # "temp" and "exit" are not keys of completer.COMMANDS (only the
        # canonical "/temperature" and "/quit" are) but handle() still
        # treats them as built-in via TEMPERATURE_COMMANDS/QUIT_COMMANDS —
        # /skills must flag them as shadowed too, or the listing misleads
        # the user into thinking the slug is reachable.
        make_skill_dir(Path(self.tmp.name), "temp", "---\ndescription: d\n---\nbody\n")
        make_skill_dir(Path(self.tmp.name), "exit", "---\ndescription: d\n---\nbody\n")
        self.store.scan()
        self.assertIsNone(self.handler.handle("/skills"))
        lines = self.output().splitlines()

        def line_for(slug: str) -> str:
            return next(line for line in lines if line.strip().startswith(f"/{slug} "))

        self.assertIn("(shadowed by a built-in command)", line_for("temp"))
        self.assertIn("(shadowed by a built-in command)", line_for("exit"))
        self.assertNotIn("shadowed", line_for("deploy"))


class TestUppercaseSlugReachable(unittest.TestCase):
    """Regression: a skill directory with an uppercase name must still be
    reachable via /slug — sanitize_slug lowercases so the store key matches
    what CommandHandler.handle produces after it lowercases the typed
    command, and what the completer offers matches what's actually usable."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        user_dir = Path(self.tmp.name)
        make_skill_dir(user_dir, "Deploy", "---\ndescription: deploy it\n---\nDeploy now.\n")
        self.store = SkillStore(user_dir=user_dir, project_dirs=[], registry=ToolRegistry())
        self.store.scan()
        set_skill_store(self.store)
        self.addCleanup(set_skill_store, None)
        self.ui = MagicMock()
        self.ui.console = Console(record=True, width=120, theme=theme)
        self.handler = CommandHandler(self.ui)

    def test_lowercase_command_reaches_uppercase_named_skill(self):
        result = self.handler.handle("/deploy")
        self.assertIsInstance(result, SkillAction)
        self.assertEqual(result.text, "Deploy now.\n")

    def test_completer_offers_the_actually_reachable_slug(self):
        completer = CommandCompleter(self.ui)
        completions = [c.text for c in completer.get_completions(Document("/dep"), None)]
        self.assertIn("/deploy", completions)
        self.assertNotIn("/Deploy", completions)


if __name__ == "__main__":
    unittest.main()
