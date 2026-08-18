import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.skills.render import render_body  # noqa: E402

DIR = Path("/tmp/sk")


class TestRenderBody(unittest.TestCase):
    def test_positional_and_all(self):
        out = render_body("do $1 then $2 with all: $ARGUMENTS", ["a", "b"], [], DIR)
        self.assertEqual(out, "do a then b with all: a b")

    def test_named_arguments(self):
        out = render_body("ticket=$ticket branch=$branch", ["T-1", "main"], ["ticket", "branch"], DIR)
        self.assertEqual(out, "ticket=T-1 branch=main")

    def test_name_boundary_not_partial(self):
        out = render_body("$ticket_id stays", ["T-1"], ["ticket"], DIR)
        self.assertEqual(out, "$ticket_id stays")

    def test_missing_arguments_become_empty(self):
        self.assertEqual(render_body("[$1][$9][$ARGUMENTS]", [], [], DIR), "[][][]")

    def test_skill_dir_and_claude_alias(self):
        out = render_body("a=${SKILL_DIR} b=${CLAUDE_SKILL_DIR}", [], [], DIR)
        self.assertEqual(out, f"a={DIR} b={DIR}")

    def test_backslash_escapes(self):
        self.assertEqual(render_body(r"price \$1.00 and $1", ["x"], [], DIR), "price $1.00 and x")
        self.assertEqual(render_body(r"\$ARGUMENTS", ["x"], [], DIR), "$ARGUMENTS")

    def test_single_pass_no_rescan(self):
        # A substituted value containing $2 must not be expanded again.
        self.assertEqual(render_body("$1", ["$2"], [], DIR), "$2")


if __name__ == "__main__":
    unittest.main()
