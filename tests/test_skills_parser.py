import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.skills.skill import (  # noqa: E402
    MAX_SKILL_DESCRIPTION_CHARS,
    parse_skill,
    sanitize_slug,
)


def make_skill_dir(root: Path, name: str, content: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(content, encoding="utf-8")
    return directory


FULL = """---
name: Humanizer Pro
description: Remove AI writing patterns.
when_to_use: Use when editing text.
arguments: [target, style]
argument-hint: "[file] [style]"
disable-model-invocation: true
user-invocable: true
license: MIT
allowed-tools: Read Write
unknown-field: whatever
---

# Humanizer

Body text here.
"""


class TestParseSkill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_full_frontmatter(self):
        directory = make_skill_dir(self.root, "humanizer", FULL)
        skill = parse_skill(directory, "user")
        self.assertEqual(skill.slug, "humanizer")
        self.assertEqual(skill.display_name, "Humanizer Pro")
        self.assertEqual(skill.description, "Remove AI writing patterns. Use when editing text.")
        self.assertEqual(skill.argument_names, ["target", "style"])
        self.assertEqual(skill.argument_hint, "[file] [style]")
        self.assertFalse(skill.model_invocable)
        self.assertTrue(skill.user_invocable)
        self.assertEqual(skill.scope, "user")
        self.assertIn("Body text here.", skill.load_body())

    def test_minimal_uses_first_paragraph_as_description(self):
        directory = make_skill_dir(self.root, "min", "---\nname: min\n---\n\n# Title\nFirst paragraph.\n\nSecond.\n")
        skill = parse_skill(directory, "project")
        self.assertIn("First paragraph.", skill.description)
        self.assertTrue(skill.model_invocable)
        self.assertTrue(skill.user_invocable)

    def test_no_frontmatter_at_all(self):
        directory = make_skill_dir(self.root, "plain", "Just instructions, no frontmatter.\n")
        skill = parse_skill(directory, "user")
        self.assertIn("Just instructions", skill.description)
        self.assertEqual(skill.load_body().strip(), "Just instructions, no frontmatter.")

    def test_arguments_as_string(self):
        directory = make_skill_dir(self.root, "s", "---\ndescription: d\narguments: ticket branch\n---\nbody\n")
        self.assertEqual(parse_skill(directory, "user").argument_names, ["ticket", "branch"])

    def test_description_cap(self):
        directory = make_skill_dir(self.root, "big", f"---\ndescription: {'x' * 3000}\n---\nbody\n")
        self.assertLessEqual(len(parse_skill(directory, "user").description), MAX_SKILL_DESCRIPTION_CHARS)

    def test_broken_yaml_returns_none(self):
        directory = make_skill_dir(self.root, "bad", "---\ndescription: [unclosed\n---\nbody\n")
        self.assertIsNone(parse_skill(directory, "user"))

    def test_missing_file_returns_none(self):
        directory = self.root / "ghost"
        directory.mkdir()
        self.assertIsNone(parse_skill(directory, "user"))

    def test_empty_body_returns_none(self):
        directory = make_skill_dir(self.root, "empty", "---\ndescription: d\n---\n\n")
        self.assertIsNone(parse_skill(directory, "user"))

    def test_flag_string_values(self):
        directory = make_skill_dir(self.root, "f", "---\ndescription: d\nuser-invocable: 'false'\n---\nbody\n")
        self.assertFalse(parse_skill(directory, "user").user_invocable)


class TestSanitizeSlug(unittest.TestCase):
    def test_keeps_dashes_and_underscores(self):
        self.assertEqual(sanitize_slug("k8s-debug"), "k8s-debug")
        self.assertEqual(sanitize_slug("my skill!"), "my_skill")
        self.assertEqual(sanitize_slug("///"), "x")


if __name__ == "__main__":
    unittest.main()
