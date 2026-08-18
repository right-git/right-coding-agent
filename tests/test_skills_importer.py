import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.skills.importer import find_candidates, import_skills  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402


class TestImporter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.foreign = base / "claude" / "skills"
        self.foreign.mkdir(parents=True)
        self.target = base / "ours"
        self.target.mkdir()
        skill_dir = make_skill_dir(self.foreign, "guide", "---\ndescription: a guide\n---\nbody\n")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "extra.md").write_text("ref", encoding="utf-8")
        make_skill_dir(self.foreign, "dupe", "---\ndescription: already ours\n---\nbody\n")
        (self.foreign / "not-a-skill").mkdir()  # no SKILL.md -> not a candidate

    def test_find_candidates(self):
        candidates = find_candidates([("claude-user", self.foreign)], existing_slugs={"dupe"})
        by_slug = {candidate.slug: candidate for candidate in candidates}
        self.assertIn("guide", by_slug)
        self.assertFalse(by_slug["guide"].collides)
        self.assertTrue(by_slug["dupe"].collides)
        self.assertNotIn("not-a-skill", by_slug)

    def test_import_copies_whole_directory(self):
        candidates = find_candidates([("claude-user", self.foreign)], existing_slugs=set())
        copied, skipped, failed = import_skills(candidates, self.target, names=["guide"])
        self.assertEqual(copied, ["guide"])
        self.assertTrue((self.target / "guide" / "references" / "extra.md").is_file())

    def test_never_overwrites(self):
        (self.target / "guide").mkdir()
        (self.target / "guide" / "SKILL.md").write_text("mine", encoding="utf-8")
        candidates = find_candidates([("claude-user", self.foreign)], existing_slugs=set())
        copied, skipped, failed = import_skills(candidates, self.target, names=["guide"])
        self.assertEqual(copied, [])
        self.assertEqual(skipped, ["guide"])
        self.assertEqual((self.target / "guide" / "SKILL.md").read_text(encoding="utf-8"), "mine")

    def test_all_means_non_colliding(self):
        candidates = find_candidates([("claude-user", self.foreign)], existing_slugs={"dupe"})
        copied, skipped, failed = import_skills(candidates, self.target, names=None)
        self.assertEqual(copied, ["guide"])
        self.assertEqual(skipped, ["dupe"])


if __name__ == "__main__":
    unittest.main()
