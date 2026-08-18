# Skills Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An Agent Skills layer (SKILL.md standard, Claude Code-compatible): skills discovered from `.agents/skills` + `~/.right-agent/skills`, registered as `skill__<slug>` pseudo-tools whose invocation delivers the rendered SKILL.md body through a dedicated `skills` channel in `run_tools` results and history recaps, `/slug` + `/skills` REPL commands, an importer for Claude Code/Codex skills, and a `skills` CLI.

**Architecture:** New package `src/llm/tools/skills/` mirroring `mcp/`. A `SkillStore` scans directories, parses frontmatter, and registers one `StructuredTool` per skill with source `skill:<slug>`. The pseudo-tool records the body into a per-run ContextVar bucket (the `get_tool`-contracts mechanic); `run_tools` ships it as a `skills` JSON field exempt from the generic clip; `compact_finished_turn` carries it verbatim in recaps under budgets. REPL slugs route through `CommandHandler` returning a `SkillAction` the main loop feeds in as the user turn.

**Tech Stack:** Python 3.12, LangChain `StructuredTool`, PyYAML (already in the tree), unittest (pytest is NOT installed), `uv`.

**Spec:** `docs/superpowers/specs/2026-08-18-skills-support-design.md`

## Global Constraints

- Tests: `unittest` only; run with `uv run python -m unittest tests.<module>`; every test file prepends the repo root to `sys.path`. Full suite baseline on main: 1 failure + 2 errors + 8 skips pre-existing — the gate is **no NEW failures**.
- Lint: `bash lint.sh` (black, 120 columns, then flake8) must pass before each commit.
- Hermeticity: no test and no module import may scan the real home directory or repo skill dirs; every path is a constructor/function parameter; tests use `tempfile.TemporaryDirectory`.
- The system prompt must stay byte-stable across turns: skill data NEVER goes into it. Registry churn is fine (tool count is frozen on first `session_context` build).
- Skill-supplied text (descriptions, bodies) is untrusted: all REPL prints of it use `markup=False, highlight=False`.
- Tools never raise: failures return strings (`[skill error] ...`).
- Exact values: tool prefix `skill__`, source `skill:<slug>`, `MAX_SKILL_DESCRIPTION_CHARS = 1536`, `MAX_SKILL_BODY_CHARS = 12_000`, `RECAP_SKILL_CHARS = 8_000`, `RECAP_SKILLS_TOTAL_CHARS = 16_000`, `MAX_TOOL_NAME_LENGTH = 64` with `_<8-hex>` hash suffix on overflow.
- Import never overwrites an existing skill directory.
- Commit after every task; end commit messages with the repo's `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.

---

### Task 1: Skill dataclass and SKILL.md parser

**Files:**
- Create: `src/llm/tools/skills/__init__.py`
- Create: `src/llm/tools/skills/skill.py`
- Test: `tests/test_skills_parser.py`

**Interfaces:**
- Consumes: nothing (leaf module; stdlib + yaml + loguru only).
- Produces: `Skill` dataclass (fields below), `parse_skill(directory: Path, scope: str) -> Skill | None`, `sanitize_slug(raw: str) -> str`, `split_frontmatter(text: str) -> tuple[str | None, str]`, `MAX_SKILL_DESCRIPTION_CHARS = 1536`. `Skill.load_body() -> str` re-reads the file (may raise `OSError`). `Skill` fields: `slug: str` (sanitized dir name), `directory: Path`, `scope: str` ("project"|"user"), `display_name: str`, `description: str` (combined+capped), `argument_names: list[str]`, `argument_hint: str`, `model_invocable: bool`, `user_invocable: bool`.

- [ ] **Step 1: Write the failing tests** — `tests/test_skills_parser.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure** — `uv run python -m unittest tests.test_skills_parser -v` — expected: `ModuleNotFoundError: No module named 'src.llm.tools.skills'`.

- [ ] **Step 3: Implement.** `src/llm/tools/skills/__init__.py`:

```python
"""Agent Skills: SKILL.md discovery, rendering, and registry integration."""
```

`src/llm/tools/skills/skill.py`:

```python
"""The `Skill` record and the SKILL.md parser.

A skill is a directory with a SKILL.md file: YAML frontmatter between ---
markers, then a markdown body of instructions. Unknown frontmatter fields
(including Claude Code extensions we don't support) are ignored — an
imported foreign skill must always load. Broken skills are skipped with a
warning, never an error. The body is NOT cached on the dataclass: it is
re-read from disk at every invocation so edits take effect immediately.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config.logging import logger

MAX_SKILL_DESCRIPTION_CHARS = 1536
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_slug(raw: str) -> str:
    """Directory name -> command/tool-safe slug; falls back to 'x'."""
    return _SLUG_RE.sub("_", str(raw or "").strip()).strip("_-") or "x"


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """(frontmatter yaml text or None, body) — never raises."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return None, text
    return match.group(1), text[match.end():]


def _first_paragraph(body: str) -> str:
    for block in re.split(r"\n\s*\n", body):
        text = " ".join(line.lstrip("# ").strip() for line in block.strip().splitlines()).strip()
        if text:
            return text
    return ""


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _argument_names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split() if part]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


@dataclass(frozen=True)
class Skill:
    slug: str
    directory: Path
    scope: str  # "project" | "user"
    display_name: str
    description: str
    argument_names: list[str]
    argument_hint: str
    model_invocable: bool
    user_invocable: bool

    def load_body(self) -> str:
        """The instruction body, freshly read from disk (may raise OSError)."""
        text = (self.directory / "SKILL.md").read_text(encoding="utf-8")
        return split_frontmatter(text)[1]


def parse_skill(directory: Path, scope: str) -> Skill | None:
    """Parse one skill directory; None (with a logged warning) on any defect."""
    path = directory / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        logger.warning("Skill {} skipped: {}", directory, error)
        return None
    meta_text, body = split_frontmatter(text)
    meta: dict = {}
    if meta_text is not None:
        try:
            loaded = yaml.safe_load(meta_text)
            if not isinstance(loaded, dict):
                raise ValueError(f"frontmatter is {type(loaded).__name__}, not a mapping")
            meta = loaded
        except Exception as error:
            logger.warning("Skill {} skipped: bad frontmatter: {}", directory, error)
            return None
    if not body.strip():
        logger.warning("Skill {} skipped: empty body", directory)
        return None

    description = str(meta.get("description") or "").strip() or _first_paragraph(body)
    when_to_use = str(meta.get("when_to_use") or "").strip()
    if when_to_use:
        description = f"{description} {when_to_use}".strip()
    description = " ".join(description.split())[:MAX_SKILL_DESCRIPTION_CHARS]

    slug = sanitize_slug(directory.name)
    return Skill(
        slug=slug,
        directory=directory,
        scope=scope,
        display_name=str(meta.get("name") or "").strip() or slug,
        description=description,
        argument_names=_argument_names(meta.get("arguments")),
        argument_hint=str(meta.get("argument-hint") or meta.get("argument_hint") or "").strip(),
        model_invocable=not _as_bool(meta.get("disable-model-invocation"), default=False),
        user_invocable=_as_bool(meta.get("user-invocable"), default=True),
    )
```

- [ ] **Step 4: Run to verify pass** — `uv run python -m unittest tests.test_skills_parser -v` — expected: all PASS.
- [ ] **Step 5: Lint and commit**

```bash
bash lint.sh
git add src/llm/tools/skills/ tests/test_skills_parser.py
git commit -m "feat(skills): Skill dataclass and SKILL.md parser"
```

---

### Task 2: Body rendering (argument substitution)

**Files:**
- Create: `src/llm/tools/skills/render.py`
- Test: `tests/test_skills_render.py`

**Interfaces:**
- Consumes: nothing (pure module).
- Produces: `render_body(body: str, arguments: list[str], names: list[str], skill_dir: Path) -> str`.

- [ ] **Step 1: Write the failing tests** — `tests/test_skills_render.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure** — `uv run python -m unittest tests.test_skills_render -v` — expected: `ModuleNotFoundError`.
- [ ] **Step 3: Implement** — `src/llm/tools/skills/render.py`:

```python
"""Substitutions applied to a skill body at invocation time.

One regex pass (re.sub never rescans its own output): $1..$n, $ARGUMENTS,
declared $name tokens, ${SKILL_DIR} (with ${CLAUDE_SKILL_DIR} accepted as an
alias so imported Claude Code skills work unmodified). A backslash directly
before a substitutable token escapes it; missing arguments render empty.
"""

import re
from pathlib import Path


def render_body(body: str, arguments: list[str], names: list[str], skill_dir: Path) -> str:
    values = {name: (arguments[i] if i < len(arguments) else "") for i, name in enumerate(names)}
    name_alt = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    token = r"ARGUMENTS(?![A-Za-z0-9_])|\d+"
    if name_alt:
        token += rf"|(?:{name_alt})(?![A-Za-z0-9_])"
    pattern = re.compile(rf"(\\?)\$({token})|\$\{{(SKILL_DIR|CLAUDE_SKILL_DIR)\}}")

    def replace(match: re.Match) -> str:
        if match.group(3):
            return str(skill_dir)
        if match.group(1):  # escaped: drop the backslash, keep the token
            return "$" + match.group(2)
        found = match.group(2)
        if found == "ARGUMENTS":
            return " ".join(arguments)
        if found.isdigit():
            index = int(found) - 1
            return arguments[index] if 0 <= index < len(arguments) else ""
        return values.get(found, "")

    return pattern.sub(replace, body)
```

- [ ] **Step 4: Run to verify pass** — `uv run python -m unittest tests.test_skills_render -v` — expected: all PASS.
- [ ] **Step 5: Lint and commit**

```bash
bash lint.sh
git add src/llm/tools/skills/render.py tests/test_skills_render.py
git commit -m "feat(skills): body rendering with argument substitution"
```

---

### Task 3: Shared naming helper, delivery channel, and the pseudo-tool adapter

**Files:**
- Create: `src/llm/tools/naming.py`
- Create: `src/llm/tools/skills/channel.py`
- Create: `src/llm/tools/skills/tool.py`
- Modify: `src/llm/tools/mcp/adapter.py:23-41` (delegate `_identifier` to the shared helper)
- Test: `tests/test_skills_tool.py`

**Interfaces:**
- Consumes: `Skill` (Task 1), `render_body` (Task 2).
- Produces: `naming.safe_part(value: str) -> str`, `naming.hashed_identifier(readable: str, raw_key: str) -> str`, `naming.MAX_TOOL_NAME_LENGTH = 64`; `channel.collecting_skill_bodies()` (context manager yielding `dict[str, str]`), `channel.record_skill_body(slug: str, body: str) -> bool`, `channel.MAX_SKILL_BODY_CHARS = 12_000`; `tool.build_skill_tool_name(slug: str) -> str`, `tool.build_skill_tool(skill: Skill, seen_hashes: dict[str, str]) -> StructuredTool`, `tool.SKILL_TOOL_PREFIX = "skill__"`.

- [ ] **Step 1: Write the failing tests** — `tests/test_skills_tool.py`:

```python
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.naming import hashed_identifier  # noqa: E402
from src.llm.tools.skills.channel import collecting_skill_bodies, record_skill_body  # noqa: E402
from src.llm.tools.skills.skill import parse_skill  # noqa: E402
from src.llm.tools.skills.tool import build_skill_tool, build_skill_tool_name  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402


class TestNaming(unittest.TestCase):
    def test_short_name_untouched(self):
        self.assertEqual(build_skill_tool_name("k8s-debug"), "skill__k8s-debug")

    def test_long_name_hash_truncated(self):
        name = build_skill_tool_name("s" * 100)
        self.assertLessEqual(len(name), 64)
        self.assertRegex(name, r"_[0-9a-f]{8}$")

    def test_hashed_identifier_stable(self):
        self.assertEqual(hashed_identifier("x" * 100, "key"), hashed_identifier("x" * 100, "key"))

    def test_mcp_naming_still_works(self):
        from src.llm.tools.mcp.adapter import build_tool_name

        self.assertEqual(build_tool_name("pw", "browser_click"), "mcp__pw__browser_click")
        self.assertLessEqual(len(build_tool_name("s" * 50, "t" * 50)), 64)


class TestChannel(unittest.TestCase):
    def test_record_without_channel_returns_false(self):
        self.assertFalse(record_skill_body("a", "body"))

    def test_record_inside_channel(self):
        with collecting_skill_bodies() as bucket:
            self.assertTrue(record_skill_body("a", "body"))
        self.assertEqual(bucket, {"a": "body"})


class TestSkillTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        directory = make_skill_dir(
            Path(self.tmp.name), "demo", "---\ndescription: demo skill\narguments: [target]\n---\nUse $target now.\n"
        )
        self.skill = parse_skill(directory, "user")
        self.seen: dict[str, str] = {}
        self.tool = build_skill_tool(self.skill, self.seen)

    async def test_first_call_records_body(self):
        with collecting_skill_bodies() as bucket:
            confirmation = await self.tool.ainvoke({"target": "the file", "force": False})
        self.assertIn("loaded", confirmation)
        self.assertEqual(bucket["demo"], "Use the file now.\n")
        self.assertIn("demo", self.seen)

    async def test_second_identical_call_dedupes(self):
        with collecting_skill_bodies():
            await self.tool.ainvoke({"target": "x", "force": False})
        with collecting_skill_bodies() as bucket:
            confirmation = await self.tool.ainvoke({"target": "x", "force": False})
        self.assertIn("already loaded", confirmation)
        self.assertEqual(bucket, {})

    async def test_force_resends(self):
        with collecting_skill_bodies():
            await self.tool.ainvoke({"target": "x", "force": False})
        with collecting_skill_bodies() as bucket:
            await self.tool.ainvoke({"target": "x", "force": True})
        self.assertIn("demo", bucket)

    async def test_changed_arguments_resend(self):
        with collecting_skill_bodies():
            await self.tool.ainvoke({"target": "x", "force": False})
        with collecting_skill_bodies() as bucket:
            await self.tool.ainvoke({"target": "y", "force": False})
        self.assertEqual(bucket["demo"], "Use y now.\n")

    async def test_no_channel_returns_body_inline(self):
        confirmation = await self.tool.ainvoke({"target": "z", "force": False})
        self.assertIn("Use z now.", confirmation)

    async def test_missing_file_returns_error_string(self):
        (self.skill.directory / "SKILL.md").unlink()
        with collecting_skill_bodies():
            result = await self.tool.ainvoke({"target": "x", "force": False})
        self.assertTrue(result.startswith("[skill error]"))

    def test_positional_field_order(self):
        # Interpreter maps positional args onto schema field order: declared
        # names first, force last — skill__demo("x") must bind target="x".
        self.assertEqual(list(self.tool.args), ["target", "force"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `uv run python -m unittest tests.test_skills_tool -v` — expected: `ModuleNotFoundError: No module named 'src.llm.tools.naming'`.
- [ ] **Step 3: Implement.** `src/llm/tools/naming.py`:

```python
"""Shared identifier building for dynamically-registered tools (MCP, skills)."""

import hashlib
import re

_HASH_LENGTH = 8
MAX_TOOL_NAME_LENGTH = 64
_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9]+")


def safe_part(value: str) -> str:
    """Normalize a component to alphanumeric + underscores; fallback to 'x'."""
    normalized = _SAFE_PART_RE.sub("_", str(value or "").strip()).strip("_")
    return normalized or "x"


def hashed_identifier(readable: str, raw_key: str) -> str:
    """`readable` unchanged when short; hash-suffixed truncation past the ceiling."""
    if len(readable) <= MAX_TOOL_NAME_LENGTH:
        return readable
    digest = hashlib.sha256(raw_key.encode()).hexdigest()[:_HASH_LENGTH]
    suffix = f"_{digest}"
    return readable[: MAX_TOOL_NAME_LENGTH - len(suffix)].rstrip("_") + suffix
```

In `src/llm/tools/mcp/adapter.py` replace the `_HASH_LENGTH`/`MAX_TOOL_NAME_LENGTH`/`_SAFE_PART_RE`/`_safe_part`/`_identifier` block (keep `MAX_TOOL_NAME_LENGTH` importable from adapter for existing tests):

```python
from ..naming import MAX_TOOL_NAME_LENGTH, hashed_identifier, safe_part  # noqa: F401


def _identifier(server: str, tool: str) -> str:
    """Build an identifier from server and tool names; hash if too long."""
    return hashed_identifier(f"mcp__{safe_part(server)}__{safe_part(tool)}", f"{server}\x1f{tool}")
```

(`_safe_part` callers inside adapter.py, if any remain, switch to `safe_part`.)

`src/llm/tools/skills/channel.py`:

```python
"""The per-run delivery channel for skill bodies.

Mirrors the get_tool contracts mechanic: a pseudo-tool records the rendered
body here instead of returning it, run_tools ships the bucket as the result
JSON's `skills` field, and history compaction carries it verbatim. When no
channel is open (library use outside run_tools) `record_skill_body` returns
False and the caller falls back to returning the body inline.
"""

from contextlib import contextmanager
from contextvars import ContextVar

MAX_SKILL_BODY_CHARS = 12_000

_BUCKET: ContextVar[dict | None] = ContextVar("skill_bodies", default=None)


@contextmanager
def collecting_skill_bodies():
    bucket: dict[str, str] = {}
    token = _BUCKET.set(bucket)
    try:
        yield bucket
    finally:
        _BUCKET.reset(token)


def record_skill_body(slug: str, body: str) -> bool:
    bucket = _BUCKET.get()
    if bucket is None:
        return False
    if len(body) > MAX_SKILL_BODY_CHARS:
        dropped = len(body) - MAX_SKILL_BODY_CHARS
        body = body[:MAX_SKILL_BODY_CHARS] + f"… [+{dropped} chars truncated — the skill file exceeds the budget]"
    bucket[slug] = body
    return True
```

`src/llm/tools/skills/tool.py`:

```python
"""Skill -> registry pseudo-tool: calling it delivers the rendered body."""

import hashlib

from langchain_core.tools import StructuredTool

from ..naming import hashed_identifier
from .channel import record_skill_body
from .render import render_body
from .skill import Skill

SKILL_TOOL_PREFIX = "skill__"
SKILL_TOOL_NOTE = (
    "Calling this tool loads the skill's instructions — they arrive in this "
    "result's `skills` field; follow them for the current task."
)


def build_skill_tool_name(slug: str) -> str:
    return hashed_identifier(f"{SKILL_TOOL_PREFIX}{slug}", f"skill:{slug}")


def _args_schema(skill: Skill) -> dict:
    properties: dict = {}
    for name in skill.argument_names or ["arguments"]:
        properties[name] = {"type": "string", "description": f"value for ${name} in the skill body"}
    properties["force"] = {"type": "boolean", "default": False, "description": "resend even if already loaded"}
    return {"type": "object", "properties": properties}


def build_skill_tool(skill: Skill, seen_hashes: dict) -> StructuredTool:
    async def call(**kwargs) -> str:
        try:
            force = bool(kwargs.pop("force", False))
            body = skill.load_body()
            if skill.argument_names:
                positional = [str(kwargs.get(name) or "") for name in skill.argument_names]
            else:
                free = str(kwargs.get("arguments") or "")
                positional = [free] if free else []
            rendered = render_body(body, positional, skill.argument_names, skill.directory)
            digest = hashlib.sha256(rendered.encode()).hexdigest()
            if not force and seen_hashes.get(skill.slug) == digest:
                return f"skill '{skill.slug}' already loaded earlier this session — pass force=True to resend"
            if not record_skill_body(skill.slug, rendered):
                return rendered  # library use outside run_tools
            seen_hashes[skill.slug] = digest
            return f"skill '{skill.slug}' loaded — instructions arrive with this result"
        except Exception as error:
            return f"[skill error] {error}"

    return StructuredTool(
        name=build_skill_tool_name(skill.slug),
        description=f"{skill.description}\n\n{SKILL_TOOL_NOTE}",
        args_schema=_args_schema(skill),
        coroutine=call,
    )
```

- [ ] **Step 4: Run to verify pass** — `uv run python -m unittest tests.test_skills_tool tests.test_mcp_adapter -v` — expected: all PASS (MCP naming behavior pinned by existing tests must not change).
- [ ] **Step 5: Lint and commit**

```bash
bash lint.sh
git add src/llm/tools/naming.py src/llm/tools/skills/ src/llm/tools/mcp/adapter.py tests/test_skills_tool.py
git commit -m "feat(skills): pseudo-tool adapter, delivery channel, shared tool naming"
```

---

### Task 4: SkillStore (scan, precedence, refresh, registration)

**Files:**
- Create: `src/llm/tools/skills/store.py`
- Test: `tests/test_skills_store.py`

**Interfaces:**
- Consumes: `parse_skill`, `Skill`, `sanitize_slug` (Task 1); `build_skill_tool`, `build_skill_tool_name` (Task 3); `ToolRegistry` from `src/llm/tools/meta/registry.py`; `render_body` (Task 2).
- Produces: `SkillStore(user_dir: Path | None, project_dirs: list[Path], registry: ToolRegistry)` with `.scan() -> None`, `.refresh() -> bool`, `.get(slug) -> Skill | None`, `.skills: dict[str, Skill]`, `.seen_hashes: dict[str, str]`, `.user_commands() -> list[tuple[str, str, str]]` (slug, hint, description), `.render_for_user(slug: str, arg_text: str) -> str` (shlex-split, render, mark seen; raises `KeyError` on unknown slug); module functions `project_skills_dirs(cwd: Path) -> list[Path]` (nearest first, walking up to the git root or filesystem root), `get_skill_store() -> SkillStore | None`, `set_skill_store(store) -> None`, `start_skill_store(cwd: Path | None = None) -> SkillStore` (builds defaults, scans, installs singleton); constants `DEFAULT_USER_SKILLS_DIR = Path.home() / ".right-agent" / "skills"`, `PROJECT_SKILLS_SUBPATH = Path(".agents") / "skills"`.

- [ ] **Step 1: Write the failing tests** — `tests/test_skills_store.py`:

```python
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.meta.registry import ToolRegistry  # noqa: E402
from src.llm.tools.skills.store import SkillStore, project_skills_dirs  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402

BODY = "---\ndescription: d{n}\n---\nbody {n}\n"


class TestProjectDirs(unittest.TestCase):
    def test_walk_up_to_git_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            nested = root / "apps" / "web"
            nested.mkdir(parents=True)
            (root / ".agents" / "skills").mkdir(parents=True)
            (nested / ".agents" / "skills").mkdir(parents=True)
            dirs = project_skills_dirs(nested)
            self.assertEqual(dirs, [nested / ".agents" / "skills", root / ".agents" / "skills"])


class TestSkillStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.user_dir = base / "user"
        self.project_dir = base / "project"
        self.user_dir.mkdir()
        self.project_dir.mkdir()
        self.registry = ToolRegistry()

    def store(self) -> SkillStore:
        return SkillStore(user_dir=self.user_dir, project_dirs=[self.project_dir], registry=self.registry)

    def test_scan_registers_with_source(self):
        make_skill_dir(self.user_dir, "alpha", BODY.format(n=1))
        store = self.store()
        store.scan()
        self.assertIsNotNone(self.registry.get("skill__alpha"))
        self.assertEqual(self.registry.source_of("skill__alpha"), "skill:alpha")

    def test_project_beats_user(self):
        make_skill_dir(self.user_dir, "dup", "---\ndescription: user one\n---\nuser body\n")
        make_skill_dir(self.project_dir, "dup", "---\ndescription: project one\n---\nproject body\n")
        store = self.store()
        store.scan()
        self.assertEqual(store.get("dup").scope, "project")

    def test_disable_model_invocation_not_registered(self):
        make_skill_dir(self.user_dir, "manual", "---\ndescription: d\ndisable-model-invocation: true\n---\nbody\n")
        store = self.store()
        store.scan()
        self.assertIsNone(self.registry.get("skill__manual"))
        self.assertIsNotNone(store.get("manual"))  # still user-invocable

    def test_refresh_detects_add_and_remove(self):
        store = self.store()
        store.scan()
        self.assertFalse(store.refresh())
        directory = make_skill_dir(self.user_dir, "late", BODY.format(n=2))
        self.assertTrue(store.refresh())
        self.assertIsNotNone(self.registry.get("skill__late"))
        (directory / "SKILL.md").unlink()
        directory.rmdir()
        self.assertTrue(store.refresh())
        self.assertIsNone(self.registry.get("skill__late"))

    def test_refresh_detects_frontmatter_change(self):
        directory = make_skill_dir(self.user_dir, "mut", BODY.format(n=3))
        store = self.store()
        store.scan()
        import os
        import time

        content = "---\ndescription: changed\nuser-invocable: false\n---\nbody\n"
        (directory / "SKILL.md").write_text(content, encoding="utf-8")
        os.utime(directory / "SKILL.md", (time.time() + 5, time.time() + 5))
        self.assertTrue(store.refresh())
        self.assertFalse(store.get("mut").user_invocable)

    def test_render_for_user_marks_seen(self):
        make_skill_dir(self.user_dir, "task", "---\ndescription: d\n---\ndo $ARGUMENTS\n")
        store = self.store()
        store.scan()
        rendered = store.render_for_user("task", 'fix "the bug"')
        self.assertEqual(rendered, "do fix the bug\n")
        self.assertIn("task", store.seen_hashes)

    def test_user_commands_lists_only_user_invocable(self):
        make_skill_dir(self.user_dir, "visible", BODY.format(n=4))
        make_skill_dir(self.user_dir, "hidden", "---\ndescription: d\nuser-invocable: false\n---\nbody\n")
        store = self.store()
        store.scan()
        slugs = [slug for slug, _, _ in store.user_commands()]
        self.assertIn("visible", slugs)
        self.assertNotIn("hidden", slugs)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `uv run python -m unittest tests.test_skills_store -v` — expected: `ModuleNotFoundError`.
- [ ] **Step 3: Implement** — `src/llm/tools/skills/store.py`:

```python
"""SkillStore: discovery, precedence, live refresh, registry registration.

Precedence on slug collision: project (nearest .agents/skills first, walking
from cwd up to the git root) beats user (~/.right-agent/skills); the loser is
skipped with a warning. Only frontmatter is cached — bodies are read at
invocation. refresh() is cheap (directory listings + SKILL.md mtimes) and is
called at the top of every user turn; a change triggers a full rescan with
register/unregister against the shared ToolRegistry. The session dedupe map
(seen_hashes) survives rescans on purpose: an unchanged body stays "already
delivered" across reloads.
"""

import shlex
from pathlib import Path

from src.config.logging import logger

from ..meta.registry import ToolRegistry
from .render import render_body
from .skill import Skill, parse_skill
from .tool import build_skill_tool, build_skill_tool_name

DEFAULT_USER_SKILLS_DIR = Path.home() / ".right-agent" / "skills"
PROJECT_SKILLS_SUBPATH = Path(".agents") / "skills"

_store: "SkillStore | None" = None


def project_skills_dirs(cwd: Path) -> list[Path]:
    """Every existing .agents/skills from cwd up to the git root, nearest first."""
    dirs: list[Path] = []
    current = cwd.resolve()
    while True:
        candidate = current / PROJECT_SKILLS_SUBPATH
        if candidate.is_dir():
            dirs.append(candidate)
        if (current / ".git").exists() or current.parent == current:
            return dirs
        current = current.parent


class SkillStore:
    def __init__(self, *, user_dir: Path | None, project_dirs: list[Path], registry: ToolRegistry) -> None:
        self.user_dir = user_dir
        self.project_dirs = list(project_dirs)
        self.registry = registry
        self.skills: dict[str, Skill] = {}
        self.seen_hashes: dict[str, str] = {}
        self._registered: dict[str, str] = {}  # slug -> tool name
        self._fingerprint: tuple = ()

    # ------------------------------------------------------------- scanning

    def _scoped_dirs(self) -> list[tuple[str, Path]]:
        scoped = [("project", directory) for directory in self.project_dirs]
        if self.user_dir is not None:
            scoped.append(("user", self.user_dir))
        return scoped

    def _skill_files(self) -> list[Path]:
        files = []
        for _, directory in self._scoped_dirs():
            if not directory.is_dir():
                continue
            for child in sorted(directory.iterdir()):
                if (child / "SKILL.md").is_file():
                    files.append(child / "SKILL.md")
        return files

    def _current_fingerprint(self) -> tuple:
        entries = []
        for path in self._skill_files():
            try:
                entries.append((str(path), path.stat().st_mtime_ns))
            except OSError:
                continue
        return tuple(entries)

    def scan(self) -> None:
        found: dict[str, Skill] = {}
        for scope, directory in self._scoped_dirs():
            if not directory.is_dir():
                continue
            for child in sorted(directory.iterdir()):
                if not (child / "SKILL.md").is_file():
                    continue
                skill = parse_skill(child, scope)
                if skill is None:
                    continue
                if skill.slug in found:
                    logger.warning("Skill {} shadowed by {}", child, found[skill.slug].directory)
                    continue
                found[skill.slug] = skill

        for slug in list(self._registered):
            if slug not in found or not found[slug].model_invocable:
                self.registry.unregister(self._registered.pop(slug))
        for slug, skill in found.items():
            if not skill.model_invocable:
                continue
            name = build_skill_tool_name(slug)
            if slug in self._registered:
                self.registry.unregister(name)  # re-register with fresh frontmatter
            try:
                self.registry.register(build_skill_tool(skill, self.seen_hashes), source=f"skill:{slug}")
                self._registered[slug] = name
            except ValueError as error:
                logger.warning("Skill {} not registered: {}", slug, error)
        self.skills = found
        self._fingerprint = self._current_fingerprint()
        logger.info("Skill scan: {} skill(s), {} registered", len(found), len(self._registered))

    def refresh(self) -> bool:
        """Rescan only when the directories changed; True when they did."""
        current = self._current_fingerprint()
        if current == self._fingerprint:
            return False
        self.scan()
        return True

    # ------------------------------------------------------------ REPL side

    def get(self, slug: str) -> Skill | None:
        return self.skills.get(slug)

    def user_commands(self) -> list[tuple[str, str, str]]:
        return [
            (skill.slug, skill.argument_hint, skill.description)
            for skill in sorted(self.skills.values(), key=lambda s: s.slug)
            if skill.user_invocable
        ]

    def render_for_user(self, slug: str, arg_text: str) -> str:
        """Rendered body for a /slug invocation; marks the content as seen."""
        skill = self.skills[slug]
        try:
            arguments = shlex.split(arg_text)
        except ValueError:
            arguments = arg_text.split()
        rendered = render_body(skill.load_body(), arguments, skill.argument_names, skill.directory)
        import hashlib

        self.seen_hashes[slug] = hashlib.sha256(rendered.encode()).hexdigest()
        return rendered


def get_skill_store() -> SkillStore | None:
    return _store


def set_skill_store(store: SkillStore | None) -> None:
    global _store
    _store = store


def start_skill_store(cwd: Path | None = None) -> SkillStore:
    """Build the real store (default dirs), scan, and install the singleton."""
    from ..meta.defaults import get_registry

    base = cwd or Path.cwd()
    store = SkillStore(
        user_dir=DEFAULT_USER_SKILLS_DIR,
        project_dirs=project_skills_dirs(base),
        registry=get_registry(),
    )
    store.scan()
    set_skill_store(store)
    return store
```

- [ ] **Step 4: Run to verify pass** — `uv run python -m unittest tests.test_skills_store -v` — expected: all PASS.
- [ ] **Step 5: Lint and commit**

```bash
bash lint.sh
git add src/llm/tools/skills/store.py tests/test_skills_store.py
git commit -m "feat(skills): SkillStore with precedence, refresh, and registration"
```

---

### Task 5: Meta-layer integration (`[Skill]` tag, `only_skills`, `skills` result field)

**Files:**
- Modify: `src/llm/tools/meta/registry.py` (add `SKILL_SOURCE_PREFIX`, tag in `brief()`)
- Modify: `src/llm/tools/meta/tool.py` (`search_tools` filter, `run_tools` channel + docstring)
- Test: `tests/test_skills_meta.py`

**Interfaces:**
- Consumes: `collecting_skill_bodies` (Task 3), `build_skill_tool`/`parse_skill` for fixtures, `set_registry` from `meta/defaults.py`.
- Produces: `registry.SKILL_SOURCE_PREFIX = "skill:"`; `search_tools(query, only_mcp=False, server="", only_skills=False)`; run_tools result JSON gains a `skills` field that survives the generic clip.

- [ ] **Step 1: Write the failing tests** — `tests/test_skills_meta.py`:

```python
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.meta.defaults import set_registry  # noqa: E402
from src.llm.tools.meta.registry import ToolRegistry  # noqa: E402
from src.llm.tools.meta.tool import run_tools, search_tools  # noqa: E402
from src.llm.tools.skills.skill import parse_skill  # noqa: E402
from src.llm.tools.skills.tool import build_skill_tool  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402


class TestSkillsMeta(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        directory = make_skill_dir(
            Path(self.tmp.name), "guide", "---\ndescription: coding guide for widgets\n---\nAlways use widgets.\n"
        )
        self.skill = parse_skill(directory, "user")
        self.registry = ToolRegistry()
        self.registry.register(build_skill_tool(self.skill, {}), source="skill:guide")
        set_registry(self.registry)
        self.addCleanup(set_registry, None)

    def test_brief_tags_skill(self):
        line = self.registry.brief(self.registry.get("skill__guide"))
        self.assertIn("[Skill]", line)

    def test_only_skills_filter(self):
        listing = asyncio.run(search_tools("widgets", only_skills=True))
        self.assertIn("skill__guide", listing)

    def test_only_both_filters_rejected(self):
        listing = asyncio.run(search_tools("x", only_mcp=True, only_skills=True))
        self.assertIn("only one", listing)

    def test_run_tools_ships_skills_field(self):
        content = asyncio.run(run_tools.ainvoke({"code": 'skill__guide()\nreturn "done"'}))
        payload = json.loads(content)
        self.assertEqual(payload["skills"]["guide"], "Always use widgets.\n")
        self.assertEqual(payload["result"], "done")

    def test_skills_field_survives_generic_clip(self):
        # A script that prints far past MAX_RESULT_CHARS: the clipped JSON is
        # rebuilt so the skills field stays intact and the payload stays JSON.
        code = 'skill__guide(force=True)\nfor i in range(60):\n    print("y" * 1000)\nreturn "ok"'
        content = asyncio.run(run_tools.ainvoke({"code": code}))
        payload = json.loads(content)
        self.assertEqual(payload["skills"]["guide"], "Always use widgets.\n")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `uv run python -m unittest tests.test_skills_meta -v` — expected: FAIL (`brief` has no `[Skill]` tag, `search_tools` has no `only_skills`, no `skills` field).
- [ ] **Step 3: Implement.** In `src/llm/tools/meta/registry.py` add below `MCP_SOURCE_PREFIX`:

```python
SKILL_SOURCE_PREFIX = "skill:"
```

and in `brief()` after the MCP branch:

```python
        elif source.startswith(SKILL_SOURCE_PREFIX):
            line += " [Skill]"
```

In `src/llm/tools/meta/tool.py`:

1. Import: `from .registry import SEARCH_LIMIT, SKILL_SOURCE_PREFIX, ToolRegistry` and `from ..skills.channel import collecting_skill_bodies`.
2. `search_tools` becomes:

```python
async def search_tools(query: str, only_mcp: bool = False, server: str = "", only_skills: bool = False) -> str:
```

with, at the top of the body (after the `server` branch):

```python
    if only_mcp and only_skills:
        return "Pass only one of only_mcp / only_skills."
    source_prefix = "mcp:" if only_mcp else (SKILL_SOURCE_PREFIX if only_skills else None)
```

and the final no-matches return:

```python
    if not matches:
        if only_mcp:
            return "No MCP tools are registered."
        if only_skills:
            return "No skills are registered. Skills live in .agents/skills and ~/.right-agent/skills."
        return "No tools are registered."
```

3. In `run_tools`, widen the `with` line and rebuild the payload after clipping:

```python
        with collecting_images() as images, counting_script_calls() as calls, collecting_skill_bodies() as skill_bodies:
            outcome = await interpreter.run(code)
```

and replace the final `return _clip(...)` line with:

```python
        serialized = _clip(json.dumps(outcome, ensure_ascii=False, default=repr))
        if skill_bodies:
            try:
                rebuilt = json.loads(serialized)
            except ValueError:  # the generic clip cut mid-structure
                rebuilt = {"result": serialized}
            rebuilt["skills"] = dict(skill_bodies)
            serialized = json.dumps(rebuilt, ensure_ascii=False)
        return serialized, images
```

Add `len(skill_bodies)` to the `logger.info` line (`... contracts [{}] skills [{}] images [{}] ...`).

4. Docstring additions to `run_tools` (the model-facing contract): in the discovery sentence, after the `only_mcp` clause, add: `pass only_skills=True to browse only skills — reusable instruction packs listed with a [Skill] marker. Calling a skill__<name> tool loads its instructions into this result's `skills` field (dedup: an unchanged skill answers "already loaded"; pass force=True to resend); follow those instructions for the task.` In the `Returns:` section, after `contracts (...)`, add: `` `skills` (instruction bodies of every skill the script invoked), ``.

- [ ] **Step 4: Run to verify pass** — `uv run python -m unittest tests.test_skills_meta tests.test_meta_tools -v` — expected: all PASS (existing meta tests must not regress).
- [ ] **Step 5: Lint and commit**

```bash
bash lint.sh
git add src/llm/tools/meta/registry.py src/llm/tools/meta/tool.py tests/test_skills_meta.py
git commit -m "feat(skills): [Skill] tag, only_skills filter, skills result channel"
```

---

### Task 6: History recap carries skill bodies

**Files:**
- Modify: `src/llm/history.py`
- Test: `tests/test_skills_history.py`

**Interfaces:**
- Consumes: recap machinery in `history.py` (`compact_finished_turn`, `_extract_contracts` pattern, `_clip`).
- Produces: `RECAP_SKILL_CHARS = 8_000`, `RECAP_SKILLS_TOTAL_CHARS = 16_000`, `_extract_skills(content) -> tuple[str, dict[str, str]]`; recaps carry a `skill instructions (kept):` section with per-skill and total budgets, newest first, and a drop note naming `force=True`.

- [ ] **Step 1: Write the failing tests** — `tests/test_skills_history.py`:

```python
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.llm.history import RECAP_SKILLS_TOTAL_CHARS, compact_finished_turn  # noqa: E402


def turn_with_skills(skills: dict) -> list:
    payload = json.dumps({"result": "ok", "logs": [], "error": None, "skills": skills})
    return [
        HumanMessage(content="do it", id="u1"),
        AIMessage(content="", id="a1", tool_calls=[{"name": "run_tools", "args": {"code": "x"}, "id": "c1"}]),
        ToolMessage(content=payload, tool_call_id="c1", name="run_tools", id="t1"),
        AIMessage(content="done", id="a2"),
    ]


class TestSkillRecap(unittest.TestCase):
    def test_body_carried_verbatim(self):
        result = compact_finished_turn(turn_with_skills({"guide": "Always use widgets."}))
        recap = next(m for m in result if isinstance(m, ToolMessage))
        self.assertIn("skill instructions (kept):", recap.content)
        self.assertIn("Always use widgets.", recap.content)

    def test_skills_not_in_result_slices(self):
        result = compact_finished_turn(turn_with_skills({"guide": "SECRETBODY"}))
        recap = next(m for m in result if isinstance(m, ToolMessage))
        self.assertEqual(recap.content.count("SECRETBODY"), 1)  # once in the skills section only

    def test_total_budget_drops_oldest_with_note(self):
        skills = {"old": "o" * 9000, "new": "n" * 9000}  # 16k total: newest fits, oldest partially
        result = compact_finished_turn(turn_with_skills(skills))
        recap = next(m for m in result if isinstance(m, ToolMessage))
        self.assertIn("n" * 100, recap.content)
        self.assertLessEqual(recap.content.count("o"), RECAP_SKILLS_TOTAL_CHARS)
        self.assertIn("force=True", recap.content)  # per-skill clip or drop note teaches recovery

    def test_no_skills_no_section(self):
        result = compact_finished_turn(turn_with_skills({}))
        recap = next(m for m in result if isinstance(m, ToolMessage))
        self.assertNotIn("skill instructions", recap.content)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `uv run python -m unittest tests.test_skills_history -v` — expected: ImportError on `RECAP_SKILLS_TOTAL_CHARS` / missing section.
- [ ] **Step 3: Implement.** In `src/llm/history.py` add constants below `RECAP_CONTRACT_CHARS`:

```python
RECAP_SKILL_CHARS = 8_000
RECAP_SKILLS_TOTAL_CHARS = 16_000
SKILL_DROP_NOTE = "[skill '{slug}' body dropped from history — re-invoke skill__{slug}(force=True) to reload]"
```

Add `_extract_skills` next to `_extract_contracts`:

```python
def _extract_skills(content: str) -> tuple[str, dict[str, str]]:
    """A run_tools result without its `skills` field, plus the skill bodies.

    Skill bodies are durable instructions (the analog of contracts): the
    recap carries them verbatim under budgets instead of trimming them into
    the generic result slice."""
    try:
        payload = json.loads(content)
    except ValueError:
        return content, {}
    if not isinstance(payload, dict) or not isinstance(payload.get("skills"), dict):
        return content, {}
    skills = {str(slug): str(body) for slug, body in payload.pop("skills").items() if str(body).strip()}
    return json.dumps(payload, ensure_ascii=False), skills
```

In `compact_finished_turn`: add `skill_bodies: dict[str, str] = {}` next to `contracts`; in the `ToolMessage` branch after the `_extract_contracts` call:

```python
            if name == "run_tools":
                content, extracted = _extract_contracts(content)
                for contract in extracted:
                    if contract not in contracts:
                        contracts.append(contract)
                content, extracted_skills = _extract_skills(content)
                for slug, body in extracted_skills.items():
                    skill_bodies.pop(slug, None)  # re-delivery moves it to newest
                    skill_bodies[slug] = body
```

After the contracts block in the digest assembly:

```python
    if skill_bodies:
        blocks: list[str] = []
        remaining = RECAP_SKILLS_TOTAL_CHARS
        for slug, body in reversed(list(skill_bodies.items())):  # newest first
            if remaining <= 0:
                blocks.append(SKILL_DROP_NOTE.format(slug=slug))
                continue
            kept = _clip(body, min(RECAP_SKILL_CHARS, remaining))
            if len(kept) < len(body):
                kept += "\n" + SKILL_DROP_NOTE.format(slug=slug).replace("dropped from", "truncated in")
            remaining -= len(kept)
            blocks.append(f"### skill: {slug}\n{kept}")
        digest += "\nskill instructions (kept):\n" + "\n\n".join(blocks)
```

- [ ] **Step 4: Run to verify pass** — `uv run python -m unittest tests.test_skills_history tests.test_history -v` (if `tests/test_history.py` does not exist, run the closest existing history test module found via `grep -l compact_finished_turn tests/*.py`) — expected: all PASS.
- [ ] **Step 5: Lint and commit**

```bash
bash lint.sh
git add src/llm/history.py tests/test_skills_history.py
git commit -m "feat(skills): recaps carry skill bodies verbatim under budgets"
```

---

### Task 7: REPL — `/slug` invocation, `/skills`, completer

**Files:**
- Modify: `src/ui/commands.py` (add `SkillAction`, `/skills` handler, slug routing, help lines)
- Modify: `src/ui/completer.py` (skill slugs as commands, `/skills` subcommands)
- Test: `tests/test_skills_commands.py`

**Interfaces:**
- Consumes: `get_skill_store`/`set_skill_store`, `SkillStore` (Task 4); importer functions arrive in Task 8 — the `/skills import` branch here calls `self._skills_import(argument)` which this task implements as a stub printing `  import arrives with the importer — use: uv run python -m src.main skills import` (Task 8 replaces it).
- Produces: `@dataclass(frozen=True) class SkillAction: text: str` in `src/ui/commands.py`; `CommandHandler.handle` returns `SkillAction` for `/slug` inputs; `/skills` prints the status table; completer offers `/slug` with `argument-hint` and `/skills` with `reload`/`import`.

- [ ] **Step 1: Write the failing tests** — `tests/test_skills_commands.py`:

```python
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console  # noqa: E402

from src.llm.tools.meta.registry import ToolRegistry  # noqa: E402
from src.llm.tools.skills.store import SkillStore, set_skill_store  # noqa: E402
from src.ui.commands import CommandHandler, SkillAction  # noqa: E402
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
        self.ui.console = Console(record=True, width=120)
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `uv run python -m unittest tests.test_skills_commands -v` — expected: ImportError on `SkillAction`.
- [ ] **Step 3: Implement.** In `src/ui/commands.py`:

1. Below `McpAction` add:

```python
@dataclass(frozen=True)
class SkillAction:
    """A `/slug` skill invocation: `text` is the rendered body the main loop
    feeds in as the user turn (the MCP-prompt mechanic, but synchronous)."""

    text: str
```

2. In `handle()` after the `/tool` branch and before the `unknown command` print:

```python
        if command == "/skills":
            return self._skills(argument)
        skill_action = self._skill_prompt(command, argument)
        if skill_action is not None:
            return skill_action
```

Note: built-ins win automatically because this check is last. But `_skill_prompt` returning `None` must fall through to `unknown command` ONLY when the slug is unknown; a known slug whose render fails prints its own error and the method returns a `SkillAction` or `None` — to avoid the double print, `_skill_prompt` returns the sentinel `"handled"`-style by printing and returning a fresh `SkillAction` only on success; on a render failure it prints the error and the caller must not print again. Implement with an explicit marker: `_skill_prompt` returns `SkillAction | None`, and sets `self._skill_handled = True` when it printed an error; the unknown-command print is guarded by `if not getattr(self, "_skill_handled", False)` and the flag reset at the top of `handle()`. Concretely, at the very start of `handle()` add `self._skill_handled = False`, and change the final block to:

```python
        if not self._skill_handled:
            self.console.print(f"  unknown command: {command} — try /help", style="error")
        return None
```

3. New methods at the bottom of the class:

```python
    # ------------------------------------------------------------- skills

    def _skill_store(self):
        from src.llm.tools.skills.store import get_skill_store

        return get_skill_store()

    def _skill_prompt(self, command: str, argument: str) -> "SkillAction | None":
        store = self._skill_store()
        if store is None:
            return None
        slug = command[1:]
        skill = store.get(slug)
        if skill is None or not skill.user_invocable:
            return None
        try:
            return SkillAction(text=store.render_for_user(slug, argument))
        except Exception as error:
            self._skill_handled = True
            self.console.print(f"  skill '{slug}' failed: {error}", style="error", markup=False, highlight=False)
            return None

    def _skills(self, argument: str) -> None:
        store = self._skill_store()
        if store is None or (not store.skills and not argument):
            self.console.print(
                "  no skills found — put them in .agents/skills or ~/.right-agent/skills, "
                "or import with /skills import",
                style="info",
                markup=False,
                highlight=False,
            )
            return None
        sub, _, rest = argument.partition(" ")
        if sub == "reload":
            store.scan()
            self.console.print(f"  reloaded: {len(store.skills)} skill(s)", style="success")
            return None
        if sub == "import":
            return self._skills_import(rest.strip())
        if sub:
            self.console.print("  usage: /skills, /skills reload, /skills import [all|names...]", style="error")
            return None
        from src.ui.completer import COMMANDS as _BUILTINS

        self.console.print()
        for skill in sorted(store.skills.values(), key=lambda item: item.slug):
            who = {(True, True): "user+model", (True, False): "user", (False, True): "model"}.get(
                (skill.user_invocable, skill.model_invocable), "none (dead)"
            )
            head = skill.description[:70]
            line = f"  /{skill.slug:<22} {skill.scope:<8} {who:<11} {head}"
            if f"/{skill.slug}" in _BUILTINS:
                line += "  (shadowed by a built-in command)"
            self.console.print(line, style="info", markup=False, highlight=False)
        self.console.print()
        return None

    def _skills_import(self, argument: str) -> None:
        self.console.print(
            "  import arrives with the importer — use: uv run python -m src.main skills import",
            style="info",
        )
        return None
```

4. Help table: add `("/skills", "skills: list, reload, import; invoke one with /<skill-name> [args]")` after the `/tool` row.

In `src/ui/completer.py`:

1. `COMMANDS` dict: add `"/skills": "skills: list / reload / import"`.
2. In `get_completions`, in the no-space branch after the MCP prompt loop:

```python
            for slug, hint, description in self._skill_commands():
                candidate = f"/{slug}"
                if candidate.startswith(needle) and candidate not in COMMANDS:
                    meta = f"skill {hint}".strip() if hint else description[:40]
                    yield Completion(candidate, start_position=-len(text), display_meta=meta)
```

3. In `_argument_completions`, add:

```python
        if command == "/skills":
            yield from self._option_completions(word, ("reload", "import"))
            return
```

4. New static method:

```python
    @staticmethod
    def _skill_commands() -> list[tuple[str, str, str]]:
        try:
            from src.llm.tools.skills.store import get_skill_store

            store = get_skill_store()
            return store.user_commands() if store is not None else []
        except Exception:
            return []
```

- [ ] **Step 4: Run to verify pass** — `uv run python -m unittest tests.test_skills_commands tests.test_mcp_commands tests.test_tool_pin -v` — expected: all PASS (existing command tests must not regress).
- [ ] **Step 5: Lint and commit**

```bash
bash lint.sh
git add src/ui/commands.py src/ui/completer.py tests/test_skills_commands.py
git commit -m "feat(skills): /<slug> invocation, /skills table and reload, completions"
```

---

### Task 8: Importer (Claude Code / Codex migration)

**Files:**
- Create: `src/llm/tools/skills/importer.py`
- Modify: `src/ui/commands.py` (replace the `_skills_import` stub)
- Test: `tests/test_skills_importer.py`

**Interfaces:**
- Consumes: `parse_skill`, `sanitize_slug` (Task 1); `SkillStore` (Task 4).
- Produces: `ImportCandidate` dataclass (`slug: str`, `source: str`, `path: Path`, `description: str`, `collides: bool`); `default_foreign_sources(home: Path, repo_root: Path | None) -> list[tuple[str, Path]]` (labels: `claude-user`, `codex-user`, `agents-user`, `claude-project`); `find_candidates(sources: list[tuple[str, Path]], existing_slugs: set[str]) -> list[ImportCandidate]`; `import_skills(candidates: list[ImportCandidate], target_dir: Path, names: list[str] | None = None) -> tuple[list[str], list[str], list[str]]` (copied, skipped, failed) — `names=None` means all non-colliding.

- [ ] **Step 1: Write the failing tests** — `tests/test_skills_importer.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure** — `uv run python -m unittest tests.test_skills_importer -v` — expected: `ModuleNotFoundError`.
- [ ] **Step 3: Implement** — `src/llm/tools/skills/importer.py`:

```python
"""Copy-based migration of Claude Code / Codex skills into our directories.

Only direct children with a SKILL.md are candidates; the whole skill
directory (references/, scripts/, assets/) is copied; an existing target is
never overwritten — collisions are skipped and reported.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.config.logging import logger

from .skill import parse_skill, sanitize_slug


@dataclass(frozen=True)
class ImportCandidate:
    slug: str
    source: str
    path: Path
    description: str
    collides: bool


def default_foreign_sources(home: Path, repo_root: Path | None) -> list[tuple[str, Path]]:
    sources = [
        ("claude-user", home / ".claude" / "skills"),
        ("codex-user", home / ".codex" / "skills"),
        ("agents-user", home / ".agents" / "skills"),
    ]
    if repo_root is not None:
        sources.append(("claude-project", repo_root / ".claude" / "skills"))
    return sources


def find_candidates(sources: list[tuple[str, Path]], existing_slugs: set[str]) -> list[ImportCandidate]:
    candidates: list[ImportCandidate] = []
    seen: set[str] = set()
    for label, directory in sources:
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            if not (child / "SKILL.md").is_file():
                continue
            slug = sanitize_slug(child.name)
            if slug in seen:
                continue
            seen.add(slug)
            parsed = parse_skill(child, "user")
            description = parsed.description if parsed else "(unreadable skill)"
            candidates.append(
                ImportCandidate(
                    slug=slug,
                    source=label,
                    path=child,
                    description=description,
                    collides=slug in existing_slugs,
                )
            )
    return candidates


def import_skills(
    candidates: list[ImportCandidate], target_dir: Path, names: list[str] | None = None
) -> tuple[list[str], list[str], list[str]]:
    """(copied, skipped, failed); names=None copies every non-colliding candidate."""
    wanted = set(names) if names is not None else None
    copied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        if wanted is not None and candidate.slug not in wanted:
            continue
        destination = target_dir / candidate.slug
        if candidate.collides or destination.exists():
            skipped.append(candidate.slug)
            continue
        try:
            shutil.copytree(candidate.path, destination)
            copied.append(candidate.slug)
        except Exception as error:
            logger.warning("Skill import of {} failed: {}", candidate.path, error)
            failed.append(candidate.slug)
    return copied, skipped, failed
```

Replace `_skills_import` in `src/ui/commands.py`:

```python
    def _skills_import(self, argument: str) -> None:
        from pathlib import Path

        from src.llm.tools.skills.importer import default_foreign_sources, find_candidates, import_skills
        from src.llm.tools.skills.store import DEFAULT_USER_SKILLS_DIR

        store = self._skill_store()
        existing = set(store.skills) if store is not None else set()
        repo_root = Path.cwd()
        candidates = find_candidates(default_foreign_sources(Path.home(), repo_root), existing)
        if not candidates:
            self.console.print("  no foreign skills found (Claude Code / Codex)", style="info")
            return None
        if not argument:
            for candidate in candidates:
                note = "  (already exists — skipped)" if candidate.collides else ""
                line = f"  {candidate.slug:<24} {candidate.source:<15} {candidate.description[:50]}{note}"
                self.console.print(line, style="info", markup=False, highlight=False)
            self.console.print(
                "  copy with: /skills import all — or /skills import <name> <name> (add --project for .agents/skills)",
                style="info",
            )
            return None
        words = argument.split()
        to_project = "--project" in words
        words = [word for word in words if word != "--project"]
        from src.llm.tools.skills.store import PROJECT_SKILLS_SUBPATH

        target = (repo_root / PROJECT_SKILLS_SUBPATH) if to_project else DEFAULT_USER_SKILLS_DIR
        names = None if words == ["all"] else words
        copied, skipped, failed = import_skills(candidates, target, names=names)
        if store is not None:
            store.scan()
        summary = f"  imported {len(copied)}: {', '.join(copied) or '—'}"
        if skipped:
            summary += f"; skipped (exists): {', '.join(skipped)}"
        if failed:
            summary += f"; failed: {', '.join(failed)}"
        self.console.print(summary, style="success" if copied else "info", markup=False, highlight=False)
        return None
```

- [ ] **Step 4: Run to verify pass** — `uv run python -m unittest tests.test_skills_importer tests.test_skills_commands -v` — expected: all PASS. Note: `/skills import` in the commands test environment scans the REAL home — add a test to `tests/test_skills_commands.py` only if it injects sources; otherwise leave `/skills import` covered by the importer tests (the command method is a thin shell). Do NOT write a command test that depends on the contents of `~/.claude/skills`.
- [ ] **Step 5: Lint and commit**

```bash
bash lint.sh
git add src/llm/tools/skills/importer.py src/ui/commands.py tests/test_skills_importer.py
git commit -m "feat(skills): importer for Claude Code / Codex skills"
```

---

### Task 9: `skills` CLI

**Files:**
- Create: `src/llm/tools/skills/cli.py`
- Modify: `src/main.py:488-495` (`cli_main` dispatch)
- Test: `tests/test_skills_cli.py`

**Interfaces:**
- Consumes: `SkillStore`, `project_skills_dirs`, `DEFAULT_USER_SKILLS_DIR` (Task 4); importer (Task 8); `ToolRegistry`.
- Produces: `run_skills_cli(argv: list[str], *, user_dir: Path | None = None, project_root: Path | None = None, home: Path | None = None) -> int` — keyword seams so tests never touch the real home.

- [ ] **Step 1: Write the failing tests** — `tests/test_skills_cli.py`:

```python
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.skills.cli import run_skills_cli  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402


class TestSkillsCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.user_dir = base / "ours"
        self.home = base / "home"
        (self.home / ".claude" / "skills").mkdir(parents=True)
        make_skill_dir(self.home / ".claude" / "skills", "found", "---\ndescription: foreign\n---\nbody\n")
        self.user_dir.mkdir()
        make_skill_dir(self.user_dir, "mine", "---\ndescription: local\n---\nbody\n")

    def run_cli(self, argv) -> tuple[int, str]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_skills_cli(argv, user_dir=self.user_dir, project_root=None, home=self.home)
        return code, buffer.getvalue()

    def test_list(self):
        code, output = self.run_cli(["list"])
        self.assertEqual(code, 0)
        self.assertIn("mine", output)

    def test_import_listing(self):
        code, output = self.run_cli(["import"])
        self.assertEqual(code, 0)
        self.assertIn("found", output)

    def test_import_all(self):
        code, output = self.run_cli(["import", "--all"])
        self.assertEqual(code, 0)
        self.assertTrue((self.user_dir / "found" / "SKILL.md").is_file())

    def test_import_by_name_unknown(self):
        code, output = self.run_cli(["import", "nosuch"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `uv run python -m unittest tests.test_skills_cli -v` — expected: `ModuleNotFoundError`.
- [ ] **Step 3: Implement** — `src/llm/tools/skills/cli.py`:

```python
"""`python -m src.main skills ...` — list skills and import foreign ones.

Never starts the REPL or touches models. The keyword seams (user_dir,
project_root, home) exist for tests; the real entry point passes nothing.
"""

import argparse
from pathlib import Path

from ..meta.registry import ToolRegistry
from .importer import default_foreign_sources, find_candidates, import_skills
from .store import DEFAULT_USER_SKILLS_DIR, PROJECT_SKILLS_SUBPATH, SkillStore, project_skills_dirs


def _build_store(user_dir: Path, project_root: Path | None) -> SkillStore:
    project_dirs = project_skills_dirs(project_root) if project_root is not None else []
    store = SkillStore(user_dir=user_dir, project_dirs=project_dirs, registry=ToolRegistry())
    store.scan()
    return store


def run_skills_cli(
    argv: list[str],
    *,
    user_dir: Path | None = None,
    project_root: Path | None = None,
    home: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="skills", description="Manage agent skills")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list discovered skills")
    import_parser = commands.add_parser("import", help="import Claude Code / Codex skills")
    import_parser.add_argument("names", nargs="*", help="skill names to import")
    import_parser.add_argument("--all", action="store_true", dest="import_all", help="import every new skill")
    import_parser.add_argument("--project", action="store_true", help="copy into ./.agents/skills instead")

    args = parser.parse_args(argv)
    user_dir = user_dir if user_dir is not None else DEFAULT_USER_SKILLS_DIR
    home = home if home is not None else Path.home()
    if project_root is None and user_dir == DEFAULT_USER_SKILLS_DIR:
        project_root = Path.cwd()
    store = _build_store(user_dir, project_root)

    if args.command == "list":
        if not store.skills:
            print("no skills found")
            return 0
        for skill in sorted(store.skills.values(), key=lambda item: item.slug):
            print(f"{skill.slug:<24} {skill.scope:<8} {skill.description[:60]}")
        return 0

    candidates = find_candidates(default_foreign_sources(home, project_root), set(store.skills))
    if not args.import_all and not args.names:
        if not candidates:
            print("no foreign skills found")
            return 0
        for candidate in candidates:
            note = "  (already exists)" if candidate.collides else ""
            print(f"{candidate.slug:<24} {candidate.source:<15} {candidate.description[:50]}{note}")
        print("import with: skills import --all  |  skills import <name>...")
        return 0
    known = {candidate.slug for candidate in candidates}
    unknown = [name for name in args.names if name not in known]
    if unknown:
        print(f"unknown skill(s): {', '.join(unknown)}")
        return 1
    target = (project_root or Path.cwd()) / PROJECT_SKILLS_SUBPATH if args.project else user_dir
    copied, skipped, failed = import_skills(candidates, target, names=args.names or None)
    print(f"imported {len(copied)}: {', '.join(copied) or '—'}")
    if skipped:
        print(f"skipped (already exist): {', '.join(skipped)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    return 0 if not failed else 1
```

In `src/main.py`'s `cli_main`, after the `mcp` dispatch:

```python
    if len(sys.argv) > 1 and sys.argv[1] == "skills":
        from src.llm.tools.skills.cli import run_skills_cli

        raise SystemExit(run_skills_cli(sys.argv[2:]))
```

- [ ] **Step 4: Run to verify pass** — `uv run python -m unittest tests.test_skills_cli -v` — expected: all PASS.
- [ ] **Step 5: Lint and commit**

```bash
bash lint.sh
git add src/llm/tools/skills/cli.py src/main.py tests/test_skills_cli.py
git commit -m "feat(skills): skills list/import CLI"
```

---

### Task 10: REPL wiring, settings, startup hint/auto-import, docs, full-suite gate

**Files:**
- Modify: `src/config/settings.py` (add `skills_auto_import`)
- Modify: `src/main.py` (store startup + per-turn refresh + `SkillAction` branch + startup notices)
- Modify: `evaluation/main.py` (same store startup, guarded)
- Modify: `CLAUDE.md` (Skills section)
- Test: `tests/test_skills_startup.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `settings.skills_auto_import: bool` (env `SKILLS_AUTO_IMPORT`, default False); `skills_startup_report(store, *, auto_import: bool, home: Path | None = None, repo_root: Path | None = None) -> str | None` in `src/llm/tools/skills/store.py` — pure-ish helper returning the one-line notice (or None) so main.py just prints it.

- [ ] **Step 1: Write the failing tests** — `tests/test_skills_startup.py`:

```python
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.tools.meta.registry import ToolRegistry  # noqa: E402
from src.llm.tools.skills.store import SkillStore, skills_startup_report  # noqa: E402
from tests.test_skills_parser import make_skill_dir  # noqa: E402


class TestStartupReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.user_dir = base / "ours"
        self.user_dir.mkdir()
        self.home = base / "home"
        (self.home / ".codex" / "skills").mkdir(parents=True)
        make_skill_dir(self.home / ".codex" / "skills", "foreign", "---\ndescription: f\n---\nbody\n")
        self.store = SkillStore(user_dir=self.user_dir, project_dirs=[], registry=ToolRegistry())
        self.store.scan()

    def test_hint_when_foreign_found(self):
        report = skills_startup_report(self.store, auto_import=False, home=self.home, repo_root=None)
        self.assertIn("/skills import", report)

    def test_auto_import_copies_and_reports(self):
        report = skills_startup_report(self.store, auto_import=True, home=self.home, repo_root=None)
        self.assertIn("foreign", report)
        self.assertTrue((self.user_dir / "foreign" / "SKILL.md").is_file())
        self.assertIsNotNone(self.store.get("foreign"))  # rescanned after the copy

    def test_silent_when_nothing_foreign(self):
        empty_home = Path(self.tmp.name) / "empty"
        empty_home.mkdir()
        self.assertIsNone(skills_startup_report(self.store, auto_import=False, home=empty_home, repo_root=None))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `uv run python -m unittest tests.test_skills_startup -v` — expected: ImportError on `skills_startup_report`.
- [ ] **Step 3: Implement.**

1. `src/config/settings.py` — after the `mcp_oauth_port` field:

```python
    skills_auto_import: bool = Field(
        default=False,
        description="Copy newly found Claude Code/Codex skills into ~/.right-agent/skills at startup.",
    )
```

2. `src/llm/tools/skills/store.py` — append:

```python
def skills_startup_report(
    store: SkillStore, *, auto_import: bool, home: Path | None = None, repo_root: Path | None = None
) -> str | None:
    """The one-line startup notice: an import hint, an auto-import summary, or None."""
    from .importer import default_foreign_sources, find_candidates, import_skills

    home = home if home is not None else Path.home()
    candidates = find_candidates(default_foreign_sources(home, repo_root), set(store.skills))
    fresh = [candidate for candidate in candidates if not candidate.collides]
    if not fresh:
        return None
    if not auto_import:
        return f"found {len(fresh)} Claude Code/Codex skill(s) — /skills import to migrate them"
    if store.user_dir is None:
        return None
    copied, _, failed = import_skills(candidates, store.user_dir, names=None)
    if copied:
        store.scan()
    line = f"auto-imported {len(copied)} skill(s): {', '.join(copied)}" if copied else None
    if failed:
        line = (line or "skill auto-import") + f" — {len(failed)} failed (see logs.log)"
    return line
```

3. `src/main.py` — in `main()`, after the `mcp_task = asyncio.create_task(...)` line:

```python
    from src.llm.tools.skills.store import get_skill_store, skills_startup_report, start_skill_store

    try:
        skill_store = start_skill_store()
        notice = skills_startup_report(skill_store, auto_import=settings.skills_auto_import)
        if notice:
            ui.console.print(f"  {notice}", style="dim", markup=False, highlight=False)
    except Exception:
        logger.exception("Skill store startup failed")
```

In the main loop, directly after `user_content = await ui.get_input()` and the empty check:

```python
            store = get_skill_store()
            if store is not None:
                try:
                    store.refresh()
                except Exception:
                    logger.exception("Skill refresh failed")
```

Import `SkillAction` next to `McpAction` (`from src.ui.commands import McpAction, SkillAction, run_mcp_action`) and restructure the command-result block:

```python
            if user_content.startswith("/"):
                result = ui.handle_command(user_content)
                if result == "clear":
                    messages = []
                model = ui.model
                if isinstance(result, SkillAction):
                    user_content = result.text  # fall through into the turn below
                elif isinstance(result, McpAction):
                    prompt_text = await run_mcp_action(result, mcp_manager, ui.console)
                    if prompt_text is None:
                        continue
                    user_content = prompt_text  # fall through into the turn below
                else:
                    continue
```

4. `evaluation/main.py` — mirror the startup (read the file first; insert after its UI/agents construction, before the loop):

```python
    try:
        from src.llm.tools.skills.store import start_skill_store

        start_skill_store()
    except Exception:
        logger.exception("Skill store startup failed (evaluation)")
```

5. `CLAUDE.md` — add a `### Skills` subsection after the "MCP servers" section:

```markdown
### Skills

`src/llm/tools/skills/` is the Agent Skills layer (SKILL.md standard, Claude Code-compatible): `skill.py` (parser; unknown frontmatter fields — including CC's `allowed-tools`/`context` — are ignored so foreign skills always load; broken skills are skipped with a warning), `render.py` ($1/$ARGUMENTS/named/${SKILL_DIR} substitutions, one pass, `${CLAUDE_SKILL_DIR}` accepted as an alias), `store.py` (`SkillStore` — project `.agents/skills` from cwd up to the git root beats user `~/.right-agent/skills`; slug = sanitized directory name; bodies are read from disk at every invocation so edits need no reload; `refresh()` at the top of each turn is an mtime fingerprint check that rescans only on change; `set_skill_store()` is the test seam and every path is injected — nothing scans the real home in tests), `tool.py` (each model-invocable skill registers as `skill__<slug>` with source `skill:<slug>`, `[Skill]`-tagged in listings, `only_skills=True` in search_tools), `importer.py` + `cli.py` (`uv run python -m src.main skills list|import`, `/skills import` — copies whole skill directories from `~/.claude/skills`, `~/.codex/skills`, `~/.agents/skills`, project `.claude/skills`; never overwrites; `SKILLS_AUTO_IMPORT=1` does it silently at startup). Invocation is the get_tool-contracts mechanic: calling `skill__x()` inside a script records the rendered body into the run's `skills` result field (per-body cap 12k chars, session-level sha256 dedupe answers "already loaded", `force=True` resends) and history recaps carry the bodies verbatim (8k per skill / 16k total, newest first, a drop note teaches `force=True` recovery). `/slug [args]` in the REPL renders the body as the next user turn via `SkillAction` (the MCP-prompt mechanic, but synchronous); built-in commands win name collisions. Skill descriptions/bodies are untrusted file text: REPL prints use `markup=False`.
```

- [ ] **Step 4: Run the FULL suite and lint** — `uv run python -m unittest discover -s tests` — expected: same baseline as main (1 pre-existing failure + 2 pre-existing errors + 8 skips), zero NEW failures; then `bash lint.sh` — clean.
- [ ] **Step 5: Manual smoke** — `uv run python -m src.main skills list` prints the repo's skills (or "no skills found") and exits without starting the REPL.
- [ ] **Step 6: Commit**

```bash
git add src/config/settings.py src/main.py evaluation/main.py src/llm/tools/skills/store.py CLAUDE.md tests/test_skills_startup.py
git commit -m "feat(skills): REPL wiring, startup import hint, SKILLS_AUTO_IMPORT, docs"
```
