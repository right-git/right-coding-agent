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
    """Directory name -> command/tool-safe slug; falls back to 'x'.

    Lowercased so the slug is reachable everywhere it's used verbatim —
    store keys, `skill__<slug>` tool names, `/<slug>` commands, and importer
    candidate names: `CommandHandler.handle` lowercases the typed command
    before matching, so an uppercase-preserving slug (e.g. a `Deploy/`
    directory) would be listed and completed but never actually invocable.
    """
    return _SLUG_RE.sub("_", str(raw or "").strip()).strip("_-").lower() or "x"


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """(frontmatter yaml text or None, body) — never raises."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return None, text
    return match.group(1), text[match.end() :]


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
