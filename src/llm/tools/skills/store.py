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
    """Every .agents/skills candidate from cwd up to the git root, nearest first,
    regardless of current existence. scan() and _skill_files() skip non-dirs lazily."""
    dirs: list[Path] = []
    # Resolve only when necessary: an already-absolute cwd (the normal case —
    # Path.cwd() is always absolute) is used as-is so we don't normalize away
    # symlinks the caller's path legitimately contains (e.g. macOS's
    # /var/folders -> /private/var/folders under $TMPDIR, which would
    # otherwise make a caller-supplied path and our returned dirs disagree).
    current = cwd if cwd.is_absolute() else cwd.resolve()
    while True:
        candidate = current / PROJECT_SKILLS_SUBPATH
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
                # Bookkeeping must match reality: the tool above was just
                # unregistered (or never registered) and registration failed,
                # so `_registered` must not keep claiming this slug is live —
                # otherwise the skill silently vanishes from the registry
                # with nothing left to trigger a retry on a later scan.
                self._registered.pop(slug, None)
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

    def reset_session(self) -> None:
        """Forget delivered-body dedupe. Call when conversation history is
        wiped (e.g. /clear) — otherwise a later skill call answers "already
        loaded earlier this session" against a history that no longer
        contains the body, and the model proceeds without instructions."""
        self.seen_hashes.clear()


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
