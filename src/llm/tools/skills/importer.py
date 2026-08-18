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
