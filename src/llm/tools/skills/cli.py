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
    if args.project and project_root is None:
        print("--project requires a project root")
        return 1
    target = project_root / PROJECT_SKILLS_SUBPATH if args.project else user_dir
    copied, skipped, failed = import_skills(candidates, target, names=args.names or None)
    print(f"imported {len(copied)}: {', '.join(copied) or '—'}")
    if skipped:
        print(f"skipped (already exist): {', '.join(skipped)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    return 0 if not failed else 1
