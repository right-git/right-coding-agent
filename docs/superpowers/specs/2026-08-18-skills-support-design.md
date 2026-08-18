# Skills Support Design

## Goal

Give the agent an Agent Skills layer modeled closely on Claude Code: skills are directories with a `SKILL.md` file (YAML frontmatter + markdown instructions), discovered from user and project directories, invocable both by the user (`/skill-name [args]` in the REPL) and by the model (through the existing meta layer). A skill "call" does not execute code — it delivers the rendered SKILL.md body into the conversation as instructions. Skills join the `ToolRegistry` with a `skill:<slug>` source label, so any number of skills adds **zero** per-turn context cost: the model finds them through `search_tools` (new `only_skills=True` filter) exactly like built-in and MCP tools.

Migration is a first-class concern: the SKILL.md format is the open Agent Skills standard (agentskills.io), shared by Claude Code and Codex. The layer reads our own canonical directories, and an importer copies existing Claude Code / Codex skills in with one command.

Prior art in this repo: the MCP layer (`src/llm/tools/mcp/`, spec `2026-08-18-mcp-support-design.md`) established every pattern this layer reuses — source-labeled registry entries, name sanitization with hash truncation, the contracts channel that carries fetched text verbatim through history compaction, `McpAction` for REPL commands that feed text into a turn, explicit-path test seams.

## New package

`src/llm/tools/skills/` following the repo's tool-package pattern:

- `skill.py` — the `Skill` dataclass and the SKILL.md parser (frontmatter via PyYAML, body as raw markdown).
- `store.py` — `SkillStore`: directory scanning, precedence, live refresh, registry registration, session dedupe state; `get_skill_store()`/`set_skill_store()` process-wide seam.
- `render.py` — argument and variable substitution over the body.
- `tool.py` — skill → registry pseudo-tool adapter (`skill__<slug>` `StructuredTool`s).
- `importer.py` — foreign-directory scan and copy-based migration.
- `cli.py` — the `skills` argparse subcommands.

No new dependencies: PyYAML is already in the tree via LangChain. Everything filesystem-facing takes explicit paths; the unit suite runs against temp directories only.

## Discovery and scopes

Two canonical scopes; on slug collision **project beats user** (consistent with `.mcp.json` precedence in the MCP layer):

- **project**: `.agents/skills/<slug>/SKILL.md` — every `.agents/skills` directory from the REPL's cwd up to the git repo root (inclusive). Nearest-to-cwd wins when two project directories define the same slug. `.agents/skills` is the tool-neutral path of the emerging standard (Codex reads it too), so repo-level skills are portable without import.
- **user**: `~/.right-agent/skills/<slug>/SKILL.md`.

The **slug is the directory name** (Claude Code behavior: frontmatter `name` is only a display label). Slugs are sanitized to `[A-Za-z0-9_-]` for tool/command names; the registry tool name is `skill__<slug>`, hash-truncated past the 64-char ceiling using the same scheme as MCP tool names (a `_<8-hex>` suffix replaces the tail). A directory without a readable `SKILL.md` is silently skipped (it is not a skill). If two different directories sanitize to the same slug, the precedence rules above pick one and the loser is skipped with a warning in `logs.log`.

## SKILL.md format

YAML frontmatter between `---` markers, then the markdown body. v1 fields:

| Field | Required | Behavior |
|---|---|---|
| `name` | No | Display label in listings. Never the identity — the slug is the directory name. |
| `description` | Recommended | When the model should use the skill. If missing, the first paragraph of the body is used (Claude Code behavior). |
| `when_to_use` | No | Appended to `description` in listings. Combined cap: 1536 chars (`MAX_SKILL_DESCRIPTION_CHARS`, Claude Code's number). |
| `arguments` | No | Named positional arguments — a YAML list or space-separated string. Names map to argument positions in order and become `$name` substitutions. |
| `argument-hint` | No | Autocomplete hint shown in the REPL, e.g. `[issue-number]`. |
| `disable-model-invocation` | No | `true` → the skill is **not registered** in the registry at all; only the `/slug` command reaches it. Default `false`. |
| `user-invocable` | No | `false` → no slash command is created; only the model can invoke it. Default `true`. |
| `license`, `compatibility`, `metadata` | No | Accepted and ignored (Agent Skills spec compatibility). |

Parsing rules:

- **Unknown fields — including Claude Code extensions we don't support (`allowed-tools`, `model`, `context`, `paths`, `agent`, `background`) — are ignored with a debug log, never an error.** An imported foreign skill must always load.
- Broken YAML, unreadable file, or empty body → the skill is skipped with a warning in `logs.log`; scanning never raises.
- A skill with both `disable-model-invocation: true` and `user-invocable: false` is dead; it is skipped and flagged in `/skills` output.
- Frontmatter text (description, hints) is untrusted file content: REPL prints use `markup=False` (the MCP lesson), and the registry brief shows it as plain text.

The **body is read from disk at every invocation**, not cached at scan time — editing a skill's instructions takes effect immediately without any reload. The scan caches only frontmatter.

## Rendering (substitutions)

Applied to the body at invocation time, single pass, in `render.py`:

- `$1`..`$n` — positional arguments.
- `$ARGUMENTS` — all arguments joined with single spaces.
- `$name` — for each name declared in `arguments`, bound by position.
- `${SKILL_DIR}` — absolute path of the skill's directory. `${CLAUDE_SKILL_DIR}` is accepted as an alias so imported Claude Code skills work unmodified.
- Missing arguments substitute the empty string (Claude Code behavior).
- `\$` directly before a digit, `ARGUMENTS`, or a declared name escapes the substitution (the backslash is consumed); a backslash anywhere else is left alone.

Dynamic context injection (`` !`command` `` lines) is **out of scope for v1**: the line reaches the model as literal text.

## Registry integration

`SkillStore` registers each model-visible skill as a `StructuredTool` named `skill__<slug>` with source `skill:<slug>` (`SKILL_SOURCE_PREFIX = "skill:"`):

- Tool description = `description` + `when_to_use`, capped at 1536 chars, so `search_tools` scoring sees exactly what Claude Code's skill listing would show.
- `registry.brief()` tags entries `[Skill]` (mirroring `[MCP: <server>]`).
- `search_tools` gains `only_skills: bool = False` — the generic source-prefix filter the MCP work prepared for. Skills never join the MCP per-server grouping; they are flat entries. `only_mcp=True` together with `only_skills=True` returns an explanatory error string.
- The `run_tools` docstring (the language contract shown to the model) documents `only_skills` and the skill-call semantics below.
- `/tool skill__<slug>` pinning works with no extra code — skills are ordinary registry entries.

**Argument schema** of the pseudo-tool: the declared `arguments` names become optional string fields (visible in the `get_tool` contract); with no declaration, a single optional `arguments: str` field feeds `$ARGUMENTS`. Every skill tool also takes `force: bool = False` (below).

## Invocation semantics and the skills result channel

An in-script call `skill__humanizer()` follows the `get_tool` mechanic exactly — the body never rides the return value:

1. The pseudo-tool re-reads the body from disk, renders substitutions, and computes `sha256(rendered)`.
2. **New content** (hash not seen this session): the body is recorded into the per-run skills collector — a ContextVar bucket opened by `run_tools` per call, mirroring `collecting_images()` and the contracts recorder — and the call returns a short confirmation: `skill 'humanizer' loaded — instructions arrive with this result`.
3. **Already delivered** (same hash seen earlier this session): no body is recorded; the call returns `skill 'humanizer' already loaded earlier this session — pass force=True to resend`.
4. `force=True` skips the dedupe and resends unconditionally.

`run_tools` appends the collected bodies to its result JSON as a `skills` field (`{slug: body}`), **after** the generic `MAX_RESULT_CHARS` clip is applied to `result`/`logs` — skill bodies have their own budget and are never truncated by the generic clip, exactly like the `contracts` field. The dedupe map (`slug → sha256`) lives on `SkillStore` per session; a changed file or changed arguments produce a new hash and the body ships again automatically.

Skill loads count as normal script tool calls in the usage footer (they are real work the user should see); the meta discovery calls (`search_tools`/`get_tool`) keep their existing exemption.

## History persistence

`compact_finished_turn` (`src/llm/history.py`) learns the `skills` field the same way it carries `contracts`: recap messages carry skill bodies **verbatim**, budgeted:

- `MAX_SKILL_RECAP_CHARS = 8_000` per skill body (truncated with a note if longer).
- `MAX_SKILLS_RECAP_TOTAL = 16_000` across all skills in a recap, spent newest-first.
- A body dropped or truncated by the budget is replaced by / suffixed with: `[skill '<slug>' body dropped from history — re-invoke skill__<slug>(force=True) to reload]` — self-teaching, like the contracts design: the model learns the recovery path from the message itself.

This is the analog of Claude Code's compaction policy (5k tokens per skill, 25k total), deliberately tighter: every recap character is re-sent on every subsequent turn.

## REPL commands

**`/slug [args]`** — one top-level command per user-invocable skill, autocompleted by `CommandCompleter` with `argument-hint` as the meta text. Routing in `CommandHandler.handle`: built-in commands and `/mcp__...` prompt commands win collisions (the shadowed skill is flagged in `/skills`); otherwise the skill store resolves the slug, arguments are split shell-style, and the handler returns a `SkillAction("prompt", rendered_body)` — the same shape as `McpAction`: the main loop feeds the rendered body in as the user turn. Rendering failures return an error string printed safely, never a raised exception.

**`/skills`** — status table: slug, scope (project/user), invocable-by (user/model/both/none-dead), description head (markup-safe). Subcommands:

- `/skills reload` — force `SkillStore.refresh()` now.
- `/skills import` — list import candidates; `/skills import all` / `/skills import name1 name2` copies them (next section).

## Live reload

No watchers, no threads: `SkillStore.refresh()` re-stats the skill directories (directory listing + each `SKILL.md` mtime) and re-registers what changed — new skill → register, removed → unregister, changed frontmatter → re-register. It is called at the start of every user turn from the main loop (a handful of `stat` calls, sub-millisecond) and by `/skills reload`. Body edits need no reload at all (bodies are read at invocation). Registry churn never touches the system prompt — the tool count there is frozen on first build — so the provider prompt-cache prefix stays intact.

## Import (migration)

`importer.py` scans foreign skill directories for direct children containing a `SKILL.md`:

- `~/.claude/skills` (Claude Code personal)
- `~/.codex/skills` (Codex legacy)
- `~/.agents/skills` (Agent Skills standard, user scope)
- `<repo>/.claude/skills` (Claude Code project)

Project `.agents/skills` needs no import — it is a canonical read path already.

Three ways in:

1. **`/skills import`** — lists candidates: slug, source path, description head, and whether the slug already exists in our directories (collision). `/skills import all` or `/skills import <names...>` copies each skill's **whole directory** (references/, scripts/, assets/ included) into `~/.right-agent/skills/` (`--project` targets `<repo>/.agents/skills/` instead). **Existing skills are never overwritten** — collisions are skipped and reported.
2. **Startup hint** — after the initial scan, if any foreign skill's slug is absent from our directories, one dim line prints once per session: `Found N Claude Code/Codex skills — /skills import to migrate them`. Suppressed when auto-import is on.
3. **`SKILLS_AUTO_IMPORT=1`** (settings, default off) — at startup, new foreign skills are copied silently (collisions skipped), with one summary line in the REPL and details in `logs.log`.

## CLI subcommands

`src/main.py` dispatches `sys.argv[1] == "skills"` alongside the existing `mcp` dispatch:

```
uv run python -m src.main skills list
uv run python -m src.main skills import                     # list candidates
uv run python -m src.main skills import --all [--project]
uv run python -m src.main skills import <name>... [--project]
```

`skills list` prints the same table as `/skills`. The CLI never starts the REPL or touches models.

## Settings and wiring

`src/config/settings.py`:

- `skills_user_dir: Path = ~/.right-agent/skills` (constant default; tests inject paths through `SkillStore`, not env).
- `skills_auto_import: bool = False` (env `SKILLS_AUTO_IMPORT`).

No enable/disable env gate: the layer costs nothing (no models, no network, no subprocesses).

Wiring in `main()`: construct `SkillStore(user_dir=..., project_root=cwd, registry=get_registry())`, run the initial scan **before the first turn** (so the frozen tool count in `Prompts.session_context` includes skills), install via `set_skill_store()`, print the startup hint if applicable, run auto-import if enabled. The main loop calls `store.refresh()` at the top of each turn. `evaluation/direct_agent.py` gets the same store wiring so the comparison harness stays fair.

Hermeticity (the `_mcp_servers_configured` lesson): nothing in `defaults.py` or module import paths scans the real home directory. `SkillStore` and `importer` take every path explicitly; only `main()` supplies real defaults.

## Error handling

- Parser: broken skill → skip + warning; never raises out of a scan.
- Pseudo-tool: never raises — a skill deleted between scan and call, a render failure, an oversized body all return `[skill error] ...` strings.
- Importer: per-skill copy failures reported per skill; one failure never aborts the batch.
- Commands and CLI: all user-facing prints of skill-supplied text are markup-safe; `report_usage`-style swallow-and-log applies to the startup hint and auto-import (a skills problem must never break the REPL).

## Testing

`unittest`, temp directories, no gates needed (fully hermetic):

- `tests/test_skills_parser.py` — frontmatter variants (full, minimal, missing description → first body paragraph, `arguments` as list and string), broken YAML skipped with warning, unknown/CC-extension fields ignored, dead-skill detection, description cap; a realistic fixture shaped like the humanizer skill.
- `tests/test_skills_render.py` — all substitutions, `${CLAUDE_SKILL_DIR}` alias, missing args → empty string, `\$` escapes, single-pass (substituted output is not re-scanned).
- `tests/test_skills_store.py` — scan precedence (project > user, nearest project dir wins), refresh detects add/remove/frontmatter change via mtime, register/unregister against a private registry, `disable-model-invocation` invisibility, session dedupe hashes, `force` resend, body re-read at invocation.
- `tests/test_skills_meta.py` — `[Skill]` tag in `brief()`, `only_skills` filter (and the `only_mcp`+`only_skills` error), `skills` field present in run_tools JSON and exempt from the generic clip, recap carry with both budgets and the drop note (extends the history tests).
- `tests/test_skills_commands.py` — `/slug` routing → `SkillAction`, argument splitting, built-in collision, `/skills` table, `reload`, markup safety on hostile description text.
- `tests/test_skills_importer.py` — candidate scan, whole-directory copy, never-overwrite, `--project` target, auto-import summary.
- `tests/test_skills_cli.py` — `list`, `import` listing and copying, exit codes.

## Documentation

CLAUDE.md gains a "Skills" section covering: directory layout and precedence, the SKILL.md contract, the `skill__<slug>` call semantics and the `skills` recap channel, `/skills` and import, and the hermeticity seams.

## Out of scope (v2+)

- Dynamic context injection (`` !`command` ``) — executing file-supplied commands needs its own safety design.
- `allowed-tools`, `model`, `context: fork`, `agent`, `paths` — no per-tool permissions or subagents exist in this agent yet.
- Plugin/marketplace distribution, claude.ai sync, nested lazily-activated subdirectory skills (Claude Code's monorepo feature).
- `only_skills` for future skill *sources* beyond directories (the source-prefix mechanism already generalizes).
