# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install dependencies: `uv sync` (uv is the environment manager; torch/torchvision come from the pytorch-cu128 index on Windows)
- Run the agent REPL: `uv run python -m src.main`
  - Requires a `.env` at the repo root with `ENV`, `LLM_API_KEY`, `LLM_API_BASE`. Importing `src.config.base` fails without them, so tests and scripts that touch `src.config` also need it.
- Run all tests: `uv run python -m unittest discover -s tests`
- Run one test file: `uv run python -m unittest tests.test_computer_use`
- Run one test: `uv run python -m unittest tests.test_computer_use.ClassName.test_name`

pytest is **not** installed — the test suite is `unittest`-based (the README's pytest command does not work). Each test file prepends the repo root to `sys.path` itself, so discovery works from the repo root without packaging. The whole suite runs in a few seconds; nothing in it touches the real desktop, network, or GPU.

Logging goes to `logs.log` via loguru (`src/config/logging.py`); stdout logging is off by default, so check that file when debugging runtime behavior.

## Architecture

This is an async coding-assistant agent built on LangChain, with a desktop computer-use capability driven by a local vision model.

### Chat loop and LLM layer

`src/main.py` runs the terminal chat REPL (`src/ui/chat.py`, prompt-toolkit + rich). It holds the LangChain message history across turns and calls `Agents.right_coding_agent()` once per user turn. On errors it rolls the history back, and it always passes history through `trim_incomplete_tool_calls` (`src/utils/functions.py`) so a turn that died mid-tool-call doesn't leave a dangling tool-call block that would make the next API call invalid.

`main()` also warms the NVIDIA LocateAnything vision model in a background thread at startup (`preload_vision_model` → `computer_tools.warm_up_computer()`, which calls the locator's `load()`); failures are logged and the locator falls back to lazy loading on first use. Tests patch `src.main.preload_vision_model` so the suite never loads the model. Because that load runs while the prompt is live, the locator wraps loading and inference in `silenced()` (`src/utils/silence.py`) — `sys.stdout`/`sys.stderr` are replaced once with thread-routing proxies, so the loading thread's transformers warnings, remote-code prints, and progress bars are dropped while the UI thread keeps printing normally (`quiet_transformers()` additionally kills transformers' logger and progress bars). Never use `redirect_stdout` for this — it is process-global and would eat the UI's output.

After each successful turn `report_usage` prints a footer — a colored context progress bar (`ChatUI._context_bar`, green → yellow → red by fill), turn tokens, dollar cost, session totals. Token counts come from `usage_metadata` on the turn's new AI messages (`src/llm/usage.py`); history returned in the response is excluded by message id, and the context figure is the **last** call's input+output (its input already contains the whole history). Context lengths and per-token USD prices come from the public OpenRouter API via `OpenRouterCatalog` (`src/llm/openrouter.py`) — fetched in a background task at startup, cached for the session, with a failure cooldown so an offline machine isn't re-polled every turn, and injectable `fetch_payload`/`clock` seams for tests. `ChatUI.handle_command` is synchronous, so the catalog is handed to the UI as a plain dict via `set_model_catalog()`: `/models` renders context + $/M pricing from it, and `/model` switches by exact or unique-partial match over the curated list plus the whole catalog (unverified switches are allowed when the catalog is empty). Usage reporting must never break a finished turn — `report_usage` swallows and logs its own failures.

`src/llm/base.py` — `LLMClient` owns provider configuration and the resilience loop: for each configured `LLMProvider` (see `src/llm/types.py`) it retries `num_retries` times with a cooldown, then fails over to the next provider. `ask_agent()` builds a fresh LangChain `create_agent` per call (model + tools + middleware + optional `response_format`). Passing a `thread_id` turns on the `MemorySaver` checkpointer and interrupt support: an interrupted run comes back with `__interrupted__` set, and `resume_agent()` continues it via `Command(resume=...)` using a per-thread agent cache.

`src/llm/agents.py` — `Agents(LLMClient)` defines concrete agents. `right_coding_agent()` composes:
- the system prompt from `src/config/prompts.py`;
- `META_TOOLS` — the agent's **entire** tool surface (see "Meta tools" below); there are no direct file, shell, web, or screen tools;
- `AttachedImagesMiddleware` (surfaces tool screenshots as vision messages, see "Meta tools");
- `SummarizationMiddleware` (summarizes history past 40k tokens using a small model);
- `MessageLogMiddleware` (`src/llm/log_middleware.py`) — logs every model request to `logs.log` as one JSON line. Registered **last** so it sees the message list exactly as the model does. Its `scrub_text` replaces data URIs and long base64 runs with `<... stripped, N chars>` placeholders and truncates remaining long text; the terminal UI reuses it for tool-result previews. Tests inject `emit=` to capture lines.

Models are addressed by provider-prefixed name (e.g. `google/gemini-3.7-flash`) and routed through one OpenAI-compatible endpoint (`LLM_API_BASE`); the `available_models` list lives at the top of `src/main.py`.

To add an agent: add a method on `Agents` calling `self.ask_agent()`. To add a tool: a `@tool(parse_docstring=True)` async function in `src/llm/tools.py` or `src/llm/computer_tools.py`, then register it in the default registry in `src/llm/meta_tools.py: get_registry()` — tool docstrings become the LLM-facing schema, and tools catch their own exceptions and return the error as a string rather than raising.

### Meta tools (tool discovery + scripted orchestration)

The agent does **not** see individual tool schemas. `src/llm/meta_tools.py` exposes exactly three tools — `search_tools` (keyword search over the registry), `get_tool` (full contract of one tool), and `run_tools` (execute a Python-subset script) — so the context cost stays flat no matter how many tools are registered. Real tools live in a `ToolRegistry` (default: `web_search` + the six `screen_*` tools) and are called *by bare name from inside `run_tools` scripts*, executed by the sandboxed tree-walking interpreter in `src/tools/base.py` (`Interpreter`): whitelisted AST nodes/builtins/methods only, no imports or dunder access, op/sleep/wall-clock/memory budgets, `sleep()` for token-free polling and a `parallel(...)` special form for concurrent tool calls.

Integration details that matter when changing this layer:
- `ToolRegistry` adapts LangChain tools to plain async callables (`_as_script_callable`) — positional args are mapped onto schema field order, then routed through `ainvoke` so argument validation stays in the path.
- The interpreter resolves names as scope → builtins → tools, so a tool named like a builtin would be silently unreachable; `register()` rejects `RESERVED_SCRIPT_NAMES` for that reason.
- `run_tools` returns JSON `{result, logs, error}` clipped to `MAX_RESULT_CHARS`; a fresh `Interpreter` is built per call so concurrent runs don't share mutable state.
- `set_registry()` is the test seam (mirrors `set_computer()`); tests install a registry of fake tools and never touch the real desktop or network. The `run_tools` docstring is the language contract shown to the model — keep it in sync with what `Interpreter._validate`/`_eval` actually allow.

**Screenshots and vision (`src/llm/attachments.py`).** Base64 in a tool's text result is invisible to the model — providers only read images from `image_url` content blocks in user-role messages, and OpenAI-compatible APIs don't accept images inside tool messages. The pipeline: tools call `attach_image()` into a ContextVar bucket that `run_tools` opens per call (`collecting_images()`); the images leave the run as the ToolMessage's `artifact` (kept in graph state, never sent to the provider); `AttachedImagesMiddleware.before_model` then appends one HumanMessage of data-URI `image_url` blocks for the latest tool round (idempotent via an `attached_images` marker in `additional_kwargs`, so retries don't duplicate). `screen_screenshot` and `screen_locate(return_screen=True)` feed it; attachments are capped at `MAX_ATTACHED_IMAGES` per run with the drop count reported in the run_tools JSON. When no channel is open (library use outside the agent), the screen tools fall back to raw base64 in their text result.

### Computer use

`src/tools/computer_use/` gives the agent eyes and hands on the Windows desktop. `base.py` — the `ComputerUse` facade — combines perception (screenshot → natural-language locate → boxes) with control (mouse, keyboard, clipboard) and a "show, don't touch" marker overlay (`mark_object` outlines an element and shows a tooltip instead of clicking).

Every OS-facing dependency is an injected backend behind a protocol (`types.py`): `locator.py` (the nvidia/LocateAnything-3B runtime — the **only** module that imports torch, loaded lazily on first locate), `screen.py` (DPI awareness + capture), `pointer.py` (Win32 SendInput), `overlay.py` (Tk), `clipboard.py`, `windows.py` (focus). `fakes.py` provides `StaticScreen`, `RecordingPointer`, and `ScriptedLocator`, which is how every test runs without a desktop — construct `ComputerUse(locator=..., screen=..., pointer=..., overlay=NullOverlay(), ...)` with fakes. Pointer/DPI/clipboard are Windows-only; detection, parsing, and annotation are platform-independent.

Two behaviors worth knowing before touching locating code:
- **Two-pass refinement:** inference downscales the screen to 768px, so small targets drift by ~100px. `locate_object` does a coarse full-screen pass, then re-locates inside a native-resolution crop around each small hit (`detection.py: needs_refinement`). Passing `region=` skips straight to native resolution when the neighbourhood is known.
- **Screenshot-keyed cache:** results are cached per (description, region, mode, refine) and reused only while the screen is byte-identical. Decoding is greedy on purpose — sampling gave nondeterministic boxes.

`src/llm/computer_tools.py` exposes this to the agent as `screen_*` LangChain tools over a process-wide `ComputerUse` singleton (`get_computer()` / `set_computer()` — the latter is the test/headless seam). Sync `ComputerUse` calls are wrapped in `asyncio.to_thread`. `cli.py` holds the interactive locator REPL loop (`run_interactive_loop`) used for manual testing of the locator.

### Other pieces

- `src/tools/web_parser/` — httpx + BeautifulSoup page fetcher, converts HTML to Markdown; wrapped by the `web_search` tool in `src/llm/tools.py`.
- `src/tools/base.py` — the sandboxed Python-subset interpreter behind `run_tools` (see "Meta tools" above). It is **not** a base class for tools; don't confuse it with `src/tools/computer_use/base.py`.
- `docs/superpowers/plans/` and `docs/superpowers/specs/` — dated design docs for features in progress; check there for intent before reworking the screen-locator behavior.
