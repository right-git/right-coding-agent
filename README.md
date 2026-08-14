# Right Coding Agent

`right-coding-agent` is an asynchronous agent supporting multiple model providers, built on LangChain and designed as a programmer's assistant.

## Overview

- Multi-provider LLM client with retry/failover logic, Azure OpenAI support, and structured output via LangChain.
- Built-in agent and tools: `Agents` adds system prompts, `LLMClient` manages model/tool initialization, and `src/llm/tools.py` contains `@tool`-decorated functions.
- Scripted tool orchestration: the agent sees only three meta tools (`search_tools`, `get_tool`, `run_tools`) and drives every other tool from a sandboxed Python-subset script — flat context cost, parallel calls, and token-free polling (see below).
- The `WebParser` tool (`src/tools/web_parser/`) fetches web pages over HTTP, parses them with BeautifulSoup, and converts to Markdown.
- The `ComputerUse` tool (`src/tools/computer_use/`) gives the agent eyes and hands on the desktop: it finds on-screen elements from a plain-language description, drives the mouse and keyboard, and can point at an element with an on-screen tooltip instead of clicking it.

## Quick Start

1. Install dependencies using `uv`:
   ```bash
   uv sync
   ```
2. Create a `.env` file at the project root and specify your variables:
   ```env
   ENV=<value>
   LLM_API_KEY=<key>
   LLM_API_BASE=<base URL>
   ```
3. Start the agent:
   ```bash
   uv run python -m src.main
   ```

## Chat commands and usage footer

- `/models` — list models with their context window and $/M pricing (metadata from the public OpenRouter API);
- `/model <name>` — switch models: exact or partial match over the curated list **and** the whole OpenRouter catalog, so any OpenRouter model id works;
- `/log-level [name]`, `/clear`, `/help`, `/quit`.

After every response a dim footer reports what the turn cost:

```
ctx 14,204/1,048,576 (1.4%) · turn 13,900 in + 304 out ($0.0058) · session 28,400 tokens ($0.0116)
```

That is: how full the current model's context window is, tokens and dollars
for this turn, and session totals. Token counts come from the provider's
`usage_metadata`; context length and per-token prices come from OpenRouter
(fetched once in the background at startup, cached for the session, and the
footer degrades gracefully — `limit unknown` / `price unknown` — when the
catalog is unavailable).

## Project Structure

- `src/main.py` — entry point; configures `Agents` with the model provider and starts the REPL interface.
- `src/config/base.py` — configuration via `pydantic-settings`, `.env` loading, and `loguru` setup (logs to `logs.log`).
- `src/config/prompts.py` — stores system prompts.
- `src/llm/` — agent core:
  - `base.py` manages the LLM client;
  - `agents.py` implements specific agents (such as `right_coding_agent()`);
  - `tools.py` — utility LangChain tools;
  - `computer_tools.py` — `screen_*` LangChain tools wrapping `ComputerUse`;
  - `meta_tools.py` — the tool registry and the `search_tools` / `get_tool` / `run_tools` meta tools.
- `src/tools/web_parser/` — standalone HTTP client and HTML-to-Markdown parser.
- `src/tools/computer_use/` — screen understanding and desktop control (see below).
- `test.py` — interactive screen-locator REPL built on `ComputerUse`.
- Tests are located in `tests/`.

## Tool orchestration

Exposing every tool schema to the model bloats the context and forces one
round-trip per call. Instead, only three meta tools are wired into the agent
(`src/llm/meta_tools.py`):

1. `search_tools("click button screen")` — keyword search over the tool
   registry (currently `web_search` plus the `screen_*` tools);
2. `get_tool("screen_click")` — the full contract of one tool;
3. `run_tools(code)` — a Python-subset script, executed server-side by the
   sandboxed interpreter in `src/tools/base.py`, where registered tools are
   called by bare name:

   ```python
   status = job_status("j1")
   while status == "running":
       sleep(5)                # token-free polling
       status = job_status("j1")
   pages = parallel(web_search("https://a"), web_search("https://b"))
   return [page[:200] for page in pages]
   ```

The interpreter whitelists AST nodes, builtins, and methods (no imports, no
dunder access), enforces op/sleep/wall-clock/memory budgets, and returns
`{result, logs, error}` to the model. Intermediate tool output stays inside
the script; only what it `return`s or `print`s reaches the conversation.

## Computer use

`ComputerUse` (`src/tools/computer_use/`) combines a vision locator with mouse
and keyboard control. Detection runs on `nvidia/LocateAnything-3B`, which is
loaded lazily on the first query and reused afterwards.

```python
from src.tools.computer_use import ComputerUse

computer = ComputerUse()

computer.get_screenshot()                       # capture the primary display
computer.locate_object("the render button")     # -> [Detection(label, box)]
computer.locate_point("the render button")      # -> clickable (x, y)

computer.left_click(x, y)                       # also right/middle/double/triple
computer.click_object("the save button")        # locate by description, then click
computer.type_text("hello")                     # unicode-safe typing
computer.key("ctrl+shift+s")                    # shortcuts, space-separated sequences
computer.scroll("down", 3)
computer.drag((100, 200), (400, 200))

computer.mark_object(                           # show, don't touch
    "кнопка рендера",
    "Запускает просчёт таймлайна. Нажатие начнёт экспорт.",
)
computer.close()                                # release the overlay window
```

`mark_object` is the "just show me where it is" path: it moves the pointer onto
the element, outlines it, and shows a click-through tooltip next to the cursor
with your explanation. It never clicks.

### Locating accurately

The screen is squeezed to 768 px before the model sees it, so a 42-px text
field arrives as a 17-px smear and the returned box drifts by ~100 px — enough
to click outside the element. `locate_object` therefore runs a coarse pass over
the whole screen to find the neighbourhood, then a second pass over a crop of it
at native resolution: measured 99 px → 5 px on that field.

Pass `region=` whenever the neighbourhood is already known (an app window, a
panel found earlier). A region that fits in 768 px is searched at native
resolution in a single pass — both faster and as accurate:

```python
panel = computer.find_object("the export panel")
computer.click_object("the render button", region=panel.box)
```

Repeating a query while the screen has not changed is served from cache without
touching the model. Decoding is greedy: on queries the model is unsure about,
sampling returned five different boxes in five runs.

Module layout:

- `base.py` — the `ComputerUse` facade;
- `locator.py` — LocateAnything runtime (the only module that imports torch);
- `screen.py` — DPI awareness and primary-display capture;
- `pointer.py` — Win32 `SendInput` mouse and keyboard control;
- `overlay.py` — the Tk marker overlay, on a single long-lived Tk loop;
- `detection.py` — model-output parsing and box geometry;
- `annotation.py` — annotated JPEG output;
- `fakes.py` — inert backends for dry runs, headless use, and tests.

Platform notes: pointer control and DPI awareness are Windows-only; detection,
parsing, and annotation are platform-independent. Every OS call sits behind an
injected seam, so tests never touch the real desktop.

### Interactive REPL

```bash
uv run test.py
```

Commands: `:mode first`, `:mode all`, `:mark <описание> | <подсказка>`,
`:click <описание>`, `exit`.

## Extension

- **New agent:** Add a method to `src/llm/agents.py` that calls `self.ask_agent()` with a custom system prompt, tool list, and model.
- **New tool:** Define a `@tool`-decorated async function in `src/llm/tools.py`, connect any logic from `src/tools/`, and register it in the default registry (`get_registry()` in `src/llm/meta_tools.py`) so the agent can discover it via `search_tools` and call it from `run_tools` scripts.

## Development

- Environment manager: `uv`.
- Logging: `loguru` writes to `logs.log`.
- To run tests (if any), use standard pytest/uv commands:
  ```bash
  uv run python -m pytest
  ```

## Additional Info

Please submit PRs and discussions via GitHub. Ensure your `.env` is excluded from the repository and `.gitignore` is in use.
