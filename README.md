# Right Coding Agent

`right-coding-agent` is an asynchronous agent supporting multiple model providers, built on LangChain and designed as a programmer's assistant.

## Overview

- Multi-provider LLM client with retry/failover logic, Azure OpenAI support, and structured output via LangChain.
- Built-in agent and tools: `Agents` adds system prompts, `LLMClient` manages model/tool initialization, and `src/llm/tools/` contains `@tool`-decorated functions.
- Scripted tool orchestration: the agent sees only three meta tools (`search_tools`, `get_tool`, `run_tools`) and drives every other tool from a sandboxed Python-subset script — flat context cost, parallel calls, and token-free polling (see below).
- The `WebParser` tool (`src/llm/tools/parser/`) fetches web pages over HTTP, parses them with BeautifulSoup, and converts to Markdown.
- The `ComputerUse` tool (`src/llm/tools/computer/`) gives the agent eyes and hands on the desktop: it finds on-screen elements from a plain-language description, drives the mouse and keyboard, and can point at an element with an on-screen tooltip instead of clicking it.

## Installation

### Prerequisites

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — the only tool you install yourself. It manages the virtual environment, every dependency, **and Python itself**: the first `uv sync` downloads Python 3.12 (pinned in `.python-version`), so you do not need a system Python.
- **[Git](https://git-scm.com/downloads)** — to clone the repository.
- **NVIDIA GPU driver** (optional, Windows/Linux) — for GPU inference of the vision locator. The project uses CUDA 12.8 PyTorch wheels, which require a driver from the R570 series or newer: download at <https://www.nvidia.com/drivers>. You do **not** need to install the CUDA Toolkit — the wheels bundle the CUDA runtime; only the driver matters. Without an NVIDIA GPU everything still works on CPU (and on Apple Silicon the locator runs on Metal/MPS).

The right PyTorch build is selected **automatically** per OS — `[tool.uv.sources]` in `pyproject.toml` routes `torch`/`torchvision` to the `pytorch-cu128` index on Windows, while Linux and macOS use the regular PyPI wheels (CUDA-bundled on Linux, MPS on macOS). No manual `pip install torch` on any platform.

### Windows

```powershell
# 1. Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone the project
git clone <repo-url> right-code
cd right-code

# 3. Create .env (see below)

# 4. Install everything — pulls the CUDA 12.8 PyTorch build automatically
uv sync

# 5. Run the agent
uv run python -m src.main
```

### Linux

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the project
git clone <repo-url> right-code
cd right-code

# 3. Create .env (see below)

# 4. Install everything — PyPI torch bundles CUDA on Linux; CPU works without a GPU
uv sync

# 5. Run the agent
uv run python -m src.main
```

The screen tools expect an X11 session; on Wayland, capture works through `mss`, input support varies by compositor. The clipboard needs `xclip` or `xsel`; window focus needs `wmctrl` (plus `xdotool` to read the active window title).

### macOS

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the project
git clone <repo-url> right-code
cd right-code

# 3. Create .env (see below)

# 4. Install everything — torch runs on Metal (MPS); CUDA does not exist on macOS
uv sync

# 5. Run the agent
uv run python -m src.main
```

For the screen tools, grant your terminal two permissions in **System Settings → Privacy & Security**: **Accessibility** (mouse/keyboard control) and **Screen Recording** (capture).

### .env

Create a `.env` file at the repo root before the first run — importing `src.config.settings` fails without it:

```env
ENV=dev
LLM_API_KEY=<your key>
LLM_API_BASE=<OpenAI-compatible endpoint, e.g. https://openrouter.ai/api/v1>
LLM_DEFAULT_MODEL=google/gemini-3.7-flash   # optional: the model the chat starts with
```

### Verify the install

```bash
uv run python -m unittest discover -s tests   # runs in seconds; needs no GPU or desktop
bash lint.sh                                  # black + flake8
```

### Platform support

| Capability | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Chat agent, web parser, run_tools sandbox | ✅ | ✅ | ✅ |
| Vision locator | ✅ CUDA / CPU | ✅ MPS / CPU | ✅ CUDA / CPU |
| Mouse + keyboard (`screen_click`, `screen_type`, `screen_key`) | ✅ native (SendInput) | ✅ portable (pynput) | ✅ portable (pynput, X11) |
| Screen capture | ✅ | ✅ (needs permission) | ✅ (X11 / Wayland via mss) |
| Clipboard (copy / paste) | ✅ Win32 | ✅ | ✅ (xclip / xsel) |
| Window focus (`focus_window`) | ✅ | ✅ AppleScript | ✅ wmctrl |

The OS backends live in `src/llm/tools/computer/platforms/` — native Win32 under `windows/`, a pynput/mss/pyperclip fallback under `portable/` — and are picked automatically at runtime.

## Architecture evaluation

`evaluation/` holds a control-group agent for measuring the meta-tool
architecture: the same agent with every tool schema wired directly into the
model (`uv run python -m evaluation.main`). Give the same task to it and to
the production agent, then compare the usage footers — see
`evaluation/README.md` for the methodology.

## Chat commands and usage footer

- `/models [query]` — list the curated models, or search the whole OpenRouter catalog; search only offers **tool-capable** models (the agent's workflow is tool calls) and tags reasoning-capable ones;
- `/model <name> [effort]` — switch models by exact or partial match over the curated list **and** the catalog; a model the catalog marks as unable to call tools is refused; an optional trailing effort (`/model o5 high`) sets reasoning effort in the same command;
- `/effort [minimal|low|medium|high|none]` — show or change the reasoning effort of the current model (refused when the catalog says the model has no reasoning support);
- `/temperature <0..2|none>` (alias `/temp`) — show or change the sampling temperature;
- `/log-level [name]`, `/clear`, `/help`, `/quit`.

Commands autocomplete as you type: `/` suggests command names, `/model ` suggests tool-capable model ids straight from the catalog (with ctx/price shown), `/effort` its levels, `/log-level` its levels.

Session settings (model, effort, temperature) apply from the next turn on and are shown after every `/model` switch and in `/models`. "default" means the parameter is not sent — the provider decides; when OpenRouter publishes the model's own default temperature it is shown in parentheses, e.g. `temperature default (1)`.

While a turn runs, the terminal shows live progress, Claude-Code-style: a
stopwatch for the current phase (`thinking… 12s` / `running tools… 3s`), every
tool call and its result printed the moment they happen (results carry their
duration), a dim `✻ thought for Ns` line before each action, and a tail of the
answer as it streams in.

After every response a dim footer reports what the turn cost:

```
ctx █░░░░░░░░░░░░░░░░░░░ 14,204/1,048,576 (1.4%) · turn 13,900 in + 304 out ($0.0058) · took 26s · tools 2 (+5 in scripts) · session 28,400 tokens ($0.0116, 1m 12s)
```

The context bar is colored by fill: green below 70%, yellow below 90%, red
above. `took` is the wall-clock time this turn spent processing (model calls,
tools, everything); the session parenthesis accumulates it. `tools` counts
the model's direct tool calls this turn, plus the registry tools its
`run_tools` scripts invoked internally; the segment is omitted on turns that
called no tools.

That is: how full the current model's context window is, tokens and dollars
for this turn, and session totals. Token counts come from the provider's
`usage_metadata`; context length and per-token prices come from OpenRouter
(fetched once in the background at startup, cached for the session, and the
footer degrades gracefully — `limit unknown` / `price unknown` — when the
catalog is unavailable).

## Project Structure

- `src/main.py` — entry point; configures `Agents` with the model provider and starts the REPL interface.
- `src/config/` — `settings.py` (pydantic-settings, `.env` loading), `logging.py` (loguru setup, logs to `logs.log`), `prompts.py` (system prompts).
- `src/llm/` — agent core:
  - `client.py` manages the LLM client; `agents.py` implements specific agents (such as `right_coding_agent()`);
  - `types.py` — `LLMProvider`, `TurnUsage`, shared type aliases; `utils.py` — client-side helpers (history hygiene, formatting);
  - `statistics/` — accounting and instrumentation: `usage.py` (token/cost per turn and session), `script_calls.py` (script tool-call counter), future metrics;
  - `providers/` — provider integrations (`openrouter.py`: model catalog with context windows and pricing);
  - `middlewares/` — agent middlewares (`attachments.py`: screenshots as vision messages; `message_log.py`: JSON request/response logging);
  - `tools/` — the tool layer, one subpackage per concern:
    - `meta/` — the meta layer: `tool.py` (the `search_tools` / `get_tool` / `run_tools` meta tools), `registry.py` + `defaults.py` (the tool registry), `attachments.py` (the image channel out of a run), `sandbox/` (the Python-subset interpreter executing `run_tools` scripts);
    - `parser/` — web-page fetching (`service.py`: the `WebParser` class; `utils.py`; `tool.py`: the `web_search` `@tool`);
    - `computer/` — screen understanding and desktop control (`service.py`: the `ComputerUse` facade; `tool.py`: the `screen_*` `@tool`s; `platforms/`: per-OS backends — native `windows/`, portable `portable/` for macOS/Linux — see below).
- `test.py` — interactive screen-locator REPL built on `ComputerUse`.
- Tests are located in `tests/`.

## Tool orchestration

Exposing every tool schema to the model bloats the context and forces one
round-trip per call. Instead, only three meta tools are wired into the agent
(`src/llm/tools/meta/tool.py`):

1. `search_tools("click button screen")` — keyword search over the tool
   registry (currently `web_search` plus the `screen_*` tools);
2. `get_tool(["screen_click", "screen_type"])` — full contracts of one or
   more tools in a single call;
3. `run_tools(code)` — a Python-subset script, executed server-side by the
   sandboxed interpreter in `src/llm/tools/meta/sandbox/`, where registered tools are
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

### Seeing the screen

Base64 in a tool's text output is invisible to an LLM — providers only read
images from `image_url` content blocks. So screenshot-capturing tools
(`screen_screenshot`, `screen_locate` with `return_screen=True`) *attach*
their image instead: it rides out of `run_tools` as the tool message's
artifact, and a middleware injects it into the conversation as a proper
vision message right after the tool result, so the model actually sees the
picture. Raw base64 is still available (`screen_screenshot(return_base64=True)`,
or automatically when the tools are used outside the agent). The
LocateAnything vision model is preloaded in the background at startup so the
first screen query doesn't pay the load time.

## Computer use

`ComputerUse` (`src/llm/tools/computer/`) combines a vision locator with mouse
and keyboard control. Detection runs on `nvidia/LocateAnything-3B`, which is
loaded lazily on the first query and reused afterwards.

```python
from src.llm.tools.computer import ComputerUse

computer = ComputerUse()

computer.get_screenshot()                       # capture the primary display
computer.screenshot_base64()                    # base64 JPEG, downscaled to ≤1280px
computer.annotated_base64()                     # last view with detection boxes drawn
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

Detection reports include a coarse position per match (`at top-left`,
`at bottom-center`), and the agent-facing `screen_click` / `screen_mark`
tools refuse to act on an ambiguous description ("the input field" on a
screen with an address bar *and* a search box): they list the candidates
with positions and require either a more concrete description or an explicit
`match=<n>` pick — which is deterministic, because the unchanged screen is
served from cache in the same order.

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
- **New tool:** Create a subpackage under `src/llm/tools/` following the `parser/` / `computer/` pattern — `service.py` with the class doing the real work, an optional `utils.py`, and `tool.py` with the `@tool`-decorated async functions the LLM receives — then register the tools in the default registry (`get_registry()` in `src/llm/tools/meta/defaults.py`) so the agent can discover them via `search_tools` and call them from `run_tools` scripts.

## Development

- Environment manager: `uv`.
- Logging: `loguru` writes to `logs.log`. Every model request is also logged
  there as one JSON line with base64/data-URI payloads stripped
  (`MessageLogMiddleware`).
- Vision-model loading noise (transformers warnings, progress bars) is muted
  per-thread (`src/utils/silence.py`) so background loads never print over
  the prompt.
- To run tests (if any), use standard pytest/uv commands:
  ```bash
  uv run python -m pytest
  ```

## Additional Info

Please submit PRs and discussions via GitHub. Ensure your `.env` is excluded from the repository and `.gitignore` is in use.
