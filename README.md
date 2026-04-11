# right_code

`right_code` is an asynchronous agent supporting multiple model providers, built on LangChain and designed as a programmer's assistant.

## Overview

- Multi-provider LLM client with retry/failover logic, Azure OpenAI support, and structured output via LangChain.
- Built-in agent and tools: `Agents` adds system prompts, `LLMClient` manages model/tool initialization, and `src/llm/tools.py` contains `@tool`-decorated functions.
- The `WebParser` tool (`src/tools/web_parser/`) fetches web pages over HTTP, parses them with BeautifulSoup, and converts to Markdown.

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

## Project Structure

- `src/main.py` — entry point; configures `Agents` with the model provider and starts the REPL interface.
- `src/config/base.py` — configuration via `pydantic-settings`, `.env` loading, and `loguru` setup (logs to `logs.log`).
- `src/config/prompts.py` — stores system prompts.
- `src/llm/` — agent core:
  - `base.py` manages the LLM client;
  - `agents.py` implements specific agents (such as `right_code()`);
  - `tools.py` — utility LangChain tools.
- `src/tools/web_parser/` — standalone HTTP client and HTML-to-Markdown parser.
- Tests are located in `tests/`.

## Extension

- **New agent:** Add a method to `src/llm/agents.py` that calls `self.ask_agent()` with a custom system prompt, tool list, and model.
- **New tool:** Define a `@tool`-decorated async function in `src/llm/tools.py`, and connect any logic from `src/tools/`.

## Development

- Environment manager: `uv`.
- Logging: `loguru` writes to `logs.log`.
- To run tests (if any), use standard pytest/uv commands:
  ```bash
  uv run python -m pytest
  ```

## Additional Info

Please submit PRs and discussions via GitHub. Ensure your `.env` is excluded from the repository and `.gitignore` is in use.
