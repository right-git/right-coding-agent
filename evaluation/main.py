"""REPL for the direct-wiring agent: `uv run python -m evaluation.main`.

Identical to `src.main` — same chat loop, usage footer, catalog, and vision
preload — except the agent sends every tool schema with every model call.
Give the same task here and to `uv run python -m src.main`, then compare the
session lines of the two footers.
"""

import asyncio

from src.config.logging import logger
from src.config.settings import settings
from src.llm.providers import OpenRouterCatalog
from src.llm.tools import META_TOOLS
from src.llm.types import LLMProvider
from src.llm.statistics import SessionUsage
from src.main import available_models, preload_vision_model, process_user_turn
from src.ui import ChatUI

from .direct_agent import DirectAgents
from .direct_tools import DIRECT_TOOLS, schema_token_estimate


def print_architecture_banner(ui: ChatUI) -> None:
    direct_estimate = schema_token_estimate(DIRECT_TOOLS)
    meta_estimate = schema_token_estimate(META_TOOLS)
    ui.console.print(
        "  architecture: DIRECT — every tool schema is sent with every " "model call",
        style="info",
    )
    ui.console.print(
        f"  schema cost per call: direct {len(DIRECT_TOOLS)} tools "
        f"~{direct_estimate:,} tokens · meta {len(META_TOOLS)} tools "
        f"~{meta_estimate:,} tokens",
        style="info",
    )
    ui.console.print(
        "  the meta agent runs with: uv run python -m src.main",
        style="info",
    )
    ui.console.print()


async def main():
    agents = DirectAgents(
        [
            LLMProvider(
                provider_name="openai",
                api_key=settings.llm_api_key,
                api_base=settings.llm_api_base,
            )
        ]
    )

    catalog = OpenRouterCatalog()
    session_usage = SessionUsage()

    messages = []
    model = available_models[0]
    ui = ChatUI(model=model, available_models=available_models)
    ui.print_welcome()
    print_architecture_banner(ui)

    async def load_catalog() -> None:
        ui.set_model_catalog(await catalog.models())

    catalog_task = asyncio.create_task(load_catalog())
    vision_task = asyncio.create_task(asyncio.to_thread(preload_vision_model))
    logger.info("Started the DIRECT-architecture evaluation REPL")

    try:
        while True:
            user_content = await ui.get_input()

            if not user_content.strip():
                continue

            if catalog_task.done() and not ui.model_catalog:
                ui.set_model_catalog(await catalog.models())

            if user_content.startswith("/"):
                result = ui.handle_command(user_content)
                if result == "clear":
                    messages = []
                    session_usage = SessionUsage()
                    print_architecture_banner(ui)
                model = ui.model
                continue

            messages = await process_user_turn(
                agents=agents,
                ui=ui,
                messages=messages,
                model=model,
                user_content=user_content,
                catalog=catalog,
                session_usage=session_usage,
            )
    finally:
        catalog_task.cancel()
        vision_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
