"""Provider-specific integrations, one package/module per provider.

Only OpenRouter lives here for now (model catalog: context windows and
pricing); a future `openai.py` / `anthropic.py` with provider-specific
client quirks belongs alongside it.
"""

from .openrouter import ModelInfo, OpenRouterCatalog, parse_models

__all__ = ["ModelInfo", "OpenRouterCatalog", "parse_models"]
