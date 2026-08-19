"""Provider-specific integrations, one package/module per provider.

OpenRouter's model catalog (context windows and pricing) lives here, and
`reasoning.py` holds the OpenAI-compatible client quirk of keeping provider
reasoning deltas; a future `anthropic.py` belongs alongside them.
"""

from .openrouter import ModelInfo, OpenRouterCatalog, parse_models
from .reasoning import ReasoningChatOpenAI, reasoning_delta

__all__ = [
    "ModelInfo",
    "OpenRouterCatalog",
    "parse_models",
    "ReasoningChatOpenAI",
    "reasoning_delta",
]
