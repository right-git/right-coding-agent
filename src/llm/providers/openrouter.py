"""Model metadata — context window and prices — from the public OpenRouter API.

One unauthenticated GET returns every model OpenRouter serves, including
`context_length` and per-token USD pricing. The catalog caches that list for
the session (with a cooldown after failures, so an offline machine is not
hammered once per turn) and is safe to consult after every model call.
"""

import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.config.logging import logger

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_TTL = 3600.0
FAILURE_COOLDOWN = 60.0
REQUEST_TIMEOUT = 8.0


@dataclass(frozen=True)
class ModelInfo:
    """One model's metadata; prices are USD per single token."""

    id: str
    name: str
    context_length: int | None
    prompt_price: float | None
    completion_price: float | None

    def cost_of(self, input_tokens: int, output_tokens: int) -> float | None:
        """Dollar cost of one call, or None when pricing is unknown."""
        if self.prompt_price is None or self.completion_price is None:
            return None
        return input_tokens * self.prompt_price + output_tokens * self.completion_price


def _parse_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    # OpenRouter marks dynamically priced routes with negative sentinels.
    return price if price >= 0 else None


def _parse_context_length(entry: dict) -> int | None:
    raw = entry.get("context_length")
    if raw is None:
        raw = (entry.get("top_provider") or {}).get("context_length")
    try:
        length = int(raw)
    except (TypeError, ValueError):
        return None
    return length if length > 0 else None


def parse_models(payload: Any) -> dict[str, ModelInfo]:
    """The `{"data": [...]}` payload as a mapping of model id to ModelInfo."""
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}

    models: dict[str, ModelInfo] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        pricing = entry.get("pricing") or {}
        models[model_id] = ModelInfo(
            id=model_id,
            name=str(entry.get("name") or model_id),
            context_length=_parse_context_length(entry),
            prompt_price=_parse_price(pricing.get("prompt")),
            completion_price=_parse_price(pricing.get("completion")),
        )
    return models


async def _fetch_models_payload(url: str) -> Any:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


class OpenRouterCatalog:
    """Cached view of OpenRouter's model list.

    A successful fetch is reused for `ttl` seconds; a failed one keeps
    whatever was cached before and blocks retries for `failure_cooldown`
    seconds, so callers may ask on every turn without paying for a dead
    network each time.
    """

    def __init__(
        self,
        *,
        url: str = OPENROUTER_MODELS_URL,
        fetch_payload=None,
        ttl: float = DEFAULT_TTL,
        failure_cooldown: float = FAILURE_COOLDOWN,
        clock=time.monotonic,
    ) -> None:
        self._url = url
        self._fetch_payload = fetch_payload or _fetch_models_payload
        self._ttl = ttl
        self._failure_cooldown = failure_cooldown
        self._clock = clock
        self._models: dict[str, ModelInfo] = {}
        self._fetched_at: float | None = None
        self._failed_at: float | None = None

    async def models(self) -> dict[str, ModelInfo]:
        now = self._clock()
        if self._fetched_at is not None and now - self._fetched_at < self._ttl:
            return self._models
        if self._failed_at is not None and now - self._failed_at < self._failure_cooldown:
            return self._models

        try:
            parsed = parse_models(await self._fetch_payload(self._url))
        except Exception as error:
            self._failed_at = self._clock()
            logger.warning("OpenRouter model catalog fetch failed: {}", error)
            return self._models

        if not parsed:
            self._failed_at = self._clock()
            logger.warning("OpenRouter model catalog came back empty")
            return self._models

        self._models = parsed
        self._fetched_at = self._clock()
        self._failed_at = None
        logger.info("Loaded [{}] models from OpenRouter", len(parsed))
        return self._models

    async def get(self, model_id: str) -> ModelInfo | None:
        return (await self.models()).get(model_id)
